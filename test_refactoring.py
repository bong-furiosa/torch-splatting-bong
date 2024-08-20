#!/usr/bin/env python
import time
import math
import glm
import torch
import numpy as np
import gaussian_splatting.util_gau as util_gau
from gaussian_splatting.utils.sh_utils import eval_sh
from gaussian_splatting.gauss_render import homogeneous
import gaussian_splatting.utils as utils

class Camera:
    def __init__(self, h, w, position_x, position_y, position_z):
        self.znear = 0.01
        self.zfar = 100
        self.image_height = self.h = h
        self.image_width = self.w = w
        self.fovy = np.pi / 2

        self.FoVy = np.pi/2
        tany = math.tan(self.fovy/2)
        tanx = tany/self.h*self.w
        self.FoVx = math.atan(tanx)*2
        self.focal_x = self.w / (2*np.tan(self.FoVx/2))
        self.focal_y = self.h / (2*np.tan(self.FoVy/2))

        self.position = torch.tensor([position_x, position_y, position_z], dtype=torch.float32).cuda()
        self.target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).cuda()
        self.up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32).cuda()
        self.yaw = -np.pi / 2
        self.pitch = 0

        self.is_pose_dirty = True
        self.is_intrin_dirty = True

        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True

        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False

        self.rot_sensitivity = 0.02
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity = 0.08
        self.roll_sensitivity = 0.03
        self.target_dist = 3.

        self.world_view_transform = torch.from_numpy(self.get_view_matrix()).T.cuda()
        self.projection_matrix = torch.from_numpy(self.get_project_matrix()).T.cuda()

    def get_view_matrix(self):
        view_matrix = np.array(glm.lookAt(self.position.cpu().numpy(), self.target.cpu().numpy(), self.up.cpu().numpy()))
        scale_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]],dtype=np.float32)
        return np.matmul(scale_matrix, view_matrix)

    def get_project_matrix(self):
        # htanx, htany, focal = self.get_htanfovxy_focal()
        # f_n = self.zfar - self.znear
        # proj_mat = np.array([
        #     1 / htanx, 0, 0, 0,
        #     0, 1 / htany, 0, 0,
        #     0, 0, self.zfar / f_n, - 2 * self.zfar * self.znear / f_n,
        #     0, 0, 1, 0
        # ])
        project_mat = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(project_mat).astype(np.float32)


###################################################
# PROJECTION PHASE FUNCTION
def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    # (bong-furisoa)
    # 영역 밖의 불필요한 Gaussian이 포함되면 색깔 계산에 잘못 누적되는 것이 아닌가 생각이 들었습니다.
    # 그래서 in_mask를 부활시키겠습니다.
    in_mask = p_view[..., 2] >= 0.2

    return p_proj, p_view, in_mask

###################################################
# BUILD COLOR PHASE FUNCTION
def build_color(ply, means3D, shs, camera):
    active_sh_degree = int(np.round(np.sqrt(ply.sh_dim/3)))-1
    rays_o = camera.position # torch.from_numpy(camera.position).cuda()   
    rays_d = means3D - rays_o
    color = eval_sh(active_sh_degree, shs.permute(0,2,1), rays_d)
    color = (color + 0.5).clip(min=0.0)

    return color

###################################################
# (Preprocess) BUILD COV3D PHASE FUNCTION
def build_rotation(r):
    #norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    #q = r / norm[:, None]
    q = r

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L
def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

###################################################
# BUILD COV2D PHASE FUNCTION
def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3

    return cov2d[:, :2, :2] + filter[None]

###################################################
# RENDERING PHASE FUNCTION
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max

def render(camera, means2D, cov2d, color, opacity, depths):
    radii = get_radius(cov2d)
    rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)

    pix_coord = torch.stack(torch.meshgrid(torch.arange(camera.image_width), torch.arange(camera.image_height), indexing='xy'), dim=-1).to('cuda')
    
    render_color = torch.ones(*pix_coord.shape[:2], 3).to('cuda')
    render_depth = torch.zeros(*pix_coord.shape[:2], 1).to('cuda')
    render_alpha = torch.zeros(*pix_coord.shape[:2], 1).to('cuda')

    TILE_SIZE = 32
    BATCH_SIZE = 8
    tile_coords_batch = []
    h_batch = []
    w_batch = []

    for h in range(0, camera.image_height, TILE_SIZE):
        for w in range(0, camera.image_width, TILE_SIZE):
            tile_coord = pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
            tile_coords_batch.append(tile_coord)
            h_batch.append(h)
            w_batch.append(w)
                                    
            if len(tile_coords_batch) == BATCH_SIZE:
                h_batch_tensor = torch.tensor(h_batch).view(BATCH_SIZE, 1, 1)
                w_batch_tensor = torch.tensor(w_batch).view(BATCH_SIZE, 1, 1)

                over_tl = rect[0][..., 0].clip(min=w_batch_tensor), rect[0][..., 1].clip(min=h_batch_tensor)
                over_br = rect[1][..., 0].clip(max=w_batch_tensor + TILE_SIZE - 1), rect[1][..., 1].clip(max=h_batch_tensor + TILE_SIZE - 1)

                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 

                P = in_mask.sum(dim=-1) # in_mask 1개 shape = torch.Size([2634990])
                max_P = P.max()

                padded_depths = torch.full((BATCH_SIZE, max_P), torch.inf, device=depths.device)
                padded_means2D = torch.ones((BATCH_SIZE, max_P, 2), device=means2D.device)
                # padded_cov2d = torch.randn((BATCH_SIZE, max_P, 2, 2), device=cov2d.device)
                padded_cov2d = torch.eye(2).expand(BATCH_SIZE, max_P, 2, 2).clone()
                padded_opacity = torch.zeros((BATCH_SIZE, max_P), device=opacity.device)
                padded_color = torch.zeros((BATCH_SIZE, max_P, 3), device=color.device)

                for i in range(BATCH_SIZE):
                    padded_depths[i, :P[i]] = depths[in_mask[i, 0]]
                    padded_means2D[i, :P[i]] = means2D[in_mask[i, 0]]
                    padded_cov2d[i, :P[i]] = cov2d[in_mask[i, 0]]
                    padded_opacity[i, :P[i]] = opacity[in_mask[i, 0]].view(-1)
                    padded_color[i, :P[i]] = color[in_mask[i, 0]]
               
                sorted_depths, indices = torch.sort(padded_depths, dim=-1)
                sorted_means2D = torch.gather(padded_means2D, 1, indices.unsqueeze(-1).expand(BATCH_SIZE, -1, 2))
                sorted_cov2d = torch.gather(padded_cov2d, 1, indices.unsqueeze(-1).unsqueeze(-1).expand(BATCH_SIZE, -1, 2, 2))
                sorted_conic = sorted_cov2d.inverse()
                sorted_opacity = torch.gather(padded_opacity, 1, indices)
                sorted_color = torch.gather(padded_color, 1, indices.unsqueeze(-1).expand(BATCH_SIZE, -1, 3))

                tile_coords_batch = torch.stack(tile_coords_batch) # shape: (BATCH_SIZE, TILE_SIZE^2, 2)
                dx = (-tile_coords_batch[:, :, None, :] + sorted_means2D[:, None, :])
                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, :, 0]**2 * sorted_conic[:, :, 0, 0].unsqueeze(1) +
                    dx[:, :, :, 1]**2 * sorted_conic[:, :, 1, 1].unsqueeze(1) +
                    dx[:, :, :, 0] * dx[:, :, :, 1] * sorted_conic[:, :, 0, 1].view(-1, 1, max_P) +
                    dx[:, :, :, 0] * dx[:, :, :, 1] * sorted_conic[:, :, 1, 0].view(-1, 1, max_P)
                ))

                alpha = (gauss_weight * sorted_opacity[:, None, :]).clip(max=0.99)  # shape: (BATCH_SIZE, TILE_SIZE^2, max_P)
                T = torch.cat([torch.ones_like(alpha[:, :, :1]), 1 - alpha[:, :, :-1]], dim=2).cumprod(dim=2)
                acc_alpha = (alpha * T).sum(dim=2)
                
                tile_color = ((T * alpha).unsqueeze(-1) * sorted_color.unsqueeze(1).expand(-1, TILE_SIZE * TILE_SIZE, -1, -1)).sum(dim=2)
                tile_depth = ((T * alpha) * sorted_depths.unsqueeze(1).expand(-1, TILE_SIZE * TILE_SIZE, -1)).sum(dim=2)

                # render 결과 업데이트
                for idx, (_h, _w) in enumerate(zip(h_batch_tensor, w_batch_tensor)):
                    render_color[_h:_h+TILE_SIZE, _w:_w+TILE_SIZE] = tile_color[idx].view(TILE_SIZE, TILE_SIZE, 3)
                    render_depth[_h:_h+TILE_SIZE, _w:_w+TILE_SIZE] = tile_depth[idx].view(TILE_SIZE, TILE_SIZE, 1)
                    render_alpha[_h:_h+TILE_SIZE, _w:_w+TILE_SIZE] = acc_alpha[idx].view(TILE_SIZE, TILE_SIZE, 1)

                # (bong-furiosa)
                # 다음 배치를 위해 초기화합니다.
                tile_coords_batch = []
                h_batch = []
                w_batch = []

    # (bong-furiosa)
    # TODO: 마지막 남은 타일들 처리 (배치 크기보다 작을 때)
    if tile_coords_batch:
        # tile_coords_batch = torch.stack(tile_coords_batch)
        '''to do'''

    return {
        "render": render_color,
        "depth": render_depth,
        "alpha": render_alpha,
        "visiility_filter": radii > 0,
        "radii": radii
    }


def main(ply, means3D, opacity, scales, rotations, cov3d, shs, camera):
    ###################################################
    # PROJECTION PHASE FUNCTION
    mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
            viewmatrix=camera.world_view_transform, 
            projmatrix=camera.projection_matrix)
    # (bong-furisoa)
    # Gaussian Sphere들을 in_mask를 사용해서 필터링하겠습니다.
    mean_ndc = mean_ndc[in_mask] # mean_ndc = mean_ndc * in_mask.unsqueeze(1) 
    mean_view = mean_view[in_mask] # mean_view = mean_view * in_mask.unsqueeze(1)
    depths = mean_view[:,2]

    means3D = means3D[in_mask] # means3D = means3D * in_mask.unsqueeze(1)
    opacity = opacity[in_mask] # opacity = opacity * in_mask.unsqueeze(1)
    # cov3d = cov3d[in_mask] # cov3d는 연산 순서를 지키기 위해, 일단, covariance 3d를 계산하는 곳에서 in_mask 처리할 것!
    shs = shs[in_mask] # shs = shs * in_mask.unsqueeze(1).unsqueeze(1)

    ###################################################
    # BUILD COLOR PHASE
    color = build_color(ply=ply, means3D=means3D, shs=shs, camera=camera)

    ###################################################
    # BUILD COV3D PHASE
    # TODO: main 함수 외부로 빼내기 -> ✅
    # cov3d = build_covariance_3d(scales, rotations)
    cov3d = cov3d[in_mask] # cov3d = cov3d * in_mask.unsqueeze(1).unsqueeze(1)

    ###################################################
    # BUILD COV2D PHASE
    cov2d = build_covariance_2d(
        # (bong-furisoa)
        # Gaussian Sphere들을 in_mask를 사용해서 필터링하겠습니다.
        mean3d=means3D, 
        cov3d=cov3d, 
        viewmatrix=camera.world_view_transform,
        fov_x=camera.FoVx, 
        fov_y=camera.FoVy, 
        focal_x=camera.focal_x, 
        focal_y=camera.focal_y)

    mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
    mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
    means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)    

    ###################################################
    # RENDERING PHASE
    out = render(
                    camera = camera, 
                    means2D=means2D,
                    cov2d=cov2d,
                    color=color,
                    opacity=opacity, 
                    depths=depths,
                )

    return out

if __name__ == "__main__":
    torch.set_default_device("cuda:0")
    ply = util_gau.load_ply('bicycle.ply')
    means3D = torch.from_numpy(ply.xyz).cuda()
    opacity = torch.from_numpy(ply.opacity).cuda()
    scales = torch.from_numpy(ply.scale).cuda()
    rotations = torch.from_numpy(ply.rot).cuda()
    shs = torch.from_numpy(ply.sh.reshape((rotations.shape[0], -1, 3))).cuda()
    
    cov3d = build_covariance_3d(scales, rotations)
    
    positions_z = np.arange(-3.0, 3.0 + 0.03, 0.06)
    camera_positions = [(0.0, 0.0, z) for z in positions_z]    

    for idx, camera_position in enumerate([camera_positions[0], camera_positions[40], camera_positions[-41], camera_positions[-1]]):
        # (bong-furiosa)
        # TODO: 렌더링 이미지 크기를 TILE_SIZE = 32의 배수로 맞추어야 오류가 발생하지 않음
        # 해당 문제를 버그로 여기고, 추후 다양한 이미지 크기에 대해서도 정상적으로 동작하게 수정할 것!
        camera = Camera(736,1280, camera_position[0], camera_position[1], camera_position[2])
        out = main(ply, means3D, opacity, scales, rotations, cov3d, shs, camera)
        image = out['render'].detach().cpu().numpy()
        utils.imwrite(str(f'./refactoring_results/test_{idx:03d}.png'), image)