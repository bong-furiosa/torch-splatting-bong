#!/usr/bin/env python
import gaussian_splatting.util_gau as util_gau
from PIL import Image
import math
import glm
import torch
import numpy as np
from gaussian_splatting.utils.sh_utils import eval_sh
from gaussian_splatting.gauss_render import homogeneous
import gaussian_splatting.utils as utils


#ply = util_gau.load_ply("/home/bongwon/Desktop/3DGS_playground/GaussianSplattingViewer/models/bicycle/point_cloud/iteration_7000/point_cloud.ply")
ply = util_gau.load_ply('bicycle.ply')
#ply = util_gau.naive_gaussian()

means3D = torch.from_numpy(ply.xyz).cuda()
opacity = torch.from_numpy(ply.opacity).cuda()
scales = torch.from_numpy(ply.scale).cuda()
rotations = torch.from_numpy(ply.rot).cuda()
shs = torch.from_numpy(ply.sh.reshape((rotations.shape[0], -1, 3))).cuda()
# (bong-furiosa)
# 기존 torch-splatting의 입력 값을 확인했을 때, dim=1에서 맨 앞만 유효한 shs 값이고 나머지는 전부 0입니다.
shs[:, 1:, :] = 0

class Camera:
    def __init__(self, h, w):
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

        self.position = np.array([0.0, 0.0, 3.0]).astype(np.float32)
        self.target = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        self.up = np.array([0.0, -1.0, 0.0]).astype(np.float32)
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

        self.world_view_transform = torch.from_numpy(self.get_view_matrix()).T.inverse().cuda()
        self.projection_matrix = torch.from_numpy(self.get_project_matrix()).T.cuda()

    def get_view_matrix(self):
        view_matrix = np.array(glm.lookAt(self.position, self.target, self.up))
        scale_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])
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

camera=  Camera(720,1280)

###################################################
# PROJECTION PHASE
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

mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
        viewmatrix=camera.world_view_transform, 
        projmatrix=camera.projection_matrix)
# (bong-furisoa)
# Gaussian Sphere들을 in_mask를 사용해서 필터링하겠습니다.
mean_ndc = mean_ndc[in_mask]
mean_view = mean_view[in_mask]
depths = mean_view[:,2]

###################################################
# BUILD COLOR PHASE
active_sh_degree = int(np.round(np.sqrt(ply.sh_dim/3)))-1
def build_color(means3D, shs, camera):
    rays_o = torch.from_numpy(camera.position).cuda()
    rays_d = means3D - rays_o
    color = eval_sh(active_sh_degree, shs.permute(0,2,1), rays_d)
    color = (color + 0.5).clip(min=0.0)
    return color

# (bong-furisoa)
# Gaussian Sphere들을 in_mask를 사용해서 필터링하겠습니다.
color = build_color(means3D=means3D[in_mask], shs=shs[in_mask], camera=camera)

###################################################
# BUILD COV3D PHASE
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

# (bong-furisoa)
# Gaussian Sphere들을 in_mask를 사용해서 필터링하겠습니다.
cov3d = build_covariance_3d(scales, rotations)
cov3d = cov3d[in_mask]

###################################################
# BUILD COV2D PHASE
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

cov2d = build_covariance_2d(
    # (bong-furisoa)
    # Gaussian Sphere들을 in_mask를 사용해서 필터링하겠습니다.
    mean3d=means3D[in_mask], 
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

    TILE_SIZE = 16
    for h in range(0, camera.image_height, TILE_SIZE):
        for w in range(0, camera.image_width, TILE_SIZE):

            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
            
            if not in_mask.sum() > 0:
                continue

            P = in_mask.sum()
            tile_coord = pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
            sorted_depths, index = torch.sort(depths[in_mask])
            sorted_means2D = means2D[in_mask][index]
            sorted_cov2d = cov2d[in_mask][index] # P 2 2
            sorted_conic = sorted_cov2d.inverse() # inverse of variance
            sorted_opacity = opacity[in_mask][index]
            sorted_color = color[in_mask][index]
            #dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
            # ADDED xy - pixf
            dx = (-tile_coord[:,None,:] + sorted_means2D[None,:]) # B P 2
            
            gauss_weight = torch.exp(-0.5 * (
                dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))

            # ADDED added large value filtering 
            #gauss_weight = gauss_weight * (gauss_weight <= 1)
            
            alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
            T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
            acc_alpha = (alpha * T).sum(dim=1)
            #tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
            tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * 0
            tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
            render_color[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
            render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
            render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)

    return {
        "render": render_color,
        "depth": render_depth,
        "alpha": render_alpha,
        "visiility_filter": radii > 0,
        "radii": radii
    }


out = render(
                camera = camera, 
                means2D=means2D,
                cov2d=cov2d,
                color=color,
                # (bong-furisoa)
                # Gaussian Sphere들을 in_mask를 사용해서 필터링하겠습니다.
                opacity=opacity[in_mask], 
                depths=depths,
            )

image = out['render'].detach().cpu().numpy()
utils.imwrite(str('test.png'), image)

image = out['depth'].detach().cpu().numpy()
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
image = np.squeeze(to8b(image))
im = Image.fromarray(image, mode='L')
im.save('depth.png')
print(means2D)
