# ./venv/bin/python3 psnr_calculator.py --image1 {image1 path} --image2 {image2 path}
import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(img1, img2):
    # MSE(mean squared error)
    mse = np.mean((img1 - img2) ** 2)
    # if the MSE value is equal to 0, then the PSNR value is infinity
    if mse == 0:
        return float('inf')
    # maximum pixel value
    max_pixel_value = 255.0
    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
         print(f"Image file in {image_path} is not found")
         exit(-1)

    return image

def main(args):
    # Load images
    image1 = load_image(args.image1)
    image2 = load_image(args.image2)
    
    # Compare images shape
    if image1.shape != image2.shape:
        print("Input images must have the same shape.")
        exit(-1)

    # Convert channels order from BGR to RGB, as OpenCV loads images in BGR channels
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Calculate PSNR and SSIM
    # https://xoft.tistory.com/3
    psnr_value = peak_signal_noise_ratio(image1, image2)
    ssim_value = structural_similarity(image1, image2, channel_axis=-1)
    
    # Higher value means better quality
    print(f"PSNR: {psnr_value:0.3f}") # 0 ~ inf
    print(f"SSIM: {ssim_value:0.3f}") # 0 ~ 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PSNR between 2 images.")
    parser.add_argument("--image1", type=str, required=True, help="Path of the first image")
    parser.add_argument("--image2", type=str, required=True, help="Path of the second image")
    args = parser.parse_args()

    main(args)
