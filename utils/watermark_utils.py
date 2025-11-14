import cv2
import torch

def extract(im):
    extract_length = 256 * 256 * 3 * 8
    height, width, _ = im.shape
    # 将图像展平成一维张量，并提取每个像素的最低有效位
    # 展平为 (height * width, 3) 的张量
    image_flat = im.reshape(-1)

    assert image_flat.numel() >= extract_length, "The image is too small to contain the watermark."

    lsb_flat = (image_flat[:extract_length] & 1).to(torch.uint8)

    # 将最低有效位还原成水印图像的像素值
    watermark_bits = torch.zeros((extract_length // 8), dtype=torch.uint8, device=im.device)
    for i in range(8):
        watermark_bits |= (lsb_flat[i::8] << i)

    # 将一维水印像素值还原成三维图像
    watermark = watermark_bits.view(256, 256, 3)

    return watermark