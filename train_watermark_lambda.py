import os
import numpy as np
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo running in gpu $CUDA_VISIBLE_DEVICES')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from random import randint
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim, psnr
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import lpips
import random
import time
from Noise import Noise
import pandas as pd

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# 彩色水印预处理函数
def preprocess_color_image(image_path, xfm, ifm, device):
    """
    处理彩色水印图像，确保小波变换后重建尺寸一致
    :param image_path: 水印图像路径
    :param xfm: 小波正变换对象
    :param ifm: 小波逆变换对象
    :param device: 设备
    :return: RGB张量
    """
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0).to(device)  # 增加批次维度 [1, 3, H, W]

    # 进行小波变换和重建，确保尺寸一致
    Yl, Yh = xfm(img)
    reconstructed_img = ifm((Yl, Yh))

    # 检查重建后的尺寸是否与原始尺寸一致
    if img.shape != reconstructed_img.shape:
        print(f"警告: 重建水印尺寸不匹配! 原始: {img.shape}, 重建: {reconstructed_img.shape}")
        print("调整水印尺寸以匹配重建尺寸")
        img = reconstructed_img

    return img


# SVD分解函数
def svd_decompose(matrix):
    """对矩阵进行SVD分解"""
    U, S, Vh = torch.svd(matrix)
    return U, S, Vh


# 训练准备和记录
def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # 设置输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建Tensorboard写入器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


# 注意力机制模块
class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        return x * ca * sa


# 增强的水印提取器
class EnhancedWatermarkDecoder(nn.Module):
    def __init__(self, output_length, in_channels=3, channels=64, blocks=4, r=8):
        """
        增强的水印提取网络
        :param output_length: 输出长度 (3×水印奇异值维度)
        :param in_channels: 输入通道数 (LL2子带)
        :param channels: 初始通道数
        :param blocks: SENet块的数量
        :param r: SE模块的压缩比
        """
        super().__init__()
        self.output_length = output_length

        # 特征提取模块
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(),
            CBAM(channels, reduction_ratio=r),
            nn.MaxPool2d(2),

            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
            CBAM(channels * 2, reduction_ratio=r),
            nn.MaxPool2d(2),

            nn.Conv2d(channels * 2, channels * 4, 3, padding=1),
            nn.ReLU(),
            CBAM(channels * 4, reduction_ratio=r),
            nn.AdaptiveAvgPool2d(1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(channels * 4, channels * 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(channels * 8, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, output_length),
            nn.Sigmoid()  # 确保输出在0-1范围
        )

    def forward(self, x):
        """输入: [B, C, H, W], 输出: [B, output_length]"""
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 归一化和逆变换函数
def normalize_singular_values(S, epsilon=1e-8):
    """归一化奇异值到0-1范围"""
    min_val = torch.min(S)
    max_val = torch.max(S)
    return (S - min_val) / (max_val - min_val + epsilon), min_val, max_val


def log_transform(S, epsilon=1e-8):
    """应用对数变换压缩奇异值范围"""
    return torch.log(S + epsilon)


def inverse_log_transform(S_log, epsilon=1e-8):
    """对数变换的逆变换"""
    return torch.exp(S_log) - epsilon


# 水印提取函数
def extract_watermark(S_pred, U_wm, V_wm, ifm, Yh_wm, min_val, max_val):
    """增强的水印提取函数"""
    device = S_pred.device
    channels = []
    idx = 0

    # 逆归一化
    S_pred_denorm = S_pred * (max_val - min_val) + min_val

    # 逆对数变换
    S_pred_restored = inverse_log_transform(S_pred_denorm)

    for i, (U, V) in enumerate(zip(U_wm, V_wm)):
        U = U.to(device)
        V = V.to(device)
        H_low, _ = U.shape
        _, W_low = V.shape
        k = min(H_low, W_low)

        # 切出本通道的k个预测奇异值
        S_c = S_pred_restored[idx: idx + k]
        idx += k

        # 添加权重 - 更重视较大的奇异值
        weights = torch.linspace(1.0, 0.5, k, device=device)
        S_c_weighted = S_c * weights

        # 重构水印通道
        channel_ll = U[:, :k] @ torch.diag(S_c_weighted) @ V[:, :k].t()
        channels.append(channel_ll)

    # 堆叠成低频子带 [1, C, H_low, W_low]
    ll_band = torch.stack(channels, dim=0).unsqueeze(0)

    # 恢复全分辨率
    full_wm = ifm((ll_band, Yh_wm))

    # 添加后处理
    full_wm = torch.clamp(full_wm, 0, 1)

    return full_wm


# 训练函数
def training(dataset, opt, pipe, args):
    print('--------------------------Train SVD-Based Watermark of Gaussian Scene--------------------------------------')
    print(f'Training watermark model with J={args.J}')

    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussian = GaussianModel(dataset.sh_degree)
    scene = Scene(args, gaussian)

    # 加载检查点
    checkpoint = os.path.join(args.model_path, args.start_checkpoint)
    if not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint missing: {checkpoint}")

    print(f'Loading checkpoint from {args.start_checkpoint}')
    (model_params, _) = torch.load(checkpoint)
    gaussian.restore(model_params, opt)
    gaussian.training_watermark_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 定义噪声阶段
    noise_stages = [
        ["Combined([Identity(), GN(), Crop(0.6, 0.6)])"],

        ["Combined([Identity(), GN(), Resize((0.75, 0.9)), Crop(0.6, 0.6), Rotation(30)])"],
        [
            "Combined([Identity(), GN(), GF(0.1), Resize((0.75, 0.9)), Jpeg(50), Crop(0.6, 0.6), Rotation(30)])"],
    ]
    noise = Noise(noise_stages[0])

    # 进行DWT分解获取原始低频分量
    xfm = DWTForward(J=args.J, mode='zero', wave='haar').to('cuda')
    ifm = DWTInverse(mode='zero', wave='haar').to('cuda')
    if args.wm_name == "":
        WM_DIR = "WM_images/"  # 风格图像所在文件夹
        wm_files = [f for f in os.listdir(WM_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        assert len(wm_files) > 0, "风格图像文件夹为空！"

        # 随机选择一张风格图像
        random.seed(time.time())
        selected_style_filename = random.choice(wm_files)
        print(f'selected_style_filename is {selected_style_filename}')
        WM_PATH = os.path.join(WM_DIR, selected_style_filename)
        wm_tensor = preprocess_color_image(WM_PATH, xfm, ifm, device='cuda')
    else:
        wm_tensor = preprocess_color_image(os.path.join("WM_images", args.wm_name), xfm, ifm, device='cuda')
    torchvision.utils.save_image(wm_tensor, os.path.join(args.source_path, f"watermark_{args.J}.png"))
    Yl_wm, Yh_wm = xfm(wm_tensor)  # [1, 3, H_low, W_low]
    print(f'Yl_wm.shape {Yl_wm.shape}')

    # 对水印进行SVD分解 - 作为密钥
    U_wm = []
    S_wm = []
    Vh_wm = []
    for c in range(Yl_wm.shape[1]):  # 对每个通道
        # 获取该通道的低频分量
        channel_tensor = Yl_wm[0, c].squeeze()
        U, S, Vh = svd_decompose(channel_tensor)
        U_wm.append(U)
        S_wm.append(S)  # 真实奇异值，用于训练提取器
        Vh_wm.append(Vh)

    # 原始奇异值
    raw_target_S = torch.cat(S_wm)
    print(f'singul is {raw_target_S.shape}')
    # 应用对数变换压缩范围
    log_target_S = log_transform(raw_target_S)

    # 归一化到0-1范围
    target_S, min_val, max_val = normalize_singular_values(log_target_S)
    print(f'target_S range: min={target_S.min().item():.4f}, max={target_S.max().item():.4f}')
    zero_target = torch.zeros_like(target_S, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()

    # 使用增强的水印提取器
    extractor = EnhancedWatermarkDecoder(target_S.shape[0])
    extractor.to('cuda')
    extractor.train()

    # 初始化损失函数
    criterion = nn.MSELoss()
    lpips_model = lpips.LPIPS(net="vgg").to("cuda")
    # 优化器
    extractor_optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-4)

    progress_bar = tqdm(range(first_iter, opt.water_iterations), desc="Training progress")
    first_iter += 1
    print(f'opt.water_iterations is {opt.water_iterations}, first_iter is {first_iter}')

    for iteration in range(first_iter, opt.water_iterations + 1):
        iter_start.record()

        # 更新噪声阶段
        if iteration == 10000:
            noise.reset_layers(noise_stages[1])
            print("Reset noise layers to stage 1")
        elif iteration == 20000:
            noise.reset_layers(noise_stages[2])
            print("Reset noise layers to stage 2")

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染图像
        render_pkg = render(viewpoint_cam, gaussian, pipe, background)
        render_image = render_pkg["render"]  # [3, H, W]

        # 获取嵌入水印后的图像
        gt_image = viewpoint_cam.original_image.cuda()

        # 渲染损失 - 在resize之前计算
        loss_l1 = l1_loss(gt_image, render_image)
        loss_render = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * (1.0 - ssim(gt_image, render_image))

        # 应用噪声
        noised_render = noise(render_image.unsqueeze(0))
        noised_gt = noise(gt_image.unsqueeze(0))

        # 混合正负样本
        batch_inputs = []
        batch_targets = []

        # 正样本 - 带水印的渲染图
        batch_inputs.append(noised_render)
        batch_targets.append(target_S)

        # 负样本 - 原始GT图 (不带水印)
        batch_inputs.append(noised_gt)
        batch_targets.append(zero_target)

        # 合成小批量
        target_batch = torch.stack(batch_targets)  # [batch_size, target_size]

        # 预测奇异值
        # 对图像进行DWT分解
        Yl_embed_render, _ = xfm(noised_render)  # 低频分量 [1, C, H_low, W_low]
        Yl_embed_gt, _ = xfm(noised_gt)  # 低频分量 [1, C, H_low, W_low]

        S_pred = extractor(torch.cat([Yl_embed_render, Yl_embed_gt], dim=0))

        # 计算损失
        loss_pred = criterion(S_pred, target_batch)
        w_loss = criterion(S_pred[0], target_batch[0])
        # wo_loss = criterion(S_pred[1], target_batch[1])
        # 提取水印并计算水印重建损失
        recover_watermark = extract_watermark(S_pred[0], U_wm, Vh_wm, ifm, Yh_wm, min_val, max_val)
        recover_zero = extract_watermark(S_pred[1], U_wm, Vh_wm, ifm, Yh_wm, min_val, max_val)
        # 渲染损失 - 在resize之前计算
        loss_wm = l1_loss(wm_tensor, recover_watermark)
        loss_render_wm = (1.0 - opt.lambda_dssim) * loss_wm + opt.lambda_dssim * (
                    1.0 - ssim(wm_tensor, recover_watermark))

        loss_zero_wm = 1 - l1_loss(wm_tensor, recover_zero)

        loss_lp = lpips_model(gt_image, render_image)

        total_loss = args.lambda_render * loss_render + args.lambda_wm * loss_render_wm + args.lambda_w * w_loss + args.lambda_neg * loss_zero_wm + loss_lp
        total_loss.backward()

        # 更新水印提取器
        extractor_optimizer.step()
        extractor_optimizer.zero_grad()

        # 更新高斯模型
        gaussian.optimizer.step()
        gaussian.optimizer.zero_grad(set_to_none=True)

        iter_end.record()

        # 进度条更新
        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "Total Loss": f"{total_loss.item():.6f}",
                "Render Loss": f"{loss_render.item():.6f}",
                "Pred Loss": f"{loss_pred.item():.6f}",
                "loss_lp": f"{loss_lp.item():.6f}",
                "Watermark Loss": f"{loss_render_wm.item():.6f}"
            })
            progress_bar.update(10)

        # 可视化监控
        if iteration == opt.water_iterations:
            progress_bar.close()

        # 保存模型
        if iteration in args.save_iterations or iteration == opt.water_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians and Watermark Network")
            torch.save((gaussian.capture(), iteration),
                       os.path.join(args.model_path, f"chkpnt_wm_order_{args.J}_{args.lambda_render}_{args.lambda_wm}_{args.lambda_w}.pth"))

            # 保存提取器网络
            torch.save(extractor.state_dict(),
                       os.path.join(args.model_path, f"svd_watermark_extractor_order_{args.J}_{args.lambda_render}_{args.lambda_wm}_{args.lambda_w}.pth"))

            # 保存密钥
            torch.save({
                'U_wm': U_wm,
                'Vh_wm': Vh_wm,
                'wm_tensor': wm_tensor,
                'Yh_wm': Yh_wm,
                'min_val': min_val,
                'max_val': max_val,
                'raw_target_S': raw_target_S
            }, os.path.join(args.model_path, f"svd_keys_order_{args.J}_{args.lambda_render}_{args.lambda_wm}_{args.lambda_w}.pt"))

            print(f"Saved SVD watermark extractor to {args.model_path}/svd_watermark_extractor_order_{args.J}_{args.lambda_render}_{args.lambda_wm}_{args.lambda_w}.pth")

    # 测试阶段
    with torch.no_grad():
        # 加载模型
        extractor_path = os.path.join(args.model_path, f"svd_watermark_extractor_order_{args.J}_{args.lambda_render}_{args.lambda_wm}_{args.lambda_w}.pth")
        extractor.load_state_dict(torch.load(extractor_path))
        extractor.eval()

        # 加载密钥
        keys = torch.load(os.path.join(args.model_path, f"svd_keys_order_{args.J}_{args.lambda_render}_{args.lambda_wm}_{args.lambda_w}.pt"))
        U_wm = keys['U_wm']
        Vh_wm = keys['Vh_wm']
        wm_tensor = keys['wm_tensor']
        Yh_wm = keys['Yh_wm']
        min_val = keys['min_val']
        max_val = keys['max_val']

        # 测试噪声
        noises = [
            "Identity()", "GN()", "GF(0.1)", "Resize((0.75, 0.75))",
            "Jpeg(50)", "Crop(0.6, 0.6)", "Rotation(30)"
        ]

        # 创建结果表格
        results = []

        for n in noises:
            noise_layers = [n]
            add_noise = Noise(noise_layers)
            test_output_dir = os.path.join(args.model_path, f"test_results_{args.J}_{n}")
            os.makedirs(test_output_dir, exist_ok=True)
            render_psnrs = []  # 带水印图像重建水印的质量
            render_ssims = []
            render_lpipss = []
            # 指标列表
            wm_psnrs = []  # 带水印图像重建水印的质量
            wm_ssims = []
            wm_lpipss = []

            no_wm_psnrs = []  # 无水印图像重建水印的质量（应接近0）
            no_wm_ssims = []
            no_wm_lpipss = []

            # 遍历测试相机
            test_cameras = scene.getTrainCameras()
            for i, test_cam in enumerate(test_cameras):
                # ================== 处理带水印的渲染图像 ==================
                # 渲染图像
                render_pkg = render(test_cam, gaussian, pipe, background)
                render_image = render_pkg["render"]  # [1, 3, H, W]
                render_image = render_image.unsqueeze(0)  # [1, 3, H, W]
                gt_image = test_cam.original_image.unsqueeze(0)
                render_psnrs.append(psnr(gt_image, render_image))
                render_ssims.append(ssim(gt_image, render_image))
                render_lpipss.append(lpips_model(gt_image, render_image).mean().item())


                # 应用噪声
                noised_image = add_noise(render_image.clone())

                # 对图像进行DWT分解获取低频分量
                Yl_render, _ = xfm(noised_image)  # [1, 3, H_low, W_low]

                # 预测奇异值
                S_pred = extractor(Yl_render)

                # 提取水印
                extracted_wm = extract_watermark(S_pred[0], U_wm, Vh_wm, ifm, Yh_wm, min_val, max_val)

                # 计算指标
                wm_psnrs.append(psnr(wm_tensor, extracted_wm))
                wm_ssims.append(ssim(wm_tensor, extracted_wm))
                wm_lpipss.append(lpips_model(wm_tensor, extracted_wm).mean().item())

                # ================== 处理无水印的原始图像 ==================
                # 获取原始图像（无水印）
                gt_image = test_cam.original_image.cuda().unsqueeze(0)

                # 应用噪声
                noised_gt = add_noise(gt_image.clone())

                # 对图像进行DWT分解获取低频分量
                Yl_gt, _ = xfm(noised_gt)  # [1, 3, H_low, W_low]

                # 预测奇异值
                S_pred_gt = extractor(Yl_gt)

                # 提取水印
                extracted_no_wm = extract_watermark(S_pred_gt[0], U_wm, Vh_wm, ifm, Yh_wm, min_val, max_val)

                # 计算指标
                no_wm_psnrs.append(psnr(wm_tensor, extracted_no_wm))
                no_wm_ssims.append(ssim(wm_tensor, extracted_no_wm))
                no_wm_lpipss.append(lpips_model(wm_tensor, extracted_no_wm).mean().item())

            # 计算平均指标
            avg_wm_psnr = float(torch.tensor(wm_psnrs).mean())
            avg_wm_ssim = float(torch.tensor(wm_ssims).mean())
            avg_wm_lpips = float(torch.tensor(wm_lpipss).mean())

            avg_no_wm_psnr = float(torch.tensor(no_wm_psnrs).mean())
            avg_no_wm_ssim = float(torch.tensor(no_wm_ssims).mean())
            avg_no_wm_lpips = float(torch.tensor(no_wm_lpipss).mean())

            avg_render_psnr = float(torch.tensor(render_psnrs).mean())
            avg_render_ssim = float(torch.tensor(render_ssims).mean())
            avg_render_lpips = float(torch.tensor(render_lpipss).mean())

            # 添加到结果表
            results.append({
                "scene": os.path.basename(args.source_path),
                "noise": n,
                "wm_psnr": avg_wm_psnr,
                "wm_ssim": avg_wm_ssim,
                "wm_lpips": avg_wm_lpips,
                "no_wm_psnr": avg_no_wm_psnr,
                "no_wm_ssim": avg_no_wm_ssim,
                "no_wm_lpips": avg_no_wm_lpips,
                "avg_render_psnr": avg_render_psnr,
                "avg_render_ssim": avg_render_ssim,
                "avg_render_lpips": avg_render_lpips,
            })

            # 打印结果
            print(f"\nNoise: {n}")
            print(f"  Watermarked Images:")
            print(f"    PSNR: {avg_wm_psnr:.4f} | SSIM: {avg_wm_ssim:.4f} | LPIPS: {avg_wm_lpips:.4f}")
            print(f"  Non-Watermarked Images:")
            print(f"    PSNR: {avg_no_wm_psnr:.4f} | SSIM: {avg_no_wm_ssim:.4f} | LPIPS: {avg_no_wm_lpips:.4f}")
            print(f"  Render Images:")
            print(f"    PSNR: {avg_render_psnr:.4f} | SSIM: {avg_render_ssim:.4f} | LPIPS: {avg_render_lpips:.4f}")

    # 保存最终结果
    results_path = os.path.join(args.model_path, f"final_results_order_{args.J}_{args.lambda_render}_{args.lambda_wm}_{args.lambda_w}.txt")
    with open(results_path, "w") as f:
        for res in results:
            f.write(f"Noise: {res['noise']}\n")
            f.write(
                f"  Watermarked: PSNR={res['wm_psnr']:.4f}, SSIM={res['wm_ssim']:.4f}, LPIPS={res['wm_lpips']:.4f}\n")
            f.write(
                f"  Non-Watermarked: PSNR={res['no_wm_psnr']:.4f}, SSIM={res['no_wm_ssim']:.4f}, LPIPS={res['no_wm_lpips']:.4f}\n\n")
            f.write(
                f"  Render: PSNR={res['avg_render_psnr']:.4f}, SSIM={res['avg_render_ssim']:.4f}, LPIPS={res['avg_render_lpips']:.4f}\n\n")

    if len(results) > 0:
        df = pd.DataFrame(results)

        # 创建保存目录
        save_dir = os.path.join("results")
        os.makedirs(save_dir, exist_ok=True)

        # 构造保存路径，建议每次运行使用不同的 exp_name 区分实验
        save_path = os.path.join(save_dir, f"final_results.xlsx")

        # 如果文件已存在则合并
        if os.path.exists(save_path):
            old_df = pd.read_excel(save_path)
            df = pd.concat([old_df, df], ignore_index=True)

        # 写入 Excel 文件
        df.to_excel(save_path, index=False)
        print(f"✅ 评估指标已保存至: {save_path}")
    print(f"Saved final results to {results_path}")


# 主函数
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="SVD-Based Watermark Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--start_checkpoint", type=str, default="chkpnt30000.pth")
    parser.add_argument("--J", type=int, default=1)
    parser.add_argument("--lambda_render", type=float, default=12)
    parser.add_argument("--lambda_wm", type=float, default=12)
    parser.add_argument("--lambda_w", type=float, default=1)
    parser.add_argument("--lambda_neg", type=float, default=10)
    parser.add_argument("--wm_name", type=str, default="airplane.jpg")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.water_iterations)

    print("Optimizing " + args.model_path)

    # 初始化系统状态
    safe_state(args.quiet)

    # 开始训练
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # 训练完成
    print("\nSVD-based watermark training complete.")
