import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"
                 ):
        super(Camera, self).__init__()
        # 第n个图像
        self.uid = uid
        # colmap_id: camera_id
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")
        # image是 image文件夹里的文件
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 第四个通道
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            # 不知道是在干啥，操作前和操作后的数组是相同的
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        # 四元数和平移向量的作用是为了将图像中的特征点映射到三维空间中的真实世界坐标
        # 根据相机的姿态计算出物体从世界坐标转化为相机视图坐标的矩阵
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # 将三维场景投影到二维屏幕的投影矩阵
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        # unsqueeze(0) 在矩阵前边添加一个维度
        # bmm：批量矩阵乘法，第一个维度不变，第二三个维度相乘
        # 没看出来干啥的，先升维再降维，看起来好像是批量操作时才会起作用
        # 将三维场景投影到二维屏幕的矩阵
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 相机在世界坐标上的位置
        self.camera_center = self.world_view_transform.inverse()[3, :3]
