import torch
import torch.nn.functional as F
import mast3r_slam.image as img_utils
from mast3r_slam.config import config
import mast3r_slam_backends


def match(X11, X21, D11, D21, idx_1_to_2_init=None):
    idx_1_to_2, valid_match2 = match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init)
    return idx_1_to_2, valid_match2


def pixel_to_lin(p1, w):
    """
    将像素坐标转换为线性索引。

    参数:
        p1 (numpy.ndarray): 具有形状(..., 2)的像素坐标数组。
        w (int): 图像的宽度。

    返回:
        numpy.ndarray: 与输入像素坐标对应的线性索引数组。
    """
    idx_1_to_2 = p1[..., 0] + (w * p1[..., 1])
    return idx_1_to_2


def lin_to_pixel(idx_1_to_2, w):
    """
    将线性索引转换为像素坐标。

    参数:
        idx_1_to_2 (torch.Tensor): 包含线性索引的张量。
        w (int): 图像的宽度。

    返回:
        torch.Tensor: 包含像素坐标 (u, v) 的张量。
        (b, h * w, 2)
    """
    u = idx_1_to_2 % w
    v = idx_1_to_2 // w
    p = torch.stack((u, v), dim=-1)
    return p


def prep_for_iter_proj(X11, X21, idx_1_to_2_init):
    """"
    输入:点图X11, X21, 初始索引映射idx_1_to_2_init
    输出:点图和梯度图rays_with_grad_img(b,h,w,c1+cx+cy), 归一化X21点图(b,h*w,c),
    初始投影p_init(b, h * w,2) 
    """
    b, h, w, _ = X11.shape
    device = X11.device

    # 归一化点图
    rays_img = F.normalize(X11, dim=-1)
    rays_img = rays_img.permute(0, 3, 1, 2)  # 转换为 (b,c,h,w) 形状
    gx_img, gy_img = img_utils.img_gradient(rays_img)  # sobel滤波器计算图像梯度
    rays_with_grad_img = torch.cat((rays_img, gx_img, gy_img), dim=1)  # 拼接光线图像和梯度,通道数变,其他不变
    rays_with_grad_img = rays_with_grad_img.permute(0, 2, 3, 1).contiguous()  # 转换为 (b,h,w,c) 形状

    # 归一化3D点
    X21_vec = X21.view(b, -1, 3)  # 展平为 (b, h*w, 3)
    pts3d_norm = F.normalize(X21_vec, dim=-1)  # 归一化3D点

    # 初始化投影的初始猜测
    if idx_1_to_2_init is None:
        # 如果没有初始索引映射，初始化创建了一个形状为 (b, h * w) 的张量
        idx_1_to_2_init = torch.arange(h * w, device=device)[None, :].repeat(b, 1)
    p_init = lin_to_pixel(idx_1_to_2_init, w)  # 将线性索引转换为像素坐标
    p_init = p_init.float()  # 转换为浮点型

    return rays_with_grad_img, pts3d_norm, p_init  # 返回准备好的数据


def match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init=None):
    # 获取匹配配置
    cfg = config["matching"]
    b, h, w = X21.shape[:3]
    device = X11.device

    # 准备迭代投影所需的数据
    rays_with_grad_img, pts3d_norm, p_init = prep_for_iter_proj(
        X11, X21, idx_1_to_2_init
    )
    
    # 调用后端函数进行迭代投影匹配
    # 返回优化后的点集p1, 以及对应的收敛程度valid_proj2
    p1, valid_proj2 = mast3r_slam_backends.iter_proj(
        rays_with_grad_img,
        pts3d_norm,
        p_init,
        cfg["max_iter"],
        cfg["lambda_init"],
        cfg["convergence_thresh"],
    )
    p1 = p1.long()

    # 基于距离检查遮挡情况
    # 计算匹配点的距离(b,h,w)
    batch_inds = torch.arange(b, device=device)[:, None].repeat(1, h * w)
    # 计算匹配点的距离(p1[..., 1], p1[..., 0]) (b,h*w,3)和X21(b,h*w,3)的距离
    # 即匹配上了,然后点云的距离求和(粗匹配)
    dists2 = torch.linalg.norm(
        X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3) - X21, dim=-1
    )
    valid_dists2 = (dists2 < cfg["dist_thresh"]).view(b, -1)
    valid_proj2 = valid_proj2 & valid_dists2

    # 如果配置中有半径参数，则进一步精细匹配
    if cfg["radius"] > 0:
        (p1,) = mast3r_slam_backends.refine_matches(
            D11.half(),
            D21.view(b, h * w, -1).half(),
            p1,
            cfg["radius"],
            cfg["dilation_max"],
        )

    # 转换为线性索引
    idx_1_to_2 = pixel_to_lin(p1, w)

    return idx_1_to_2, valid_proj2.unsqueeze(-1)
