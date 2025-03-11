import torch
import torch.nn.functional as F


def img_gradient(img):
    """

    计算图像梯度，使用Sobel滤波器。

    参数:
        img (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)，其中
                            B 是批量大小，C 是通道数，
                            H 是高度，W 是宽度。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 包含x方向和y方向梯度的元组。
                                           每个张量的形状与输入图像相同。
    """
    device = img.device
    dtype = img.dtype
    b, c, h, w = img.shape
    device = img.device
    dtype = img.dtype
    b, c, h, w = img.shape

    # 定义用于计算x方向梯度的Sobel滤波器核
    gx_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gx_kernel = gx_kernel.repeat(c, 1, 1, 1)
    
    # 定义用于计算y方向梯度的Sobel滤波器核
    gy_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gy_kernel = gy_kernel.repeat(c, 1, 1, 1)

    # 尺度变化 (h+2p-k)/s+1  ,进行滤波
    gx = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gx_kernel,
        groups=img.shape[1],
    )

    gy = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gy_kernel,
        groups=img.shape[1],
    )
    # 定义用于计算x方向梯度的Sobel滤波器核
    gx_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gx_kernel = gx_kernel.repeat(c, 1, 1, 1)
    
    # 定义用于计算y方向梯度的Sobel滤波器核
    gy_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gy_kernel = gy_kernel.repeat(c, 1, 1, 1)

    # 尺度变化 (h+2p-k)/s+1  ,进行滤波
    gx = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gx_kernel,
        groups=img.shape[1],
    )

    gy = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gy_kernel,
        groups=img.shape[1],
    )

    return gx, gy
