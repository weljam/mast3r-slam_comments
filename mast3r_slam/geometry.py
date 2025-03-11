import lietorch
import torch

# 计算反对称矩阵
def skew_sym(x):
    b = x.shape[:-1]
    x, y, z = x.unbind(dim=-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).view(*b, 3, 3)

# 计算点到原点的距离
def point_to_dist(X):
    d = torch.linalg.norm(X, dim=-1, keepdim=True)
    return d

# 计算点到射线的距离
def point_to_ray_dist(X, jacobian=False):
    b = X.shape[:-1]

    d = point_to_dist(X)
    d_inv = 1.0 / d
    r = d_inv * X
    rd = torch.cat((r, d), dim=-1)  # 维度4
    if not jacobian:
        return rd
    else:
        d_inv_2 = d_inv**2
        I = torch.eye(3, device=X.device, dtype=X.dtype).repeat(*b, 1, 1)
        dr_dX = d_inv.unsqueeze(-1) * (
            I - d_inv_2.unsqueeze(-1) * (X.unsqueeze(-1) @ X.unsqueeze(-2))
        )
        dd_dX = r.unsqueeze(-2)
        drd_dX = torch.cat((dr_dX, dd_dX), dim=-2)
        return rd, drd_dX

# 将点约束到射线上
def constrain_points_to_ray(img_size, Xs, K):
    uv = get_pixel_coords(Xs.shape[0], img_size, device=Xs.device, dtype=Xs.dtype).view(
        *Xs.shape[:-1], 2
    )
    Xs = backproject(uv, Xs[..., 2:3], K)
    return Xs

# Sim3变换作用于点
def act_Sim3(X: lietorch.Sim3, pC: torch.Tensor, jacobian=False):
    pW = X.act(pC)
    if not jacobian:
        return pW
    dpC_dt = torch.eye(3, device=pW.device).repeat(*pW.shape[:-1], 1, 1)
    dpC_dR = -skew_sym(pW)
    dpc_ds = pW.reshape(*pW.shape[:-1], -1, 1)
    return pW, torch.cat([dpC_dt, dpC_dR, dpc_ds], dim=-1)  # view(-1, mdim)

# 分解内参矩阵K
def decompose_K(K):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    return fx, fy, cx, cy

# 投影校准
def project_calib(P, K, img_size, jacobian=False, border=0, z_eps=0.0):
    b = P.shape[:-1]

    K_rep = K.repeat(*b, 1, 1)

    p = (K_rep @ P[..., None]).squeeze(-1)
    p = p / p[..., 2:3]
    p = p[..., :2]

    u, v = p.split([1, 1], dim=-1)
    x, y, z = P.split([1, 1, 1], dim=-1)

    # 检查像素是否在图像内
    valid_u = (u > border) & (u < img_size[1] - 1 - border)
    valid_v = (v > border) & (v < img_size[0] - 1 - border)
    # 检查是否在相机前方
    valid_z = z > z_eps
    # 获取总的有效性
    valid = valid_u & valid_v & valid_z

    # 深度变换
    logz = torch.log(z)
    invalid_z = torch.logical_not(valid_z)
    logz[invalid_z] = 0.0  # 需要避免nans

    # 输出
    pz = torch.cat((p, logz), dim=-1)

    if not jacobian:
        return pz, valid
    else:
        fx, fy, cx, cy = decompose_K(K)
        z_inv = 1.0 / z[..., 0]
        dpz_dP = torch.zeros(*b + (3, 3), device=P.device, dtype=P.dtype)
        dpz_dP[..., 0, 0] = fx
        dpz_dP[..., 1, 1] = fy
        dpz_dP[..., 0, 2] = -fx * x[..., 0] * z_inv
        dpz_dP[..., 1, 2] = -fy * y[..., 0] * z_inv
        dpz_dP *= z_inv[..., None, None]
        dpz_dP[..., 2, 2] = z_inv  # 仅在底行中z本身
        return pz, dpz_dP, valid

# 反投影
def backproject(p, z, K):
    tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
    tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
    dP_dz = torch.empty(p.shape[:-1] + (3, 1), device=z.device, dtype=K.dtype)
    dP_dz[..., 0, 0] = tmp1
    dP_dz[..., 1, 0] = tmp2
    dP_dz[..., 2, 0] = 1.0
    P = torch.squeeze(z[..., None, :] * dP_dz, dim=-1)
    return P

# 获取像素坐标
def get_pixel_coords(b, img_size, device, dtype):
    h, w = img_size
    u, v = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    uv = torch.stack((u, v), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    uv = uv.to(device=device, dtype=dtype)
    return uv
