import torch
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric


class FrameTracker:
    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device

        self.reset_idx_f2k()
        self.confidence_log = {}

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def track(self, frame: Frame):
        # 获取最新的关键帧
        keyframe = self.keyframes.last_keyframe()

        # 进行不对称匹配，获取匹配像素一维索引、匹配置信度、匹配点云、匹配点云置信度、匹配描述符置信度
        # 都是以当前坐标系下的值
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        # 保存匹配像素点以供下次使用
        self.idx_f2k = idx_f2k.clone()

        # 去除维度
        # print(f'idx_f2k: {idx_f2k.shape}')
        idx_f2k = idx_f2k[0]
        # print(f'idx_f2k: {idx_f2k.shape}')
        valid_match_k = valid_match_k[0]
        # print(f'valid_match_k: {valid_match_k.shape}')
        # with open('/home/narwal/mast3r-slam/MASt3R-SLAM/match_output.txt', 'a') as f:
        #     f.write(f'idx_f2k: {idx_f2k.shape}\n')
        #     f.write(f'valid_match_k: {valid_match_k.shape}\n')

        # 计算匹配上的点图对应的置信度,利用两个像素点特征的置信度乘积的平方根
        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # 在更新当前帧的点云图,是在当前坐标系下的点云
        frame.update_pointmap(Xff, Cff)

        # 检查是否使用标定
        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # 获取位姿、点对应关系和置信度,此处已经将当前帧点图与关键帧匹配上了
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # 获取有效点
        # 使用规范化置信度平均值
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]
        # print(f'Cf: {Cf}')
        # print(f'Ck: {Ck}')
        # print(f'Qk: {Qk}')
        # 保存置信度
        self.confidence_log = {
            'frame': frame.frame_id,
            'valid_Cf': valid_Cf,
            'valid_Ck': valid_Ck,
            'valid_Q': valid_Q
        }

        # 判断匹配后的像素点是否有效
        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        # print(f'valid_opt: {valid_opt.shape}')
        valid_kf = valid_match_k & valid_Q
        # print(f'valid_kf: {valid_kf.shape}')

        # 计算匹配比例
        match_frac = valid_opt.sum() / valid_opt.numel()
        # print(f"Match frac: {match_frac}")
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            # 进行跟踪
            if not use_calib:
                #这里的点图实际上已经匹配上了,所以可以对位姿态进行优化
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
                print(f'T_WCf: {T_WCf.shape}')
                print(f'T_CkCf: {T_CkCf.shape}')
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        # 更新帧的位姿
        frame.T_WC = T_WCf

        # 使用位姿变换点以更新关键帧
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        # 写回过滤后的点云图
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # 关键帧选择
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        # 有效特征点索引(重复的用unique去重)/总特征点数
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        # 判断是否需要新的关键帧
        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # 如果是新关键帧，重置索引
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        #这里获取的点云是各自坐标系下的点云
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        #位姿
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence,这里如果是多次融合,则需要使用平均值
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        """
        使用加权残差和鲁棒Huber损失函数求解线性系统。

        参数:
            sqrt_info (torch.Tensor): 方根信息矩阵。
            r (torch.Tensor): 残差向量。
            J (torch.Tensor): 雅可比矩阵。

        返回:
            tuple: 包含以下内容的元组:
            - tau_j (torch.Tensor): 解向量。(1,7)
            - cost (float): 代价值。
        """
        # 计算加权残差
        whitened_r = sqrt_info * r
        # 使用Huber损失函数计算鲁棒加权信息矩阵
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        # 计算雅可比矩阵和残差
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        # 计算Hessian矩阵和梯度
        H = A.T @ A
        g = -A.T @ b
        # 计算代价
        cost = 0.5 * (b.T @ b).item()

        # 使用Cholesky分解求解线性系统
        L = torch.linalg.cholesky(H, upper=False) # H = L*L^T
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1) # 求解更新量(1,7)
        print(f'g: {g.shape}')
        print(f'H: {H.shape}')
        print(f'L: {L.shape}')
        print(f'tau_j: {tau_j.shape}')

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        """
        通过光线距离和相似变换优化位姿。

        参数:
        Xf (torch.Tensor): 当前帧的特征点。
        Xk (torch.Tensor): 关键帧的特征点。
        T_WCf (SE3): 当前帧的位姿。
        T_WCk (SE3): 关键帧的位姿。
        Qk (torch.Tensor): 像素点描述子的置信度。
        valid (torch.Tensor): 有效点的掩码。

        返回:
        T_WCf (SE3): 优化后的当前帧位姿。
        T_CkCf (Sim3): 优化后的相对位姿。
        """
        # 初始化最后的误差
        last_error = 0
        # 计算光线(点图)和距离的置信度信息
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)
        # print(f'sqrt_info_ray: {sqrt_info_ray.shape}')
        # print(f'sqrt_info_dist: {sqrt_info_dist.shape}')
        # print(f'sqrt_info: {sqrt_info.shape}')


        # 计算相对位姿（不包含尺度）
        T_CkCf = T_WCk.inv() * T_WCf

        # 预计算关键帧观测点的距离和光线,得到归一化的向量
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            # 计算当前帧在关键帧坐标系下的点和sim相似变换的雅可比矩阵,包含s,R,t
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            # 计算光线距离和雅可比矩阵
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # 计算残差 r = z - h(x)
            r = rd_k - rd_f_Ck
            # 计算雅可比矩阵
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            # 求解更新量 tau_ij_sim3 和新的代价
            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            # 更新相对位姿
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            # 检查是否收敛
            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            # 如果达到最大迭代次数，打印最后的误差
            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # 根据相对位姿更新当前帧的位姿
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
