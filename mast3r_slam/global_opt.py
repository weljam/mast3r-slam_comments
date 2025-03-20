import lietorch
import torch
from mast3r_slam.config import config
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric
import mast3r_slam_backends


class FactorGraph:
    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):
        # 初始化因子图
        self.model = model  # 模型
        self.frames = frames  # 共享关键帧
        self.device = device  # 设备，默认为 "cuda"
        self.cfg = config["local_opt"]  # 配置文件中的局部优化配置
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)  # 边的起点索引
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)  # 边的终点索引
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)  # 起点到终点的匹配索引
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)  # 终点到起点的匹配索引
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)  # 有效匹配标志（终点）
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)  # 有效匹配标志（起点）
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)  # 匹配质量（起点到终点）
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)  # 匹配质量（终点到起点）
        self.window_size = self.cfg["window_size"]  # 窗口大小

        self.K = K  # 相机内参

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        # 当前帧的索引 ,如果重定位索引到3,且当前帧为27 则 [27 27 27]
        kf_ii = [self.frames[idx] for idx in ii]  
        # 索引到的重定位帧的索引
        kf_jj = [self.frames[idx] for idx in jj]   
        #获取对应的特征点以及位置 
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

        # 对当前帧率与重定位帧进行匹配
        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # NOTE: 需要两个边方向都超过阈值才能接受任意一个
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        #判断是不是连续的帧
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        #不是连续帧,且匹配小于阈值的边是无效的)
        invalid_edges = (~consecutive_edges) & invalid_edges
        #invalid_edges只要有一条边是无效的,那么就重定位失败
        if invalid_edges.any() and is_reloc:
            return False
        #反之,如果是连续帧,则一定是有效边,并且如果不是连续帧,且匹配程度都大于阈值,则也是有效边
        valid_edges = ~invalid_edges
        #根据有效边的索引,获取有效的匹配
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        #设置边相关的信息
        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        #返回是否添加了新的边
        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def get_unique_kf_idx(self):
        # 获取唯一的关键帧索引,并升序,例如ii = {12,14,16}  jj = {18,18,18} ,则返回{12,14,16,18}
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    def prep_two_way_edges(self):
        # 准备双向边
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        # 每个关键帧的位姿和点以及置信读
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    def solve_GN_rays(self):
        # 使用高斯牛顿法求解射线
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        #除了自身的关键帧外,还有其他的关键帧
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)
        # 准备双向边,及[11 12 14 18 18 18 ] [18 18 18 11 12 14] 分别计算双边误差
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # 更新关键帧 T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self):
        # 使用高斯牛顿法求解标定
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # 将点约束到射线上
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size

        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # 更新关键帧 T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
