import dataclasses
from enum import Enum
from typing import Optional
import lietorch
import torch
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config


class Mode(Enum):
    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@dataclasses.dataclass
class Frame:
    frame_id: int
    img: torch.Tensor
    img_shape: torch.Tensor
    img_true_shape: torch.Tensor
    uimg: torch.Tensor
    T_WC: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    X_canon: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    N: int = 0
    N_updates: int = 0
    K: Optional[torch.Tensor] = None

    def get_score(self, C):
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            score = torch.median(C)  # Is this slower than mean? Is it worth it?
        elif filtering_score == "mean":
            score = torch.mean(C)
        return score

    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        # 获取配置中的过滤模式
        filtering_mode = config["tracking"]["filtering_mode"]

        # 如果这是第一次更新点云地图
        if self.N == 0:
            self.X_canon = X.clone()  # 克隆输入的点云
            self.C = C.clone()  # 克隆输入的置信度
            self.N = 1  # 设置点云数量为1
            self.N_updates = 1  # 设置更新次数为1
            if filtering_mode == "best_score":
                self.score = self.get_score(C)  # 计算并存储当前置信度的得分
            return

        # 根据不同的过滤模式更新点云地图
        if filtering_mode == "first":
            if self.N_updates == 1:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
        elif filtering_mode == "recent":
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
        elif filtering_mode == "best_score":
            new_score = self.get_score(C)
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.score = new_score
        elif filtering_mode == "indep_conf":
            new_mask = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            self.C = self.C + C
            self.N += 1
        elif filtering_mode == "weighted_spherical":

            def cartesian_to_spherical(P):
                """
                将笛卡尔坐标转换为球面坐标。

                参数:
                    P (torch.Tensor): 输入的笛卡尔坐标张量，形状为 (..., 3)。

                返回:
                    torch.Tensor: 转换后的球面坐标张量，形状为 (..., 3)，
                                  其中包含径向距离 r、方位角 phi 和极角 theta。

                注意:
                    - r 是径向距离，表示点到原点的距离。
                    - phi 是方位角，表示点在 xy 平面上的投影与 x 轴的夹角。
                    - theta 是极角，表示点与 z 轴的夹角。
                """
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                phi = torch.atan2(y, x)
                theta = torch.acos(z / r)
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            def spherical_to_cartesian(spherical):
                """
                将球坐标转换为笛卡尔坐标。

                参数:
                spherical (torch.Tensor): 包含球坐标的张量，形状为 (..., 3)，
                                          其中最后一个维度包含 (r, phi, theta)。
                                          r 是半径，phi 是方位角，theta 是极角。

                返回:
                torch.Tensor: 包含笛卡尔坐标的张量，形状与输入相同，最后一个维度包含 (x, y, z)。

                示例:
                >>> spherical = torch.tensor([[1, 0, 0], [1, 3.14159, 1.5708]])
                >>> cartesian = spherical_to_cartesian(spherical)
                >>> print(cartesian)
                tensor([[1.0000, 0.0000, 0.0000],
                        [-1.0000, 0.0000, 0.0000]])
                """
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                P = torch.cat((x, y, z), dim=-1)
                return P

            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            self.X_canon = spherical_to_cartesian(spherical)
            self.C = self.C + C
            self.N += 1

        self.N_updates += 1
        return

    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None


def create_frame(i, img, T_WC, img_size=512, device="cuda:0"):
    img = resize_img(img, img_size)
    rgb = img["img"].to(device=device)
    img_shape = torch.tensor(img["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)
    return frame


class SharedStates:
    def __init__(self, manager, h, w, dtype=torch.float32, device="cuda"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()  # 线程锁，用于同步访问
        self.paused = manager.Value("i", 0)  # 用于表示是否暂停的标志
        self.mode = manager.Value("i", Mode.INIT)  # 当前模式
        self.reloc_sem = manager.Value("i", 0)  # 重定位信号量
        self.global_optimizer_tasks = manager.list()  # 全局优化任务队列
        self.edges_ii = manager.list()  # 边的起点索引列表
        self.edges_jj = manager.list()  # 边的终点索引列表

        self.feat_dim = 1024  # 特征维度
        self.num_patches = h * w // (16 * 16)  # 图像块数量

        # fmt:off
        # 当前帧的共享状态（用于重定位/可视化）
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()  # 数据集索引
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()  # 图像
        self.uimg = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_()  # 未归一化的图像
        self.img_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()  # 图像形状
        self.img_true_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()  # 图像真实形状
        self.T_WC = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()  # 世界到相机的变换矩阵
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()  # 点云
        self.C = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()  # 置信度
        self.feat = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()  # 特征
        self.pos = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()  # 位置
        # fmt: on

    def set_frame(self, frame):
        with self.lock:
            # 设置当前帧的共享状态
            self.dataset_idx[:] = frame.frame_id
            self.img[:] = frame.img
            self.uimg[:] = frame.uimg
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.T_WC[:] = frame.T_WC.data
            self.X[:] = frame.X_canon
            self.C[:] = frame.C
            self.feat[:] = frame.feat
            self.pos[:] = frame.pos

    def get_frame(self):
        with self.lock:
            # 获取当前帧的共享状态
            frame = Frame(
                int(self.dataset_idx[0]),
                self.img,
                self.img_shape,
                self.img_true_shape,
                self.uimg,
                lietorch.Sim3(self.T_WC),
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.feat = self.feat
            frame.pos = self.pos
            return frame

    def queue_global_optimization(self, idx):
        with self.lock:
            # 将索引添加到全局优化任务队列
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        with self.lock:
            # 增加重定位信号量
            self.reloc_sem.value += 1

    def dequeue_reloc(self):
        with self.lock:
            # 减少重定位信号量
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self):
        with self.lock:
            # 获取当前模式
            return self.mode.value

    def set_mode(self, mode):
        with self.lock:
            # 设置当前模式
            self.mode.value = mode

    def pause(self):
        with self.lock:
            # 暂停
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            # 取消暂停
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            # 检查是否暂停
            return self.paused.value == 1


class SharedKeyframes:
    def __init__(self, manager, h, w, buffer=512, dtype=torch.float32, device="cuda"):
        self.lock = manager.RLock()  # 线程锁，用于同步访问
        self.n_size = manager.Value("i", 0)  # 关键帧数量

        self.h, self.w = h, w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device

        self.feat_dim = 1024  # 特征维度
        self.num_patches = h * w // (16 * 16)  # 图像块数量

        # fmt:off
        # 初始化共享内存
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()  # 数据集索引
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()  # 图像
        self.uimg = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_()  # 未归一化的图像
        self.img_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()  # 图像形状
        self.img_true_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()  # 图像真实形状
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype).share_memory_()  # 世界到相机的变换矩阵
        self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()  # 点云
        self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()  # 置信度
        self.N = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()  # 点云数量
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()  # 更新次数
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()  # 特征
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()  # 位置
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()  # 脏标记
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()  # 内参矩阵
        # fmt: on

    def __getitem__(self, idx) -> Frame:
        with self.lock:
            # 将所有数据放入一个帧中
            kf = Frame(
                int(self.dataset_idx[idx]),
                self.img[idx],
                self.img_shape[idx],
                self.img_true_shape[idx],
                self.uimg[idx],
                lietorch.Sim3(self.T_WC[idx]),
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.feat = self.feat[idx]
            kf.pos = self.pos[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if config["use_calib"]:
                kf.K = self.K
            return kf

    def __setitem__(self, idx, value: Frame) -> None:
        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)

            # 设置属性
            self.dataset_idx[idx] = value.frame_id
            self.img[idx] = value.img
            self.uimg[idx] = value.uimg
            self.img_shape[idx] = value.img_shape
            self.img_true_shape[idx] = value.img_true_shape
            self.T_WC[idx] = value.T_WC.data
            self.X[idx] = value.X_canon
            self.C[idx] = value.C
            self.feat[idx] = value.feat
            self.pos[idx] = value.pos
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            self.is_dirty[idx] = True
            return idx

    def __len__(self):
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame):
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self):
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Optional[Frame]:
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1]

    def update_T_WCs(self, T_WCs, idx) -> None:
        with self.lock:
            self.T_WC[idx] = T_WCs.data

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K):
        assert config["use_calib"]
        with self.lock:
            self.K[:] = K

    def get_intrinsics(self):
        assert config["use_calib"]
        with self.lock:
            return self.K
