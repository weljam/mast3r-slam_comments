# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Whitener and RetrievalModel
# --------------------------------------------------------
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images

default_device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')


# from https://github.com/gtolias/how/blob/4d73c88e0ffb55506e2ce6249e2a015ef6ccf79f/how/utils/whitening.py#L20
"""
用于从给定的描述符中学习带有收缩的PCA白化。
PCA数据降维，而白化对数据进行线性变换，使得变换后的数据具有单位方差且不相关。
"""
def pcawhitenlearn_shrinkage(X, s=1.0):
    """Learn PCA whitening with shrinkage from given descriptors"""
    N = X.shape[0]

    # Learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True) # 计算均值
    Xc = X - m # 中心化
    Xcov = np.dot(Xc.T, Xc) # 计算协方差矩阵
    Xcov = (Xcov + Xcov.T) / (2 * N) # 对称化写方差矩阵并归一化
    eigval, eigvec = np.linalg.eig(Xcov) # 计算特征值和特征向量
    order = eigval.argsort()[::-1] #对特征值进行降序排序
    eigval = eigval[order]  # 排序后的特征值
    eigvec = eigvec[:, order] # 排序后的特征向量

    eigval = np.clip(eigval, a_min=1e-14, a_max=None) # 限制特征值的最小值
    P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5 * s))), eigvec.T) # 计算PCA变换矩阵

    return m, P.T # 返回均值和PCA变换矩阵


class Dust3rInputFromImageList(torch.utils.data.Dataset):
    def __init__(self, image_list, imsize=512):
        super().__init__()
        self.image_list = image_list
        assert imsize == 512
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return load_images([self.image_list[index]], size=self.imsize, verbose=False)[0]


class Whitener(nn.Module):
    """
    Whitener 是一个用于数据白化（去相关性和归一化）的神经网络模块。

    参数:
    dim (int): 输入数据的维度。
    l2norm (int, 可选): 如果不为 None，则沿给定维度应用 L2 归一化。

    属性:
    m (torch.nn.Parameter): 数据中心化的均值参数，初始化为全零向量。
    p (torch.nn.Parameter): PCA 变换矩阵，初始化为单位矩阵。
    l2norm (int, 可选): 如果不为 None，则沿给定维度应用 L2 归一化。

    方法:
    forward(x):
        对输入数据 x 进行白化处理。
        
        参数:
        x (torch.Tensor): 输入数据张量。
        
        返回:
        torch.Tensor: 白化后的数据张量。
    """
    def __init__(self, dim, l2norm=None):
        super().__init__()
        self.m = torch.nn.Parameter(torch.zeros((1, dim)).double())  # 数据中心化的均值参数，初始化为全零向量
        self.p = torch.nn.Parameter(torch.eye(dim, dim).double()) #PCA 变换矩阵，初始化为单位矩阵
        self.l2norm = l2norm  # if not None, apply l2 norm along a given dimension

    def forward(self, x):
        with torch.autocast(self.m.device.type, enabled=False):
            shape = x.size() # 获取输入数据的形状
            input_type = x.dtype # 获取输入数据的类型
            x_reshaped = x.view(-1, shape[-1]).to(dtype=self.m.dtype) # 将输入数据展平
            # Center the input data
            x_centered = x_reshaped - self.m # 中心化
            # Apply PCA transformation
            pca_output = torch.matmul(x_centered, self.p) # 应用PCA变换
            # reshape back
            pca_output_shape = shape  # list(shape[:-1]) + [shape[-1]]
            pca_output = pca_output.view(pca_output_shape) # 将数据恢复原来的形状
            # Apply L2 normalization
            if self.l2norm is not None:
                return torch.nn.functional.normalize(pca_output, dim=self.l2norm).to(dtype=input_type)
            return pca_output.to(dtype=input_type)


def weighted_spoc(feat, attn):
    """
    feat: BxNxC
    attn: BxN
    output: BxC L2-normalization weighted-sum-pooling of features
    feat*attn[:, :, None]: BxNxC * BxNx1 -> BxNxC 加权
    (feat*attn[:, :, None]).sum(dim=1): BxNxC -> BxC 求和
    torch.nn.functional.normalize(...): BxC -> BxC L2归一化
    """
    return torch.nn.functional.normalize((feat * attn[:, :, None]).sum(dim=1), dim=1)


def how_select_local(feat, attn, nfeat):
    """
    feat: BxNxC
    attn: BxN
    nfeat: nfeat to keep
    """
    # get nfeat
    if nfeat < 0:
        assert nfeat >= -1.0
        nfeat = int(-nfeat * feat.size(1)) #视为要保留的特征比例。例如，nfeat = -0.5 表示保留 50% 的特征
    else:
        nfeat = int(nfeat) 
    # asort
    #使用torch.topk 从 attn 中选择前nfeat个最高得分的特征。topk_attn 是这些特征的得分，topk_indices 是这些特征的索引。
    # topk_attn: B x min(nfeat, N) -> B x min(nfeat, N) 
    topk_attn, topk_indices = torch.topk(attn, min(nfeat, attn.size(1)), dim=1)
    #扩展维度 B x min(nfeat,N) -> B x min(nfeat, N) x C
    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, feat.size(2))
    topk_features = torch.gather(feat, 1, topk_indices_expanded)  # 获取对应的特征
    return topk_features, topk_attn, topk_indices


class RetrievalModel(nn.Module):
    def __init__(self, backbone, freeze_backbone=1, prewhiten=None, hdims=[1024], residual=False, postwhiten=None,
                 featweights='l2norm', nfeat=300, pretrained_retrieval=None):
        super().__init__()
        #添加骨干内容
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        #冻结骨干,训练时不更新骨干参数
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        #获取骨干维度
        self.backbone_dim = backbone.enc_embed_dim
        #添加白化层
        self.prewhiten = nn.Identity() if prewhiten is None else Whitener(self.backbone_dim)
    
        self.prewhiten_freq = prewhiten
        #如果prewhiten不为None且不为-1，则冻结白化层参数
        if prewhiten is not None and prewhiten != -1:
            for p in self.prewhiten.parameters():
                p.requires_grad = False
        # 残差链接
        self.residual = residual
        #该投影层用于对骨干网络提取的特征进行进一步变换。hdims定义了投影层隐藏层的维度，residual决定是否使用残差连接。
        self.projector = self.build_projector(hdims, residual)
        self.dim = hdims[-1] if len(hdims) > 0 else self.backbone_dim
        self.postwhiten_freq = postwhiten
        self.postwhiten = nn.Identity() if postwhiten is None else Whitener(self.dim)
        if postwhiten is not None and postwhiten != -1:
            assert len(hdims) > 0
            for p in self.postwhiten.parameters():
                p.requires_grad = False
        self.featweights = featweights
        if featweights == 'l2norm':
            self.attention = lambda x: x.norm(dim=-1)
        else:
            raise NotImplementedError(featweights)
        self.nfeat = nfeat
        self.pretrained_retrieval = pretrained_retrieval
        if self.pretrained_retrieval is not None:
            ckpt = torch.load(pretrained_retrieval, 'cpu')
            msg = self.load_state_dict(ckpt['model'], strict=False)
            assert len(msg.unexpected_keys) == 0 and all(k.startswith('backbone')
                                                         or k.startswith('postwhiten') for k in msg.missing_keys)

    def build_projector(self, hdims, residual):
        #判断是否使用残差连接,如果使用残差连接，则最后一层的维度应该等于骨干网络的维度,即特征向量维度[1024]
        if self.residual:
            assert hdims[-1] == self.backbone_dim
        #特征向量维度[1024]，
        d = self.backbone_dim
        #如果隐藏层维度为空，则不进行投影变换
        if len(hdims) == 0:
            return nn.Identity()
        layers = []
        #根据hdims的长度构建一系列线性层、层归一化层和 GELU 激活函数，最后返回一个包含这些层的nn.Sequential对象。
        for i in range(len(hdims) - 1):
            layers.append(nn.Linear(d, hdims[i]))
            d = hdims[i]
            layers.append(nn.LayerNorm(d))
            layers.append(nn.GELU())
        layers.append(nn.Linear(d, hdims[-1]))
        return nn.Sequential(*layers)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """
        重写state_dict方法，用于获取模型的状态字典。如果freeze_backbone为真，
        从状态字典中过滤掉以backbone开头的键值对，即不保存骨干网络的参数。
        """
        ss = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.freeze_backbone:
            ss = {k: v for k, v in ss.items() if not k.startswith('backbone')}
        return ss

    def reinitialize_whitening(self, epoch, train_dataset, nimgs=5000, log_writer=None, max_nfeat_per_image=None, seed=0, device=default_device):
        """
         根据prewhiten_freq和postwhiten_freq决定是否重新初始化预处理白化层和后处理白化层。
    
        """
        do_prewhiten = self.prewhiten_freq is not None and self.pretrained_retrieval is None and \
            (epoch == 0 or (self.prewhiten_freq > 0 and epoch % self.prewhiten_freq == 0))
        do_postwhiten = self.postwhiten_freq is not None and ((epoch == 0 and self.postwhiten_freq in [0, -1])
                                                              or (self.postwhiten_freq > 0 and
                                                                  epoch % self.postwhiten_freq == 0 and epoch > 0))
        if do_prewhiten or do_postwhiten:
            self.eval()
            imdataset = train_dataset.imlist_dataset_n_images(nimgs, seed)
            loader = torch.utils.data.DataLoader(imdataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        if do_prewhiten:
            print('Re-initialization of pre-whitening')
            t = time.time()
            with torch.no_grad():
                features = []
                for d in tqdm(loader):
                    feat = self.backbone._encode_image(d['img'][0, ...].to(device),
                                                       true_shape=d['true_shape'][0, ...])[0]
                    feat = feat.flatten(0, 1)
                    if max_nfeat_per_image is not None and max_nfeat_per_image < feat.size(0):
                        l2norms = torch.linalg.vector_norm(feat, dim=1)
                        feat = feat[torch.argsort(-l2norms)[:max_nfeat_per_image], :]
                    features.append(feat.cpu())
            features = torch.cat(features, dim=0)
            features = features.numpy()
            m, P = pcawhitenlearn_shrinkage(features)
            self.prewhiten.load_state_dict({'m': torch.from_numpy(m), 'p': torch.from_numpy(P)})
            prewhiten_time = time.time() - t
            print(f'Done in {prewhiten_time:.1f} seconds')
            if log_writer is not None:
                log_writer.add_scalar('time/prewhiten', prewhiten_time, epoch)
        if do_postwhiten:
            print(f'Re-initialization of post-whitening')
            t = time.time()
            with torch.no_grad():
                features = []
                for d in tqdm(loader):
                    backbone_feat = self.backbone._encode_image(d['img'][0, ...].to(device),
                                                                true_shape=d['true_shape'][0, ...])[0]
                    backbone_feat_prewhitened = self.prewhiten(backbone_feat)
                    proj_feat = self.projector(backbone_feat_prewhitened) + \
                        (0.0 if not self.residual else backbone_feat_prewhitened)
                    proj_feat = proj_feat.flatten(0, 1)
                    if max_nfeat_per_image is not None and max_nfeat_per_image < proj_feat.size(0):
                        l2norms = torch.linalg.vector_norm(proj_feat, dim=1)
                        proj_feat = proj_feat[torch.argsort(-l2norms)[:max_nfeat_per_image], :]
                    features.append(proj_feat.cpu())
                features = torch.cat(features, dim=0)
                features = features.numpy()
            m, P = pcawhitenlearn_shrinkage(features)
            self.postwhiten.load_state_dict({'m': torch.from_numpy(m), 'p': torch.from_numpy(P)})
            postwhiten_time = time.time() - t
            print(f'Done in {postwhiten_time:.1f} seconds')
            if log_writer is not None:
                log_writer.add_scalar('time/postwhiten', postwhiten_time, epoch)

    def extract_features_and_attention(self, x):
        """
            该方法用于提取图像的特征和注意力分数。首先通过骨干网络提取特征，然后进行预处理白化，接着通过投影层进行变换，
            并根据是否使用残差连接进行处理。之后计算注意力分数，最后进行后处理白化，返回处理后的特征和注意力分数。
        """
        backbone_feat = self.backbone._encode_image(x['img'], true_shape=x['true_shape'])[0]
        backbone_feat_prewhitened = self.prewhiten(backbone_feat)
        proj_feat = self.projector(backbone_feat_prewhitened) + \
            (0.0 if not self.residual else backbone_feat_prewhitened)
        attention = self.attention(proj_feat)
        proj_feat_whitened = self.postwhiten(proj_feat)
        return proj_feat_whitened, attention

    def forward_local(self, x):
        feat, attn = self.extract_features_and_attention(x)
        print(f"feat: {feat.shape}, attn: {attn.shape}")
        return how_select_local(feat, attn, self.nfeat)

    def forward_global(self, x):
        feat, attn = self.extract_features_and_attention(x)
        return weighted_spoc(feat, attn)

    def forward(self, x):
        return self.forward_global(x)


def identity(x):  # to avoid Can't pickle local object 'extract_local_features.<locals>.<lambda>'
    return x


@torch.no_grad()
def extract_local_features(model, images, imsize, seed=0, tocpu=False, max_nfeat_per_image=None,
                           max_nfeat_per_image2=None, device=default_device):
    model.eval()
    imdataset = Dust3rInputFromImageList(images, imsize=imsize) if isinstance(images, list) else images
    loader = torch.utils.data.DataLoader(imdataset, batch_size=1, shuffle=False,
                                         num_workers=8, pin_memory=True, collate_fn=identity)
    with torch.no_grad():
        features = []
        imids = []
        for i, d in enumerate(tqdm(loader)):
            dd = d[0]
            dd['img'] = dd['img'].to(device, non_blocking=True)
            feat, _, _ = model.forward_local(dd)
            feat = feat.flatten(0, 1)
            if max_nfeat_per_image is not None and feat.size(0) > max_nfeat_per_image:
                feat = feat[torch.randperm(feat.size(0))[:max_nfeat_per_image], :]
            if max_nfeat_per_image2 is not None and feat.size(0) > max_nfeat_per_image2:
                feat = feat[:max_nfeat_per_image2, :]
            features.append(feat)
            if tocpu:
                features[-1] = features[-1].cpu()
            imids.append(i * torch.ones_like(features[-1][:, 0]).to(dtype=torch.int64))
    features = torch.cat(features, dim=0)
    imids = torch.cat(imids, dim=0)
    return features, imids