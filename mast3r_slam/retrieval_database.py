import torch
import numpy as np
from mast3r.retrieval.processor import Retriever
from mast3r.retrieval.model import how_select_local

from asmk import io_helpers


class RetrievalDatabase(Retriever):
    def __init__(self, modelname, backbone=None, device="cuda"):
        super().__init__(modelname, backbone, device)

        self.ivf_builder = self.asmk.create_ivf_builder()

        self.kf_counter = 0
        self.kf_ids = []

        self.query_dtype = torch.float32
        self.query_device = device
        self.centroids = torch.from_numpy(self.asmk.codebook.centroids).to(
            device=self.query_device, dtype=self.query_dtype
        )

    # Mirrors forward_local in extract_local_features from retrieval/model.py
    def prep_features(self, backbone_feat):
        retrieval_model = self.model

        # 提取特征和注意力，不进行编码
        backbone_feat_prewhitened = retrieval_model.prewhiten(backbone_feat)
        proj_feat = retrieval_model.projector(backbone_feat_prewhitened) + (
            0.0 if not retrieval_model.residual else backbone_feat_prewhitened
        )
        attention = retrieval_model.attention(proj_feat)
        proj_feat_whitened = retrieval_model.postwhiten(proj_feat)

        # 选择局部特征
        topk_features, _, _ = how_select_local(
            proj_feat_whitened, attention, retrieval_model.nfeat
        )

        return topk_features

    def update(self, frame, add_after_query, k, min_thresh=0.0):
        feat = self.prep_features(frame.feat)
        id = self.kf_counter  # 使用自己的计数器，否则会弄乱 IVF

        feat_np = feat[0].cpu().numpy()  # 假设一次只有一帧
        id_np = id * np.ones(feat_np.shape[0], dtype=np.int64)

        database_size = self.ivf_builder.ivf.n_images
        # print("Database size: ", database_size, self.kf_counter)

        # 只有在已有图像时才进行查询
        topk_image_inds = []
        topk_codes = None  # 如果实际查询则更改此项
        if self.kf_counter > 0:
            ranks, ranked_scores, topk_codes = self.query(feat_np, id_np)

            scores = np.empty_like(ranked_scores)
            scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
            scores = torch.from_numpy(scores)[0]

            topk_images = torch.topk(scores, min(k, database_size))

            valid = topk_images.values > min_thresh
            topk_image_inds = topk_images.indices[valid]
            topk_image_inds = topk_image_inds.tolist()

        if add_after_query:
            self.add_to_database(feat_np, id_np, topk_codes)

        return topk_image_inds

    # 需要这个函数的原因是手动更新 ivf_builder 时未定义 kernel 和倒排文件
    def query(self, feat, id):
        step_params = self.asmk.params.get("query_ivf")

        images2, ranks, scores, topk = self.accumulate_scores(
            self.asmk.codebook,
            self.ivf_builder.kernel,
            self.ivf_builder.ivf,
            feat,
            id,
            params=step_params,
        )

        return ranks, scores, topk

    def add_to_database(self, feat_np, id_np, topk_codes):
        self.add_to_ivf_custom(feat_np, id_np, topk_codes)

        # 记录
        self.kf_ids.append(id_np[0])
        self.kf_counter += 1

    def quantize_custom(self, qvecs, params):
        # 使用高效距离矩阵的技巧
        l2_dists = (
            torch.sum(qvecs**2, dim=1)[:, None]
            + torch.sum(self.centroids**2, dim=1)[None, :]
            - 2 * (qvecs @ self.centroids.mT)
        )
        k = params["quantize"]["multiple_assignment"]
        topk = torch.topk(l2_dists, k, dim=1, largest=False)
        return topk.indices

    def accumulate_scores(self, cdb, kern, ivf, qvecs, qimids, params):
        """根据给定的码书、核、倒排文件和参数，累积每个查询图像（qvecs, qimids）的得分。"""
        similarity_func = lambda *x: kern.similarity(*x, **params["similarity"])

        acc = []
        slices = list(io_helpers.slice_unique(qimids))
        for imid, seq in slices:
            # 计算 qvecs 到质心的距离矩阵（不形成差异）
            qvecs_torch = torch.from_numpy(qvecs[seq]).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, params)
            topk_inds = topk_inds.cpu().numpy()
            quantized = (qvecs, topk_inds)

            aggregated = kern.aggregate_image(*quantized, **params["aggregate"])
            ranks, scores = ivf.search(
                *aggregated, **params["search"], similarity_func=similarity_func
            )
            acc.append((imid, ranks, scores, topk_inds))

        imids_all, ranks_all, scores_all, topk_all = zip(*acc)

        return (
            np.array(imids_all),
            np.vstack(ranks_all),
            np.vstack(scores_all),
            np.vstack(topk_all),
        )

    def add_to_ivf_custom(self, vecs, imids, topk_codes=None):
        """将描述符和对应的图像 ID 添加到 IVF

        :param np.ndarray vecs: 局部描述符的二维数组
        :param np.ndarray imids: 图像 ID 的一维数组
        :param bool progress: 更新进度打印的步骤（None 表示禁用）
        """
        ivf_builder = self.ivf_builder

        step_params = self.asmk.params.get("build_ivf")

        if topk_codes is None:
            qvecs_torch = torch.from_numpy(vecs).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, step_params)
            topk_inds = topk_inds.cpu().numpy()
        else:
            # 重用之前计算的！只取前 1
            # 注意：假设构建参数的多重分配小于查询
            k = step_params["quantize"]["multiple_assignment"]
            topk_inds = topk_codes[:, :k]

        quantized = (vecs, topk_inds, imids)

        aggregated = ivf_builder.kernel.aggregate(
            *quantized, **ivf_builder.step_params["aggregate"]
        )
        ivf_builder.ivf.add(*aggregated)
