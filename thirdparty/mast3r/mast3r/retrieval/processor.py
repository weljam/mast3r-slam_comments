# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main Retriever class
# --------------------------------------------------------
import os
import argparse
import numpy as np
import torch

from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.model import RetrievalModel, extract_local_features

try:
    import faiss
    faiss.StandardGpuResources()  # when loading the checkpoint, it will try to instanciate FaissGpuL2Index
except AttributeError as e:
    import asmk.index

    class FaissCpuL2Index(asmk.index.FaissL2Index):
        def __init__(self, gpu_id):
            super().__init__()
            self.gpu_id = gpu_id

        def _faiss_index_flat(self, dim):
            """Return initialized faiss.IndexFlatL2"""
            return faiss.IndexFlatL2(dim)

    asmk.index.FaissGpuL2Index = FaissCpuL2Index

from asmk import asmk_method  # noqa


def get_args_parser():
    parser = argparse.ArgumentParser('Retrieval scores from a set of retrieval', add_help=False, allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help="shortname of a retrieval model or path to the corresponding .pth")
    parser.add_argument('--input', type=str, required=True,
                        help="directory containing images or a file containing a list of image paths")
    parser.add_argument('--outfile', type=str, required=True, help="numpy file where to store the matrix score")
    return parser


def get_impaths(imlistfile):
    with open(imlistfile, 'r') as fid:
        impaths = [f for f in imlistfile.read().splitlines() if not f.startswith('#')
                   and len(f) > 0]  # ignore comments and empty lines
    return impaths


def get_impaths_from_imdir(imdir, extensions=['png', 'jpg', 'PNG', 'JPG']):
    assert os.path.isdir(imdir)
    impaths = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if any(f.endswith(ext) for ext in extensions)]
    return impaths


def get_impaths_from_imdir_or_imlistfile(input_imdir_or_imlistfile):
    if os.path.isfile(input_imdir_or_imlistfile):
        return get_impaths(input_imdir_or_imlistfile)
    else:
        return get_impaths_from_imdir(input_imdir_or_imlistfile)


class Retriever(object):
    def __init__(self, modelname, backbone=None, device='cuda'):
        # 加载模型
        assert os.path.isfile(modelname), modelname  # 确认模型文件存在
        print(f'Loading retrieval model from {modelname}')  # 打印加载模型信息
        ckpt = torch.load(modelname, 'cpu')  
        ckpt_args = ckpt['args']  # 获取检查点中的参数
        if backbone is None:
            backbone = AsymmetricMASt3R.from_pretrained(ckpt_args.pretrained)  # 如果没有提供backbone，则从预训练模型中加载
        self.model = RetrievalModel(
            backbone, freeze_backbone=ckpt_args.freeze_backbone, prewhiten=ckpt_args.prewhiten,
            hdims=list(map(int, ckpt_args.hdims.split('_'))) if len(ckpt_args.hdims) > 0 else "",
            residual=getattr(ckpt_args, 'residual', False), postwhiten=ckpt_args.postwhiten,
            featweights=ckpt_args.featweights, nfeat=ckpt_args.nfeat
        ).to(device)  # 初始化检索模型
        self.device = device  # 设置设备
        msg = self.model.load_state_dict(ckpt['model'], strict=False)  # 加载模型状态字典
        assert all(k.startswith('backbone') for k in msg.missing_keys)  # 确认所有缺失的键都以'backbone'开头
        assert len(msg.unexpected_keys) == 0  # 确认没有意外的键
        self.imsize = ckpt_args.imsize  # 设置图像大小

        # 加载asmk码书
        dname, bname = os.path.split(modelname)  # 获取模型文件的目录和文件名
        bname_splits = bname.split('_')  # 分割文件名
        cache_codebook_fname = os.path.join(dname, '_'.join(bname_splits[:-1]) + '_codebook.pkl')  # 构建码书缓存文件名
        assert os.path.isfile(cache_codebook_fname), cache_codebook_fname  # 确认码书缓存文件存在
        asmk_params = {'index': {'gpu_id': 0}, 'train_codebook': {'codebook': {'size': '64k'}},
                       'build_ivf': {'kernel': {'binary': True}, 'ivf': {'use_idf': False},
                                     'quantize': {'multiple_assignment': 1}, 'aggregate': {}},
                       'query_ivf': {'quantize': {'multiple_assignment': 5}, 'aggregate': {},
                                     'search': {'topk': None},
                                     'similarity': {'similarity_threshold': 0.0, 'alpha': 3.0}}}  # 初始化asmk参数
        asmk_params['train_codebook']['codebook']['size'] = ckpt_args.nclusters  # 设置码书大小
        self.asmk = asmk_method.ASMKMethod.initialize_untrained(asmk_params)  # 初始化未训练的asmk方法
        self.asmk = self.asmk.train_codebook(None, cache_path=cache_codebook_fname)  # 训练码书

    def __call__(self, input_imdir_or_imlistfile, outfile=None):
        # 获取图像路径
        if isinstance(input_imdir_or_imlistfile, str):
            impaths = get_impaths_from_imdir_or_imlistfile(input_imdir_or_imlistfile)  # 从目录或文件中获取图像路径
        else:
            impaths = input_imdir_or_imlistfile  # 假设传入的是一个列表
        print(f'Found {len(impaths)} images')  # 打印找到的图像数量

        # 构建数据库
        feat, ids = extract_local_features(self.model, impaths, self.imsize, tocpu=True, device=self.device)  # 提取局部特征
        feat = feat.cpu().numpy()  # 将特征转换为numpy数组
        ids = ids.cpu().numpy()  # 将ID转换为numpy数组
        asmk_dataset = self.asmk.build_ivf(feat, ids)  # 构建IVF数据库

        # 实际上我们检索的是同一组图像
        metadata, query_ids, ranks, ranked_scores = asmk_dataset.query_ivf(feat, ids)  # 查询IVF数据库

        # 分数实际上是根据排名重新排序的...
        # 所以我们反过来重新做...
        scores = np.empty_like(ranked_scores)  # 创建一个空的分数数组
        scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores  # 根据排名重新排序分数

        # 保存
        if outfile is not None:
            if os.path.isdir(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile), exist_ok=True)  # 创建输出目录
            np.save(outfile, scores)  # 保存分数矩阵
            print(f'Scores matrix saved in {outfile}')  # 打印保存信息
        return scores  # 返回分数