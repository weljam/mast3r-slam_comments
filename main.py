import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # 我们正在添加然后从关键帧中移除，所以需要小心。
    # 锁会减慢可视化速度，但这样更安全...
    with keyframes.lock:
        kf_idx = []
        # 更新检索数据库，不在查询后添加
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        # print("Retrieved: ", retrieval_inds)
        kf_idx += retrieval_inds # 添加索引到的关键帧
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)  # 添加当前帧到关键帧
            n_kf = len(keyframes)  # 获取关键帧数量,即当前帧索引+1
            kf_idx = list(kf_idx)  # 转换为列表,主要是获取索引图像的数量,代码yaml中设置为3 
            frame_idx = [n_kf - 1] * len(kf_idx)  # 创建当前帧索引列表,若当前帧率为27,则[27 27 27]
            # print("Adding  kf ", n_kf - 1)
            # print("Adding factors against kf ", frame_idx.shape)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            # 添加新边，如果成功则进行后续操作
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                # 更新检索数据库，在查询后添加
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                # 将当前帧的位姿设置为与第一个匹配关键帧相同
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()  # 移除最后一个关键帧
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()  # 使用校准数据进行优化
            else:
                factor_graph.solve_GN_rays()  # 使用射线进行优化
        return successful_loop_closure  # 返回重定位是否成功


def run_backend(config_path, model, states, keyframes, K):
    # 加载配置文件
    load_config(config_path)

    device = keyframes.device  # 获取设备
    factor_graph = FactorGraph(model, keyframes, K, device)  # 创建因子图
    retrieval_database = load_retriever(model)  # 加载检索数据库

    mode = states.get_mode()  # 获取当前模式
    while mode is not Mode.TERMINATED:  # 如果模式不是终止
        mode = states.get_mode()  # 获取当前模式
        if mode == Mode.INIT or states.is_paused():  # 如果模式是初始化或暂停
            time.sleep(0.01)  # 等待0.01秒
            continue
        if mode == Mode.RELOC:  # 如果模式是重定位
            frame = states.get_frame()  # 获取当前帧
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)  # 进行重定位
            if success:  # 如果重定位成功
                states.set_mode(Mode.TRACKING)  # 设置模式为跟踪
            states.dequeue_reloc()  # 从重定位队列中移除
            continue
        idx = -1
        with states.lock:  # 加锁
            if len(states.global_optimizer_tasks) > 0:  # 如果有全局优化任务
                idx = states.global_optimizer_tasks[0]  # 获取第一个任务的索引
        if idx == -1:  # 如果没有任务
            time.sleep(0.01)  # 等待0.01秒
            continue

        # 图构建
        kf_idx = []
        n_consec = 1  # 前n个连续关键帧,这里可以设置需要构建的连续关键帧数量
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)  # 添加前n个连续关键帧的索引
        frame = keyframes[idx]  # 获取当前帧
        #对当前帧率的共视关键进行检索
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )  # 更新检索数据库
        kf_idx += retrieval_inds  # 添加检索到的关键帧索引

        lc_inds = set(retrieval_inds)  # 转换为集合去重
        lc_inds.discard(idx - 1)  # 移除当前帧的前一帧索引,主要为了显示检索结果
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)  # 打印检索结果

        kf_idx = set(kf_idx)  # 去重
        kf_idx.discard(idx)  # 移除当前帧索引,避免自己跟自己匹配
        kf_idx = list(kf_idx)  # 转换为列表
        frame_idx = [idx] * len(kf_idx)  # 创建当前帧索引列表
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )  # 添加因子

        with states.lock:  # 加锁,用于可视化部分
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()  # 更新边的索引
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()  # 更新边的索引

        if config["use_calib"]:
            factor_graph.solve_GN_calib()  # 使用校准数据进行优化
        else:
            factor_graph.solve_GN_rays()  # 使用射线进行优化

        with states.lock:  # 加锁
            if len(states.global_optimizer_tasks) > 0:  # 如果有全局优化任务
                idx = states.global_optimizer_tasks.pop(0)  # 移除第一个任务


if __name__ == "__main__":
    # 设置多进程启动方法为“spawn”
    mp.set_start_method("spawn")
    # 允许CUDA的TF32矩阵乘法
    torch.backends.cuda.matmul.allow_tf32 = True
    # 禁用梯度计算
    torch.set_grad_enabled(False)
    device = "cuda:0"  # 使用的设备
    save_frames = False  # 是否保存帧
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")  # 当前时间

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    # 加载配置文件
    load_config(args.config)
    print(args.dataset)
    print(config)

    # 创建多进程管理器和消息队列
    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    # 加载数据集
    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])

    h, w = dataset.get_img_shape()[0]  # 获取图像高度和宽度

    # 如果提供了校准数据，则使用校准数据
    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )
    keyframes = SharedKeyframes(manager, h, w ,168)  # 共享关键帧,这里的关键帧可以减少,进而减少显存
    states = SharedStates(manager, h, w)  # 共享状态

    # 如果不禁用可视化，启动可视化进程
    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(states, keyframes, main2viz, viz2main, args.config),
        )
        viz.start()

    # 加载模型
    model = load_mast3r(device=device)
    model.share_memory()  # 共享内存

    has_calib = dataset.has_calib()  # 数据集是否有校准数据
    use_calib = config["use_calib"]  # 是否使用校准数据
    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)  # 设置相机内参

    # 删除之前运行的轨迹文件
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.pt"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)  # 帧跟踪器
    last_msg = WindowMsg()  # 最后一条消息

    # 启动后端进程
    backend = mp.Process(
        target=run_backend, args=(args.config, model, states, keyframes, K)
    )
    backend.start()

    i = 0
    fps_timer = time.time()  # FPS计时器

    frames = []  # 存储帧的列表

    while True:
        mode = states.get_mode()  # 获取当前模式
        msg = try_get_msg(viz2main)  # 获取消息
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img ,_= dataset[i]  # 获取时间戳和归一化图像
        frames.append(img)

        # 获取上一帧的相机位姿
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # 通过单目推理初始化，并为数据库编码特征,对应点图X_init和置信度图C_init
            X_init, C_init = mast3r_inference_mono(model, frame)
            # 根据初始化结果更新点图和置信度图
            frame.update_pointmap(X_init, C_init)
            # 设置关键帧
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            # 设置当前帧
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # 在单线程模式下，确保每帧都进行重定位
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # 在单线程模式下，等待后端完成
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        # 记录时间
        if i % 5 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    # 保存结果
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_ATE(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir, f"{seq_name}.pt", dataset.timestamps, keyframes
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        print(len(frames))
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
