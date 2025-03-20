import dataclasses
import weakref
from pathlib import Path

import imgui
import lietorch
import torch
import moderngl
import moderngl_window as mglw
import numpy as np
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from mast3r_slam.frame import Mode
from mast3r_slam.geometry import get_pixel_coords
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)
from mast3r_slam.config import load_config, config


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


class Window(WindowEvents):
    title = "MASt3R-SLAM"
    window_size = (1960, 1080)

    def __init__(self, states, keyframes, main2viz, viz2main, **kwargs): #**kwargs表示将参数以字典的形式传入
        super().__init__(**kwargs) 
        self.ctx.gc_mode = "auto" # 设置垃圾回收模式为自动
        # bit hacky, but detect whether user is using 4k monitor
        self.scale = 1.0 # 初始化缩放比例
        if self.wnd.buffer_size[0] > 2560: # 如果窗口缓冲区宽度大于2560，假定使用4k显示器
            self.set_font_scale(2.0) # 设置字体缩放比例为2.0
            self.scale = 2 # 设置缩放比例为2
        self.clear = hex2rgba("#1E2326", alpha=1) # 设置清除颜色
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve()) # 注册资源目录

        self.line_prog = self.load_program("programs/lines.glsl") # 加载线条渲染程序
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl") # 加载surfelmap渲染程序
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl") # 加载trianglemap渲染程序
        self.pointmap_prog = self.surfelmap_prog # 默认使用surfelmap渲染程序

        width, height = self.wnd.size # 获取窗口宽度和高度
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100), # 设置投影矩阵
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])), # 设置相机位置和朝向
        )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale) # 创建坐标轴对象
        self.frustums = Frustums(self.line_prog) # 创建视锥对象
        self.lines = Lines(self.line_prog) # 创建线条对象

        self.viewport = ViewportWindow("Scene", self.camera) # 创建视口窗口
        self.state = WindowMsg() # 初始化窗口消息
        self.keyframes = keyframes # 关键帧数据
        self.states = states # 系统状态

        self.show_all = True # 是否显示所有内容
        self.show_keyframe_edges = True # 是否显示关键帧边缘
        self.culling = True # 是否启用面剔除
        self.follow_cam = True # 是否跟随相机

        self.depth_bias = 0.001 # 深度偏移
        self.frustum_scale = 0.05 # 视锥缩放比例

        self.dP_dz = None # 初始化深度梯度

        self.line_thickness = 3 # 线条厚度
        self.show_keyframe = True # 是否显示关键帧
        self.show_curr_pointmap = True # 是否显示当前点云图
        self.show_axis = True # 是否显示坐标轴

        self.textures = dict() # 初始化纹理字典
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime # 获取渲染程序的修改时间
        self.curr_img, self.kf_img = Image(), Image() # 初始化当前图像和关键帧图像
        self.curr_img_np, self.kf_img_np = None, None # 初始化图像数据

        self.main2viz = main2viz # 主进程到可视化进程的通信队列
        self.viz2main = viz2main # 可视化进程到主进程的通信队列

    def render(self, t: float, frametime: float):
        self.viewport.use()  # 使用视口
        self.ctx.enable(moderngl.DEPTH_TEST)  # 启用深度测试
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)  # 启用面剔除
        self.ctx.clear(*self.clear)  # 清除颜色缓冲区

        self.ctx.point_size = 2  # 设置点大小
        if self.show_axis:
            self.axis.render(self.camera)  # 渲染坐标轴

        curr_frame = self.states.get_frame()  # 获取当前帧
        h, w = curr_frame.img_shape.flatten()  # 获取图像高度和宽度
        self.frustums.make_frustum(h, w)  # 创建视锥

        self.curr_img_np = curr_frame.uimg.numpy()  # 获取当前帧图像数据
        self.curr_img.write(self.curr_img_np)  # 写入当前帧图像

        cam_T_WC = as_SE3(curr_frame.T_WC).cpu()  # 获取当前帧的相机位姿
        if self.follow_cam:
            #利用相机位姿计算相机跟随位姿,先沿着z轴负方向移动2个单位
            T_WC = cam_T_WC.matrix().numpy().astype(
                dtype=np.float32
            ) @ translation_matrix(np.array([0, 0, -2], dtype=np.float32))  # 计算相机跟随位姿
            #对相机位姿求逆,得到跟随相机
            self.camera.follow_cam(np.linalg.inv(T_WC))  # 跟随相机
        else:
            self.camera.unfollow_cam()  # 取消跟随相机
        self.frustums.add(
            cam_T_WC,
            scale=self.frustum_scale,
            color=[0, 1, 0, 1],
            thickness=self.line_thickness * self.scale,
        )  # 添加当前帧的视锥

        with self.keyframes.lock:
            N_keyframes = len(self.keyframes)  # 获取关键帧数量
            dirty_idx = self.keyframes.get_dirty_idx()  # 获取脏关键帧索引

        #将脏关键帧(用于标记是否未写入,若已经写入则标志位变化)的数据写入纹理
        for kf_idx in dirty_idx:
            keyframe = self.keyframes[kf_idx]  # 获取关键帧
            h, w = keyframe.img_shape.flatten()  # 获取关键帧图像高度和宽度
            X = self.frame_X(keyframe)  # 获取关键帧的点云数据
            C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)  # 获取关键帧的置信度数据

            if keyframe.frame_id not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)  # 创建点云纹理
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4) # 创建置信度纹理
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4) # 创建图像纹理
                self.textures[keyframe.frame_id] = ptex, ctex, itex   # 添加关键帧纹理
                ptex, ctex, itex = self.textures[keyframe.frame_id]
                itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())  # 写入关键帧图像数据
            # 获取关键帧纹理,并写入数据
            ptex, ctex, itex = self.textures[keyframe.frame_id]
            ptex.write(X.tobytes())  # 写入关键帧点云数据
            ctex.write(C.tobytes())  # 写入关键帧置信度数据

        for kf_idx in range(N_keyframes):
            keyframe = self.keyframes[kf_idx]  # 获取关键帧
            h, w = keyframe.img_shape.flatten()  # 获取关键帧图像高度和宽度
            if kf_idx == N_keyframes - 1:
                self.kf_img_np = keyframe.uimg.numpy()  # 获取最后一个关键帧的图像数据
                self.kf_img.write(self.kf_img_np)  # 写入最后一个关键帧的图像数据

            color = [1, 0, 0, 1]  # 设置颜色
            if self.show_keyframe:
                self.frustums.add(
                    as_SE3(keyframe.T_WC.cpu()),
                    scale=self.frustum_scale,
                    color=color,
                    thickness=self.line_thickness * self.scale,
                )  # 添加关键帧的视锥

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            if self.show_all:
                self.render_pointmap(keyframe.T_WC.cpu(), w, h, ptex, ctex, itex)  # 渲染关键帧点云图

        if self.show_keyframe_edges:
            with self.states.lock:
                ii = torch.tensor(self.states.edges_ii, dtype=torch.long)
                jj = torch.tensor(self.states.edges_jj, dtype=torch.long)
                if ii.numel() > 0 and jj.numel() > 0:
                    T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                    T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
            if ii.numel() > 0 and jj.numel() > 0:
                t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
                t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )  # 添加关键帧边缘线条
        if self.show_curr_pointmap and self.states.get_mode() != Mode.INIT:
            if config["use_calib"]:
                curr_frame.K = self.keyframes.get_intrinsics()  # 获取相机内参
            h, w = curr_frame.img_shape.flatten()  # 获取当前帧图像高度和宽度
            X = self.frame_X(curr_frame)  # 获取当前帧的点云数据
            C = curr_frame.C.cpu().numpy().astype(np.float32)  # 获取当前帧的置信度数据
            if "curr" not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures["curr"] = ptex, ctex, itex
            ptex, ctex, itex = self.textures["curr"]
            ptex.write(X.tobytes())  # 写入当前帧点云数据
            ctex.write(C.tobytes())  # 写入当前帧置信度数据
            itex.write(depth2rgb(X[..., -1], colormap="turbo"))  # 写入当前帧深度图
            self.render_pointmap(
                curr_frame.T_WC.cpu(),
                w,
                h,
                ptex,
                ctex,
                itex,
                use_img=True,
                depth_bias=self.depth_bias,
            )  # 渲染当前帧点云图

        self.lines.render(self.camera)  # 渲染线条
        self.frustums.render(self.camera)  # 渲染视锥
        self.render_ui()  # 渲染UI

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        # get window size and full screen
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()
        imgui.begin("GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        new_state = WindowMsg()
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )

        imgui.spacing()

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)

        imgui.spacing()
        shader_options = [
            "surfelmap.glsl",
            "trianglemap.glsl",
        ]
        current_shader = shader_options.index(
            self.pointmap_prog.extra["meta"].resolved_path.name
        )

        for i, shader in enumerate(shader_options):
            if imgui.radio_button(shader, current_shader == i):
                current_shader = i

        selected_shader = shader_options[current_shader]
        if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
            self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

        imgui.spacing()

        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        imgui.spacing()

        _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
            "show_normal", self.pointmap_prog["show_normal"].value
        )
        imgui.same_line()
        _, self.culling = imgui.checkbox("culling", self.culling)
        if "radius" in self.pointmap_prog:
            _, self.pointmap_prog["radius"].value = imgui.drag_float(
                "radius",
                self.pointmap_prog["radius"].value,
                0.0001,
                min_value=0.0,
                max_value=0.1,
            )
        if "slant_threshold" in self.pointmap_prog:
            _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                "slant_threshold",
                self.pointmap_prog["slant_threshold"].value,
                0.1,
                min_value=0.0,
                max_value=1.0,
            )
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_curr_pointmap = imgui.checkbox(
            "show_curr_pointmap", self.show_curr_pointmap
        )
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )

        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        imgui.spacing()

        gui_size = imgui.get_content_region_available()
        scale = gui_size[0] / self.curr_img.texture.size[0]
        scale = min(self.scale, scale)
        size = (
            self.curr_img.texture.size[0] * scale,
            self.curr_img.texture.size[1] * scale,
        )
        image_with_text(self.kf_img, size, "kf", same_line=False)
        image_with_text(self.curr_img, size, "curr", same_line=False)

        imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def render_pointmap(self, T_WC, w, h, ptex, ctex, itex, use_img=True, depth_bias=0):
        """
        渲染点云图。

        :param T_WC: 相机位姿矩阵。
        :param w: 图像宽度。
        :param h: 图像高度。
        :param ptex: 点云纹理。
        :param ctex: 置信度纹理。
        :param itex: 图像纹理。
        :param use_img: 是否使用图像纹理。
        :param depth_bias: 深度偏移。
        """
        w, h = int(w), int(h)  # 将宽度和高度转换为整数
        ptex.use(0)  # 绑定点云纹理到纹理单元 0
        ctex.use(1)  # 绑定置信度纹理到纹理单元 1
        itex.use(2)  # 绑定图像纹理到纹理单元 2
        model = T_WC.matrix().numpy().astype(np.float32).T  # 获取相机位姿矩阵并转置

        #这里矩阵转置是因为在GLSL中矩阵是列主序的,而在Python中矩阵是行主序的
        vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)  # 创建顶点数组对象
        vao.program["m_camera"].write(self.camera.gl_matrix())  # 写入相机矩阵
        vao.program["m_model"].write(model)  # 写入模型矩阵
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())  # 写入投影矩阵

        vao.program["pointmap"].value = 0  # 设置点云纹理单元
        vao.program["confs"].value = 1  # 设置置信度纹理单元
        vao.program["img"].value = 2  # 设置图像纹理单元
        vao.program["width"].value = w  # 设置图像宽度
        vao.program["height"].value = h  # 设置图像高度
        vao.program["conf_threshold"] = self.state.C_conf_threshold  # 设置置信度阈值
        vao.program["use_img"] = use_img  # 设置是否使用图像纹理
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias  # 设置深度偏移
        vao.render(mode=moderngl.POINTS, vertices=w * h)  # 渲染点云图
        vao.release()  # 释放顶点数组对象

    def frame_X(self, frame):
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]

        return frame.X_canon.cpu().numpy().astype(np.float32)


def run_visualization(states, keyframes, main2viz, viz2main, config_path) -> None:
    """
        运行可视化的主函数。

        :param states: 包含系统状态的对象。
        :param keyframes: 关键帧数据对象。
        :param main2viz: 从主进程到可视化进程的通信队列。
        :param viz2main: 从可视化进程到主进程的通信队列。
        :param config_path: 配置文件的路径。
    """
    # 加载配置文件
    load_config(config_path)

    config_cls = Window #config指向Window类
    backend = "glfw"   #设置窗口管理的后端为 glfw
    window_cls = mglw.get_local_window_cls(backend) #获取后端对应的窗口类

    # 创建窗口实例
    window = window_cls(
        title=config_cls.title, #窗口标题
        size=config_cls.window_size, #窗口大小
        fullscreen=False, #非全屏模式
        resizable=True, #可调整窗口大小
        visible=True, #窗口可见
        gl_version=(3, 3), #OpenGL版本
        aspect_ratio=None, #窗口宽高比
        vsync=True, #开启垂直同步
        samples=4, #抗拒齿采样
        cursor=True, #显示鼠标指针
        backend=backend, #后端
    )
    window.print_context_info() #打印上下文信息
    mglw.activate_context(window=window) #激活上下文
    window.ctx.gc_mode = "auto" #垃圾回收模式
    timer = Timer() #计时器
    # 创建窗口配置
    window_config = config_cls(
        states=states,  #系统状态
        keyframes=keyframes, #关键帧数据
        main2viz=main2viz,  #主进程到可视化进程的通信队列
        viz2main=viz2main,  #可视化进程到主进程的通信队列
        ctx=window.ctx,   #窗口上下文
        wnd=window,       #窗口对象
        timer=timer,    #计时器
    )
    # 暂时避免在属性设置器中分配事件
    # 我们希望事件分配发生在 WindowConfig.__init__ 中
    # 这样用户可以自由地在自己的 __init__ 中分配它们。
    window._config = weakref.ref(window_config)

    # 在开始主循环之前交换一次缓冲区。
    # 这可以触发额外的调整大小事件，报告更准确的缓冲区大小
    window.swap_buffers()
    window.set_default_viewport()   #设置默认视口

    timer.start() #开始计时器

    # 主循环
    while not window.is_closing:
        current_time, delta = timer.next_frame()  #获取当前时间和时间间隔
        if window_config.clear_color is not None:
            # 如果设置了清除颜色，使用该颜色清除窗口
            window.clear(*window_config.clear_color)

        # 在调用 render 之前始终绑定窗口帧缓冲区
        window.use()

        window.render(current_time, delta) #渲染窗口
        if not window.is_closing:
            window.swap_buffers() #交换缓冲区,显示渲染结果

    _, duration = timer.stop() #停止计时器
    window.destroy() #销毁窗口
    viz2main.put(WindowMsg(is_terminated=True)) #向主进程发送终止消息
