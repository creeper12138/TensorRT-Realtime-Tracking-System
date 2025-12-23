import time
import ctypes
import os
import numpy as np
import dxcam
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from pynput.keyboard import Key, Listener
from multiprocessing import Value, Process
import psutil

# 假设 PID 模块现在是一个通用的控制器类
import PID 

# ================= ⚙️ 系统配置 (System Config) =================
# [工程化命名]: 避免使用 absolute path，改用相对路径
# [功能描述]: 加载包含 CUDA 预处理/后处理 Kernel 的动态链接库
KERNEL_LIB_PATH = os.path.join(os.path.dirname(__file__), "libs", "acceleration_backend.dll")
ENGINE_PATH = os.path.join("models", "yolov8m_int8.engine")

# [视觉伺服配置]
# 定义感兴趣区域 (Region of Interest, ROI)
ROI_REGION = (960, 480, 1600, 1120)

# [控制算法参数]
# Kp: 比例增益 (Proportional Gain)。
# 0.25 表示以当前误差的 25% 速率进行收敛，避免过冲震荡。
TRACKING_GAIN_SMOOTH = 0.25 
TRACKING_GAIN_AGGRESSIVE = 0.45 

# 追踪死区 (Deadzone): 误差小于此像素值时停止修正，防止伺服抖动
DEADZONE_THRESHOLD = 1.5

# ================= 全局状态控制 =================
# system_active: 控制追踪系统是否介入
system_active = Value('b', True)
stop_signal = Value('b', False)

def input_listener(key):
    """
    监听键盘输入以切换系统状态
    Home: 切换 追踪模式 / 待机模式
    End:  安全退出系统
    """
    if key == Key.home:
        system_active.value = not system_active.value
        state = "【ACTIVE】Tracking Enabled" if system_active.value else "【STANDBY】Idle"
        print(f"[System State] changed to: {state}")
    elif key == Key.end:
        stop_signal.value = True
        return False
    return True

def tracking_pipeline(active_flag, stop_signal):
    """
    主追踪流水线 (Main Tracking Pipeline)
    流程: Screen Capture -> CUDA Preprocessing -> TensorRT Inference -> CUDA Postprocessing -> PID Control
    """
    # 1. 进程优先级提升 (Real-time Priority)
    try:
        psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)
        print("[System] Process priority set to HIGH.")
    except Exception as e:
        print(f"[System] Warning: Failed to set priority: {e}")

    # 2. 加载加速后端 (CUDA Kernels)
    if not os.path.exists(KERNEL_LIB_PATH):
        print(f"[Error] Kernel library not found at: {KERNEL_LIB_PATH}")
        return
        
    lib = ctypes.CDLL(KERNEL_LIB_PATH)
    # 定义 Kernel 函数签名 (预处理 & 后处理)
    lib.launch_preprocess.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.launch_find_best.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]

    # 3. 加载 TensorRT 引擎
    logger = trt.Logger(trt.Logger.WARNING)
    try:
        with open(ENGINE_PATH, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
    except FileNotFoundError:
        print(f"[Error] Engine file not found: {ENGINE_PATH}")
        return

    context = engine.create_execution_context()
    stream = cuda.Stream()

    # 4. 分配显存 (Memory Allocation)
    src_w = ROI_REGION[2] - ROI_REGION[0]
    src_h = ROI_REGION[3] - ROI_REGION[1]
    model_input_w, model_input_h = 640, 640

    # Device Memory (GPU)
    d_raw_frame = cuda.mem_alloc(src_w * src_h * 4)       # 原始截图数据
    d_input_tensor = cuda.mem_alloc(3 * model_input_w * model_input_h * 4) # 归一化后的 Tensor
    d_output_tensor = cuda.mem_alloc(8400 * 56 * 4)       # 模型原始输出
    d_best_target = cuda.mem_alloc(64 * 3 * 4)            # 筛选后的最佳目标
    
    # Host Memory (CPU - Pinned for DMA)
    h_target_data = cuda.pagelocked_empty(64 * 3, dtype=np.float32)

    bindings = [int(d_input_tensor), int(d_output_tensor)]
    
    # 误差积分累加器 (用于亚像素级移动)
    accum_error_x = 0.0
    accum_error_y = 0.0
    
    camera = None
    print(f">>> Tracking System Initialized. ROI: {ROI_REGION}")

    try:
        while not stop_signal.value:
            # --- 阶段 A: 图像采集 (Acquisition) ---
            if camera is None:
                try:
                    # 使用 DXGI 进行低延迟捕获
                    camera = dxcam.create(output_idx=0, output_color="BGRA")
                    camera.start(target_fps=120, region=ROI_REGION)
                except Exception:
                    time.sleep(1)
                    continue

            if not active_flag.value:
                time.sleep(0.01)
                accum_error_x, accum_error_y = 0.0, 0.0
                continue

            frame = camera.get_latest_frame()
            if frame is None: continue
            
            # 使用 ascontiguousarray 确保内存布局连续，便于 CUDA 拷贝
            frame = np.ascontiguousarray(frame)

            try:
                # --- 阶段 B: 异构计算流水线 (Heterogeneous Pipeline) ---
                
                # 1. Host to Device (DMA Copy)
                cuda.memcpy_htod_async(d_raw_frame, frame, stream)
                
                # 2. Pre-processing (CUDA Kernel: Resize + Normalize + Color Convert)
                lib.launch_preprocess(int(d_raw_frame), int(d_input_tensor), src_w, src_h, model_input_w, model_input_h)
                
                # 3. Inference (TensorRT)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                
                # 4. Post-processing (CUDA Kernel: Decode + NMS + Best Selection)
                # 直接在 GPU 上筛选置信度 > 0.65 的目标
                lib.launch_find_best(int(d_output_tensor), int(d_best_target), 8400, 56, 320.0, 320.0, 0.65)
                
                # 5. Device to Host (Result Copy)
                cuda.memcpy_dtoh_async(h_target_data, d_best_target, stream)
                stream.synchronize()

                # --- 阶段 C: 视觉伺服控制 (Visual Servoing Control) ---
                candidates = h_target_data.reshape(-1, 3) # [x, y, confidence]
                
                # 过滤无效数据 (Confidence > 1.0 为标记位)
                valid_mask = (candidates[:, 2] > 1.0) & (candidates[:, 2] < 100000.0)
                valid_targets = candidates[valid_mask]

                if len(valid_targets) > 0:
                    # 简单策略：选择置信度最高的目标
                    best_idx = np.argmin(valid_targets[:, 2])
                    target = valid_targets[best_idx]

                    # 坐标映射 (ROI -> Screen Space)
                    raw_x = target[0] + ROI_REGION[0]
                    raw_y = target[1] + ROI_REGION[1]
                    
                    # 目标中心点 (Target Center)
                    target_center_y = raw_y 
                    # 如果需要追踪特定部位(如车辆顶部)，可在此处添加 ROI Offset 计算
                    # target_center_y -= VERTICAL_OFFSET 

                    # 计算位置误差 (Position Error)
                    # 假设画面中心为 (1280, 800)
                    error_x = raw_x - 1280
                    error_y = target_center_y - 800

                    # 动态增益调度 (Gain Scheduling)
                    # 大误差使用高响应，小误差使用平滑响应
                    current_gain = TRACKING_GAIN_SMOOTH
                    
                    # 死区检测 (Deadzone Check)
                    if abs(error_x) < DEADZONE_THRESHOLD and abs(error_y) < DEADZONE_THRESHOLD:
                        accum_error_x, accum_error_y = 0.0, 0.0
                        continue

                    # 计算控制量 (Control Signal Calculation)
                    control_x = error_x * current_gain
                    control_y = error_y * current_gain * 0.7 

                    # 亚像素累积 (Sub-pixel Accumulation)
                    accum_error_x += control_x
                    accum_error_y += control_y
                    
                    output_x = int(accum_error_x)
                    output_y = int(accum_error_y)
                    
                    accum_error_x -= output_x
                    accum_error_y -= output_y

                    if output_x != 0 or output_y != 0:
                        # 发送控制信号
                        PID.move(output_x, output_y)

                else:
                    accum_error_x, accum_error_y = 0.0, 0.0

            except Exception as e:
                # 异常处理：重启摄像头
                if camera:
                    camera.stop()
                    del camera
                    camera = None

    except KeyboardInterrupt: pass
    finally:
        if camera:
            camera.stop() 

if __name__ == "__main__":
    listener = Listener(on_release=input_listener)
    listener.start()
    
    # 启动追踪进程
    p = Process(target=tracking_pipeline, args=(system_active, stop_signal))
    p.start()
    
    try:
        while not stop_signal.value: time.sleep(1)
    except KeyboardInterrupt: stop_signal.value = True
    
    p.join()
    listener.stop()