#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cfloat> // for FLT_MAX

// 定义一个结构体，用来存“候选人”信息
// 我们只需要知道它的位置、距离、和是否有效
struct Target {
    float x;
    float y;
    float dist; // 距离屏幕中心的距离 (越小越好)
    int valid;  // 1=有效, 0=无效
};

// ---------------------- 核心 Kernel: 寻找最近目标 ----------------------
__global__ void find_best_target_kernel(
    const float* __restrict__ output, // TRT输出: [1, 56, 8400]
    float* __restrict__ best_arr,     // 暂存每个 Block 的冠军: [GridDim, 3]
    int num_anchors,                  // 8400
    int num_channels,                 // 56
    float center_x, float center_y,   // 屏幕/瞄准中心 (如 320, 320)
    float conf_thres                // 置信度阈值 (0.6)
) {
    // 1. 声明共享内存：用来在 Block 内部打擂台
    // 假设 BlockSize = 256，我们需要存 256 个 Target
    __shared__ Target smem[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 每个线程先读取自己的那个框 (Load)
    // 初始化为“无效”
    smem[tid].dist = FLT_MAX;
    smem[tid].valid = 0;

    if (idx < num_anchors) {
        // YOLO 输出布局通常是 [Batch, Channel, Anchor] -> Flat array
        // Channel 0: x, 1: y, 2: w, 3: h, 4: conf ...
        // 索引计算：channel * num_anchors + idx

        float score = output[4 * num_anchors + idx]; // 读取置信度

        if (score > conf_thres) {
            float bx = output[0 * num_anchors + idx]; // cx
            float by = output[1 * num_anchors + idx]; // cy
            // 注意：如果模型输出是归一化的(0-1)，这里需要乘图宽图高
            // 假设你的 TRT 输出已经是像素坐标 (640x640)

            // 计算距离 (欧氏距离平方，不开根号更省资源)
            float dx = bx - center_x;
            float dy = by - center_y;
            float d2 = dx * dx + dy * dy;

            // 写入 Shared Memory
            smem[tid].x = bx;
            smem[tid].y = by;
            smem[tid].dist = d2;
            smem[tid].valid = 1;
        }
    }
    __syncthreads(); // 等大家读完

    // 3. 并行规约 (Reduction) - 擂台赛开始！
    // 每一轮，比较的人数减半。
    // 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 只有前一半的线程干活
            // 比较 smem[tid] 和 smem[tid + s]
            // 谁的距离更小，谁就晋级留在 smem[tid]
            if (smem[tid + s].dist < smem[tid].dist) {
                smem[tid] = smem[tid + s];
            }
        }
        __syncthreads(); // 每一轮比完都要同步
    }

    // 4. Block 冠军诞生
    // 现在的 smem[0] 就是本 Block 里距离最小的那个
    if (tid == 0) {
        // 把结果写到 Global Memory 的临时数组里
        // 格式：[BlockID * 3 + 0/1/2] -> x, y, dist
        if (smem[0].valid) {
            best_arr[blockIdx.x * 3 + 0] = smem[0].x;
            best_arr[blockIdx.x * 3 + 1] = smem[0].y;
            best_arr[blockIdx.x * 3 + 2] = smem[0].dist;
        }
        else {
            // 如果整个 Block 都没有有效框
            best_arr[blockIdx.x * 3 + 2] = FLT_MAX;
        }
    }
}

// ---------------------- 导出接口 ----------------------
extern "C" __declspec(dllexport)
void launch_find_best(
    void* output_ptr,      // TRT 输出指针
    void* best_arr_ptr,    // 结果暂存指针
    int num_anchors,       // 8400
    int num_channels,      // 56
    float cx, float cy,    // 瞄准中心
    float conf_thres       // 阈值
) {
    const float* d_output = (const float*)output_ptr;
    float* d_best = (float*)best_arr_ptr;

    int threads = 256;
    int blocks = (num_anchors + threads - 1) / threads;

    find_best_target_kernel << <blocks, threads >> > (
        d_output, d_best,
        num_anchors, num_channels,
        cx, cy, conf_thres
        );
}