#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
    } \
}

// ---------------------- 核心 Kernel (不变) ----------------------
// 这里依然是你那段 0.03ms 的双线性插值代码
__global__ void preprocess_bilinear_kernel(
    const uchar4* __restrict__ src,
    float* __restrict__ dst,
    int src_width, int src_height,
    int dst_width, int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    int x_low = (int)floorf(src_x);
    int y_low = (int)floorf(src_y);
    int x_high = x_low + 1;
    int y_high = y_low + 1;

    x_low = max(0, min(x_low, src_width - 1));
    y_low = max(0, min(y_low, src_height - 1));
    x_high = max(0, min(x_high, src_width - 1));
    y_high = max(0, min(y_high, src_height - 1));

    float lx = src_x - x_low;
    float ly = src_y - y_low;
    float hx = 1.0f - lx;
    float hy = 1.0f - ly;

    uchar4 v1 = src[y_low * src_width + x_low];
    uchar4 v2 = src[y_low * src_width + x_high];
    uchar4 v3 = src[y_high * src_width + x_low];
    uchar4 v4 = src[y_high * src_width + x_high];

    float b = (v1.x * hx * hy + v2.x * lx * hy + v3.x * hx * ly + v4.x * lx * ly);
    float g = (v1.y * hx * hy + v2.y * lx * hy + v3.y * hx * ly + v4.y * lx * ly);
    float r = (v1.z * hx * hy + v2.z * lx * hy + v3.z * hx * ly + v4.z * lx * ly);

    int area = dst_width * dst_height;
    int idx = y * dst_width + x;

    dst[idx] = r / 255.0f;
    dst[idx + area] = g / 255.0f;
    dst[idx + area * 2] = b / 255.0f;
}

// ---------------------- 导出接口 (Python 调用这个) ----------------------
// extern "C": 告诉 C++ 编译器保持函数名不变
// __declspec(dllexport): 告诉 Windows 这是一个可以被外部调用的 DLL 函数
extern "C" __declspec(dllexport)
void launch_preprocess(
    void* src_ptr,      // 输入显存指针 (Python 会传过来一个整数地址)
    void* dst_ptr,      // 输出显存指针
    int src_width, int src_height,
    int dst_width, int dst_height
) {
    // 强制转换指针类型
    const uchar4* d_src = (const uchar4*)src_ptr;
    float* d_dst = (float*)dst_ptr;

    // 计算 Grid 和 Block
    dim3 block(32, 32);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

    // 启动 Kernel (这里是异步的，Python 调用完立刻返回，不用等 GPU 算完)
    preprocess_bilinear_kernel << <grid, block >> > (d_src, d_dst, src_width, src_height, dst_width, dst_height);

    // 只有调试时才加同步检查错误，生产环境为了速度可以去掉
    CHECK_CUDA(cudaGetLastError());
    // cudaDeviceSynchronize(); // 生产环境不要加这个，让 Python 去做流水线
}