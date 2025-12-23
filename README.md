# 🚀 High-Performance Visual Servo System (TensorRT & CUDA)

[![Language](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.6-orange.svg)](https://developer.nvidia.com/tensorrt)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **An ultra-low latency (<5ms) computer vision pipeline featuring custom CUDA kernels, TensorRT INT8 quantization, and a closed-loop PID controller.**

---

## 📖 Introduction (项目简介)

This project implements a high-frequency **Visual Servo Control System** designed for real-time object tracking and actuation. 

Unlike traditional Python-based pipelines that suffer from CPU bottlenecks and memory copy overhead, this project leverages **Heterogeneous Computing**. By offloading preprocessing (Resize/Normalize/HWC2CHW) and postprocessing (NMS/Parallel Reduction) to custom **CUDA Kernels**, and utilizing **TensorRT INT8** inference, the system achieves physical hardware limits in terms of latency and throughput.

**Key Applications:**
* Industrial Robot Visual Servoing (High-speed sorting/grabbing)
* UAV/Drone Target Tracking
* Optical Feedback Control Systems

---

## ⚡ Performance Benchmark (性能对决)

The core optimization goal was to minimize **End-to-End Latency** (Photon-to-Action).

| Pipeline Stage | Baseline (OpenCV CPU) | **Optimized (CUDA + TensorRT)** | **Speedup** |
| :--- | :--- | :--- | :--- |
| **Preprocessing** | ~10.00 ms | **0.03 ms** | **~300x** 🚀 |
| **Inference** | ~15.00 ms (FP32) | **2.10 ms (INT8)** | **~7x** |
| **Postprocessing** | ~2.00 ms (CPU NMS) | **0.05 ms (GPU Reduction)** | **~40x** |
| **Total Latency** | **> 30 ms** | **< 5 ms** | **~6x** |

### 📊 Nsight Systems Profiling Evidence

**Before Optimization (CPU Bound):**
*Note the sparse GPU utilization and large gaps due to GIL and PCI-e transfer overhead.*
![Baseline Profiling](assets/baseline_cpu.png)

**After Optimization (Fully Saturated Pipeline):**
*GPU is fully utilized with tightly packed kernels. Preprocessing and Inference are fused in a single CUDA Stream.*
![Optimized Profiling](assets/ScreenShot_2025-12-23_195325_523.png)![Optimized Profiling](assets/ScreenShot_2025-12-23_195338_663.png)![Optimized Profiling](assets/ScreenShot_2025-12-23_195417_448.png)

---

## 🏗️ System Architecture (系统架构)

The system adopts a **Producer-Consumer model**. The Python layer handles the high-level control logic (PID), while the C++ shared library handles the heavy lifting on the GPU.

```mermaid
graph TD
    subgraph "Host (CPU) - Control Plane"
        A[DXCam Sensor Input] -->|Raw Frame| B(Pinned Memory Buffer)
        G[Data Parsing] -->|Candidates| H{PID Controller}
        H -->|Tracking Signal| I[Actuator Interface]
    end

    subgraph "Device (GPU) - Compute Plane"
        B -->|HtoD Async Copy| C[CUDA Stream]
        subgraph "Custom Kernels"
            C --> D[Preprocess Kernel<br/>Bilinear Interpolation]
            D --> E[TensorRT Engine<br/>INT8 Inference]
            E --> F[Postprocess Kernel<br/>Parallel Reduction Top-K]
        end
        F -->|DtoH Async Copy| G
    end

    style D fill:#f96,stroke:#333,stroke-width:2px
    style F fill:#f96,stroke:#333,stroke-width:2px
    style E fill:#ff9,stroke:#333,stroke-width:4px
🛠️ Key Technologies (核心技术)
1. Custom CUDA Preprocessing
Replaced cv2.resize and cv2.cvtColor with a fused kernel written in CUDA C++.

Technique: Hand-written Bilinear Interpolation.

Optimization: Uses uchar4 vector loading for coalesced global memory access. Performs normalization and layout transposition (NHWC -> NCHW) in registers.

2. Parallel Reduction Post-Processing
Instead of copying 8400 anchor boxes to CPU for NMS, I implemented a Tree-Based Reduction kernel on GPU.

Technique: Each CUDA block computes the local best candidate using Shared Memory (__shared__).

Result: Only top-k candidates are transferred back to Host, reducing DtoH bandwidth usage by 99%.

3. TensorRT INT8 Quantization
Calibrated the YOLO model on a custom dataset to generate an INT8 engine.

Achieved 40% latency reduction compared to FP16 with negligible accuracy loss (<1% mAP drop).

4. Sub-pixel PID Controller
Implemented a custom PID logic with Sub-pixel Accumulation.

Handles floating-point control signals and converts them to discrete actuator steps, ensuring smooth tracking even at high frame rates (144Hz+).

💻 Build & Run (构建指南)
This project is optimized for Windows and Visual Studio.

Prerequisites
Windows 10/11

Visual Studio 2019/2022 (with C++ Desktop Development)

NVIDIA CUDA Toolkit 11.8+

TensorRT 8.x

Python 3.8+

Step 1: Compile the Core Library (DLL)
Open the solution file DeepVisualServo.sln in Visual Studio.

Select Release configuration and x64 platform.

Right-click Project -> Properties:

C/C++ / General / Additional Include Directories: Add paths to CUDA include and TensorRT include.

Linker / General / Additional Library Directories: Add paths to CUDA lib/x64 and TensorRT lib.

Linker / Input: Ensure cudart.lib, nvinfer.lib are added.

Build the solution.

Important: Copy the generated yoloC.dll from x64/Release/ to the root directory src/.

Step 2: Install Python Dependencies
Bash

pip install -r requirements.txt
Step 3: Run the System
Bash

python src/visual_servo_loop.py
Home Key: Toggle Tracking Mode.

End Key: Safe Exit.
