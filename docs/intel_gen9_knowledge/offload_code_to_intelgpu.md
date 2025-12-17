# How to Offload Compute-Intensive Code to Intel® GPUs

---

**ID:** 861206  
**Updated:** 7/23/2020  
**Version:** Archived  
**Public**

---

## Table of Contents

1. [Overview](#overview)
2. [Intel® Processor Graphics: Architecture Overview by Gen](#intel-processor-graphics-architecture-overview-by-gen)
3. [oneAPI and DPC++](#oneapi-and-dpc)
4. [Case Study: Compute Kernel Execution on Intel® Processor Graphics](#case-study-compute-kernel-execution-on-intel-processor-graphics)
5. [Writing Better Algorithms](#writing-better-algorithms)

---

**Author:** Rama Malladi, graphics performance modeling engineer, Intel Corporation  
**Twitter:** @IntelDevTools

---

## Overview

Intel® Processor Graphics Architecture is an Intel technology that provides graphics, compute, media, and display capabilities for many of the system-on-a-chip (SoC) products from Intel. The Intel Processor Graphics Architecture is informally known as "Gen," shorthand for generation. Each release of the architecture has a corresponding version indicated after the word "Gen." For example, the latest release of Intel Processor Graphics Architecture is Gen11. Over the years, they have evolved to offer excellent graphics (3D rendering and media performance) and general-purpose compute capabilities with up to 1 TFLOPS (trillion floating-point operations per second) of performance.

In this article, we explore the general-purpose compute capabilities of the Gen9 and Gen11 Intel Processor Graphics Architectures and how to program them using Data Parallel C++ (DPC++) in the Intel® oneAPI Base Toolkit. Specifically, we look at a case study that shows programming and performance aspects of the two Gen architectures using DPC++.

---

## Intel® Processor Graphics: Architecture Overview by Gen

Intel® Processor Graphics is a power-efficient, high-performance graphics and media accelerator integrated on-die with the Intel CPU. The integrated GPU shares the last-level cache (LLC) with the CPU, which permits fine-grained, coherent data sharing at low latency and high bandwidth. The on-die integration enables much lower power consumption than a discrete graphics card.

### Figure 1: Intel Processor Graphics Gen11 SoC

![Gen11 SoC Architecture Diagram]()

*A block diagram illustrating the architecture of an Intel Processor Graphics Gen11 SoC. The diagram is divided into several sections:*

**Left side (GT - Graphics Technology):**
- Global Assets
- Media Fixed Function
- Blitter
- Multiple SubSlices containing EUs (Execution Units), Samplers, and SLMs (Shared Local Memory)
- SubSlices organized into Geometry, Raster, HZDepth, Pixel Dispatch, Pixel Backend, and L3S components

**Right side (CPU/System):**
- System Agent
- CPU Cores connected to the SoC Ring Interconnect
- LLC Cache slice
- Memory Controller
- Display Controller and PCIe connected to the SoC Ring Interconnect

*Blue arrows indicate data flow between components.*

---

### Figure 2: Gen9 GPU Architecture

![Gen9 GPU Architecture Diagram]()

*A diagram illustrating the Gen9 GPU Architecture divided into "Non-coherent" and "Coherent" sections:*

**Intel Core Processor section:**
- CPU core connected to CPU L1$ and CPU L2$

**Intel Processor Graphics section:**
- Two "Slice: 24 EUs" blocks
- Each slice contains three "Subslice: 8 EUs" blocks
- Each subslice has a "Sampler" connected to L1$ and L2$
- "Shared Local Memory (64KB/subslice)" 
- "L3 Data Cache (512 KB/slice)"
- "L3 Fabric" connecting the two L3 Data Caches

**External Memory:**
- "Shared LLC" connects to:
  - "(Optional) On-Package EDRAM"
  - "System DRAM"

**Markdown Diagram Recreation:**

```
                        NON-COHERENT                              COHERENT
┌────────────────────────────────────────────────────────────────────────────────┐
│                            INTEL CORE PROCESSOR                              │
│   ┌─────────────┐                                                             │
│   │  CPU core   │──▶ CPU L1$ ──▶ CPU L2$                                      │
│   └─────────────┘                                                             │
└────────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────────────┐
│                        INTEL PROCESSOR GRAPHICS                              │
│  ┌───────────────────────────────┐   ┌───────────────────────────────┐       │
│  │      Slice: 24 EUs            │   │      Slice: 24 EUs            │       │
│  ├───────────────────────────────┤   ├───────────────────────────────┤       │
│  │ ┌───────┐┌───────┐┌───────┐  │   │ ┌───────┐┌───────┐┌───────┐  │       │
│  │ │SubSl. ││SubSl. ││SubSl. │  │   │ │SubSl. ││SubSl. ││SubSl. │  │       │
│  │ │ 8 EUs ││ 8 EUs ││ 8 EUs │  │   │ │ 8 EUs ││ 8 EUs ││ 8 EUs │  │       │
│  │ ├───────┤├───────┤├───────┤  │   │ ├───────┤├───────┤├───────┤  │       │
│  │ │Sampler││Sampler││Sampler│  │   │ │Sampler││Sampler││Sampler│  │       │
│  │ │L1$/L2$││L1$/L2$││L1$/L2$│  │   │ │L1$/L2$││L1$/L2$││L1$/L2$│  │       │
│  │ └───────┘└───────┘└───────┘  │   │ └───────┘└───────┘└───────┘  │       │
│  └───────────────────────────────┘   └───────────────────────────────┘       │
│         │                                    │                              │
│  ┌──────┴───────────────┐ ┌───────────┴───────────────┐              │
│  │Shared Local Memory│ │Shared Local Memory │              │
│  │   (64KB/subslice) │ │   (64KB/subslice)  │              │
│  └────────────────────┘ └─────────────────────┘              │
│         │                            │                                      │
│  ┌──────┴───────────────┐ ┌───────┴────────────────┐              │
│  │  L3 Data Cache    │ │  L3 Data Cache     │              │
│  │  (512 KB/slice)   │ │  (512 KB/slice)    │              │
│  └────────┬───────────┘ └─────────┬───────────┘              │
│            └─────────┬─────────┘                                  │
│                      │  L3 Fabric                                          │
└──────────────────────┼────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌───────────────────────┐
              │      Shared LLC        │
              └───────────┬───────────┘
                          │
           ┌─────────────┼─────────────┐
           ▼                           ▼
  ┌─────────────────────┐    ┌─────────────────┐
  │  (Optional)          │    │   System DRAM   │
  │ On-Package EDRAM     │    │                 │
  └─────────────────────┘    └─────────────────┘
```

---

### Figure 3: Subslice and EU Architecture Details

![Subslice and EU Architecture]()

*A diagram illustrating the architecture of a Subslice with 8 Execution Units (EUs) and a detailed view of a single Execution Unit:*

**Subslice Diagram:**
- Top: Instruction cache and Local Thread Dispatcher
- Two columns of four EUs each (total 8 EUs)
- Bottom left: Sampler with L1 and L2 Sampler Cache (Read: 64B/cyc)
- Bottom right: Data Port (Read: 64B/cyc, Write: 64B/cyc)
- A dashed red line connects one EU to the detailed EU diagram

**EU (Execution Unit) Detail Diagram:**
- Instruction Fetch unit at top
- 28KB GRF (General Register File): 7 thrds x 128x SIMD8 x 32b
- ARF (Address Register File)
- Thread Arbiter connecting to:
  - Send unit
  - Branch unit
  - Two SIMD FPU units

**Markdown Diagram Recreation:**

```
┌───────────────────────────────────────────────────┐
│             SUBSLICE: 8 EUs                     │
├───────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌────────────────────┐  │
│  │  Instruction cache  │  │Local Thread Disp.│  │
│  └─────────────────────┘  └────────────────────┘  │
│  ┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐         │
│  │  EU   │ │  EU   │  │  EU   │ │  EU   │         │
│  └───────┘ └───────┘  └───────┘ └───────┘         │
│  ┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐         │
│  │  EU   │ │  EU   │  │  EU   │ │  EU   │──────┐  │
│  └───────┘ └───────┘  └───────┘ └───────┘     │  │
├───────────────────────┬───────────────────────────┤     │  │
│      Sampler           │        Data Port         │     │  │
│  L1 & L2 Sampler Cache │   Read: 64B/cyc          │     │  │
│  Read: 64B/cyc         │   Write: 64B/cyc         │     │  │
└───────────────────────┴───────────────────────────┘     │  │
                                                          │  │
                   ┌─────────────────────────────────────┘  │
                   │        EU: Execution Unit               │
                   ├───────────────────────────────────────┤
                   │  ┌──────────────────┐                  │
                   │  │ Instruction Fetch  │                  │
                   │  └────────┬─────────┘                  │
                   │           │                              │
                   │  ┌────────┴──────────────┐  ┌───────┐   │
                   │  │     28KB GRF          │  │  ARF  │   │
                   │  │ 7 thrds x 128x        │  │       │   │
                   │  │ SIMD8 x 32b           │  │       │   │
                   │  └──────────┬────────────┘  └───────┘   │
                   │             │                            │
                   │  ┌──────────┴────────────┐               │
                   │  │     Thread Arbiter    │               │
                   │  └─┬─────┬─────┬──────┬─┘               │
                   │    │     │     │      │                  │
                   │    ▼     ▼     ▼      ▼                  │
                   │ ┌────┐┌──────┐┌────────┐┌────────┐   │
                   │ │Send││Branch││SIMD FPU││SIMD FPU│   │
                   │ └────┘└──────┘└────────┘└────────┘   │
                   └───────────────────────────────────────┘
```

**Key EU Specifications:**

| Component | Specification |
|-----------|---------------|
| Threads per EU | 7 |
| Registers per Thread | 128 x SIMD8 x 32-bit |
| GRF Size per EU | 28KB |
| Instructions per Cycle | Up to 4 (from different threads) |
| Peak GFLOPS Formula | (EUs) × (SIMD units/EU) × (FLOPS per cycle/SIMD unit) × (Freq GHz) |

---

## oneAPI and DPC++

oneAPI is an open, free, and standards-based programming model that provides portability and performance across accelerators and generations of hardware. oneAPI includes DPC++, the core programming language for code reuse across various hardware targets. You can find more details in my previous article, Heterogeneous Programming Using oneAPI ("The Parallel Universe," issue 39).

**DPC++ includes:**

* A unified shared memory (USM) feature for easy host-device memory management
* OpenCL™ platform style NDRange subgroups to aid vectorization
* Support for generic/function pointers
* And many other features

This article presents a case study that converts a CUDA* code to DPC++.

---

## Case Study: Compute Kernel Execution on Intel® Processor Graphics

Let's look at the Hogbom Clean imaging algorithm, widely used in processing radio astronomy images. This imaging algorithm has two hot spots:

* **Find Peak**
* **SubtractPSF**

For brevity, we'll focus on the performance aspects of Find Peak. The original implementation was in C++, OpenMP*, CUDA, and OpenCL™ software. The host CPU offloads the CUDA and OpenCL™ kernels onto the GPU when available. (CUDA is a proprietary approach to offload computations to only NVIDIA* GPUs.)

---

### Figure 4: Find Peak Host Code — C++, CUDA

![CUDA Host Code]()

*A code snippet showing CUDA host code for finding a peak value. The code includes memory allocation (cudaMalloc), a kernel launch (d_findPeak), memory transfer from device to host (cudaMemcpy), and a loop to find the peak value and its position.*

```cpp
// Find Peak - CUDA Host Code
cudaMalloc((void **) &d_peak, nBlocks * sizeof(Peak));
// Find Peak - CUDA Code
d_findPeak<<<nBlocks, findPeakWidth>>>(d_image, size, d_peak);
cudaMemcpy(&peaks, d_peak, nBlocks * sizeof(Peak), cudaMemcpyDeviceToHost);
Peak p;
for (int i = 0; i < nBlocks; ++i) {
    if (abs(peaks[i].val) > abs(p.val)) {
        p.val = peaks[i].val;
        p.pos = peaks[i].pos;
    }
}
```

---

### Figure 5: Find Peak Device Code — CUDA

```cpp
// Find Peak - CUDA Device Code
__shared__ float maxVal[findPeakWidth];
__shared__ size_t maxPos[findPeakWidth];
const int column = threadIdx.x + (blockIdx.x * blockDim.x);
for (int idx = column; idx < size; idx += 4096) {
    if (abs(image[idx]) > abs(maxVal[threadIdx.x])) {
        maxVal[threadIdx.x] = image[idx];
        maxPos[threadIdx.x] = idx;
    }
}
__syncthreads();
if (threadIdx.x == 0) {
    absPeak[blockIdx.x].val = 0.0;
    for (int i = 0; i < findPeakWidth; ++i) {
        if (abs(maxVal[i]) > abs(absPeak[blockIdx.x].val)) {
            absPeak[blockIdx.x].val = maxVal[i];
        }
    }
}
```

---

### Migrating CUDA to DPC++

We can manually replace the CUDA code with DPC++, or we can use the **Intel® DPC++ Compatibility Tool**. This tool assists with migrating CUDA programs to DPC++ (figures 6 and 7). It just requires the Intel oneAPI Base Toolkit and the NVIDIA CUDA header.

**Invoking the Intel DPC++ Compatibility Tool to migrate an example.cu file is as simple as:**

```bash
dpct example.cu
```

**For migrating applications with many CUDA files:**

```bash
intercept-build make
dpct -p=<path to .json file> --out-root=dpct_output
```

**Specifically, for migrating Hogbom Clean CUDA code to DPC++:**

```bash
dpct HogbomCuda.cu --out-root=MigratedCode --cuda-include-path=<CUDA-Headers>
```

**OR using intercept-build:**

```bash
intercept-build make
dpct -p=compile_commands.json --out-root=MigratedCode --cuda-include-path=<CUDA-Headers>
```

By default, the migrated code gets the file name extension `dp.cpp`.

---

### Figure 6: Find Peak DPC++ Host Code (Migrated)

```cpp
// Find peak - DPC++ Compatibility Tool Host Code Generated
static Peak findPeak (const float* d_image, size_t size)
{
    d_peak = (Peak *)sycl::malloc_device(nBlocks * sizeof(Peak), 
                                         dpct::get_current_device(),
                                         dpct::get_default_context());

    dpct::get_default_queue().submit([&] (sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access::mode::read_write, 
                       sycl::access::target::local>
            maxVal_acc_ctl(sycl::range<1>(1024 /*findPeakWidth*/), cgh);
        sycl::accessor<size_t, 1, sycl::access::mode::read_write, 
                       sycl::access::target::local>
            maxPos_acc_ctl(sycl::range<1>(1024 /*findPeakWidth*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nBlocks) * 
                              sycl::range<3>(1, 1, findPeakWidth),
                              sycl::range<3>(1, 1, findPeakWidth)),
            [=] (sycl::nd_item<3> item_ctl) {
                d_findPeak(d_image, size, d_peak, item_ctl,
                           maxVal_acc_ctl.get_pointer(),
                           maxPos_acc_ctl.get_pointer());
            });
    });
    dpct::get_default_queue().memcpy(&peaks, d_peak, 
                                     nBlocks * sizeof(Peak)).wait();
}
```

---

### Figure 7: Comparison of CUDA Host Code vs Migrated DPC++ Host Code

![Code Comparison]()

*A side-by-side comparison showing CUDA Host Code on the left and DPC++ Host Code on the right, demonstrating the device kernel call syntax differences.*

| CUDA Host Code | DPC++ Host Code |
|----------------|------------------|
| `// device kernel call by host.` | `dpct::get_default_queue().submit([&](` |
| `d_image and size are read inside` | `sycl::handler &cgh) {...` |
| `the d_findPeak kernel code.` | `cgh.parallel_for(` |
| `Output is written to d_peak.` | `d_findPeak(d_image, size, d_peak,` |
| | `item_ctl,` |
| `d_findPeak<<<nBlocks,` | `maxVal_acc_ctl.get_pointer(),` |
| `findPeakWidth>>>(d_image, size,` | `maxPos_acc_ctl.get_pointer());` |
| `d_peak);` | `...); // device queue, parallelism` |
| | `});` |

---

### Figure 8: Find Peak DPC++ Device Code (Migrated)

```cpp
void d_findPeak(const float* image, size_t size, Peak* absPeak, 
                sycl::nd_item<3> item_ctl, float *maxVal, size_t *maxPos)
{
    const int column = item_ctl.get_local_id(2) +
        (item_ctl.get_group(2) * item_ctl.get_local_range().get(2));

    maxVal[item_ctl.get_local_id(2)] = 0.0;
    maxPos[item_ctl.get_local_id(2)] = 0;

    for (int idx = column; idx < size; idx += 4096)
    {
        if (sycl::fabs(image[idx]) > sycl::fabs(maxVal[item_ctl.get_local_id(2)]))
        {
            maxVal[item_ctl.get_local_id(2)] = image[idx];
            maxPos[item_ctl.get_local_id(2)] = idx;
        }
    }

    item_ctl.barrier();
    if (item_ctl.get_local_id(2) == 0) {
        absPeak[item_ctl.get_group(2)].val = 0.0;
        absPeak[item_ctl.get_group(2)].pos = 0;
        for (int i = 0; i < findPeakWidth; ++i) {
            if (sycl::fabs(maxVal[i]) > sycl::fabs(absPeak[item_ctl.get_group(2)].val)) {
                absPeak[item_ctl.get_group(2)].val = maxVal[i];
                absPeak[item_ctl.get_group(2)].pos = maxPos[i];
            }
        }
    }
}
```

---

### Figure 9: Comparison of CUDA Kernel vs DPC++ Migrated Kernel

![Kernel Comparison]()

*A comparison table showing CUDA Kernel code on the left and DPC++ Migrated Kernel code on the right, highlighting the syntax transformations.*

| CUDA Kernel | DPC++ Migrated Kernel |
|-------------|------------------------|
| `__global__` | *(implicit in parallel_for)* |
| `void d_findPeak (const float* image, size_t size, Peak* absPeak) // device fn.` | `void d_findPeak (const float* image, size_t size, Peak* absPeak, sycl::nd_item<3> item_ctl, float *maxVal, size_t *maxPos)` |
| `__shared__ float maxVal[findPeakWidth];` | `float *maxVal // Passed as function param` |
| `maxVal[threadIdx.x] = ... // local write` | `maxVal[item_ctl.get_local_id(2)] = ... // nd_range accessor` |
| `__syncthreads();` | `item_ctl.barrier(); // synchronization` |
| `absPeak[blockIdx.x].v... // global write` | `absPeak[item_ctl.get_group(2)].v...` |

---

### Key DPC++ Concepts

Some key aspects of a DPC++ code include:

1. **Invocation of device code using SYCL queues**
2. **A lambda function handler for executing the device code**
3. **Optionally, a `parallel_for` construct for multithreaded execution**

The migrated DPC++ code here uses the **unified shared memory (USM)** programming model and allocates memory on the device for data being read/written by the device kernels. Since this is a device allocation, explicit data copy needs to be done from host to device and vice versa. We can also allocate the memory as **shared**, and it can be accessed and updated by both the host and the device. Not shown here is non-USM code, in which data transfers are done using SYCL buffers and accessors.

**Key migration details:**

* The migrated code determines the current device and creates a queue for that device (calls to `get_current_device()` and `get_default_queue()`).
* To offload DPC++ code to the GPU, we need to create a queue with the parameter `sycl::gpu_selector`.
* The data to be processed should be made available on the device and to the kernel that executes on the GPU.
* The dimensions and size of the data being copied into and out of the GPU are specified by `sycl::range`, `sycl::nd_range`.
* When using Intel DPC++ Compatibility Tool, each source line in the CUDA code is migrated to equivalent DPC++ code.

---

### Validation and Optimization

Having migrated the code to DPC++ using Intel DPC++ Compatibility Tool, our next task is to **check correctness and efficiency**.

**Potential Issues:**
* In some cases, the tool may replace preprocessor directive variables with their values. We may need a manual fix to undo this replacement.
* We may also get compilation errors with the migrated code that indicate a fix (for example, replacing CUDA `threadIdx.x` with an equivalent `nd_range` accessor).

The Hogbom Clean application code has a correctness checker that helped us validate the results produced by the migrated DPC++ code. The correctness check was done by comparing results from the DPC++ code execution on the GPU and a baseline C++ implementation on the host CPU.

---

### Performance Analysis

Now we can determine the efficiency of the migrated DPC++ code on a GPU by analyzing its utilization (EU occupancy, use of caches, SP or DP FLOPS) and data transfer between host and device.

**Parameters that have an impact on GPU utilization:**
* Workgroup sizes and range dimensions
* In the Hogbom Clean application, for Find Peak, these are `nBlocks` and `findPeakWidth`

---

### Figure 10: Hogbom Clean Profile on Gen9

![Performance Profile Chart]()

*A bar chart showing GPU execution units, GPU usage, and CPU time for two different scenarios: (a) higher efficiency and (b) lower efficiency. The chart includes a legend for GPU Execution Units, EU Arrays, Active, Idle, and Stalled states. This shows performance profile collected using nBlocks values set to 24 and 4 respectively, with findPeakWidth set to 256.*

**Markdown Visualization:**

```
          GPU Performance Comparison: nBlocks = 24 vs 4
          
(a) nBlocks = 24 (Higher Efficiency)
┌─────────────────────────────────────────────────────┐
│ GPU EUs:  ███████████████████████ (Active)    │
│ GPU Usage:████████████████████░░░░░           │
│ CPU Time: █████ (3.61 sec)                      │
└─────────────────────────────────────────────────────┘

(b) nBlocks = 4 (Lower Efficiency)
┌─────────────────────────────────────────────────────┐
│ GPU EUs:  ███████░░░░░░░░░░░░░░░░ (Active)    │
│ GPU Usage:█████░░░░░░░░░░░░░░░░░░░░           │
│ CPU Time: ████████████ (8.65 sec)              │
└─────────────────────────────────────────────────────┘

Legend: █ = Active/Utilized  ░ = Idle/Stalled
```

---

### Table 1: Performance Metrics on Gen9 GPU for the Find Peak Hotspot

| Function | Global Size | Local Size | Execution Time (Sec.) | Instances | % GPU Array | FPU Util. % | L3 Shader BW GB/Sec. |
|----------|-------------|------------|----------------------|-----------|-------------|-------------|----------------------|
| **d_findPeak** | 6,144 | 256 | **A: 3.61** | 1,001 | **Active: 36.5** | **8.6** | **19.5** |
| | | | | | **Stalled: 48.8** | | |
| | | | | | **Idle: 14.7** | | |
| | 1,024 | 256 | **B: 8.65** | 1,001 | **Active: 13.1** | **1.7** | **8.3** |
| | | | | | **Stalled: 52.5** | | |
| | | | | | **Idle: 34.4** | | |

**Key Observations:**

* Scenario A (nBlocks=24): Higher GPU array active percentage (36.5%) and faster execution (3.61 sec)
* Scenario B (nBlocks=4): Lower GPU array active percentage (13.1%) and slower execution (8.65 sec)
* Tuning is more explicitly required when using the Intel DPC++ Compatibility Tool because the parameters that are efficient for an NVIDIA GPU using CUDA may not be the most efficient for an Intel GPU executing DPC++ code.

---

### Data Transfer Optimization

In addition to GPU utilization and efficiency optimizations, the data transfer between host and device should also be tuned. The Hogbom Clean application has multiple calls to Find Peak and SubtractPSF kernels. The data used by these kernels can be **resident on the device**. Thus, they don't require reallocation and/or copy from host to device, or vice versa.

*(We'll discuss some of these optimizations related to data transfers and USM in future articles.)*

---

## Writing Better Algorithms

Understanding the Intel Processor Graphics Architecture and DPC++ features can help you write better algorithms and portable implementations. In this article, we reviewed some details of the architecture and explored a case study using DPC++ constructs and Intel DPC++ Compatibility Tool.

**Key Takeaways:**

1. **It's important to tune the kernel parameters to get the best performance on Intel GPUs**, especially when using the Intel DPC++ Compatibility Tool.

2. **Parameters that affect performance include:**
   - Workgroup sizes
   - Range dimensions  
   - Memory allocation strategy (USM vs. buffers)
   - Data residency on device

3. **Tools available:**
   - Intel DPC++ Compatibility Tool for CUDA migration
   - Intel® VTune™ Profiler for GPU profiling
   - Intel® Tiber™ AI Cloud for development and testing

---

**Recommendation:** We recommend trying the **Intel® Tiber™ AI Cloud** to develop, test, and run applications on the latest Intel hardware and software.

---

*© Intel Corporation. All rights reserved.*

---

## Document Summary

| Property | Value |
|----------|-------|
| Document ID | 861206 |
| Last Updated | 7/23/2020 |
| Version | Archived |
| Author | Rama Malladi |
| Affiliation | Intel Corporation |
| Topics | GPU Computing, DPC++, CUDA Migration, oneAPI, Gen9, Gen11 |