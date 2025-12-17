# Graphics API Performance Guide

## For Intel® Processor Graphics Gen9

---

![Intel Logo]()
*The Intel logo, a stylized blue oval containing the word "intel" in lowercase white letters.*

![Intel Core i7 Logo]()
*A small square badge showing the Intel Core i7 processor logo with "intel" at top, "CORE i7" in the center, and "inside" at the bottom, rendered in blue and silver colors.*

**Version 2.5**  
**November 29, 2017**

---

## Abstract

Welcome to the Graphics API Performance Guide for Intel® Processor Graphics Gen9. This document provides recommendations for developers who are already engaged in a graphics project on how to take advantage of numerous new features and capabilities, resulting in significant performance gains and reduced power consumption.

Intel® driver development and software validation teams work closely with industry partners to ensure that each of the APIs discussed in this document take full advantage of the hardware improvements. Recommendations for optimization are derived by porting real-world games to the new platforms and instrumenting them with the Intel graphics analysis tools. In this guide, you will find information on structuring and optimizing OpenGL* 4.5, Direct3D* 12, Vulkan*, and Metal* 2 code through hints, tutorials, step-by-step instructions, and online references that will help with the app development process.

This guide expands on previous guides and tutorials. For additional details, or further hints, tips, and expert insights on Intel® processors and prior graphics APIs, you can find a full range of reference materials at the Intel® Software Developer Zone website for Game Developers.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Intel® Processor Graphics Gen9 Architecture](#2-intel-processor-graphics-gen9-architecture)
   - 2.1 [General Refinements](#21-general-refinements)
   - 2.2 [Conservative Rasterization](#22-conservative-rasterization)
   - 2.3 [Texture Compression](#23-texture-compression)
   - 2.4 [Memory](#24-memory)
   - 2.5 [Multiplane Overlay](#25-multiplane-overlay)
3. [Tools](#3-tools)
   - 3.1 [Intel® Graphics Performance Analyzers (Intel® GPA)](#31-intel-graphics-performance-analyzers-intel-gpa)
   - 3.2 [Intel® GPA Tools](#32-intel-gpa-tools)
   - 3.3 [Intel® VTune™ Amplifier XE](#33-intel-vtune-amplifier-xe)
4. [Performance Tips for Intel® Processor Graphics Gen9](#4-performance-tips-for-intel-processor-graphics-gen9)
   - 4.1 [Common Graphics Performance Recommendations](#41-common-graphics-performance-recommendations)
   - 4.2 [Direct3D* 12 Performance Tips](#42-direct3d-12-performance-tips)
   - 4.3 [Vulkan* Performance Tips](#43-vulkan-performance-tips)
   - 4.4 [Metal* 2 Performance Tips](#44-metal-2-performance-tips)
   - 4.5 [OpenGL* Performance Tips](#45-opengl-performance-tips)
5. [Designing for Low Power](#5-designing-for-low-power)
   - 5.1 [Idle and Active Power](#51-idle-and-active-power)
   - 5.2 [Analysis Tips](#52-analysis-tips)
   - 5.3 [Use of SIMD](#53-use-of-simd)
   - 5.4 [Power Versus Frame Rate](#54-power-versus-frame-rate)
6. [Performance Analysis with Intel® GPA](#6-performance-analysis-with-intel-gpa)
   - 6.1 [Performance Analysis with Experiments](#61-performance-analysis-with-experiments)
   - 6.2 [Performance Analysis with Hardware Metrics](#62-performance-analysis-with-hardware-metrics)
7. [Appendix: Developer Resources](#7-appendix-developer-resources)
   - 7.1 [The Intel® Software Developer Zone and Game Dev Websites](#71-the-intel-software-developer-zone-and-game-dev-websites)
   - 7.2 [DirectX 12 Resources](#72-directx-12-resources)
   - 7.3 [Vulkan Resources](#73-vulkan-resources)
   - 7.4 [Metal* 2 Resources](#74-metal-2-resources)
   - 7.5 [OpenGL* Resources](#75-opengl-resources)
8. [Notices](#8-notices)

---

## 1. Introduction

The 6th through 8th generations of Intel® Core™ processors (codenamed Skylake and Kaby Lake) incorporate a powerful new graphics processing unit—the Intel® Processor Graphics Gen9. These processors are fully capable of meeting the high-end computing and graphical performance needs of both mobile platforms and premium desktop gaming systems.

This guide highlights new features of the graphics hardware architecture of Intel® Processor Graphics Gen9 and provides expert tips and best practices to consider when leveraging their capabilities. The document also provides guidance and direction on how to get better performance from Intel® Processor Graphics using the latest graphical APIs.

The information contained in this guide will appeal to experienced programmers who are already involved in a graphics development project. You can continue using Direct3D* 11 and OpenGL* 4.3 if you prefer, but the current trend is toward low-level explicit programming of the graphics processing unit (GPU) in a new generation of graphics APIs. Details on programming for earlier generations of Intel® processors are available at the Intel® Processor Graphics document library. Download drivers from the Download Center for the latest updates.

OpenGL and Vulkan* have Linux* support, though their Linux driver implementation isn't specifically discussed in this document. Linux developers should go to the Intel® Graphics for Linux* website for drivers, whitepapers, and reference documents.

### Figure 1: Single Slice Implementation

![Single Slice Implementation Diagram]()

*A block diagram illustrating the single slice implementation of Intel® Processor Graphics Gen9.*

**Markdown Diagram Recreation:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TOP LEVEL                                       │
├─────────────────┬─────────────────────────────────┬─────────────────────────┤
│ Command         │ 3D pipeline fixed function       │ Media and other         │
│ Streamer        │ units and shader dispatch        │ units                   │
└─────────────────┴─────────────────────────────────┴─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Slice: 24 EUs                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │ Rasterizer   │  │    Hi-Z      │  │Raster ops    │   Shader dispatch     │
│  └──────────────┘  └──────────────┘  │and caches    │                       │
│                                       └──────────────┘                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                 │
│ │ Subslice: 8 EUs │ │ Subslice: 8 EUs │ │ Subslice: 8 EUs │                 │
│ ├─────────────────┤ ├─────────────────┤ ├─────────────────┤                 │
│ │ ┌──┬──┬──┬──┐   │ │ ┌──┬──┬──┬──┐   │ │ ┌──┬──┬──┬──┐   │                 │
│ │ │EU│EU│EU│EU│   │ │ │EU│EU│EU│EU│   │ │ │EU│EU│EU│EU│   │                 │
│ │ ├──┼──┼──┼──┤   │ │ ├──┼──┼──┼──┤   │ │ ├──┼──┼──┼──┤   │                 │
│ │ │EU│EU│EU│EU│   │ │ │EU│EU│EU│EU│   │ │ │EU│EU│EU│EU│   │                 │
│ │ └──┴──┴──┴──┘   │ │ └──┴──┴──┴──┘   │ │ └──┴──┴──┴──┘   │                 │
│ │  IS  Dispatcher │ │  IS  Dispatcher │ │  IS  Dispatcher │                 │
│ ├─────────────────┤ ├─────────────────┤ ├─────────────────┤                 │
│ │Sampler │  L2    │ │Sampler │  L2    │ │Sampler │  L2    │                 │
│ │Data Port        │ │Data Port        │ │Data Port        │                 │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           L3 Data Cache                                      │
├───────────────────────┬─────────────────────────────────────────────────────┤
│   Atomics, Barriers   │             Shared Local Memory                     │
└───────────────────────┴─────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 GTI: Graphics Technology Interface                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Intel® Processor Graphics Gen9 Architecture

Intel® Processor Graphics Gen9 includes many refinements throughout the microarchitecture and supporting software. Generally, these changes are across the domains of memory hierarchy, compute capability, power management, and product configuration. They are briefly summarized here, with more detail provided for significant new features.

### 2.1. General Refinements

* Conservative rasterization
* Multisample anti-aliasing (MSAA) - 2x, 4x, 8x performance improved, new 16x support
* 16-bit floating point hardware support, providing power and performance benefits when using half precision. Native support for denormal and gradual underflow
* Multiplane overlay, display controller features
* Atomics support in shared local memory and global memory
* Improved back-to-back performance to the same address
* Improved geometry throughput
* Improved pixel back-end fill rate
* Codec support for HEVC and VP8 encode/decode

#### 2.1.1 Memory Hierarchy Refinements

* Increased bandwidth for the ring interconnect and last-level cache (LLC).
* Cache hierarchy changes to support LLC, enhanced dynamic random access memory (EDRAM), and GPU data sharing. Coherent shared local memory write performance is significantly improved via new LLC cache management policies.
* The available L3 cache capacity has been increased to 768 Kbytes per slice, 512 Kbytes for application data. The sizes of both L3 and LLC request queues have been increased. This improves latency hiding to achieve better effective bandwidth against the architecture peak theoretical.
* The EDRAM memory controller has moved into the system agent, adjacent to the display controller, to support power efficient and low latency display refresh. EDRAM now acts as a memory-side cache between LLC and DRAM. Up to 128 MB on some stock keeping units (SKUs).

#### 2.1.2 Resource Refinements

* Tiled resources support for sparse textures and buffers. Unmapped tiles return zero, writes are discarded.
* Lossless compression of render target and texture data, up to 2:1 maximum compression.
* Larger texture and buffer sizes (2D mipmapped textures, up to 128Kb x 128Kb x 8 bits).
* Hardware support for Adaptive Scalable Texture Compression (ASTC) for lossless color compression to save bandwidth and power with minimal perceptive degradation.
* Bindless resources support added increasing available slots from ~256 to ~2 million (depending on API support). Reduces binding table overhead and adds flexibility.
* New shader instructions for LOD clamping and obtaining operation status.
* Standard swizzle support for more efficient texture sharing across adapters.
* Texture samplers now natively support an NV12 YUV format for improved surface sharing between compute APIs and media fixed function units.

#### 2.1.3 Compute Capability Refinements

* Preemption of compute applications is now supported at a thread level, meaning that compute threads can be preempted (and later resumed) midway through execution.
* Round robin scheduling of threads within an execution unit.
* Intel® Processor Graphics Gen9 adds new native support for the 32-bit float atomics operations of min, max, and compare/exchange. The performance of all 32-bit atomics is improved for kernel scenarios that issued multiple atomics back to back.

#### 2.1.4 Power Management Refinements

* New power gating and clock domains for more efficient dynamic power management. This can particularly improve low-power media playback modes.

### 2.2 Conservative Rasterization

Rasterization is the process of converting vector-based geometric objects into pixels on a screen. While you could simply check the center of a pixel to see if a polygon covers it (point sampling), conservative rasterization tests coverage at the corners of the pixel leading to much more accurate results. The Intel® Processor Graphics Gen9 GPU architecture adds hardware support for conservative rasterization. There is a flag in the shader to indicate whether the pixel is fully (inner conservative) or partially covered (outer conservative).

The implementation meets the requirements of tier2 hardware per the Direct3D specification. It is truly conservative with respect to floating point inputs, and is at most 1/256th of a pixel over conservative (tier 2). No covered pixels are missed or incorrectly flagged as fully covered. Post-snapped degenerate triangles are not culled. A depth coverage flag notes whether each sample was covered by rasterization and has also passed the early depth flag test (see SV_DepthCoverage in Direct3D).

### Figure 2: Conservative Rasterization

![Conservative Rasterization Diagram]()

*Two grid diagrams illustrating conservative rasterization. The left diagram shows "Outer Conservative" mode where a triangle covers a larger area including partially covered pixels at the edges. The right diagram shows "Inner Conservative" mode where only fully covered pixels are highlighted. Both diagrams display a grid of dots representing pixel sample points, with blue-filled dots indicating coverage.*

**Markdown Diagram Recreation:**

```
          OUTER CONSERVATIVE                    INNER CONSERVATIVE
    ┌─────────────────────────┐           ┌─────────────────────────┐
    │  ·  ·  ·  ·  ·  ·  ·  · │           │  ·  ·  ·  ·  ·  ·  ·  · │
    │  ·  ·  ·  ◆──────────◆  │           │  ·  ·  ·  ◆──────────◆  │
    │  ·  ·  ·  │██████████│  │           │  ·  ·  ·  │          │  │
    │  ·  ·  ·  │██████████│  │           │  ·  ·  ·  │  ████████│  │
    │  ·  ·  ◆──┼██████████│  │           │  ·  ·  ·  │  ████████│  │
    │  ·  ·  │██│██████████│  │           │  ·  ·  ·  │  ████████│  │
    │  ·  ·  │██│██████████│  │           │  ·  ·  ·  │  ████████│  │
    │  ·  ·  └──┴───────◆──┘  │           │  ·  ·  ·  └──┴───────◆  │
    │  ·  ·  ·  ·  ·  ·  ·  · │           │  ·  ·  ·  ·  ·  ·  ·  · │
    └─────────────────────────┘           └─────────────────────────┘
    
    Legend: ·  = pixel sample point
            ◆  = triangle vertex
            ██ = covered pixels
            ─  = triangle edge
```

### 2.3 Texture Compression

Texture compression is used to reduce memory requirements, decrease load times, and conserve bandwidth. When a shader program executes, the sampler retrieves the texture, then decompresses it to get RGBA. There are a number of popular compression formats in use, including ETC2, TXTC, BC1-5, BC6/7 and ASTC.

Intel® Processor Graphics Gen9 family components now have hardware support for ASTC. ASTC produces higher compression ratios than other compression technologies. When used with the low-power versions of these processors, larger ASTC tiles provide better quality than reducing texture resolution.

These enhancements implement lossless color compression with automatic compression on store to memory, and decompression on load for memory. There is a maximum peak compression ratio of 2:1.

### Figure 3: Comparing Compression Techniques

![Butterfly Compression Comparison]()

*A butterfly with yellow wings and black markings is shown against a blurred background of green foliage and pink flowers. A red rectangular box highlights a small section of the butterfly's wing. Below the main image, the text "800x600, 32BPP" indicates the resolution and color depth. Below this are comparison images showing the same wing section compressed using different techniques (RGBA8, ETC2, ASTC 4x4, ASTC 8x8) along with artifact visualization images showing red dots on gray backgrounds representing compression artifacts.*

**Compression Comparison Table:**

| Format    | RGBA8   | ETC2   | ASTC 4x4 | ASTC 8x8 |
|-----------|---------|--------|----------|----------|
| Size      | 1920 kB | 480 kB | 480 kB   | 120 kB   |
| PSNR      | -       | 31.0 dB| 35.5 dB  | 28.7 dB  |

### 2.4 Memory

Graphics resources that are allocated from system memory with write-back caching will utilize the full cache hierarchy: Level 1 (L1), Level 2 (L2), last-level cache (LLC), optional EDRAM, and finally DRAM. When accessed from the GPU, sharing can occur in LLC and further caches/memory. The Intel® Processor Graphics Gen9 GPU has numerous bandwidth improvements including increased LLC and ring bandwidth. Coupled with support for DDR4 RAM, these help hide latency.

When detecting the amount of available video memory for your graphics device, remember that different rules may apply to different devices. An integrated graphics part reports only a small amount of dedicated video memory. Because the graphics hardware uses a portion of system memory, the reported system memory is a better indication of what your game may use. This should be taken into account when using the reported memory as guidance for enabling or disabling certain features. The GPU Detect Sample shows how to obtain this information in Windows*.

### 2.5 Multiplane Overlay

Most 3D games will run faster in full screen mode than windowed mode. This is because they have the ability to bypass the System Compositor, which manages the presentation of multiple windows on a screen. This eliminates at least one copy operation per screen refresh, but introduces the requirement that scenes are rendered using a supported color space at the native resolution of the display.

The Intel® Processor Graphics Gen9 GPU now includes hardware support in the display controller subsystem to scale, convert, color correct, and composite independently for up to three layers. Using Multiplane Overlay, surfaces can come from separate swap chains using different update frequencies and resolutions. Visibility is determined through alpha channel blending.

Multiplane Overlay (MPO) can be used to avoid performance degradation of games in high-resolution rendering under demanding workloads. Consider a game running in windowed mode, with a 3D rendered scene and an overlay for scoring and charms. If the 3D layer can't keep up with the frame rate due to CPU or GPU loading, MPO can adjust resolution of that layer independently to maintain frame rate. Some use cases are battle scenes with numerous characters, special effects that release particles, and scenes that require a lot of pixel shading.

MPO is also beneficial in low power mobile applications. With the display controller handling scaling, blending, and compositing there is less work for the CPU to do, and power consumption is reduced.

### Figure 4: Intel® Processor Graphics Gen9 Display Controller

![Display Controller Flowchart]()

*A flowchart illustrating the Intel® Processor Graphics Gen9 Display Controller. The flowchart shows two main paths: "Full Screen" which goes directly to the Display Controller, and "Windowed" which starts with Layered Inputs (3D Rendered Scene, Game GUI, OS Desktop) feeding into DWM (Display Window Manager). From DWM, three parallel processing paths (Display Format Conversion, Scaling, Color Space Conversion) feed into a System Compositor where Alpha Blending occurs before reaching the Display Controller. The system supports Multiplane Overlay functionality.*

**Markdown Diagram Recreation:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DISPLAY PATHS                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐                               ┌─────────────────────────────┐
│ FULL SCREEN │ ─────────────────────────────▶│                             │
└─────────────┘                               │                             │
                                              │    DISPLAY CONTROLLER       │
┌─────────────┐      ┌─────────┐              │                             │
│  WINDOWED   │ ────▶│         │              │                             │
└─────────────┘      │         │              └─────────────────────────────┘
                     │         │                           ▲
                     │         │                           │
┌─────────────────┐  │         │  ┌─────────────────────────────────────────┐
│ LAYERED INPUTS  │  │         │  │            SYSTEM COMPOSITOR             │
├─────────────────┤  │   DWM   │  │         (With Multiplane Overlay)        │
│ • 3D Rendered   │  │         │  ├─────────────────────────────────────────┤
│   Scene         │──│ Display │──│  ┌─────────────┐  ┌─────────────────┐   │
│ • Game GUI      │  │ Window  │  │  │  Display    │  │                 │   │
│ • OS Desktop    │  │ Manager │  │  │  Format     │──│  Alpha Blending │───┤
│                 │  │         │  │  │  Conversion │  │                 │   │
│                 │  │         │  │  ├─────────────┤  │                 │   │
│                 │  │         │  │  │  Scaling    │──│                 │   │
│                 │  │         │  │  ├─────────────┤  │                 │   │
│                 │  │         │  │  │ Color Space │──│                 │   │
│                 │  │         │  │  │ Conversion  │  │                 │   │
└─────────────────┘  └─────────┘  │  └─────────────┘  └─────────────────┘   │
                                  └─────────────────────────────────────────┘
```

---

## 3. Tools

### 3.1. Intel® Graphics Performance Analyzers (Intel® GPA)

The Intel® Graphics Performance Analyzers (Intel® GPA) are a suite of graphics analysis and optimization tools that help developers make games and other graphics-intensive applications run faster. They allow programmers to identify problem areas and improve graphics API usage without the need to make changes to the source code, and to optimize a game for best performance on Intel Processor Graphics.

### 3.2. Intel® GPA Tools

#### 3.2.1 System Analyzer

System Analyzer allows you to view real-time system metrics, quickly run graphics experiments, and capture graphics streams. Metrics can be viewed in a standalone fashion, or with the head-up-display (HUD) overlay for DirectX* workflows.

### Figure 5: System Analyzer

![System Analyzer Screenshot]()

*A screenshot of the Intel GPA System Analyzer software interface displaying performance data. The interface shows numbered sections: Section 1 displays a game scene showing a 3D interior environment with wooden floors and walls. Section 2 shows real-time status information. Section 3 displays lists of metrics and settings. Section 4 presents multiple graphs showing different performance metrics over time including GPU utilization, frame time, and other parameters. Section 5 shows numerical values for frame rate and frame time statistics.*

With System Analyzer you can:

* Display the overall system metrics, or specific metrics, for a selected application.
* Select from a range of metrics from the CPU, GPU, graphics driver, and either DirectX* or OpenGL ES.
* Perform various what-if experiments using override modes to quickly isolate many common performance bottlenecks.
* Capture frames, traces, export metric values, and pause/continue data collection.
* Show the current, minimum, and maximum frame rate.

### Figure 6: HUD Display

![HUD Display Screenshot]()

*A screenshot of a video game (appears to be a racing/driving game) with a head-up display (HUD) showing real-time performance metrics. The main view shows a futuristic vehicle on a road surrounded by buildings. The HUD elements include "199" prominently displayed, along with various smaller windows showing data like "Fullscreens (F1)", "Change device (F3)", "Toggle fullscreen (F4)", and other performance-related graphs and text overlays.*

#### 3.2.2 Graphics Frame Analyzer

Graphics Frame Analyzer is a powerful, intuitive, single-frame analysis and optimization tool for Microsoft DirectX, OpenGL, and OpenGL ES game workloads. It provides deep frame performance analysis down to the draw call level, including shaders, render states, pixel history, and textures. You can conduct what-if experiments to see how changes iteratively impact performance and visuals without having to recompile your source code.

### Figure 7: Intel® GPA Graphics Frame Analyzer

![Graphics Frame Analyzer Screenshot]()

*A screenshot of the Intel GPA Graphics Frame Analyzer software. The top portion shows a bar chart with blue and orange bars representing performance metrics over time. Below the chart are several panels: a hierarchical tree view on the left listing various events and draw calls, a central panel showing shader code and register values for selected events, and a right panel with controls for experiments and visualizations including "Thread Support", "Pixel History", and "Geometry Viewer". A row of small image thumbnails appears above the detailed panels, representing different stages or views of the rendered frame.*

With Graphics Frame Analyzer you can:

* Graph draw calls with axes based on a variety of graphics metrics.
* View the history of any given pixel.
* Select regions and draw calls in a hierarchical tree.
* Implement and view the results of real-time experiments on the graphics pipeline to determine bottlenecks and isolate any unnecessary events, effects, or render passes.
* Import and modify shaders to see the visual and performance impact of a simpler or more complex shader, without modifying the game as a whole.
* Study the geometry, wireframe, and overdraw view of any frame.
* Use hardware metrics to determine bottlenecks with the GPU pipeline.

#### 3.2.3 Graphics Trace Analyzer

Graphics Trace Analyzer lets you see where your application is spending time across the CPU and GPU. This will help ensure that your software takes full advantage of the processing power available from today's Intel® platforms.

Graphics Trace Analyzer provides offline analysis of CPU and GPU metrics and workloads with a timeline view for analysis of tasks, threads, Microsoft DirectX, OpenGL ES, and GPU-accelerated media applications in context.

### Figure 8: Graphics Trace Analyzer

![Graphics Trace Analyzer Screenshot]()

*A screenshot of the Graphics Trace Analyzer software interface showing a complex timeline with multiple colored bars representing different activities and metrics related to CPU and GPU performance. The left side shows labels including "Trace Analyzer", "CPU", "GPU", and numerical thread identifiers. The main area displays a horizontal timeline filled with numerous colored segments (in various colors indicating different processes and their durations), vertical lines marking specific events or timeframes, and shaded areas. A scrollbar is visible on the right side of the timeline.*

With Graphics Trace Analyzer, you can:

* View task data in a detailed timeline.
* Identify CPU and GPU bound processes.
* Explore queued GPU tasks.
* Explore CPU thread utilization and correlate to API use if applicable.
* Correlate CPU and GPU activity based on captured platform and hardware metric data.
* Filter and isolate the timeline to focus on a specific duration in time.

### 3.3. Intel® VTune™ Amplifier XE

Intel® VTune Amplifier XE provides insights into CPU and GPU performance, threading performance, scalability, bandwidth, caching and much more. You can use the powerful VTune analysis tools to sort, filter, and visualize results on the timeline and on your source. For more on Intel® VTune Amplifier XE and key features, please see the Intel® VTune Amplifier product webpage.

### Figure 9: Intel® VTune Amplifier XE 2016

![VTune Amplifier Screenshot]()

*A screenshot of the Intel VTune Amplifier XE 2016 software interface showing a "Basic Hotspots" analysis. The interface is divided into sections with tabs at the top ("Collection Log", "Analysis Target", "Analysis Type", "Summary", "Bottom-up", "Caller/Callee", "Top-down Tree", "Platform"). A "Grouping" dropdown is set to "Function / Call Stack". The main panel displays "CPU Time" with a table showing columns for "Function / Call Stack", "Effective Time by Utilization" (with sub-columns for Idle, Poor, Ok, Ideal, Over), "Spin Time", and "Overhead Time". A bar chart visualizes the effective time for each function. Example data shows functions like "FireObject:checkCollision" with timing values.*

**Sample Data Table:**

| Function / Call Stack                     | Effective Time | Spin Time | Overhead Time |
|-------------------------------------------|----------------|-----------|---------------|
| FireObject:checkCollision                 | 7.650s         | 0s        | 0s            |
| func@0x1000e190                           | 3.318s         | 2.020s    | 0s            |
| FireObject:ProcessFireCollisionsRange     | 5.013s         | 0s        | 0s            |
| FireObject:FireCollisionCallback          | 4.025s         | 0s        | 0s            |
| FireObject:EmitterCollisionCheck          | 0.988s         | 0s        | 0s            |
| func@0x7545a064                           | 3.811s         | 0.675s    | 0s            |

#### 3.3.1 Hotspot Analysis

Intel® VTune Amplifier quickly locates code that is taking up a lot of time. Hotspot analysis features provide a sorted list of the functions that are using considerable CPU time. Fine tuning can provide significant gains, especially if your game is CPU-bound. If you have symbols and sources, you can easily drill down to find the costliest functions, and VTune will provide profiling data on your source to indicate hotspots within the function of interest.

#### 3.3.2 Locks and Waits Analysis

With Intel® VTune Amplifier, you can also locate slow, threaded code more effectively. You can use locks and waits analysis functions to resolve issues, such as having to wait too long on a lock while cores are underutilized. You can also use them to address challenges where wait is a common cause of slow performance in parallel programs.

The timeline view helps you maximize software performance by letting you easily monitor and spot lock contention, load imbalance, and inadvertent serialization. These are all common causes of poor parallel performance.

### Figure 10: Timeline Filtering

![Timeline View Screenshot]()

*A screenshot of the Intel VTune Amplifier timeline view showing various performance metrics over time. The timeline displays Frame Rate, Thread activity (including wWinMainCRTStartup, _endthreadex for multiple TIDs, and CBatchFilter:LHBatc), CPU Usage, and Thread Concurrency. On the right side is a legend with checkboxes for Frame, Frame Rate, Thread, Running, Waits, CPU Time, Spin and Overhead, CPU Sample, Tasks, and Transitions. The timeline shows colored bars and lines representing different activities and states. Yellow vertical lines indicate transitions, where high density may indicate lock contention.*

#### 3.3.3 GPU and Platform-Level Analysis

On newer Intel Core processors you can collect GPU and platform data, and correlate GPU and CPU activities. Configure VTune to explore GPU activity over time and understand whether your application is CPU- or GPU-bound.

### Figure 11: GPU and CPU Activity Correlation

![GPU and CPU Correlation Screenshot]()

*A screenshot of a performance analysis tool showing GPU and CPU activity over time. The top section displays GPU Engines Usage with sub-categories like Render and GPGPU, Video Codec, and Computing Task (GPU). The middle section shows GPU Core Activity with Core Frequency, EU Array Idle, EU Array Active, and EU Array Stalled. The bottom section shows GPU Compute Shader Activity and GPU Sampler Activity. The main graph area displays timelines for different threads (wmainCRTStartup (0xf60) and MFXVideoVPP_GetVPPStat (0xad0)) and GPU Core Activity, with a time scale from 3336ms to 3342ms.*

A sample rule of thumb: If the Timeline pane in the Graphics window shows that the GPU is busy most of the time with small, idle gaps between busy intervals, and a GPU software queue that rarely decreases to zero, your application is GPU-bound. However, if the gaps between busy intervals are big and the CPU is busy during these gaps, your application is most likely CPU-bound.

#### 3.3.4 Slow Frame Analysis

When you discover a slow spot in your Windows game play, you don't just want to know where you are spending a lot of time. You also want to know why the frame rate is slow. VTune can automatically detect DirectX frames and filter results to show you what's happening in slow frames.

#### 3.3.5 User Task-Level Analysis with Code Instrumentation

Intel VTune Amplifier XE also provides a task annotation API that you can use to annotate your source. Then, when you study your results in VTune, it can display which tasks are executing. For instance, if you label the stages of your pipeline they will be marked in the timeline, and hovering over them will reveal further details. This makes profiling data much easier to understand.

---

## 4. Performance Tips for Intel® Processor Graphics Gen9

### 4.1. Common Graphics Performance Recommendations

The new closer to the metal programming APIs give developers control over low-level design choices that used to be buried in the device driver. With great power comes great responsibility, so you are going to need debug libraries and good visualization tools to find those hotspots and stalls in your code.

There are many similarities between the different APIs that run on Intel Processor Graphics. In this section, we'll take a closer look at how to maximize performance when designing and engineering applications for use with Intel Core processors. The performance recommendations in this section are relevant to all APIs interacting with Intel Processor Graphics. Specific recommendations for OpenGL, Direct3D, Vulkan, and Metal appear in subsequent chapters.

#### 4.1.1 Optimizing Clear, Copy, and Update Operations

To achieve the best performance when performing clear, copy, or update operations on resources, please follow these guidelines:

* Use the provided API functions for clear, copy, and update needs. Do not implement your own version of the API calls in the 3D pipeline.
* Enable hardware 'fast clear' as specified, and utilize 0 or 1 for each channel in the clear color for color or depth. For example, RGBA <0,0,0,1> will clear to black using a fast clear operation.
* Copy depth and stencil surfaces only as needed instead of copying both unconditionally.

#### 4.1.2 Render Target Definition and Use

Use the following guidelines to achieve the best performance from a render target:

* Use as few render targets as you can. Combine render targets when possible.
* Define the appropriate format for rendering; that is, avoid defining unnecessary channels or higher-precision formats when not needed.
* Avoid using sRGB formats where unnecessary.

#### 4.1.3 Texture Sampling and Texture Definition

Sampling is a common shader operation, and it is important to optimize both the definition of surfaces sampled along with the sampling operations. Follow these guidelines to achieve the best performance when sampling:

* When sampling from a render target, avoid sampling across levels in the surface with instructions like sample l.
* Make use of API-defined compression formats (BC1-BC7) to reduce memory bandwidth and improve locality of memory accesses when sampling.
* If you're creating textures that are not CPU-accessible, define them so they match the render target size and alignment requirements. This will enable lossless compression when possible. Sampling is cheaper than conversion!
* Avoid dependent texture samples between sample instructions; for example, when the UV coordinate of the next sample instruction is dependent upon the results of the previous sample operation.
* Define appropriate resource types for sampling operation and filtering mode. Do not use volumetric surface options when texture 2D or 2D array could have been used.
* Avoid defining constant data in textures that could be procedurally computed in the shader, such as gradients.
* Use constant channel width formats (for example; R8B8G8A8) rather than variable channel width formats (R10G10B10A2) for operations that may be sampler bottlenecked due to poor cache utilization.
* Use non-volumetric surfaces for operations that may be sampler bottlenecked.

#### 4.1.4 Geometry Transformation

Follow these guidelines to ensure maximum performance during geometry transformation:

* Define input geometry in structure of arrays (SOA) instead of array of structures (AOS) layouts for vertex buffers by providing multiple streams (one for each attribute) versus a single stream containing all attributes.
* Do not duplicate shared vertices in mesh to enable better vertex cache reuse; that is, merge edge data to avoid duplicating the vertices. Incorrect results may appear if the vertex positions are cached but some of the other attributes are different.
* Optimize transformation shaders (VS->GS) to only output attributes consumed by downstream shader stages. For example, avoid defining unnecessary outputs from a vertex shader that are not consumed by a pixel shader.
* Avoid spatial overlap of geometry within a single draw when stencil operations are enabled. Presort geometry to minimize overlap, and use a stencil prepass for overlapped objects, otherwise the geometry may serialize in the front end as it attempts to determine if it can update the stencil buffer.

#### 4.1.5 General Shading Guidance

To achieve optimal performance when shading, follow these guidelines:

* Avoid creating shaders where the number of temporaries (dcl_temps for D3D) is ≥ 16. This is to prevent overutilizing the register space, causing the potential for slowdowns. Optimize to reduce the number of temporaries in the shader assembly.
* Structure code to avoid unnecessary dependencies (especially dependencies on high-latency operations like sample).
* Avoid flow control decisions on high-latency operations. Structure your code to hide the latency of the operation that drives control flow.
* Avoid flow control decisions using non-uniform variables, including loops. Try to ensure uniform execution among the shader threads.
* Avoid querying resource information at runtime (for example, the High Level Shading Language (HLSL) GetDimensions call) to make decisions on control flow.
* Implement fast paths in shaders to return early in algorithms where the output of the algorithm can be predetermined or computed at a lower cost.
* Use discard (or other kill pixel operations) where output will not contribute to the final color in the render target.

#### 4.1.6 Constants

Follow these guidelines when defining and using constants to achieve the best possible results from software applications:

* Structure constant buffers to improve cache locality; that is, so accesses all occur on the same cache line.
* Accesses have two modes, direct (offset known at compile time) and indirect (offset computed at runtime). Avoid defining algorithms that rely on indirect accesses, especially with control flow or tight loops. Make use of direct accesses for high-latency operations like control flow and sampling.

#### 4.1.7 Anti-Aliasing

For the best anti-aliasing performance, follow these guidelines:

* For improved performance over MSAAx4, use Conservative Morphological Anti-Aliasing (CMAA) (see the CMAA whitepaper for more information).
* Avoid querying resource information from within a loop or branch where the result is immediately consumed or duplicated across loop iterations.
* Minimize per-sample operations, and when shading in per-sample, maximize the number of cases where any kill pixel operation is used (for example, discard) to get the best surface compression.
* Minimize the use of stencil or blend when MSAA is enabled.

---

### 4.2. Direct3D* 12 Performance Tips

#### 4.2.1 What's new in Direct3D 12?

Direct3D 12 introduces a set of new resources for the rendering pipeline including pipeline state objects, command lists and bundles, descriptor heaps and tables, and explicit synchronization objects.

New features include conservative rasterization, volume-tiled resources to enable streamed, three-dimension resources to be treated as if they were all in video memory, rasterizer order views (ROVs) to enable reliable transparency rendering, setting the stencil reference within a shader to enable special shadowing and other effects, and improved texture mapping and typed unordered access views (UAV).

#### 4.2.2 Clear Operations

To get maximum performance of depth and render target clears, provide the optimized clear color (use 0 or 1 valued colors; for example, 0,0,0,1) when creating the resource, and use only that clear color when clearing any of the Render target views or Depth stencil views associated with the D3D12_CLEAR_VALUES structure at resource creation time.

#### 4.2.3 Pipeline State Objects

Direct3D 12 introduces a collection of shaders and some states known as pipeline state objects (PSOs). Follow these guidelines when defining PSOs to optimize output:

* When creating PSOs, make sure to take advantage of all the available CPU threads on the system. In previous versions of DirectX, the driver would create these threads for you, but you must create the threads yourself in DirectX 12.
* Compile similar PSOs on the same thread to improve on the de-duplication done by the driver and runtime.
* Avoid creating duplicate PSOs, and do not rely on the driver to cache PSOs.
* Define optimized shaders for PSOs instead of using combinations of generic shaders mixed with specialized shaders.
* Don't define a depth + stencil format if the stencil will not be enabled.

#### 4.2.4 Asynchronous Dispatch (3D + Compute)

Asynchronous dispatch of 3D and compute operations is not supported at this time. Therefore, it is not recommended to structure algorithms with the expectation of latency hiding by executing 3D and compute functions simultaneously. Instead, batch your compute algorithms to allow for minimal latency and context switching.

#### 4.2.5 Root Signature

Follow these guidelines to achieve maximum performance:

* Limit visibility flags to shader stages that will be bound.
* Actively use DENY flags where resources will not be referenced by a shader stage.
* Avoid generic root signature definitions where unnecessary descriptors are defined and not leveraged.

#### 4.2.6 Root Constants

Use root constants for cases where the constants are changing at a high frequency. Favor root constants over root descriptors, and favor root descriptors over descriptor tables when working with constants.

#### 4.2.7 Compiled Shader Caching

Use the features provided in the Direct3D 12 Pipeline Library support to natively cache compiled shaders to reduce CPU overhead upon startup.

#### 4.2.8 Conservative Rasterization and Raster Order Views

Use hardware conservative rasterization for full-speed rasterization and use ROVs only when you need synchronization. Also favor ROVs over atomics.

#### 4.2.9 Command Lists and Barriers

Group together as many command lists as possible when calling execute command lists as long as the CPU cost of managing the bundles does not starve the GPU. Try to group together barriers into a single call instead of many separate calls, and avoid the use of unnecessary barriers.

#### 4.2.10 Resource Creation

Prefer committed resources. They have less padding, so the memory layout is more optimal.

#### 4.2.11 Direct Compute Optimizations

For more info on general purpose GPU programming, see The Compute Architecture of Intel® Processor Graphics Gen9.

#### 4.2.12 Dispatch and Thread Occupancy

To achieve optimal dispatch and thread occupancy of EUs, there are three factors to balance when optimizing for dispatch and occupancy: thread group dimensions, single instruction multiple data (SIMD) width of execution, and shared local memory (SLM) allocation per thread group. The goal is to avoid GPU idle cycles.

**Key specifications:**

* Intel® Processor Graphics Gen9 have 56 hardware threads per subslice.
* Hardware thread utilization per thread group is determined by taking thread group dimensions (dim x * dim y * dim z) and dividing by SIMD width of shader (SIMD16 when number of temporaries are ≤ 16; otherwise, SIMD8).
* For a 16 x 16 x 1 thread group, 256 threads / 16 (SIMD width) = 16 hardware threads for each thread group. 56 total threads / 16 hardware threads per group allows for 3 thread groups to fully execute per subslice.
* SLM allocation per subslice is 64 Kbytes for up to the number of hardware threads supported by your subslice.

#### 4.2.13 Resource Access

When reading from or writing to surfaces in compute, avoid partial writes so that you can get maximum bandwidth through the graphics cache hierarchy. Partial writes are any accesses where the data written doesn't fully cover the 64-byte cache line. UAV writes commonly cause partial writes.

#### 4.2.14 Shared Local Memory

Memory is shared across threads in the thread group. Follow these guidelines for maximum performance:

* Layout elements in structured buffers as SOA instead of AOS to improve caching and reduce unnecessary memory fetch of unused/unreferenced elements.
* Split RGBA definition in a single structure into R, G, B, and A, where the array length of each R, G, B, and A is padded out to a non-multiple of 16.
* SOA removes bank collisions when accessing a single channel (for example, R). Padding out to a non-multiple of 16 removes bank collisions when accessing multiple channels (RGBA).
* If AOS is required, pad structure size to a prime number for structure sizes greater than 16 bytes.

---

### 4.3. Vulkan* Performance Tips

Vulkan was developed by the Khronos Group as an alternative to OpenGL, for a low-level, high-performance API for modern graphics and compute hardware. Like OpenGL, it is cross platform and open source; supporting Windows, Linux, and Android. Unlike OpenGL, the programmer is responsible for low-level details such as frame buffer creation, command lists, memory management, CPU/GPU synchronization, and error checking.

#### 4.3.1 What's new in Vulkan?

Vulkan shader code is loaded in a bytecode format called SPIR-V* (Standard Portable Intermediate Representation). SPIR-V allows the use of industry standard shading languages like OpenGL Shading Language (GLSL) and HLSL on the front end, then outputs a compiled intermediate binary. Additional benefits of SPIR-V are standards compliance and decreased shader load times.

Device independence is achieved in Vulkan by separating out any operating system-dependent code at the application binary interface (ABI).

#### 4.3.2 Memory Resources

Create memory resources from the same memory object. Image layouts and transitions are heavily optimized. Always use VK_IMAGE_LAYOUT_{}_Optimal. Avoid VK_IMAGE_LAYOUT_GENERAL or VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT unless really needed.

#### 4.3.3 Clears

Use RGBA <0,0,0,1> clears for best performance, but use any 0/1 clear value when possible. Use VK_ATTACHMENT_LOAD_OP+CLEAR to enable this rather than vkCmdClearColorImage.

#### 4.3.4 Barriers

Avoid sending barriers per resource. Minimize the number of memory barriers used in layout transitions. Batch pipeline barriers. Use implicit render pass barriers whenever possible.

#### 4.3.5 Command Buffers

Use primary command buffers whenever possible. Performance is better due to internal batch buffer usage. Batch your work to increase command buffer size and reduce the number of command buffers.

Secondary command buffers are less efficient than primary, especially on depth clears. Use caution with reusable command buffers as they may require a copy.

**Use:** `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT`  
**Avoid:** `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT`

#### 4.3.6 Presentation

Use full screen presentation modes whenever possible. Windowed mode requires an extra context switch and copy operation.

#### 4.3.7 Multithreading

Use multiple CPU cores to fill command buffers. Multithreading reduces single core CPU overhead and can improve CPU performance and power consumption. It may allow the GPU to benefit from the shared turbo.

#### 4.3.8 Compute

The pipeline is flushed when switching between 3D graphics rendering and compute functions. Asynchronous compute functions are not supported at this time. Batch the compute kernels into groups whenever possible.

---

### 4.4. Metal* 2 Performance Tips

#### 4.4.1 What's new in Metal 2?

Apple first introduced the Metal* API in 2014 as a low-level interface for hardware-accelerated graphics and compute applications running on modern GPUs. The latest release, Metal 2, extends the API with important new features for game developers.

#### 4.4.2 Fast Resource Clears (Render Targets/Depth Buffers)

Make sure to use loadAction = clear in the render pass descriptor for any fast-cleared resource. Make sure to also draw to the render target (RT) after clearing it. This is because the driver resolves the clear color at the end of the encoder pass that cleared it.

#### 4.4.3 Stencil Clears

Use MTLLoadActionClear to clear stencil. Do not change the value of stencil clear between clears.

#### 4.4.4 Texture Barriers

The MTLRenderCommandEncoder's textureBarrier() call flushes all render targets and the depth buffer. Instead of having one encoder that does some draws, then binds the depth buffer for reading, and then some more draws, it is better to split it into two different encoders.

#### 4.4.5 Using Proper Address Space for Buffers

Access to read-only data is much more efficient than read/write data. If possible, use the constant address space buffers instead of device address space buffers.

```metal
vertex VertexOutput
vertexShader (const device VertexInput *vIn [[ buffer (0)]],
              constant float4x4& mvp [[ buffer (1)]],
              constant Light& lights [[ buffer (2)]],
              uint vid [[ vertex_id ]] )
```

#### 4.4.6 Math Precision in Shaders—Tradeoff Speed and Correctness

The Intel backend compiler can generate better optimized code when the programmer can tolerate some imprecision in the floating-point computations. Whenever feasible, try to use the default compiler option `-ffast-math` instead of using `-fno-fast-math`.

#### 4.4.7 Using Half-Floats Efficiently

Using half-float data types, whenever possible, is advisable as it reduces the data footprint and can increase instruction throughput. However, avoid frequent mixing of float and half-float computations as it may result in unnecessary type-conversion overhead.

#### 4.4.8 Low Occupancy

If compute kernels are very short, thread launch can be an overhead. Try forcing a higher SIMD, or introduce a loop so there is more work per thread.

#### 4.4.9 Avoid uchars

Data reads are done at a 64-byte granularity. Using an uchar data type can result in partial reads and writes of cache lines, which can affect performance. Use a wider data type like int or int4.

#### 4.4.10 Array Indices

Use unsigned int for array indexes, since this helps the compiler to ensure the indices cannot have negative values, and thus can optimize send instructions.

#### 4.4.11 Other Metal 2 Optimization Recommendations

* Render drawables at the exact pixel size of your target display.
* Merge render command encoders when possible.
* Mark the resource option as untracked if there will be no overlap between multiple uses of the same buffer.
* Calculate tessellation factors once for multiple draws to reduce the amount of context switching.
* Keep execution units (EU) busy by avoiding lots of very short draw calls.
* The application should avoid redundant setFragmentBytes() calls.

---

### 4.5. OpenGL* Performance Tips

This section will lead you through some of the main performance points for OpenGL applications. The OpenGL 4.5 device driver contains many Intel® Processor Graphics Gen9 specific optimizations.

#### 4.5.1 OpenGL Shaders

* When writing GLSL shaders, use and pass only what is needed, and declare only the resources that will be used.
* Use built-in shader functions rather than other equivalents. Use compile-time constants to help the compiler generate optimal code.
* Use medium precision for OpenGL ES contexts, for improved performance.
* Each thread in the Intel® Processor Graphics Gen9 architecture has 128 registers and each register is 8 x 32 bits.
* Limit the number of uniforms in the default block to minimize register pressure.
* Don't use compute shader group sizes larger than 256.

#### 4.5.2 Textures

* Always use mipmaps to minimize the memory bandwidth used for texturing.
* Textures with power-of-two dimensions will have better performance in general.
* When uploading textures, provide textures in a format that is the same as the internal format, to avoid implicit conversions.

#### 4.5.3 Images

When using OpenGL images, textures usually provide better read performance than images.

#### 4.5.4 Shader Storage Buffer Objects

Shader Storage Buffer Objects present a universal mechanism for providing input/output both to and from shaders. Use vertex arrays where possible, as they usually offer better performance.

#### 4.5.5 Atomic Counter Buffers

Atomic counter buffers and atomic counters are internally implemented as shader storage buffer objects atomic operations.

#### 4.5.6 Frame Buffer Object

For frame buffer objects (FBOs), consider these important things:

* When switching color/depth/stencil attachments, try to use dedicated frame buffer objects for each set in use.
* Don't clear buffers that are never used.
* Skip color buffer clears if all pixels are to be rendered.
* Limit functions that switch color/depth/stencil attachments between rendering and sampling.

#### 4.5.7 State Changes

* Minimize state changes. Group similar draw calls together.
* For texturing, use texture arrays.
* Use the default uniform block rather than uniform buffer objects for small constant data that changes frequently.
* Limit functions that switch frame buffer objects and GLSL programs.
* Avoid redundant state changes.

#### 4.5.8 Avoid CPU/GPU Synchronization

Synchronization between the CPU and GPU can cause stalls.

* Avoid calls that synchronize between the CPU and GPU, for example, glReadPixels or glFinish.
* Use glFlush with caution. Use sync objects to achieve synchronization between contexts.
* Avoid updating resources that are used by the GPU.
* For creation of buffers and textures, use immutable versions of API calls: glBufferStorage() and glTexStorage*().

#### 4.5.9 Anti-Aliasing Options

OpenGL drivers for 6th, 7th, and 8th generation Intel Core processors support standard MSAA functionality. You might get better anti-aliasing performance from Conservative Morphological Anti-Aliasing (CMAA).

---

## 5. Designing for Low Power

Mobile and ultra-mobile computing are ubiquitous. As a result, battery life, device temperature, and power-limited performance have become significant issues.

### 5.1. Idle and Active Power

Processors execute in different power states, known as P-states and C-states. C-states are essentially idle states that minimize power draw by progressively shutting down more and more of the processor. P-states are performance states where the processor will consume progressively more power and run faster at a higher frequency.

When you optimize applications, try to save power in two different ways:

* Increase the amount of idle time your application uses where it makes sense.
* Improve overall power usage and balance under active use.

### 5.2. Analysis Tips

To start, begin by measuring your app's baseline power usage in multiple cases and at different loads:

* At near idle, as in the UI during videos.
* Under an average load during a typical scene with average effects.

#### 5.2.1 Investigating Idle Power

As you study power at near idle, watch for very high frame rates. If your app has high frame rates at near idle power (during cut scenes, menus, or other low-GPU-intensive parts), remember that these parts of your app will look fine if you lock the present interval to a 60Hz display refresh rate (or clamp your frame rate lower, to 30 FPS).

#### 5.2.2 Active Power and Speed Shift

While in active states, the processor and the operating system jointly decide frequencies for various parts of the system (CPUs, GPU, and memory ring, in particular). The current generation of Intel Core processors add more interaction between the operating system and the processor(s) to respond more efficiently and quickly to changes in power demand—a process referred to as Intel® Speed Shift Technology.

#### 5.2.3 When and How to Reduce Activity

There are times when the user explicitly requests trading performance for battery life, and there are things you can do to more effectively meet these demands.

#### 5.2.4 Scale Settings to Match System Power Settings and Power Profile

* Use RegisterPowerSettingNotification() with the appropriate globally unique identifier (GUID) to track changes.
* Scale your app's settings and behavior based on the power profile and whether your device is plugged in to power.

#### 5.2.5 Run as Slow as You Can, While Remaining Responsive

* Detect when you are in a power-managed mode and limit frame rate. Running at 30 Hz instead of 60 Hz can save significant power.
* Provide a way to disable the frame rate limit, for benchmarking.
* Use off-screen buffers and do smart compositing for in-game user interfaces.

#### 5.2.6 Manage Timers and Respect System Idle, Avoid Tight Polling Loops

* Reduce your app's reliance on high-resolution periodic timers.
* Avoid Sleep() calls in tight loops. Use Wait*() APIs instead.
* Avoid tight polling loops. Convert to an event-driven architecture.
* Avoid busy-wait calls.

```cpp
HRESULT res;
IDirect3DQuery9 *pQuery;
// create a query
res = pDevice->CreateQuery(. &pQuery);
//busy-wait for query data - AVOID THIS!
while ((res = pQuery->GetData(. 0)) == S_FALSE);
```

#### 5.2.7 Multithread Sensibly

Balanced threading offers performance benefits, but you need to consider how it operates alongside the GPU. Avoid affinitizing threads so that the operating system can schedule threads directly. If you must, provide hints using SetIdealProcessor().

### 5.3. Use of SIMD

Using SIMD instructions, either through the Intel® SPMD Program Compiler or intrinsics, can provide a significant power and performance boost.

### Figure 12: SIMD Instructions versus Power Consumption

![SIMD Power Graph]()

*A line graph titled "SIMD instructions versus power consumption" showing Package Power (W) on the y-axis (ranging from 2 to 20) and Thread Count on the x-axis (1-4). Four lines represent different configurations: GPU, CPU 1/4th 1 Sub Thread Affinity, CPU 1/2th 2 Sub Thread Affinity, and CPU All 4 Sub Thread Affinity. The lines show varying power consumption across different thread counts.*

However, on Intel Core processors, using SIMD instruction requires a voltage increase, in order to power the SIMD architecture block. In order to avoid power increase, Intel Core processors will then run at a lower frequency, which can decrease performance for a mostly scalar workload with a few SIMD instructions. For this reason, sporadic SIMD usage should be avoided.

### 5.4. Power Versus Frame Rate

The latest graphics APIs (DirectX 12, Vulkan, Metal 2) can dramatically reduce CPU overhead, resulting in lower CPU power consumption given a fixed frame rate (33 fps). When unconstrained by frame rate, the total power consumption is unchanged, but there is a significant performance boost due to increased GPU utilization.

### Figure 13: Power Comparison DirectX 11 vs DirectX 12

![Power Comparison Chart]()

*Two bar charts comparing power consumption for DirectX 11 and DirectX 12 under different frame rate conditions. The left chart shows "33 fps fixed frame rate" with two bars segmented into blue (CPU Power) and red (GPU Power), demonstrating lower CPU power for DirectX 12. The right chart shows "unconstrained frame rate" with similar bars showing power distribution between CPU and GPU for both APIs.*

---

## 6. Performance Analysis with Intel® GPA

### 6.1. Performance Analysis with Experiments

Select one or more events that take at least 20,000 cycles to ensure that the metrics are as accurate as possible. If you want to study short draw calls where the cycle count is <20,000, select multiple back-to-back events as long as the following conditions are met:

* Total cycle count of all selected ergs is ≥ 20,000.
* There are no state changes between the ergs.
* Events share the same render, depth, and stencil surface.

### 6.2. Performance Analysis with Hardware Metrics

Once you have selected the events of interest, you can accurately triage the hotspots associated with major graphics architectural blocks. Intel Processor Graphics perform deeply pipelined parallel execution of front-end (geometry transformation, rasterization, early depth/stencil, etc.) and back-end (pixel shading, sampling, color write, blend, and late depth/stencil) work within a single event.

### Figure 14: Hardware Metrics for 3D Workloads

![3D Metrics Flowchart]()

*A flowchart illustrating hardware metrics for 3D workloads. Starting from "Start 3D" (oval), it proceeds to a decision diamond "Is it Filtered?" If Yes, leads to "Draw not eligible". If No, proceeds through rectangular boxes in sequence: "LLC/EDRAM/DRAM (GTI)" → "Pixel Back-End (PBE)" → "Shader Execution (EU)". Then a decision diamond "EU Occupancy <= 80%" branches: If Yes, leads to "Thread Dispatch (TDL)" → "Setup Back-End (SBE)" → "Early Depth-Stencil (Z/STC)" → "Rasterization" → "Geometry Transformation". If No, leads to another decision "EU Stalled >= 10%" which branches to either "L3" → "Sampler" → "Unknown Shader Stall" (Yes) or "Unknown Execution Hotspot" (No).*

**Markdown Diagram Recreation:**

```
                              ┌─────────────┐
                              │  Start 3D   │
                              └──────┬──────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  Is it Filtered? │
                            └────────┬────────┘
                              Y /    │    \ N
                               /     │     \
                ┌─────────────┴──┐   │   ┌──┴─────────────────┐
                │Draw not eligible│   │   │ LLC/EDRAM/DRAM    │
                └────────────────┘   │   │      (GTI)         │
                                     │   └─────────┬──────────┘
                                     │             │
                                     │             ▼
                                     │   ┌──────────────────┐
                                     │   │ Pixel Back-End   │
                                     │   │     (PBE)        │
                                     │   └─────────┬────────┘
                                     │             │
                                     │             ▼
                                     │   ┌──────────────────┐
                                     │   │ Shader Execution │
                                     │   │      (EU)        │
                                     │   └─────────┬────────┘
                                     │             │
                                     │             ▼
                                     │   ┌───────────────────────┐
                                     │   │ EU Occupancy <= 80%?  │
                                     │   └───────────┬───────────┘
                                     │        Y /    │    \ N
                                     │         /     │     \
                    ┌─────────────────┴─┐      │   ┌─┴────────────────┐
                    │ Thread Dispatch   │      │   │ EU Stalled>=10%? │
                    │     (TDL)         │      │   └────────┬─────────┘
                    └─────────┬─────────┘      │      Y /    \ N
                              │                │       /      \
                              ▼                │   ┌──┴──┐  ┌──┴─────────────────┐
                    ┌─────────────────┐        │   │ L3  │  │Unknown Execution   │
                    │ Setup Back-End  │        │   └──┬──┘  │    Hotspot         │
                    │     (SBE)       │        │      │     └────────────────────┘
                    └─────────┬───────┘        │      ▼
                              │                │   ┌───────────┐
                              ▼                │   │  Sampler  │
                    ┌─────────────────┐        │   └─────┬─────┘
                    │Early Depth-     │        │         │
                    │Stencil (Z/STC)  │        │         ▼
                    └─────────┬───────┘        │   ┌───────────────────┐
                              │                │   │ Unknown Shader    │
                              ▼                │   │     Stall         │
                    ┌─────────────────┐        │   └───────────────────┘
                    │  Rasterization  │        │
                    └─────────┬───────┘        │
                              │                │
                              ▼                │
                    ┌─────────────────────┐    │
                    │ Geometry Transform  │    │
                    │    (Non-slice)      │    │
                    └─────────────────────┘    │
```

### Figure 15: Hardware Metrics for Compute Workloads

![Compute Metrics Flowchart]()

*A flowchart illustrating hardware metrics for compute workloads. Starting from "Start GPGPU" (oval), it proceeds to "Is it Filtered?" decision. If Yes, leads to "Dispatch not eligible". If No, proceeds through "LLC/EDRAM/DRAM (GTI)" → "Shader Execution (EU)" → decision "EU Occupancy <= 80%" (branches to "Thread Dispatch (TDL)" if Yes) → decision "EU Stalled >= 10%" (branches to "L3" → "Sampler" → "Unknown Shader Stall" if Yes, or "Unknown Execution Hotspot" if No).*

### Figure 16: Frame Analyzer Hotspots

![Frame Analyzer Hotspots]()

*A comparison of 3D Metrics and Compute Metrics showing pipeline stages and their status. The 3D Metrics pipeline shows stages like LLC/EDRAM/DRAM, Pixel Back-End, Shader Execution, Thread Dispatch, Setup Back-End, Early Depth/Stencil, Rasterization, and Geometry Transformation. The Compute Metrics pipeline is simpler, showing LLC/EDRAM/DRAM, Shader Execution, Thread Dispatch, and Sampler. Stages are color-coded: green (not a bottleneck), yellow (minor optimization opportunity), red (primary bottleneck).*

Each of the nodes in the flowcharts from Figures 14 and 15 are shown in the Intel® GPA Graphics Frame Analyzer metrics analysis tab. In this view:
- **Green** means that the bottleneck's criteria was not met and that part of the pipeline is not the bottleneck.
- **Red** means that this part of the GPU pipeline is the bottleneck.
- **Yellow** means that that node is not a primary bottleneck, but does have performance optimization opportunities.

#### 6.2.1 LLC/ EDRAM/ DRAM—Graphics Interface to Memory Hierarchy (GTI)

| Metric Name | Description |
|-------------|-------------|
| GTI: SQ is full | Percentage of time that the graphics-to-memory interface is fully saturated for the event(s) due to internal cache misses. |

When GTI: SQ is full more than 90 percent of the time, this is probably a primary hotspot. Improve the memory access pattern of the event(s) to reduce cache misses.

#### 6.2.2 Pixel Back-End—Color Write and Post-Pixel Shader (PS) Operations (PBE)

| Metric Name | Description |
|-------------|-------------|
| GPU / 3D Pipe: Slice <N> PS Output Available | Percentage of time that color data is ready from pixel shading to be processed by the pixel back-end for slice 'N'. |
| GPU / 3D Pipe: Slice <N> Pixel Values Ready | Percentage of time that pixel data is ready in the pixel back-end (following post-PS operations) for color write. |

#### 6.2.3 Shader Execution—Shader Execution FPU Pipe 0/1 (EU)

| Metric Name | Description |
|-------------|-------------|
| EU Array / Pipes: EU FPU0 Pipe Active | Percentage of time the Floating Point Unit (FPU) pipe is actively executing instructions. |
| EU Array / Pipes: EU FPU1 Pipe Active | Percentage of time the Extended Math (EM) pipe is active executing instructions. |

#### 6.2.4 EU Occupancy—Shader Thread EU Occupancy

| Metric Name | Description |
|-------------|-------------|
| EU Array: EU Thread Occupancy | Percentage of time that all EU threads were occupied with shader threads. |

#### 6.2.5 Thread Dispatch (TDL)

| Metric Name | Description |
|-------------|-------------|
| EU Array: EU Thread Occupancy | Percentage of time that all EU threads were occupied with shader threads. |
| GPU / Thread Dispatcher: PS Thread Ready for Dispatch on Slice <N> Subslice <M> | The percentage of time in which PS thread is ready for dispatch. |

#### 6.2.6 Setup Back-End (SBE)

| Metric Name | Description |
|-------------|-------------|
| GPU / Rasterizer / Early Depth Test: Slice<N> Post-Early Z Pixel Data Ready | Percentage of time that early depth/stencil had pixel data ready for dispatch. |

#### 6.2.7 Early Depth/Stencil (Z/STC)

| Metric Name | Description |
|-------------|-------------|
| GPU / Rasterizer: Slice <N> Rasterizer Output Ready | Percentage of time that input was available for early depth/stencil evaluation from rasterization unit. |

#### 6.2.8 Rasterization

| Metric Name | Description |
|-------------|-------------|
| GPU / Rasterizer: Slice <N> Rasterizer Input Available | Percentage of time that input was available to the rasterizer from geometry transformation. |
| GPU / 3D Pipe / Strip Fans: Polygon Data Ready | The percentage of time in which geometry pipeline output is ready. |

#### 6.2.9 Geometry Transformation (non-slice)

Reaching this point in the flow indicates that geometry transformation is taking up a significant amount of execution time.

#### 6.2.10 Shader Execution Stalled

| Metric Name | Description |
|-------------|-------------|
| EU Array: EU Stall | Percentage of time that the shader threads were stalled. |

#### 6.2.11 Unknown Shader Execution Hotspot

When you hit this point and the stall is low but the occupancy is high it indicates that there is some EU execution inefficiency associated with the workload.

#### 6.2.12 Graphics Cache (L3)

| Metric Name | Description |
|-------------|-------------|
| GTI/L3: Slice <N> L3 Bank <M> Active | Percentage of time that L3 bank 'M' on slice 'N' is servicing memory requests. |
| GTI/L3: Slice <N> L3 Bank <M> Stalled | Percentage of time that L3 bank 'M' on slice 'N' has a memory request but cannot service. |

#### 6.2.13 Sampler

| Metric Name | Description |
|-------------|-------------|
| GPU / Sampler: Slice <N> Subslice<M> Sampler Input Available | Percentage of time there is input from the EUs to the sampler. |
| GPU / Sampler: Slice <N> Subslice<M> Sampler Output Ready | Percentage of time there is output from the sampler to EUs. |

#### 6.2.14 Unknown Shader Stall

Indicates that while a stall was seen during shader execution, the root cause is not clear. Further debugging will be required.

---

## 7. Appendix: Developer Resources

### 7.1. The Intel® Software Developer Zone and Game Dev Websites

Intel regularly releases to the developer community code samples covering a variety of topics.

For the most up-to-date samples and links, please see the following resources:

* Intel® Software Developer Zone
* GitHub* Intel Repository

#### 7.1.1 GPU Detect

This DirectX sample demonstrates how to get the vendor and ID from the GPU. For Intel Processor Graphics, the sample also demonstrates a default graphics quality preset (low, medium, or high), support for DirectX 9 and DirectX 11 extensions, and the recommended method for querying the amount of video memory.

#### 7.1.2 Fast ISPC Texture Compression

This sample performs high-quality BC7, BC6H, ETC1, and ASTC compression on the CPU using the Intel® SPMD Program Compiler (Intel® SPC) to exploit SIMD instruction sets.

#### 7.1.3 Asteroids and DirectX* 12

An example of how to use the DirectX 12 graphics API to achieve performance and power benefits over previous APIs.

#### 7.1.4 Multi-Adapter Support with DirectX 12

This sample shows how to implement an explicit multi-adapter application using DirectX 12.

#### 7.1.5 Early-Z Rejection

This sample demonstrates two ways to take advantage of early Z rejection: Front to back rendering and z prepass.

#### 7.1.6 Additional Code Samples

* Dynamic Resolution Rendering
* Conservative Morphological Anti-Aliasing Article and Sample
* Software Occlusion Culling
* Sparse Procedural Volumetric Rendering
* Sample Distribution Shadow Maps
* Programmable Blend with Pixel Shader Ordering
* Adaptive Volumetric Shadow Maps
* Adaptive Transparency Paper
* CPU Texture Compositing with Instant Access
* OpenGL Fragment Shader Ordering Extension
* OpenGL Map Texture Extension

### 7.2. DirectX 12 Resources

The Microsoft Developer Network (MSDN.com) should be your first stop for Windows technical information and DirectX.

**Videos:**
- Microsoft DirectX 12 and Graphics Education (channel)

**Programming Guides:**
- Direct3D 12 Programming Guide
- Direct3D 12 Overview
- Programmer Guide for HLSL
- Gpudetect - an example of a CPU and memory check

**Reference documents:**
- Direct3D 12 Reference

**Drivers:**
- Intel Download Center

### 7.3. Vulkan Resources

The primary website for the Vulkan API is https://www.khronos.org/vulkan.

**Videos:**
- 2017 DevU - 01 Getting Started with Vulkan

**Programming Guides:**
- API without Secrets: Vulkan (in 6 parts)
- Beginners Guide to Vulkan
- Vulkan 1.0.19 + WSI Extensions
- Vulkan in 30 minutes
- Vulkan API Companion Guide
- PowerVR Documentation
- Introduction to SPIR-V Shaders

**Reference documents:**
- Vulkan 1.0.55 - A Specification (with KHR extensions)
- Vulkan API Reference
- LunarG Vulkan™ SDK
- SPIR-V Shader
- GLM Library (linear algebra)
- GLFW Library (window creation)

**Drivers:**
- Linux Open Source
- Intel Download Center

### 7.4. Metal* 2 Resources

The primary website for the Metal 2 API is https://developer.apple.com/metal.

**Videos:**
- WWDC17 - Session 601 Introducing Metal 2
- WWDC17 - Session 603 VR with Metal 2
- WWDC17 - Session 608 Using Metal 2 for Compute

**Programming Guides:**
- The Metal Programming Guide
- Metal Best Practices Guide
- Introducing Metal 2 (summary of WWDC 2017 Metal sessions)

**Reference documents:**
- The Metal API
- MetalKit
- UIView
- Metal Shading Language Specification Version 2.0
- Metal Performance Shaders

**Drivers:**
- Included in macOS*, Apple* Store

### 7.5. OpenGL* Resources

There are abundant resources for the OpenGL developer. Some of the better ones are listed below:

**OpenGL Websites:**
- OpenGL.org - SDK, tutorials, sample code, community
- The Khronos Group - API registry, forums, resources

**Videos:**
- Siggraph University: An Introduction to OpenGL Programming

**Programming Guides:**
- OpenGL Tutorial
- Learn OpenGL
- OpenGL-Tutorial

**Reference documents:**
- OpenGL 4.5
- OpenGL Extension Wrangler (GLEW) Library (Extensions)
- GLM Library (linear algebra)
- GLFW Library (window creation)

**Drivers:**
- Linux* Open Source
- Windows 7, 8.1, 10

---

## 8. Notices

No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document.

Intel disclaims all express and implied warranties, including without limitation, the implied warranties of merchantability, fitness for a particular purpose, and non-infringement, as well as any warranty arising from course of performance, course of dealing, or usage in trade.

This document contains information on products, services and/or processes in development. All information provided here is subject to change without notice. Contact your Intel representative to obtain the latest forecast, schedule, specifications and roadmaps.

Intel may make changes to specifications and product descriptions at any time, without notice. Designers must not rely on the absence or characteristics of any features or instructions marked "reserved" or "undefined". Intel reserves these for future definition and shall have no responsibility whatsoever for conflicts or incompatibilities arising from future changes to them. The information here is subject to change without notice. Do not finalize a design with this information.

The products and services described may contain defects or errors known as errata which may cause deviations from published specifications. Current characterized errata are available on request.

Contact your local Intel sales office or your distributor to obtain the latest specifications and before placing your product order. Copies of documents which have an order number and are referenced in this document may be obtained by calling 1-800-548-4725 or by visiting www.intel.com/design/literature.htm.

This sample source code is released under the Intel Sample Source Code License Agreement.

**Intel, the Intel logo, Intel Core, Intel Speed Step, and VTune are trademarks of Intel Corporation in the U.S. and/or other countries.**

*Microsoft, Windows, and the Windows logo are trademarks, or registered trademarks of Microsoft Corporation in the United States and/or other countries. Other names and brands may be claimed as the property of others.*

**© 2017 Intel Corporation.**

---

*Document: 58 Pages*