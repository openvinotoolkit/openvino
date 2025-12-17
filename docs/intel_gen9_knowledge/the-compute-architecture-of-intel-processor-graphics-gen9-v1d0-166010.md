# The Compute Architecture of Intel® Processor Graphics Gen9

**Version 1.0**

---

![Intel Chip Rendering]()

*A 3D rendering of a rectangular integrated circuit (chip) with a complex, colorful pattern of blue, green, and yellow blocks, suggesting different functional areas. The chip is oriented slightly to the right, with a subtle reflection beneath it.*

---

## External Revision History

| Version | Date | Comment |
|---------|------|--------|
| v1.0 | 8/14/2015 | IDF-2015 release of "The Compute Architecture of Intel Processor Graphics Gen9" - Stephen Junkins |

---

## Table of Contents

1. [Contents](#1-contents)
2. [Audience](#2-audience)
3. [Introduction](#3-introduction)
   - 3.1 [What is Intel Processor Graphics?](#31-what-is-intel-processor-graphics)
4. [SoC Architecture](#4-soc-architecture)
   - 4.1 [SoC Architecture](#41-soc-architecture)
   - 4.2 [Ring Interconnect](#42-ring-interconnect)
   - 4.3 [Shared LLC](#43-shared-llc)
   - 4.4 [Optional EDRAM](#44-optional-edram)
5. [The Compute Architecture of Intel Processor Graphics Gen9](#5-the-compute-architecture-of-intel-processor-graphics-gen9)
   - 5.1 [New Changes for Intel Processor Graphics Gen9](#51-new-changes-for-intel-processor-graphics-gen9)
   - 5.2 [Modular Design for Product Scalability](#52-modular-design-for-product-scalability)
   - 5.3 [Execution Unit (EUs) Architecture](#53-execution-unit-eus-architecture)
   - 5.4 [Subslice Architecture](#54-subslice-architecture)
   - 5.5 [Slice Architecture](#55-slice-architecture)
   - 5.6 [Product Architecture](#56-product-architecture)
   - 5.7 [Memory](#57-memory)
   - 5.8 [Architecture Configurations, Speeds, and Feeds](#58-architecture-configurations-speeds-and-feeds)
6. [Example Compute Applications](#6-example-compute-applications)
7. [Acknowledgements](#7-acknowledgements)
8. [More Information](#8-more-information)
9. [Notices](#9-notices)

---

## 2. Audience

Software, hardware, and product engineers who seek to understand the architecture of Intel® processor graphics gen9. More specifically, those seeking to understand the architecture characteristics relevant to compute applications on Intel processor graphics.

This gen9 whitepaper updates the material found in "The Compute Architecture of Intel Processor Graphics Gen8" so that it can stand on its own. But where necessary, specific architecture changes for gen9 are noted.

---

## 3. Introduction

Intel's on-die integrated processor graphics architecture offers outstanding real time 3D rendering and media performance. However, its underlying compute architecture also offers general purpose compute capabilities that approach **teraFLOPS** performance. The architecture of Intel processor graphics delivers a full complement of high-throughput floating-point and integer compute capabilities, a layered high bandwidth memory hierarchy, and deep integration with on-die CPUs and other on-die system-on-a-chip (SoC) devices. Moreover, it is a modular architecture that achieves scalability for a family of products that range from cellphones to tablets and laptops, to high end desktops and servers.

### 3.1 What is Intel Processor Graphics?

Intel processor graphics is the technology that provides graphics, compute, media, and display capabilities for many of Intel's processor SoC products. At Intel, architects colloquially refer to Intel processor graphics architecture as simply **"Gen"**, shorthand for Generation. A specific generation of the Intel processor graphics architecture may be referred to as "Gen7" for generation 7, or "gen8" for generation 8, etc.

**Product Examples:**
- **Gen8 Architecture:**
  - Intel HD Graphics 5600
  - Intel Iris™ Graphics 6100
  - Intel Iris Pro Graphics 6200

- **Gen9 Architecture:**
  - Intel HD Graphics 530 (first released product)

> **Note:** Graphics product naming conventions changed with gen9, from 4 digits to 3.

This whitepaper focuses on just the compute architecture components of Intel processor graphics gen9. For shorthand, in this paper we may use the term **gen9 compute architecture** to refer to just those compute components. The whitepaper also briefly discusses the gen9 derived instantiation of Intel HD Graphics 530 in the recently released Intel Core™ i7 processor 6700K for desktop form factors. Additional processor products that include Intel processor graphics gen9 will be released in the near future.

---

### Figure 1: Architecture Components Layout for Intel® Core™ i7 Processor 6700K

![Architecture Components Layout]()

*A detailed diagram showing the architecture components layout for an Intel® Core™ i7 processor 6700K for desktop systems. The diagram is divided into several sections:*

- **Left (outlined in red dashed box):** Intel® Processor Graphics, Gen9 (graphics, compute, & media) - Intel® HD Graphics 530
- **Center:** Four CPU cores and Shared LLC (outlined in blue dashed boxes)
- **Right:** System Agent with display, memory, & I/O controllers

*This SoC contains 4 CPU cores and is a one-slice instantiation of Intel processor graphics gen9 architecture.*

---

## 4. SoC Architecture

This section describes the SoC architecture within which Intel processor graphics is a component.

### Figure 2: Intel® Core™ i7 Processor 6700K SoC and Ring Interconnect Architecture

![SoC Ring Interconnect Architecture]()

*A detailed diagram illustrating the Intel® Core™ i7 processor 6700K SoC and its ring interconnect architecture:*

**Left - Intel® Processor Graphics Gen9:**
- Command Streamer
- Global Thread Dispatcher
- Rendering fixed function units
- Slice with 24 EUs (Execution Units) and FUs (Fixed Units)
- L3 Data Cache

**Center:**
- Two CPU cores connected by SoC Ring Interconnect
- Two LLC cache slices

**Right - System Agent:**
- Display Controller
- Memory Controller
- PCIe
- (Opt) EDRAM Controller
- Clock domain indication

**Markdown Diagram Recreation:**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           INTEL® CORE™ i7 6700K SoC                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
┌───────────────────────┐   ┌─────────────────────┐   ┌───────────────────────────┐
│   INTEL PROCESSOR      │   │     CPU CORES        │   │      SYSTEM AGENT         │
│   GRAPHICS Gen9        │   │                      │   │                           │
├───────────────────────┤   ├─────────────────────┤   ├───────────────────────────┤
│ ┌───────────────────┐ │   │ ┌───────┐ ┌───────┐ │   │ ┌───────────────────────┐ │
│ │ Command Streamer  │ │   │ │ Core0 │ │ Core1 │ │   │ │  Display Controller   │ │
│ └───────────────────┘ │   │ └───────┘ └───────┘ │   │ └───────────────────────┘ │
│ ┌───────────────────┐ │   │                      │   │ ┌───────────────────────┐ │
│ │ Global Thread     │ │   │                      │   │ │  Memory Controller    │ │
│ │ Dispatcher        │ │   │                      │   │ └───────────────────────┘ │
│ └───────────────────┘ │   │                      │   │ ┌───────────────────────┐ │
│ ┌───────────────────┐ │   │                      │   │ │       PCIe            │ │
│ │ Rendering fixed   │ │   │                      │   │ └───────────────────────┘ │
│ │ function units    │ │   │                      │   │ ┌───────────────────────┐ │
│ └───────────────────┘ │   │                      │   │ │ (Opt) EDRAM Controller│ │
│ ┌───────────────────┐ │   │                      │   │ └───────────────────────┘ │
│ │ Slice: 24 EUs     │ │   │ ┌───────┐ ┌───────┐ │   │                           │
│ │ + FUs            │ │   │ │ LLC$0 │ │ LLC$1 │ │   │                           │
│ └───────────────────┘ │   │ └───────┘ └───────┘ │   │                           │
│ ┌───────────────────┐ │   │                      │   │                           │
│ │ L3 Data Cache    │ │   │                      │   │                           │
│ └───────────────────┘ │   │                      │   │                           │
└────────────┬──────────┘   └──────────┬───────────┘   └─────────────┬─────────────┘
             │                    │                           │
             └────────────────────┼───────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │     SoC Ring Interconnect    │
                    └─────────────────────────────┘
```

---

### 4.1 SoC Architecture

Intel 6th generation Core processors are complex SoCs integrating multiple CPU cores, Intel processor graphics, and potentially other fixed functions all on a single shared silicon die. The architecture implements multiple unique clock domains, which have been partitioned as:

- A **per-CPU core clock domain**
- A **processor graphics clock domain**
- A **ring interconnect clock domain**

The SoC architecture is designed to be extensible for a range of products, and yet still enable efficient wire routing between components within the SoC.

### 4.2 Ring Interconnect

The on-die bus between CPU cores, caches, and Intel processor graphics is a **ring based topology** with dedicated local interfaces for each connected "agent". This SoC ring interconnect is a bi-directional ring that has:

- **32-byte wide data bus**
- Separate lines for **request**, **snoop**, and **acknowledge**

Every on-die CPU core is regarded as a unique agent. Similarly, Intel processor graphics is treated as a unique agent on the interconnect ring. A **system agent** is also connected to the ring, which bundles the DRAM memory management unit, display controller, and other off chip I/O controllers such as PCI Express. Importantly, all off-chip system memory transactions to/from CPU cores and to/from Intel processor graphics are facilitated by this interconnect, through the system agent, and the unified DRAM memory controller.

### 4.3 Shared LLC

Some SoC products include a **shared Last Level Cache (LLC)** that is also connected to the ring. In such SoCs, each on-die core is allocated a slice of cache, and that cache slice is connected as a unique agent on the ring. However, all of the slices work together as a single cache, albeit a shared and distributed cache.

An **address hashing scheme** routes data requests to the cache slice assigned to its address. This distributed LLC is also shared with Intel processor graphics. For both CPU cores and for Intel processor graphics, LLC seeks to reduce apparent latency to system DRAM and to provide higher effective bandwidth.

### 4.4 Optional EDRAM

Some SoC products may include **64-128 megabytes of embedded DRAM (EDRAM)**, bundled into the SoC's chip packaging. For example, the Intel processor graphics gen8 based Intel Iris Pro 6200 products bundle a 128 megabyte EDRAM.

**EDRAM Specifications:**
- Operates in its own clock domain
- Can be clocked up to **1.6GHz**
- Has separate buses for read and write, each capable of **32 bytes/EDRAM-cycle**
- Supports low latency display surface refresh

For the compute architecture of Intel processor graphics gen9, EDRAM further supports the memory hierarchy by serving as a **"memory-side" cache** between LLC and DRAM. Like LLC, EDRAM caching is shared by both Intel processor graphics and by CPU cores.

**EDRAM Cache Behavior:**
- On an LLC or EDRAM cache miss, data from DRAM will be filled first into EDRAM
- An optional mode also allows bypass to LLC
- As cachelines are evicted from LLC, they will be written back into EDRAM
- If compute kernels wish to read or write cachelines currently stored in EDRAM, they are quickly re-loaded into LLC, and read/writing then proceeds as usual

---

## 5. The Compute Architecture of Intel Processor Graphics Gen9

### 5.1 New Changes for Intel Processor Graphics Gen9

Intel processor graphics gen9 includes many refinements throughout the micro architecture and supporting software, over Intel processor graphics gen8. Generally, these changes are across the domains of **memory hierarchy**, **compute capability**, and **product configuration**.

#### Gen9 Memory Hierarchy Refinements:

- Coherent SVM write performance is significantly improved via new LLC cache management policies
- The available L3 cache capacity has been increased to **768 Kbytes per slice** (512 Kbytes for application data)
- The sizes of both L3 and LLC request queues have been increased. This improves latency hiding to achieve better effective bandwidth against the architecture peak theoretical
- In Gen9 EDRAM now acts as a **memory-side cache** between LLC and DRAM. Also, the EDRAM memory controller has moved into the system agent, adjacent to the display controller, to support power efficient and low latency display refresh
- Texture samplers now natively support an **NV12 YUV format** for improved surface sharing between compute APIs and media fixed function units

#### Gen9 Compute Capability Refinements:

- **Preemption** of compute applications is now supported at a thread level, meaning that compute threads can be preempted (and later resumed) midway through their execution
- **Round robin scheduling** of threads within an execution unit
- Gen9 adds new native support for the **32-bit float atomics** operations of min, max, and compare/exchange. Also the performance of all 32-bit atomics is improved for kernel scenarios that issued multiple atomics back to back
- **16-bit floating point** capability is improved with native support for denormals and gradual underflow

#### Gen9 Product Configuration Flexibility:

- Gen9 has been designed to enable products with **1, 2 or 3 slices**
- Gen9 adds new **power gating** and clock domains for more efficient dynamic power management. This can particularly improve low power media playback modes

---

### 5.2 Modular Design for Product Scalability

The gen9 compute architecture is designed for scalability across a wide range of target products. The architecture's modularity enables exact product targeting to a particular market segment or product power envelope.

**Building Blocks (from smallest to largest):**
1. **Execution Units (EUs)** - foundational compute components
2. **Subslices** - clusters of EUs
3. **Slices** - clusters of subslices

Together, execution units, subslices, and slices are the modular building blocks that are composed to create many product variants based upon Intel processor graphics gen9 compute architecture.

---

### 5.3 Execution Unit (EUs) Architecture

The foundational building block of gen9 compute architecture is the **execution unit**, commonly abbreviated as **EU**. The architecture of an EU is a combination of:

- **Simultaneous multi-threading (SMT)**
- **Fine-grained interleaved multi-threading (IMT)**

These are compute processors that drive multiple issue, single instruction, multiple data arithmetic logic units (**SIMD ALUs**) pipelined across multiple threads, for high-throughput floating-point and integer compute. The fine-grain threaded nature of the EUs ensures continuous streams of ready to execute instructions, while also enabling latency hiding of longer operations such as memory scatter/gather, sampler requests, or other system communication.

### Figure 3: The Execution Unit (EU)

![Execution Unit Diagram]()

*A block diagram illustrating the architecture of an Execution Unit (EU). The diagram is titled "EU: Execution Unit" at the top:*

- **Left:** "Instruction Fetch" block with an arrow pointing right into a central section
- **Center:** Multiple rows of green rectangular blocks representing registers or processing elements
- Below the green blocks: "28KB GRF: 7 thrds x 128x SIMD8 x 32b" and "ARF" (Architecture Register File)
- **Right:** "Thread Arbiter" block connecting to output units
- **Output units:** Send, Branch, SIMD FPU, and another SIMD FPU

**Markdown Diagram Recreation:**

```
┌────────────────────────────────────────────────────────────────────┐
│                        EU: Execution Unit                        │
└────────────────────────────────────────────────────────────────────┘
┌──────────────────┐
│ Instruction Fetch│──────┬────────────────────────────────────────────┐
└──────────────────┘      │                                            │
                         ▼                                            │
┌──────────────────────────────────────────┐                         │
│ ████████████████████████████████████████ │  Thread 0 (GRF)        │
│ ████████████████████████████████████████ │  Thread 1 (GRF)        │
│ ████████████████████████████████████████ │  Thread 2 (GRF)        │
│ ████████████████████████████████████████ │  Thread 3 (GRF)        │
│ ████████████████████████████████████████ │  Thread 4 (GRF)        │
│ ████████████████████████████████████████ │  Thread 5 (GRF)        │
│ ████████████████████████████████████████ │  Thread 6 (GRF)        │
├──────────────────────────────────────────┤                         │
│  28KB GRF: 7 thrds x 128x SIMD8 x 32b    │  ┌───────┐             │
└──────────────────────────────────────────┘  │  ARF  │             │
                                              └───────┘             │
                    │                                                │
                    ▼                                                │
┌─────────────────────────┐                                       │
│    Thread Arbiter       │                                       │
└─────┬──────┬──────┬──────┬┘                                       │
      │      │      │      │                                         │
      ▼      ▼      ▼      ▼                                         │
  ┌─────┐┌───────┐┌─────────┐┌─────────┐                            │
  │Send ││Branch ││SIMD FPU ││SIMD FPU │                            │
  └─────┘└───────┘└─────────┘└─────────┘                            │
────────────────────────────────────────────────────────────────────┘
```

**Figure 3 Caption:** Each gen9 EU has seven threads. Each thread has 128 SIMD-8 32-bit registers (GRF) and supporting architecture specific registers (ARF). The EU can co-issue to four instruction processing units including two FPUs, a branch unit, and a message send unit.

---

#### EU Specifications:

| Component | Gen9 Specification |
|-----------|--------------------|
| Threads per EU | 7 |
| Registers per Thread | 128 general purpose registers |
| Register Size | 32 bytes (SIMD 8-element vector of 32-bit data) |
| GRF per Thread | 4 Kbytes |
| GRF per EU | 28 Kbytes total |
| Max Co-issue | 4 instructions/cycle (from 4 different threads) |

Product architects may fine-tune the number of threads and number of registers per EU to match scalability and specific product design requirements. Flexible addressing modes permit registers to be addressed together to build effectively wider registers, or even to represent strided rectangular block data structures. Per-thread architectural state is maintained in a separate dedicated architecture register file (ARF).

#### 5.3.1 Simultaneous Multi-Threading and Multiple Issue Execution

Depending on the software workload, the hardware threads within an EU may all be executing the same compute kernel code, or each EU thread could be executing code from a completely different compute kernel. The execution state of each thread, including its own instruction pointers, are held in thread-specific ARF registers.

On every cycle, an EU can co-issue up to **four different instructions**, which must be sourced from four different threads. The EU's thread arbiter dispatches these instructions to one of four functional units for execution. Although the issue slots for the functional units pose some instruction co-issue constraints, the four instructions are independent, since they are dispatched from four different threads.

It is theoretically possible for **just two non-stalling threads to fully saturate** the floating-point compute throughput of the machine. More typically all seven threads are loaded to deliver more ready-to-run instructions from which the thread arbiter may choose, and thereby promote the EU's instruction-level parallelism.

#### 5.3.2 SIMD FPUs

In each EU, the primary computation units are a pair of **SIMD floating-point units (FPUs)**. Although called FPUs, they support both floating-point and integer computation.

**FPU Capabilities:**
- SIMD execute up to **four 32-bit floating-point (or integer) operations**
- SIMD execute up to **eight 16-bit integer or 16-bit floating-point operations**
- 16-bit float (half-float) support is **new for gen9** compute architecture
- Each SIMD FPU can complete simultaneous **add and multiply (MAD)** floating-point instructions every cycle

**Peak Operations per EU:**
```
16 32-bit FP ops/cycle = (add + mul) x 2 FPUs x SIMD-4
```

In gen9, both FPUs support native 32-bit integer operations. Finally, one of the FPUs provides **extended math capability** to support high-throughput transcendental math functions and **double precision 64-bit floating-point**.

**Local Bandwidth within EU:**
- MAD instructions: **96 bytes/cycle read bandwidth** (3 source operands)
- MAD instructions: **32 bytes/cycle write bandwidth** (1 destination operand)

Aggregated across the whole architecture, this bandwidth can scale linearly with the number of EUs. For gen9 products with multiple slices of EUs and higher clock rates, the aggregated theoretical peak bandwidth that is local between FPUs and GRF can approach **multiple terabytes of read bandwidth**.

#### 5.3.3 Branch and Send Units

Within the EUs:
- **Branch instructions** are dispatched to a dedicated **branch unit** to facilitate SIMD divergence and eventual convergence
- **Memory operations, sampler operations, and other longer-latency system communications** are all dispatched via **"send" instructions** executed by the message passing send unit

#### 5.3.4 EU ISA and Flexible Width SIMD

The EU Instruction Set Architecture (ISA) and associated general purpose register file are all designed to support a **flexible SIMD width**. Thus for 32-bit data types, the gen9 FPUs can be viewed as physically 4-wide. But the FPUs may be targeted with SIMD instructions and registers that are logically:

- 1-wide
- 2-wide
- 4-wide
- 8-wide
- 16-wide
- 32-wide

For example, a single operand to a **SIMD-16 wide instruction** pairs two adjacent SIMD-8 wide registers, logically addressing the pair as a single SIMD-16 wide register containing a contiguous **64 bytes**. This logically SIMD-16 wide instruction is transparently broken down by the microarchitecture into physically SIMD-4 wide FPU operations, which are iteratively executed.

From the viewpoint of a single thread, wider SIMD instructions do take more cycles to complete execution. But because the EUs and EU functional units are fully pipelined across multiple threads, **SIMD-8, SIMD-16, and SIMD-32 instructions are all capable of maximizing compute throughput** in a fully loaded system.

The instruction SIMD width choice is left to the compiler or low level programmer. Differing SIMD width instructions can be issued back to back with no performance penalty. This flexible design allows compiler heuristics and programmers to choose specific SIMD widths that precisely optimize the register allocation footprint for individual programs, balanced against the amount of work assigned to each thread.

#### 5.3.5 SIMD Code Generation for SPMD Programming Models

Compilers for **single program multiple data (SPMD)** programming models, such as:
- RenderScript
- OpenCL™¹
- Microsoft DirectX Compute Shader
- OpenGL Compute
- C++AMP

...generate SIMD code to map multiple **kernel instances**² to be executed simultaneously within a given hardware thread. The exact number of kernel instances per-thread is a heuristic driven compiler choice. We refer to this compiler choice as the **dominant SIMD-width** of the kernel. In OpenCL and DirectX Compute Shader, SIMD-8, SIMD-16, and SIMD-32 are the most common SIMD-width targets.

**Concurrent Kernel Instances per EU:**

| SIMD Width | Threads | Concurrent Kernel Instances |
|------------|---------|-----------------------------|
| SIMD-16 | 7 | 16 × 7 = **112** |
| SIMD-32 | 7 | 32 × 7 = **224** |

For a given SIMD-width, if all kernel instances within a thread are executing the same instruction, then the SIMD lanes can be maximally utilized. If one or more of the kernel instances chooses a divergent branch, then the thread will execute the two paths of the branch separately in serial. The EU branch unit keeps track of such branch divergence and branch nesting. The branch unit also generates a **"live-ness" mask** to indicate which kernel instances in the current SIMD-width need to execute (or not execute) the branch.

> ¹ OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos.
> ² We use the generic term "kernel instance" as equivalent to OpenCL work-item, or DirectX Compute Shader thread.

---

### 5.4 Subslice Architecture

In gen9 compute architecture, arrays of EUs are instantiated in a group called a **subslice**. For scalability, product architects can choose the number of EUs per subslice. For most gen9-based products, each subslice contains **8 EUs**.

**Subslice Components:**
- Local thread dispatcher unit
- Supporting instruction caches
- 8 EUs with 7 threads each = **56 simultaneous threads** per subslice
- Sampler unit
- Data port memory management unit

Compared to the Gen7.5 design which had 10 EUs per subslice, the gen8 and gen9 designs reduce the number EUs sharing each subslice's sampler and data port. From the viewpoint of each EU, this has the effect of **improving effective bandwidth local to the subslice**.

### Figure 4: The Intel Processor Graphics Gen9 Subslice

![Subslice Architecture]()

*A block diagram illustrating the architecture of a subslice with 8 Execution Units (EUs):*

- **Top:** "Instruction cache" and "Local Thread Dispatcher"
- **Center:** Two columns of four "EU" blocks each, totaling 8 EUs
- Each EU block has a small rectangular component representing local resources
- **Bottom left:** "Sampler" with "L1" and "L2 Sampler Cache" (Read: 64B/cyc)
- **Bottom right:** "Data Port" (Read: 64B/cyc, Write: 64B/cyc)

**Markdown Diagram Recreation:**

```
┌───────────────────────────────────────────────────┐
│             SUBSLICE: 8 EUs                     │
├───────────────────────────────────────────────────┤
│ ┌──────────────────┐  ┌────────────────────────┐ │
│ │Instruction Cache│  │ Local Thread Dispatcher│ │
│ └──────────────────┘  └────────────────────────┘ │
│                                                  │
│  ┌────────┐┌────────┐  ┌────────┐┌────────┐       │
│  │   EU   ││   EU   │  │   EU   ││   EU   │       │
│  └────────┘└────────┘  └────────┘└────────┘       │
│  ┌────────┐┌────────┐  ┌────────┐┌────────┐       │
│  │   EU   ││   EU   │  │   EU   ││   EU   │       │
│  └────────┘└────────┘  └────────┘└────────┘       │
│                                                  │
├───────────────────────┬──────────────────────────┤
│       SAMPLER           │         DATA PORT          │
│                         │                            │
│  L1 & L2 Sampler Cache  │   Read: 64B/cyc            │
│  Read: 64B/cyc          │   Write: 64B/cyc           │
└───────────────────────┴──────────────────────────┘
```

#### 5.4.1 Sampler

The **sampler** is a read-only memory fetch unit that may be used for sampling of tiled (or not tiled) texture and image surfaces.

**Sampler Components:**
- Level-1 sampler cache (L1)
- Level-2 sampler cache (L2)
- Dedicated logic between caches for dynamic decompression of block compression texture formats (DirectX BC1-BC7, DXT, and OpenGL compressed texture formats)
- Fixed-function logic for address conversion on image (u,v) coordinates
- Address clamping modes: mirror, wrap, border, and clamp

**Supported Filtering Modes:**
- Point
- Bilinear
- Tri-linear
- Anisotropic

#### 5.4.2 Data Port

Each subslice also contains a memory load/store unit called the **data port**. The data port supports efficient read/write operations for:

- General purpose buffer accesses
- Flexible SIMD scatter/gather operations
- Shared local memory access

To maximize memory bandwidth, the unit **dynamically coalesces** scattered memory operations into fewer operations over non-duplicated 64-byte cacheline requests.

**Example:** A SIMD-16 gather operation against 16 unique offset addresses for 16 32-bit floating-point values, might be coalesced to a single 64-byte read operation if all the addresses fall within a single cacheline.

---

### 5.5 Slice Architecture

Subslices are clustered into **slices**. For most gen9-based products, **3 subslices** are aggregated into **1 slice**. Thus a single slice aggregates a total of **24 EUs**.³

> ³ Note some gen9-based products may enable fewer than 24 EUs in a slice.

**Slice Components:**
- Thread dispatch routing logic
- Banked level-3 cache
- Smaller but highly banked shared local memory structure
- Fixed function logic for atomics and barriers
- Additional fixed function units for media and graphics capability

### Figure 5: The Intel Processor Graphics Gen9 Slice

![Slice Architecture]()

*A detailed diagram illustrating the architecture of a graphics slice with 24 Execution Units (EUs). The slice is composed of three identical subslices, each containing 8 EUs:*

- **Top:** "Fixed function units" block
- **Each subslice:** Instruction cache, Local Thread Dispatcher, 8 EU blocks, Sampler with L2 Sampler Cache, and Data Port (Read: 64B/cyc, Write: 64B/cyc)
- **Bottom:** L3 Data Cache with read/write capabilities, Atomics/Barriers block, and Shared Local Memory block

**Markdown Diagram Recreation:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SLICE: 24 EUs                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Fixed Function Units                           │
├───────────────────────┬───────────────────────┬─────────────────────────┤
│     SUBSLICE 0          │     SUBSLICE 1          │     SUBSLICE 2        │
│     (8 EUs)             │     (8 EUs)             │     (8 EUs)           │
│ ┌───────────────────┐ │ ┌───────────────────┐ │ ┌───────────────────┐ │
│ │ Instr$ | ThrdDisp│ │ │ Instr$ | ThrdDisp│ │ │ Instr$ | ThrdDisp│ │
│ ├───────────────────┤ │ ├───────────────────┤ │ ├───────────────────┤ │
│ │EU EU EU EU     │ │ │EU EU EU EU     │ │ │EU EU EU EU     │ │
│ │EU EU EU EU     │ │ │EU EU EU EU     │ │ │EU EU EU EU     │ │
│ ├───────────────────┤ │ ├───────────────────┤ │ ├───────────────────┤ │
│ │Sampler|DataPort│ │ │Sampler|DataPort│ │ │Sampler|DataPort│ │
│ └───────────────────┘ │ └───────────────────┘ │ └───────────────────┘ │
└───────────────────────┴───────────────────────┴─────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      L3 Data Cache (512KB for compute)                  │
│                       Read: 64B/cyc   Write: 64B/cyc                    │
├─────────────────────────┬─────────────────────────────────────────────────┤
│    Atomics, Barriers    │            Shared Local Memory (64KB/subslice)│
└─────────────────────────┴─────────────────────────────────────────────────┘
```

#### 5.5.1 Level-3 Data Cache

For gen9-based products, the level-3 (L3) data cache capacity has been increased to **768 Kbytes total per slice**. Each application context has flexibility as to how much of the L3 memory structure is allocated:

1. As application L3 data cache
2. As system buffers for fixed-function pipelines
3. As shared local memory

For compute application contexts on gen9 compute architecture, the typical allocation is **512 Kbytes per slice** as application data cache.

**L3 Cache Characteristics:**
- Cachelines are **64 bytes each**
- Uniformly distributed across the entire aggregate cache
- Each data port has 64 bytes/cycle read and write interface to L3
- L3 fill logic: 64 bytes/cycle to/from system memory
- Aggregate L3 bandwidth for 3-subslice slice: **192 bytes/cycle**

#### 5.5.2 Shared Local Memory

**Shared local memory** is a structure within the L3 complex that supports programmer-managed data for sharing among EU hardware threads within the same subslice.

**Specifications:**
- Read/write bus interface: **64 bytes wide** per subslice
- Latency: Similar to L3 data cache
- Banking: More highly banked than L3 for better access patterns
- Capacity: **64 Kbytes per subslice** (dedicated and available)

> **Note:** Shared local memory is **not coherent** with other memory structures.

SPMD programming model constructs such as OpenCL's local memory space or DirectX Compute Shader's shared memory space are shared across a single work-group (thread-group). For software kernel instances that use shared local memory, driver runtimes typically map all instances within a given OpenCL work-group (or a DirectX 11 threadgroup) to EU threads within a single subslice.

#### 5.5.3 Barriers and Atomics

**Barriers:**
- Dedicated logic in each slice supports implementation of barriers across groups of threads
- Available as a hardware alternative to pure compiler-based barrier implementation
- Gen9 logic can support barriers simultaneously in up to **16 active thread-groups per subslice**

**Atomics:**
- Rich suite of atomic read-modify-write memory operations
- Support operations to L3 cached global memory or to shared local memory
- Gen9-based products support **32-bit atomic operations**

#### 5.5.4 64-Byte Data Width

A foundational element of gen9 compute architecture is the **64-byte data width**:

| Component | Width |
|-----------|-------|
| SIMD-16 instruction operands | 64 bytes |
| SIMD-16 register pairs | 64 bytes |
| L3 data bus | 64 bytes |
| L3 cacheline | 64 bytes |
| L3 to LLC bus interface | 64 bytes |

---

### 5.6 Product Architecture

SoC product architects can create product families or a specific product within a family by instantiating a **single slice or groups of slices**. Members of a product family might differ primarily in the number of slices.

These slices are combined with:
- Additional front end logic to manage command submission
- Fixed-function logic to support 3D, rendering, and media pipelines
- **Graphics Technology Interface (GTI)** for interfacing with the rest of the SoC

### Figure 6: Single Slice Product Design (24 EUs)

![Single Slice Design]()

*A potential product design composed of a single slice with three subslices, for a total of 24 EUs. The Intel® Core™ i7 processor 6700K with Intel® HD Graphics 530 instantiates such a design.*

### Figure 7: Dual Slice Product Design (48 EUs)

![Dual Slice Design]()

*Another potential product design composed of two slices, of three subslices each for a total of 48 EUs.*

### Figure 8: Triple Slice Product Design (72 EUs)

![Triple Slice Design]()

*A detailed diagram of the Intel® Processor Graphics Gen9 architecture showing three main slices, each labeled "Slice: 24 EUs". Each slice contains three subslices, and each subslice contains multiple Execution Units (EUs). Below the slices are three L3 Data Caches (one for each slice) and a central "GTI: Graphics Technology Interface" connecting them. Above the slices are "Rendering fixed function units". The diagram illustrates the flow from a "Command Streamer" and "Global Thread Dispatcher" at the top, down to the EUs and L3 Data Caches, and then through the GTI.*

**Product Configuration Summary:**

| Configuration | Slices | Subslices | Total EUs |
|--------------|--------|-----------|----------|
| Single Slice | 1 | 3 | 24 |
| Dual Slice | 2 | 6 | 48 |
| Triple Slice | 3 | 9 | 72 |

#### 5.6.1 Command Streamer and Global Thread Dispatcher

**Command Streamer:**
- Efficiently parses command streams submitted from driver stacks
- Routes individual commands to their representative units

**Global Thread Dispatcher:**
- Responsible for load balancing thread distribution across the entire device
- Works in concert with local thread dispatchers in each subslice

**Two Operating Modes:**

1. **Without barriers/SLM dependencies:** Distribute workload over all available subslices to maximize throughput and utilization with global load balancing

2. **With barriers/SLM dependencies:** Assign thread-group sized portions of the workload to specific subslices to ensure localized access to barrier logic and shared local memory

#### 5.6.2 Graphics Technology Interface (GTI)

The **graphics technology interface (GTI)** is the gateway between gen9 compute architecture with the rest of the SoC, including:

- Shared LLC memory
- System DRAM
- Possibly embedded DRAM
- CPU cores and other fixed function devices

**GTI Functions:**
- Facilitates communication with CPU cores
- Implements global memory atomics (shared between GPU and CPU cores)
- Implements power management controls
- Interfaces between GTI clock domain and other SoC clock domains

**Bus Bandwidth:**
- Slice to GTI: **64 bytes/cycle read, 64 bytes/cycle write**
- GTI to LLC (high-performance config): **64 bytes/cycle read, 64 bytes/cycle write**
- GTI to LLC (low-power config): **64 bytes/cycle read, 32 bytes/cycle write**

#### 5.6.3 Unslice

The command streamer, global thread dispatcher, and graphics technology interface all exist independent of the slice instantiations, in a domain typically called the **"unslice"**.

**New to gen9:** This domain is given its own power gating and clocking that can run at the same or faster than the slice clock. This can enable intelligent power savings by dynamically diverting more power to GTI's memory bandwidth, versus the EU slices's compute capability. This can be particularly effective for low power media playback modes.

#### 5.6.4 Product EU Counts

Although gen9 subslices generally contain 8 EUs each, complete gen9-based products can **disable an EU within a subslice** to optimize product yields from silicon manufacturing. For example, a three subslice-based product can have a total of **23 EUs** by disabling an EU in one subslice.

---

### 5.7 Memory

#### 5.7.1 Unified Memory Architecture

Intel processor graphics architecture has long pioneered sharing DRAM physical memory with the CPU. This **unified memory architecture** offers advantages over PCI Express-hosted discrete memory systems:

- **Shared physical memory** enables **zero copy** buffer transfers between CPUs and gen9 compute architecture
- Shared LLC cache augments performance of memory sharing
- Benefits: Performance, conserves memory footprint, conserves system power

Shared physical memory and zero copy buffer transfers are programmable through the buffer allocation mechanisms in APIs such as **OpenCL 1.0+** and **DirectX 11.2+**.

#### 5.7.2 Shared Memory Coherency

Gen9 compute architecture supports **global memory coherency** between Intel processor graphics and the CPU cores. SoC products with Intel processor graphics gen9 integrate new hardware components to support the recently updated **Intel® Virtualization Technology (Intel® VT)** for Directed I/O (Intel® VT-d) specification.

**Key Features:**
- New page table entry formats
- Cache protocols
- Hardware snooping mechanisms for shared memory
- Maintains memory coherency and consistency for fine grained sharing throughout the memory hierarchy
- Same virtual addresses can be shared seamlessly across devices

Such memory sharing is application-programmable through emerging heterogeneous compute APIs such as the **shared virtual memory (SVM)** features specified in **OpenCL 2.0**.

The net effect is that **pointer-rich data-structures can be shared directly** between application code running on CPU cores with application code running on Intel processor graphics, without programmer data structure marshalling or cumbersome software translation techniques.

### Figure 9: SoC Chip Level Memory Hierarchy

![Memory Hierarchy Diagram]()

*A detailed diagram illustrating the SoC chip level memory hierarchy and its theoretical peak bandwidths for the compute architecture of Intel processor graphics Gen9:*

**Intel® Processor Graphics Gen9 section:**
- **Each EU:** GRF/EU R: 96B/cyc, W: 32B/cyc
- **Each Subslice:** 8 EUs connected to Sampler with L1$ and L2$ (R: 64B/cyc each)
- **Shared Local Memory:** 64KB/subslice (R: 64B/cyc, W: 64B/cyc)
- **L3$:** 512KB/slice (R: 64B/cyc, W: 32B or 64B/cyc @Gen-clock) connected to GTI
- **GTI** connected to:
  - **EDRAM:** 64-128MB (On Package, some Iris™ products) - R: 32B/cycle, W: 32B/cycle @EDRAM-clock
  - **LLC$:** 2-8 MB (Shared With CPUs)
  - **System DRAM:** 2 channels, each 8B/cyc @mem-clock

**CPU section:**
- CPU core connected to CPU core L1$ and CPU core L2$
- Each CPU Core: R: 32B/cyc, W: 32B/cyc @ring clock

**SoC Ring Interconnect** connects the CPU and Intel Processor Graphics sections.

**Legend:**
- Light blue: Coherent
- Dark blue: Not coherent
- Grey: clock domain

> **Note:** The sampler's L1 and L2 caches as well as the shared local memory structures are **not coherent**.

---

### 5.8 Architecture Configurations, Speeds, and Feeds

The following table presents the theoretical peak throughput of the compute architecture of Intel processor graphics, aggregated across the entire graphics product architecture. Values are stated as "per cycle", as final product clock rates were not available at time of this writing.

| Metric | Intel® HD Graphics 530 | Derivation/Notes |
|--------|------------------------|------------------|
| Slices | 1 | - |
| Subslices | 3 | - |
| EUs | 24 | - |
| Threads per EU | 7 | - |
| Total Threads | 168 | 24 × 7 |
| 32-bit FP ops/cycle/EU | 16 | (add+mul) × 2 FPUs × SIMD-4 |
| Total 32-bit FP ops/cycle | 384 | 24 EUs × 16 ops |
| GRF per EU | 28 KB | 7 threads × 4KB |
| Total GRF | 672 KB | 24 EUs × 28KB |
| L3 Cache | 768 KB | (512KB for compute) |
| SLM per subslice | 64 KB | - |
| Total SLM | 192 KB | 3 subslices × 64KB |

---

## 6. Example Compute Applications

### Figure 10: CyberLink PhotoDirector® Face Swap

![Face Swap Image]()

*Before and after images demonstrating CyberLink's PhotoDirector® Face Swap feature, whose OpenCL™ implementation is optimized for Intel® processor graphics. Results are generated nearly real-time using compute with extremely minimal effort from the end-user. Facial recognition and face geometry are detected to crop, scale, rotate, color correct, and blend replacement faces in an instant.*

### Figure 11: CyberLink PhotoDirector® Clarify Effect

![Clarify Effect Image]()

*A before (left) and after (right) image generated using CyberLink's PhotoDirector® Clarify effect, whose OpenCL™ implementation is optimized for Intel® processor graphics. CPU and GPU work concurrently on the same image using SVM to efficiently share the image data. Shows a landscape with mountains, demonstrating enhanced clarity while minimizing visual noise and halos.*

### Figure 12: Interactive Real-time Volumetric 3D Smoke Effect

![Smoke Effect Image]()

*Interactive Real-time volumetric rendered 3D smoke effect implemented using DirectX® 11 Compute Shader on Intel® processor graphics. Shows a building with fire and volumetric smoke rendering.*

### Figure 13: Interactive Dynamic Relighting

![Dynamic Relighting Images]()

*Three images demonstrating interactive dynamic relighting of a real-time video feed:*
1. **Raw video frame & virtual lights** - Shows a plush monkey toy with two colored lights (red and green) visible in the background
2. **Computed surface normals** - Same monkey toy rendered with computed surface normals, highlighting contours and depth in green and red lines
3. **Video frame with relighting** - Monkey toy re-lit based on virtual light sources

*A localized surface polynomial approximation solver is implemented in OpenCL™ and applied to real-time depth captures. Derived surface normals and lighting computations can then synthetically re-light the scene based on virtual light sources.*

### Figure 14: Crowd Simulation Transition Effect

![Crowd Simulation Images]()

*A series of three images showing a crowd simulation-based transition effect between two photos. Particles carry colors of the source image (blue mug) and then change the colors while moving to form the destination image (wooden barrel). Dynamic particle collision detection and response are calculated with UNC's RVO2 library ported to OpenCL™ 2.0 and running on Intel® processor graphics.*

*Intel processor graphics and OpenCL 2.0's Shared Virtual Memory enable passing the original pointer-rich data structures of the RVO2 library directly to Intel processor graphics "as is". Neither data structure redesign nor fragile software data marshaling is necessary.*

### Figure 15: OpenCV® Examples

![OpenCV Examples]()

*Two images demonstrating OpenCV® functionalities:*
1. **Face Detection** - A man's face in a video feed with green bounding boxes around detected faces, accelerated via OpenCL™ and optimized for Intel® processor graphics
2. **Optical Flow** - An outdoor scene with people in blue outlines and blue arrows indicating optical flow vectors, using OpenCV's Luckas-Kanade implementation accelerated via OpenCL

*OpenCV 3.0 is now a feature of Intel® INDE with Intel-optimized, Windows® and Android™ pre-built binaries for academic and commercial use.*

---

## 7. Acknowledgements

Intel processor graphics architecture, products, supporting software, and optimized applications are the results of many years and the efforts of many engineers and architects, too many to list here. Also many reviewers contributed to this document. Thank you all.

A particular thank-you to:
- Jim Valerio
- Murali Sundaresan
- David Blythe
- Tom Piazza

...for their technical leadership and support in writing this document.

---

## 8. More Information

- [The Compute Architecture of Intel Processor Graphics Gen7.5]()
- [Intel® Iris™ Graphics Powers Built-in Beautiful]()
- [About Intel® Processor Graphics Technology]()
- [Open source Linux documentation of Gen Graphics and Compute Architecture]()
- [Intel® SDK for OpenCL]()
- [Optimizing Heterogeneous Computing for Intel® Processor Graphics, IDF 2014 Shenzhen]()
- [Intel® 64 and IA-32 Architectures Software Developers Manual]()
- [Intel® Virtualization Technology (Intel® VT) for Directed I/O (Intel® VT-d): Enhancing Intel platforms for efficient virtualization of I/O devices]()
- [Intel® Virtualization Technology for Directed I/O - Architecture Specification]()

---

## 9. Notices

Copyright © 2015 Intel Corporation. All rights reserved.

By using this document, in addition to any agreements you have with Intel, you accept the terms set forth below.

No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document.

Intel disclaims all express and implied warranties, including without limitation, the implied warranties of merchantability, fitness for a particular purpose, and non-infringement, as well as any warranty arising from course of performance, course of dealing, or usage in trade.

This document contains information on products, services and/or processes in development. All information provided here is subject to change without notice. Contact your Intel representative to obtain the latest forecast, schedule, specifications and roadmaps.

The products and services described may contain defects or errors known as errata which may cause deviations from published specifications. Current characterized errata are available on request.

Copies of documents which have an order number and are referenced in this document may be obtained by calling 1-800-548-4725 or by visiting www.intel.com/design/literature.htm.

Software and workloads used in performance tests may have been optimized for performance only on Intel microprocessors. Performance tests, such as SYSmark™ and MobileMark™, are measured using specific computer systems, components, software, operations and functions. Any change to any of those factors may cause the results to vary. You should consult other information and performance tests to assist you in fully evaluating your contemplated purchases, including the performance of that product when combined with other products. For more information go to http://www.intel.com/performance.

**Intel, the Intel logo, Iris™, Core™ are trademarks of Intel Corporation in the U.S. and/or other countries.**

*Other names and brands may be claimed as the property of others.*

*Intel® Graphics 4600, Iris™ Graphics, and Iris™ Pro Graphics are available on select systems. Consult your system manufacturer.*

---

**Document:** The Compute Architecture of Intel® Processor Graphics Gen9  
**Version:** 1.0  
**Pages:** 21  
**Release:** IDF-2015