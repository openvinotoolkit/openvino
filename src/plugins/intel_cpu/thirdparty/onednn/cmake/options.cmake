#===============================================================================
# Copyright 2018-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Manage different library options
#===============================================================================

if(options_cmake_included)
    return()
endif()
set(options_cmake_included true)

# ========
# Features
# ========

option(DNNL_VERBOSE
    "allows oneDNN be verbose whenever DNNL_VERBOSE
    environment variable set to 1" ON) # enabled by default

option(DNNL_ENABLE_CONCURRENT_EXEC
    "disables sharing a common scratchpad between primitives.
    This option must be turned ON if there is a possibility of executing
    distinct primitives concurrently.
    CAUTION: enabling this option increases memory consumption."
    OFF) # disabled by default

option(DNNL_ENABLE_PRIMITIVE_CACHE "enables primitive cache." ON)
    # enabled by default

option(DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE "If DNNL_ENABLE_PRIMITIVE_CACHE
    is ON enables using runtime objects in the primitive cache" ON)

option(DNNL_ENABLE_MAX_CPU_ISA
    "enables control of CPU ISA detected by oneDNN via DNNL_MAX_CPU_ISA
    environment variable and dnnl_set_max_cpu_isa() function" ON)

option(DNNL_ENABLE_CPU_ISA_HINTS
    "enables control of CPU ISA specific hints by oneDNN via DNNL_CPU_ISA_HINTS
    environment variable and dnnl_set_cpu_isa_hints() function" ON)

# =============================
# Building properties and scope
# =============================

set(DNNL_LIBRARY_TYPE "SHARED" CACHE STRING
    "specifies whether oneDNN library should be SHARED or STATIC")
option(DNNL_BUILD_EXAMPLES "builds examples"  ON)
option(DNNL_BUILD_TESTS "builds tests" ON)
option(DNNL_BUILD_FOR_CI
    "specifies whether oneDNN library will use special testing environment for
    internal testing processes"
    OFF)
option(DNNL_WERROR "treat warnings as errors" OFF)

set(DNNL_TEST_SET "CI" CACHE STRING
    "specifies testing targets coverage. Supports CI, CI_NO_CORR, NIGHTLY.

    When CI option is set, it enables a subset of test targets to run. When
    CI_NO_CORR option is set, it enables same coverage as for CI option, but
    switches off correctness validation for benchdnn targets. When NIGHTLY
    option is set, it enables a broader set of test targets to run.")

set(DNNL_INSTALL_MODE "DEFAULT" CACHE STRING
    "specifies installation mode; supports DEFAULT or BUNDLE.

    When BUNDLE option is set oneDNN will be installed as a bundle
    which contains examples and benchdnn.")

set(DNNL_CODE_COVERAGE "OFF" CACHE STRING
    "specifies which supported tool for code coverage will be used
    Currently only gcov supported")
if(NOT "${DNNL_CODE_COVERAGE}" MATCHES "^(OFF|GCOV)$")
    message(FATAL_ERROR "Unsupported code coverage tool: ${DNNL_CODE_COVERAGE}")
endif()

set(DNNL_DPCPP_HOST_COMPILER "DEFAULT" CACHE STRING
    "specifies host compiler for Intel oneAPI DPC++ Compiler")

set(DNNL_LIBRARY_NAME "dnnl" CACHE STRING
    "specifies name of the library. For example, user can use this variable to
     specify a custom library names for CPU and GPU configurations to safely
     include them into their CMake project via add_subdirectory")

message(STATUS "DNNL_LIBRARY_NAME: ${DNNL_LIBRARY_NAME}")

set(DNNL_ENABLE_WORKLOAD "TRAINING" CACHE STRING
    "Specifies a set of functionality to be available at build time. Designed to
    decrease the final memory disk footprint of the shared object or application
    statically linked against the library. Valid values:
    - TRAINING (the default). Includes all functionality to be enabled.
    - INFERENCE. Includes only forward propagation kind functionality and their
      dependencies.")
if(NOT "${DNNL_ENABLE_WORKLOAD}" MATCHES "^(TRAINING|INFERENCE)$")
    message(FATAL_ERROR "Unsupported propagation kind: ${DNNL_ENABLE_WORKLOAD}")
endif()

set(DNNL_ENABLE_PRIMITIVE "ALL" CACHE STRING
    "Specifies a set of primitives to be available at build time. Valid values:
    - ALL (the default). Includes all primitives to be enabled.
    - <PRIMITIVE_NAME>. Includes only the selected primitive to be enabled.
      Possible values are: BATCH_NORMALIZATION, BINARY, CONCAT, CONVOLUTION,
      DECONVOLUTION, ELTWISE, INNER_PRODUCT, LAYER_NORMALIZATION, LRN, MATMUL,
      POOLING, PRELU, REDUCTION, REORDER, RESAMPLING, RNN, SHUFFLE, SOFTMAX,
      SUM.
    - <PRIMITIVE_NAME>;<PRIMITIVE_NAME>;... Includes only selected primitives to
      be enabled at build time. This is treated as CMake string, thus, semicolon
      is a mandatory delimiter between names. This is the way to specify several
      primitives to be available in the final binary.")

# =============
# Optimizations
# =============

set(DNNL_ARCH_OPT_FLAGS "HostOpts" CACHE STRING
    "specifies compiler optimization flags (see below for more information).
    If empty default optimization level would be applied which depends on the
    compiler being used.

    - For Intel C++ Compilers the default option is `-xSSE4.1` which instructs
      the compiler to generate the code for the processors that support SSE4.1
      instructions. This option would not allow to run the library on older
      architectures.

    - For GNU* Compiler Collection and Clang, the default option is `-msse4.1` which
      behaves similarly to the description above.

    - For all other cases there are no special optimizations flags.

    If the library is to be built for generic architecture (e.g. built by a
    Linux distributive maintainer) one may want to specify DNNL_ARCH_OPT_FLAGS=\"\"
    to not use any host specific instructions")

# ======================
# Profiling capabilities
# ======================

# TODO: restore default to ON after the issue with linking C files by 
# Intel oneAPI DPC++ Compiler is fixed. Currently this compiler issues a warning
# when linking object files built from C and C++ sources.
option(DNNL_ENABLE_JIT_PROFILING
    "Enable registration of oneDNN kernels that are generated at
    runtime with VTune Amplifier (on by default). Without the
    registrations, VTune Amplifier would report data collected inside
    the kernels as `outside any known module`."
    ON)

option(DNNL_ENABLE_ITT_TASKS
    "Enable ITT Tasks tagging feature and tag all primitive execution 
    (on by default). VTune Amplifier can group profiling results based 
    on those ITT tasks and show corresponding timeline information."
    ON)

# ===================
# Engine capabilities
# ===================

set(DNNL_CPU_RUNTIME "OMP" CACHE STRING
    "specifies the threading runtime for CPU engines;
    supports OMP (default), TBB or DPCPP (DPC++ CPU engines).

    To use Threading Building Blocks (TBB) one should also
    set TBBROOT (either environment variable or CMake option) to the library
    location.")
if(NOT "${DNNL_CPU_RUNTIME}" MATCHES "^(NONE|OMP|TBB|TBB_AUTO|SEQ|THREADPOOL|DPCPP|SYCL)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${DNNL_CPU_RUNTIME}")
endif()

set(_DNNL_TEST_THREADPOOL_IMPL "STANDALONE" CACHE STRING
    "specifies which threadpool implementation to use when
    DNNL_CPU_RUNTIME=THREADPOOL is selected. Valid values: STANDALONE, EIGEN,
    TBB")
if(NOT "${_DNNL_TEST_THREADPOOL_IMPL}" MATCHES "^(STANDALONE|TBB|EIGEN)$")
    message(FATAL_ERROR
        "Unsupported threadpool implementation: ${_DNNL_TEST_THREADPOOL_IMPL}")
endif()

set(TBBROOT "" CACHE STRING
    "path to Thread Building Blocks (TBB).
    Use this option to specify TBB installation locaton.")

set(DNNL_GPU_RUNTIME "NONE" CACHE STRING
    "specifies the runtime to use for GPU engines.
    Can be NONE (default; no GPU engines), OCL (OpenCL GPU engines)
    or DPCPP (DPC++ GPU engines).

    Using OpenCL for GPU requires setting OPENCLROOT if the libraries are
    installed in a non-standard location.")
if(NOT "${DNNL_GPU_RUNTIME}" MATCHES "^(OCL|NONE|DPCPP|SYCL)$")
    message(FATAL_ERROR "Unsupported GPU runtime: ${DNNL_GPU_RUNTIME}")
endif()

set(DNNL_GPU_VENDOR "INTEL" CACHE STRING
    "specifies target GPU vendor for GPU engines.
    Can be INTEL (default) or NVIDIA.")
if(NOT "${DNNL_GPU_VENDOR}" MATCHES "^(INTEL|NVIDIA)$")
    message(FATAL_ERROR "Unsupported GPU vendor: ${DNNL_GPU_VENDOR}")
endif()

set(OPENCLROOT "" CACHE STRING
    "path to Intel SDK for OpenCL applications.
    Use this option to specify custom location for OpenCL.")

# TODO: move logic to other cmake files?
# Shortcuts for SYCL/DPC++
if(DNNL_CPU_RUNTIME STREQUAL "DPCPP" OR DNNL_CPU_RUNTIME STREQUAL "SYCL")
    set(DNNL_CPU_SYCL true)
else()
    set(DNNL_CPU_SYCL false)
endif()

if(DNNL_GPU_RUNTIME STREQUAL "DPCPP" OR DNNL_GPU_RUNTIME STREQUAL "SYCL")
    set(DNNL_GPU_SYCL true)
    set(DNNL_SYCL_CUDA OFF)
    if(DNNL_GPU_VENDOR STREQUAL "NVIDIA")
        set(DNNL_SYCL_CUDA ON)
    endif()
else()
    set(DNNL_GPU_SYCL false)
endif()

if(DNNL_CPU_SYCL OR DNNL_GPU_SYCL)
    set(DNNL_WITH_SYCL true)
else()
    set(DNNL_WITH_SYCL false)
endif()

# =============
# Miscellaneous
# =============

option(BENCHDNN_USE_RDPMC
    "enables rdpms counter to report precise cpu frequency in benchdnn.
    CAUTION: may not work on all cpus (hence disabled by default)"
    OFF) # disabled by default

# =========================
# Developer and debug flags
# =========================

set(DNNL_USE_CLANG_SANITIZER "" CACHE STRING
    "instructs build system to use a Clang sanitizer. Possible values:
    Address: enables AddressSanitizer
    Leak: enables LeakSanitizer
    Memory: enables MemorySanitizer
    MemoryWithOrigin: enables MemorySanitizer with origin tracking
    Thread: enables ThreadSanitizer
    Undefined: enables UndefinedBehaviourSanitizer
    This feature is experimental and is only available on Linux.")

option(DNNL_ENABLE_MEM_DEBUG "enables memory-related debug functionality,
    such as buffer overflow (default) and underflow, using gtests and benchdnn.
    Additionaly, this option enables testing of out-of-memory handling by the
    library, such as failed memory allocations, using primitive-related gtests.
    This feature is experimental and is only available on Linux." OFF)

set(DNNL_USE_CLANG_TIDY "NONE" CACHE STRING
    "Instructs build system to use clang-tidy. Valid values:
    - NONE (default)
      Clang-tidy is disabled.
    - CHECK
      Enables checks from .clang-tidy.
    - FIX
      Enables checks from .clang-tidy and fix found issues.
    This feature is only available on Linux.")

# =============================
# External BLAS library options
# =============================

set(DNNL_BLAS_VENDOR "NONE" CACHE STRING
    "Use an external BLAS library. Valid values:
      - NONE (default)
        Use in-house implementation.
      - MKL
        Intel Math Kernel Library (Intel MKL)
        (https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html)
      - OPENBLAS
        (https://www.openblas.net)
      - ARMPL
        Arm Performance Libraries
        (https://developer.arm.com/tools-and-software/server-and-hpc/downloads/arm-performance-libraries)
      - ANY
        FindBLAS will search default library paths for a known BLAS installation.")

# ==============================================
# AArch64 optimizations with Arm Compute Library
# ==============================================

option(DNNL_AARCH64_USE_ACL "Enables use of AArch64 optimised functions
    from Arm Compute Library.
    This is only supported on AArch64 builds and assumes there is a
    functioning Compute Library build available at the location specified by the
    environment variable ACL_ROOT_DIR." OFF)
