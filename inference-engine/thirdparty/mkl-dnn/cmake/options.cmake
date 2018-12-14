#===============================================================================
# Copyright 2018 Intel Corporation
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

option(MKLDNN_VERBOSE
    "allows Intel(R) MKL-DNN be verbose whenever MKLDNN_VERBOSE
    environment variable set to 1" ON) # enabled by default

option(MKLDNN_ENABLE_CONCURRENT_EXEC
    "disables sharing a common scratchpad between primitives.
    This option must be turned on if there is a possibility of concurrent
    execution of primitives that were created in the same thread.
    CAUTION: enabling this option increases memory consumption"
    OFF) # disabled by default

# =============================
# Building properties and scope
# =============================

set(MKLDNN_LIBRARY_TYPE "SHARED" CACHE STRING
    "specifies whether Intel(R) MKL-DNN library should be SHARED or STATIC")
option(WITH_EXAMPLE "builds examples"  ON)
option(WITH_TEST "builds tests" ON)

set(MKLDNN_THREADING "OMP" CACHE STRING
    "specifies threading type; supports OMP (default), or TBB.
    If Intel(R) Threading Building Blocks (Intel(R) TBB) one should also
    set TBBROOT (either environement variable or CMake option) to the library
    location")

# =============
# Optimizations
# =============

set(ARCH_OPT_FLAGS "HostOpts" CACHE STRING
    "specifies compiler optimization flags (see below for more information).
    If empty default optimization level would be applied which depends on the
    compiler being used.

    - For Intel(R) C++ Compilers the default option is `-xHOST` which instructs
      the compiler to generate the code for the architecture where building is
      happening. This option would not allow to run the library on older
      architectures.

    - For GNU* Compiler Collection version 5 and newer the default options are
      `-march=native -mtune=native` which behaves similarly to the descriprion
      above.

    - For all other cases there are no special optimizations flags.

    If the library is to be built for generic architecture (e.g. built by a
    Linux distributive maintainer) one may want to specify ARCH_OPT_FLAGS=\"\"
    to not use any host specific instructions")

# ======================
# Profiling capabilities
# ======================

set(VTUNEROOT "" CACHE STRING
    "path to Intel(R) VTune(tm) Amplifier.
    Required to register Intel(R) MKL-DNN kernels that are generated at
    runtime, otherwise the profile would not be able to track the kernels and
    would report `outside any known module`.")

# =============
# Miscellaneous
# =============

option(BENCHDNN_USE_RDPMC
    "enables rdpms counter to report precise cpu frequency in benchdnn.
     CAUTION: may not work on all cpus (hence disabled by default)"
    OFF) # disabled by default
