#===============================================================================
# Copyright 2019-2021 Intel Corporation
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

set(TEST_EXE test_api)

file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
list(APPEND TEST_SOURCES ${MAIN_SRC_GTEST})

if(DNNL_USE_CLANG_SANITIZER)
    # Due to the following tests trigger memory sanitizer
    # `allocation-size-too-big`, remove them.
    list(REMOVE_ITEM TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_memory.cpp)
endif()

# Switch off C API tests for CUDA since USM model is not supported
if(NOT DNNL_SYCL_CUDA)
    register_exe(${TEST_EXE} "${TEST_SOURCES}" "test" "dnnl_gtest")
endif()

# Create DPC++ buffer target.
if(DNNL_WITH_SYCL)
    register_exe(${TEST_EXE}_buffer "${TEST_SOURCES}" "test" "dnnl_gtest")
    target_compile_definitions(${TEST_EXE}_buffer PUBLIC -DTEST_DNNL_DPCPP_BUFFER)
    append_host_compiler_options(CMAKE_CXX_FLAGS "-DTEST_DNNL_DPCPP_BUFFER")
endif()
