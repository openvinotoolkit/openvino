#===============================================================================
# Copyright 2016-2021 Intel Corporation
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

# XXX: Intel(R) oneAPI DPC++ Compiler doesn't support SEH with -fsycl option
if(WIN32 AND DNNL_WITH_SYCL)
    add_definitions(-DGTEST_HAS_SEH=0)
endif()

add_subdirectory (gtest)

set(APP_NAME "gtest")
set(MAIN_SRC_GTEST ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
list(APPEND MAIN_SRC_GTEST ${TEST_THREAD})

# Windows does not support weak/strong symbols and no guarrantees by the linker
# for out_of_memory testing to work. Not tested on macOS
if(UNIX)
    if(DNNL_ENABLE_MEM_DEBUG)
        add_definitions_with_host_compiler(-DDNNL_ENABLE_MEM_DEBUG)
        list(APPEND MAIN_SRC_GTEST ${CMAKE_CURRENT_SOURCE_DIR}/test_malloc.cpp)
    endif()
endif()

include_directories_with_host_compiler(${CMAKE_CURRENT_SOURCE_DIR}
                                       ${CMAKE_CURRENT_SOURCE_DIR}/gtest
                                       ${CMAKE_CURRENT_SOURCE_DIR}/in
                                       ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common
                                       ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpu
)

if(WIN32)
    # Correct 'jnl' macro/jit issue
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" AND ${CMAKE_CXX_COMPILER_VERSION} LESS 19.1)
        append(CMAKE_CXX_FLAGS "/Qlong-double")
    endif()
endif()

# TODO: enable me!
file(GLOB PRIM_TEST_CASES_SRC
                              test_comparison_operators.cpp
                              test_primitive_cache_mt.cpp
                              test_iface_primitive_cache.cpp
                              test_iface_pd.cpp
                              test_iface_pd_iter.cpp
                              test_iface_attr.cpp
                              test_iface_binary_bcast.cpp
                              test_iface_handle.cpp
                              test_iface_runtime_dims.cpp
                              test_iface_runtime_attr.cpp
                              test_iface_weights_format.cpp
                              test_iface_wino_convolution.cpp
                              test_dnnl_threading.cpp
                              test_memory.cpp
                              test_sum.cpp
                              test_reorder.cpp
                              test_cross_engine_reorder.cpp
                              test_concat.cpp
                              test_softmax.cpp
                              test_eltwise.cpp
                              test_lrn_forward.cpp
                              test_lrn_backward.cpp
                              test_pooling_forward.cpp
                              test_pooling_backward.cpp
                              test_pooling_v2_forward.cpp
                              test_pooling_v2_backward.cpp
                              test_batch_normalization_f32.cpp
                              test_batch_normalization_s8.cpp
                              test_inner_product_forward.cpp
                              test_inner_product_backward_data.cpp
                              test_inner_product_backward_weights.cpp
                              test_shuffle.cpp
                              test_rnn_forward.cpp
                              test_convolution_forward_f32.cpp
                              test_convolution_forward_u8s8s32.cpp
                              test_convolution_forward_u8s8fp.cpp
                              test_convolution_eltwise_forward_f32.cpp
                              test_convolution_eltwise_forward_x8s8f32s32.cpp
                              test_convolution_backward_data_f32.cpp
                              test_convolution_backward_weights_f32.cpp
                              test_deconvolution.cpp
                              test_layer_normalization.cpp
                              test_binary.cpp
                              test_logsoftmax.cpp
                              test_matmul.cpp
                              test_resampling.cpp
                              test_reduction.cpp
                              )

if(DNNL_CPU_RUNTIME STREQUAL "NONE")
    list(APPEND PRIM_TEST_CASES_SRC test_iface_gpu_only.cpp)
    set_source_files_properties(test_iface_gpu_only.cpp PROPERTIES NO_ENGINE_PARAM true)
endif()

if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    file(GLOB CPU_SPECIFIC_TESTS
        test_gemm_f16.cpp
        test_gemm_f32.cpp
        test_gemm_f16f16f32.cpp
        test_gemm_bf16bf16f32.cpp
        test_gemm_bf16bf16bf16.cpp
        test_gemm_u8s8s32.cpp
        test_gemm_s8s8s32.cpp
        test_gemm_s8u8s32.cpp
        test_gemm_u8u8s32.cpp
        test_convolution_format_any.cpp
        test_global_scratchpad.cpp
        )
    foreach(TEST_FILE ${CPU_SPECIFIC_TESTS})
        list(APPEND PRIM_TEST_CASES_SRC "${TEST_FILE}")
    endforeach()
endif()

if(NOT DNNL_USE_CLANG_SANITIZER)
    # Due to the following tests have long run-time, move them to Nightly set
    if(DNNL_TEST_SET GREATER DNNL_TEST_SET_CI)
        file(GLOB LARGE_PRIM_TEST_CASES_SRC
            test_ip_formats.cpp
            test_reorder_formats.cpp
            )
        foreach(TEST_FILE ${LARGE_PRIM_TEST_CASES_SRC})
            list(APPEND PRIM_TEST_CASES_SRC "${TEST_FILE}")
        endforeach()
    endif()
endif()


# Add X64-specific tests
if(DNNL_TARGET_ARCH STREQUAL "X64" AND NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    file(GLOB X64_PRIM_TEST_CASES_SRC
        test_isa_mask.cpp
        test_isa_hints.cpp
        test_isa_iface.cpp
        test_brgemm.cpp
        )
    foreach(TEST_FILE ${X64_PRIM_TEST_CASES_SRC})
        list(APPEND PRIM_TEST_CASES_SRC "${TEST_FILE}")
    endforeach()
endif()

# Workaround for an Intel compiler bug: stack unwinding does not restore
# some of the nonvolatile registers with /O2 optimization
if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set_source_files_properties(
            test_convolution_eltwise_forward_x8s8f32s32.cpp
            test_convolution_backward_data_f32.cpp
            test_convolution_backward_weights_f32.cpp
            PROPERTIES COMPILE_FLAGS "/O1")
endif()

# Higher optimization levels cause access violation in _CxxThrowException.
if(WIN32 AND DNNL_WITH_SYCL)
    set_source_files_properties(
            test_iface_runtime_dims.cpp
            PROPERTIES COMPILE_FLAGS "-O1")
endif()

# Tests that don't support '--engine' parameter
set_source_files_properties(
            test_cross_engine_reorder.cpp
            test_comparison_operators.cpp
            PROPERTIES NO_ENGINE_PARAM true)

function(register_gtest exe src)
    add_executable(${exe} ${MAIN_SRC_GTEST} ${src})
    add_definitions_with_host_compiler(-DNOMINMAX) # to allow std::max on Windows with parentheses
    target_link_libraries(${exe} ${DNNL_LIBRARY_NAME} dnnl_gtest ${EXTRA_SHARED_LIBS})

    get_source_file_property(no_engine_param ${src} NO_ENGINE_PARAM)

    set(DPCPP_HOST_COMPILER_FLAGS)
    # Extract host compiler flags from CMAKE_CXX_FLAGS to DPCPP_HOST_COMPILER_FLAGS
    string(REGEX MATCH "-fsycl-host-compiler-options=\".*\""  DPCPP_HOST_COMPILER_FLAGS ${CMAKE_CXX_FLAGS})
    # Erase host compiler flags from CMAKE_CXX_FLAGS
    string(REGEX REPLACE "-fsycl-host-compiler-options=\".*\""  "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

    if(${exe} MATCHES "_buffer")
        if (DNNL_WITH_SYCL)
            target_compile_definitions(${exe} PUBLIC -DTEST_DNNL_DPCPP_BUFFER)
        endif()
        # Add a flag to build gtest with buffer support
        append_host_compiler_options(DPCPP_HOST_COMPILER_FLAGS "-DTEST_DNNL_DPCPP_BUFFER")
    else()
        if(DNNL_GPU_RUNTIME STREQUAL "OCL")
            target_compile_definitions(${exe} PUBLIC -DTEST_DNNL_OCL_USM)
        endif()
    endif()

    if (NOT no_engine_param)
        target_compile_definitions(${exe} PUBLIC -DDNNL_TEST_WITH_ENGINE_PARAM)
        # Add a flag to enable gtest engine parameter
        append_host_compiler_options(DPCPP_HOST_COMPILER_FLAGS "-DDNNL_TEST_WITH_ENGINE_PARAM")
    endif()

    # Set flags for each target separately
    set_target_properties(${exe} PROPERTIES COMPILE_FLAGS "${DPCPP_HOST_COMPILER_FLAGS}")

    if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE" AND NOT no_engine_param AND NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
        if(NOT ${exe} MATCHES "_buffer" OR (${exe} MATCHES "_buffer" AND DNNL_CPU_SYCL))
            add_dnnl_test(${exe}_cpu ${exe} COMMAND ${exe} --engine=cpu)
            maybe_configure_windows_test(${exe}_cpu TEST)
        endif()

        add_dnnl_test(${exe}_gpu ${exe} COMMAND ${exe} --engine=gpu)
        maybe_configure_windows_test(${exe}_gpu TEST)
    elseif(DNNL_CPU_RUNTIME STREQUAL "NONE" AND NOT no_engine_param)
        add_dnnl_test(${exe}_gpu ${exe} COMMAND ${exe} --engine=gpu)
        maybe_configure_windows_test(${exe} TEST)
    else()
        add_dnnl_test(${exe} ${exe})
        maybe_configure_windows_test(${exe} TEST)
    endif()
endfunction()

set(skip_usm_pattern "^$") # no skip by default
if(DNNL_WITH_SYCL)
    # - cross_engine_reorder creates usm pointers on CPU and GPU that belong to
    #   the different context, which causes synchronization issues.
    #   FIXME: the library should handle this case gracefully.
    set(skip_usm_pattern "(test_cross_engine_reorder)")
endif()

foreach(TEST_FILE ${PRIM_TEST_CASES_SRC})
    get_filename_component(exe ${TEST_FILE} NAME_WE)
    if(NOT ${exe} MATCHES "${skip_usm_pattern}" AND NOT DNNL_SYCL_CUDA)
        register_gtest(${exe} ${TEST_FILE})
    endif()

    # Create additional buffer targets for SYCL and OCL.
    if(DNNL_WITH_SYCL OR DNNL_GPU_RUNTIME STREQUAL "OCL")
        register_gtest(${exe}_buffer ${TEST_FILE})
    endif()
endforeach()

add_subdirectory(api)
add_subdirectory(internals)

if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    add_subdirectory(regression)
endif()

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    add_subdirectory(ocl)
endif()

if(DNNL_WITH_SYCL)
    add_subdirectory(sycl)
endif()

# vim: et ts=4 sw=4
