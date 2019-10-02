# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "MKLDNNPlugin")

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

if (LINUX)
    set(OS_SPECIFIC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/os/lin/*.cpp)
endif()

set(CROSS_COMPILED_LAYERS
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/argmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/interp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mvn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/resample.cpp
    )

set(LAYERS
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_activation_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_batchnorm_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_bin_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_concat_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_crop_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_deconv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_def_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_depthwise_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_eltwise_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_fullyconnected_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_gemm_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_generic_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_input_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_lrn_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_memory_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_permute_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_pooling_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_power_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_quantize_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reorder_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reshape_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_rnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_roi_pooling_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_softmax_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_split_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_tensoriterator_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_tile_node.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/ctc_greedy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/depth_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/detectionoutput.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/detectionoutput_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/fill.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/gather_tree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/grn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/non_max_suppression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/scatter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/log_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/math.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/one_hot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/powerfile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorbox_clustered.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorgridgenerator_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/psroi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/region_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reorg_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reverse_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/roifeatureextractor_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/select.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/shuffle_channels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/simplernms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/space_to_depth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_fill_empty_rows.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_segment_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/squeeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/strided_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/topk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/topkrois_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/unique.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/unsqueeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/normalize.cpp
)

foreach(LAYER ${LAYERS})
    get_filename_component(LAYER_NAME ${LAYER} NAME_WE)
    string(TOUPPER ${LAYER_NAME} LAYER_UNAME)
    add_definitions(-DCOMPILED_CPU_${LAYER_UNAME})
endforeach()

foreach(LAYER ${CROSS_COMPILED_LAYERS})
    get_filename_component(LAYER_NAME ${LAYER} NAME_WE)
    string(TOUPPER ${LAYER_NAME} LAYER_UNAME)
    add_definitions(-DCOMPILED_CPU_${LAYER_UNAME})
endforeach()

file(GLOB CROSS_COMPILED_SOURCES
        ${CROSS_COMPILED_LAYERS})

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
        ${LAYERS}
        ${OS_SPECIFIC_SRC}
)

file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/*.hpp
)

addVersionDefines(mkldnn_plugin.cpp CI_BUILD_NUMBER MKL_VERSION)

include_directories(
        ${IE_MAIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/common
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include
        ${CMAKE_BINARY_DIR}/include/
)

if (GEMM STREQUAL "MKL")
    log_rpath_from_dir(MKL "${MKL}/lib")
endif()

if (THREADING STREQUAL "TBB")
    set(MKLDNN_THR MKLDNN_THR_TBB)
elseif (THREADING STREQUAL "TBB_AUTO")
    set(MKLDNN_THR MKLDNN_THR_TBB_AUTO)
elseif (THREADING STREQUAL "OMP")
    set(MKLDNN_THR MKLDNN_THR_OMP)
else()
    set(MKLDNN_THR MKLDNN_THR_SEQ)
endif()

# Define CPU capabilities 

include(cmake/CPUID.cmake)

if (ENABLE_AVX512F)
    if ((CMAKE_CXX_COMPILER_ID MATCHES MSVC) AND (MSVC_VERSION VERSION_LESS 1920))
        # 1920 version of MSVC 2019. In MSVC 2017 AVX512F not work
        set(ENABLE_AVX512F OFF)
    endif()
    if (CMAKE_CXX_COMPILER_ID STREQUAL Clang)
        set(ENABLE_AVX512F OFF)
    endif()
    if ((CMAKE_CXX_COMPILER_ID STREQUAL GNU) AND (NOT (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)))
        set(ENABLE_AVX512F OFF)
    endif()
endif()

# Not optimized layers

add_library(mkldnn_plugin_layers_no_opt OBJECT ${CROSS_COMPILED_SOURCES})
set_ie_threading_interface_for(mkldnn_plugin_layers_no_opt)

add_library(mkldnn_plugin_layers_no_opt_s OBJECT ${CROSS_COMPILED_SOURCES})
set_ie_threading_interface_for(mkldnn_plugin_layers_no_opt_s)
target_compile_definitions(mkldnn_plugin_layers_no_opt_s PRIVATE USE_STATIC_IE)

set(object_libraries mkldnn_plugin_layers_no_opt)
set(mkldnn_plugin_object_libraries mkldnn_plugin_layers_no_opt_s)

# SSE 4.2 optimized layers

if (ENABLE_SSE42)
    function(mkldnn_create_sse42_layers LIBRARY_NAME)
        add_library(${LIBRARY_NAME} OBJECT ${CROSS_COMPILED_SOURCES})

        target_compile_definitions(${LIBRARY_NAME} PRIVATE "HAVE_SSE")
        if(LIBRARY_NAME MATCHES "mkldnn_plugin_layers_.+_s$")
            target_compile_definitions(${LIBRARY_NAME} PRIVATE "USE_IE_STATIC")
        endif()
        set_ie_threading_interface_for(${LIBRARY_NAME})

        if(WIN32)
            if(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
                target_compile_options(${LIBRARY_NAME} PRIVATE "/arch:SSE4.2")
                target_compile_options(${LIBRARY_NAME} PRIVATE "/QxSSE4.2")
            endif()
            if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
                target_compile_options(${LIBRARY_NAME} PRIVATE "/arch:SSE4.2")
            endif()
        else()
            if(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
                target_compile_options(${LIBRARY_NAME} PRIVATE "-msse4.2")
                target_compile_options(${LIBRARY_NAME} PRIVATE "-xSSE4.2")
            else()
                target_compile_options(${LIBRARY_NAME} PRIVATE "-msse4.2")
            endif()
        endif()
    endfunction()

    mkldnn_create_sse42_layers(mkldnn_plugin_layers_sse42 OBJECT)
    mkldnn_create_sse42_layers(mkldnn_plugin_layers_sse42_s STATIC)

    list(APPEND object_libraries mkldnn_plugin_layers_sse42)
    list(APPEND mkldnn_plugin_object_libraries mkldnn_plugin_layers_sse42_s)
endif()

# AVX2 optimized layers

if (ENABLE_AVX2)
    function(mkldnn_create_avx2_layers LIBRARY_NAME)
        add_library(${LIBRARY_NAME} OBJECT ${CROSS_COMPILED_SOURCES})

        target_compile_definitions(${LIBRARY_NAME} PRIVATE "HAVE_SSE;HAVE_AVX2")
        if(LIBRARY_NAME MATCHES "mkldnn_plugin_layers_.+_s$")
            target_compile_definitions(${LIBRARY_NAME} PRIVATE "USE_IE_STATIC")
        endif()
        set_ie_threading_interface_for(${LIBRARY_NAME})

        if(WIN32)
            if(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
                target_compile_options(${LIBRARY_NAME} PRIVATE "/QxCORE-AVX2")
            endif()
            if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
                target_compile_options(${LIBRARY_NAME} PRIVATE "/arch:AVX2")
            endif()
        else()
            if(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
                target_compile_options(${LIBRARY_NAME} PRIVATE "-march=core-avx2")
                target_compile_options(${LIBRARY_NAME} PRIVATE "-xCORE-AVX2")
                target_compile_options(${LIBRARY_NAME} PRIVATE "-mtune=core-avx2")
            else()
                target_compile_options(${LIBRARY_NAME} PRIVATE "-mavx2")
                target_compile_options(${LIBRARY_NAME} PRIVATE "-mfma")
            endif()
        endif()
    endfunction()

    mkldnn_create_avx2_layers(mkldnn_plugin_layers_avx2 OBJECT)
    mkldnn_create_avx2_layers(mkldnn_plugin_layers_avx2_s STATIC)

    list(APPEND object_libraries mkldnn_plugin_layers_avx2)
    list(APPEND mkldnn_plugin_object_libraries mkldnn_plugin_layers_avx2_s)
endif()

# AVX512F optimized layers

if (ENABLE_AVX512F)
    function(mkldnn_create_avx512f_layers LIBRARY_NAME)
        add_library(${LIBRARY_NAME} OBJECT ${CROSS_COMPILED_SOURCES})

        target_compile_definitions(${LIBRARY_NAME} PRIVATE "HAVE_SSE;HAVE_AVX2;HAVE_AVX512F")
        if(LIBRARY_NAME MATCHES "mkldnn_plugin_layers_.+_s$")
            target_compile_definitions(${LIBRARY_NAME} PRIVATE "USE_IE_STATIC")
        endif()
        set_ie_threading_interface_for(${LIBRARY_NAME})

        if(WIN32)
            if(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
                target_compile_options(${LIBRARY_NAME} PRIVATE "/QxCOMMON-AVX512")
            endif()
            if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
                # TODO
            endif()
        else()
            if(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
                target_compile_options(${LIBRARY_NAME} PRIVATE "-xCOMMON-AVX512")
            endif()
            if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
                target_compile_options(${LIBRARY_NAME} PRIVATE "-mavx512f")
                target_compile_options(${LIBRARY_NAME} PRIVATE "-mfma")
            endif()
        endif()
    endfunction()

    mkldnn_create_avx512f_layers(mkldnn_plugin_layers_avx512 OBJECT)
    mkldnn_create_avx512f_layers(mkldnn_plugin_layers_avx512_s STATIC)

    list(APPEND object_libraries mkldnn_plugin_layers_avx512)
    list(APPEND mkldnn_plugin_object_libraries mkldnn_plugin_layers_avx512_s)
endif()

# create plugin

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CPU"
              OBJECT_LIBRARIES ${object_libraries}
              SOURCES ${SOURCES} ${HEADERS})

set_ie_threading_interface_for(${TARGET_NAME})

target_compile_definitions(${TARGET_NAME} PUBLIC -DMKLDNN_THR=${MKLDNN_THR})
target_link_libraries(${TARGET_NAME} PRIVATE inference_engine ${INTEL_ITT_LIBS} mkldnn)

#  add test object library

add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})

target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>)

set_ie_threading_interface_for(${TARGET_NAME}_obj)

target_compile_definitions(${TARGET_NAME}_obj PUBLIC -DMKLDNN_THR=${MKLDNN_THR}
                                              PRIVATE USE_STATIC_IE)

set(mkldnn_plugin_object_libraries "${mkldnn_plugin_object_libraries};${TARGET_NAME}_obj" CACHE INTERNAL "" FORCE)
