# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "MKLDNNPlugin")

if(CMAKE_COMPILER_IS_GNUCC)
    ie_add_compiler_flags(-Wno-all)
endif()

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

if (ENABLE_CPU_DEBUG_CAPS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_DEBUG_CAPS")
endif()

## TODO
set(LAYERS
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_bin_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_concat_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_convert_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_strided_slice_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_deconv_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_def_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_eltwise_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_fullyconnected_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_matmul_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_generic_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_input_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_lrn_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_memory_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_pad_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_transpose_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_pooling_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_fake_quantize_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reorder_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reshape_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_rnn.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_roi_align_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_roi_pooling_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_psroi_pooling_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_softmax_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_split_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_tensoriterator_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_tile_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_mvn_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_normalize_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_scatter_update_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_interpolate_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reduce_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reference_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_depth_to_space_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_space_to_depth_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_batch_to_space_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_space_to_batch_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_broadcast_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_embedding_bag_sum_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_embedding_bag_offset_sum_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_embedding_bag_packed_sum_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_embedding_segments_sum_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_gather_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_gather_elements_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_gather_nd_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_one_hot_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_region_yolo_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_select_node.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_roll_node.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/list.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/ctc_greedy_decoder.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/ctc_greedy_decoder_seq_len.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/ctc_loss.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/detectionoutput.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/detectionoutput_onnx.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/extract_image_patches.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/gather_tree.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/grn.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/non_max_suppression.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/log_softmax.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/math.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorgridgenerator_onnx.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal_onnx.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/range.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reorg_yolo.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reverse_sequence.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/roifeatureextractor_onnx.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/shuffle_channels.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/bucketize.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/topkrois_onnx.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/softmax.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/topk.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal_imp.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/nodes/cum_sum.cpp
)

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/rt_info/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/emitters/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_transformations/op/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/memory_desc/*.cpp
        ${LAYERS}
        ${OS_SPECIFIC_SRC}
)

# file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h
#                           ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

addVersionDefines(mkldnn_plugin.cpp CI_BUILD_NUMBER)

# create plugin

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CPU"
              SOURCES ${SOURCES} ${HEADERS})

set_ie_threading_interface_for(${TARGET_NAME})

ie_mark_target_as_cc(${TARGET_NAME})

if(SELECTIVE_BUILD STREQUAL "ON")
    # After disabling a block of code, some variables might be unused.
    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        target_compile_options(${TARGET_NAME} PRIVATE -Wno-unused-variable)
    endif()
endif()

target_link_libraries(${TARGET_NAME} PRIVATE mkldnn
                                             inference_engine
                                             inference_engine_transformations
                                             inference_engine_lp_transformations)

target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:mkldnn,INCLUDE_DIRECTORIES>)

# Cross compiled function
# TODO: The same for proposal, proposalONNX, topk
cross_compiled_file(${TARGET_NAME}
        ARCH AVX2 ANY
                    nodes/proposal_imp.cpp
        API         nodes/proposal_imp.hpp
        NAME        proposal_exec
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

#  add test object library

add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
link_system_libraries(${TARGET_NAME}_obj PUBLIC mkldnn)

target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_lp_transformations,INTERFACE_INCLUDE_DIRECTORIES>
                                              PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
                                                      $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>)
                                                
target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:mkldnn,INCLUDE_DIRECTORIES>)

set_ie_threading_interface_for(${TARGET_NAME}_obj)

target_compile_definitions(${TARGET_NAME}_obj
        PRIVATE USE_STATIC_IE IMPLEMENT_INFERENCE_ENGINE_PLUGIN
)

set_target_properties(${TARGET_NAME}_obj PROPERTIES EXCLUDE_FROM_ALL ON)

# LTO

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_obj
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
