# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "MKLDNNPlugin")

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

set(LAYERS
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_batchnorm_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_bin_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_concat_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_crop_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_deconv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_def_conv_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_eltwise_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_fullyconnected_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_gemm_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_generic_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_input_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_lrn_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_memory_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_pad_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_permute_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_pooling_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_quantize_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reorder_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reshape_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_rnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_roi_pooling_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_softmax_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_split_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_tensoriterator_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_tile_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_mvn_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_normalize_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_scatter_update_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_interpolate_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/mkldnn_reduce_node.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/list.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/batch_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/ctc_greedy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/ctc_loss.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/depth_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/detectionoutput.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/detectionoutput_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/embedding_bag_offset_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/embedding_bag_packed_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/embedding_bag_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/embedding_segments_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/extract_image_patches.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/fill.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/gather_nd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/gather_tree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/grn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/non_max_suppression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/log_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/math.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/one_hot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/powerfile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorbox_clustered.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/priorgridgenerator_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/psroi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/region_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reorg_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/reverse_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/roifeatureextractor_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/select.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/shuffle_channels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/simplernms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/space_to_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/space_to_depth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_fill_empty_rows.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_segment_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_weighted_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/sparse_to_dense.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/bucketize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/squeeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/strided_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/topkrois_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/unique.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/unsqueeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/emitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/jit_eltwise_emitters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/jit_mkldnn_emitters.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/argmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/argmax_imp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/topk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/proposal_imp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes/cum_sum.cpp
)

foreach(LAYER ${LAYERS})
    get_filename_component(LAYER_NAME ${LAYER} NAME_WE)
    string(TOUPPER ${LAYER_NAME} LAYER_UNAME)
    add_definitions(-DCOMPILED_CPU_${LAYER_UNAME})
endforeach()

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/common/*.cpp
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
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include)

include_directories(SYSTEM
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/common
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include)

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

# create plugin

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CPU"
              OBJECT_LIBRARIES ${object_libraries}
              SOURCES ${SOURCES} ${HEADERS})

set_ie_threading_interface_for(${TARGET_NAME})

target_compile_definitions(${TARGET_NAME} PUBLIC -DMKLDNN_THR=${MKLDNN_THR})

target_link_libraries(${TARGET_NAME} PRIVATE mkldnn inference_engine inference_engine_legacy
                                             inference_engine_transformations inference_engine_lp_transformations)

# Cross compiled function
# TODO: The same for proposal, proposalONNX, topk
cross_compiled_file(${TARGET_NAME}
        ARCH AVX512F AVX2 SSE42 ANY
                    nodes/argmax_imp.cpp
        API         nodes/argmax_imp.hpp
        NAME        arg_max_execute
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
)
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

target_include_directories(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:inference_engine_preproc_s,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine_lp_transformations,INTERFACE_INCLUDE_DIRECTORIES>)

set_ie_threading_interface_for(${TARGET_NAME}_obj)

target_compile_definitions(${TARGET_NAME}_obj PUBLIC -DMKLDNN_THR=${MKLDNN_THR}
                                              PRIVATE USE_STATIC_IE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

set_target_properties(${TARGET_NAME}_obj PROPERTIES EXCLUDE_FROM_ALL ON)

# LTO

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_obj
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# install

if(GEMM STREQUAL "MKL")
    install(DIRECTORY "${MKL}/include"
            DESTINATION ${IE_CPACK_IE_DIR}/external/mkltiny_lnx
            COMPONENT cpu)
    install(FILES "${MKLLIB}"
            DESTINATION ${IE_CPACK_IE_DIR}/external/mkltiny_lnx/lib
            COMPONENT cpu)
    install(FILES "${MKL}/version.info"
            DESTINATION ${IE_CPACK_IE_DIR}/external/mkltiny_lnx
            COMPONENT cpu)
endif()
