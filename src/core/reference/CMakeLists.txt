# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_reference")

set(REF_IMPL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Create static library
add_library(${TARGET_NAME} STATIC)

target_sources(${TARGET_NAME}
    PRIVATE

    # Sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/convert_x86_intrinsics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/depth_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/einsum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/experimental_detectron_detection_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/experimental_detectron_proposal_single_image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/experimental_detectron_roi_feature_extractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/fake_convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/fft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/function.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/gather_tree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/generate_proposal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/group_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/group_convolution_backprop_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/if.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/interpolate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/inverse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/irdft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/istft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/loop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/matrix_nms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/multiclass_nms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/nms_rotated.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/non_max_suppression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/random_uniform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/rdft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/reorg_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/reverse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/shuffle_channels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/space_to_depth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/stft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/strided_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/tensor_iterator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/tile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/utils/fft_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/utils/nms_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/utils/paged_cache_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/utils/rounding_guard.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/coordinate_index.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/coordinate_range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/coordinate_transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/jit_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/philox_converter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/philox_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/registers_pool.cpp

    # Headers
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/abs.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/acos.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/acosh.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/adaptive_avg_pool.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/adaptive_max_pool.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/adaptive_rkv_diversity.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/add.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/and.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/asin.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/asinh.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/atan.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/atan2.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/atanh.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/autobroadcast_binop.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/avg_pool.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/batch_norm.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/binary_convolution.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/bitwise_and.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/bitwise_left_shift.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/bitwise_not.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/bitwise_or.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/bitwise_right_shift.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/bitwise_xor.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/broadcast.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/bucketize.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/ceiling.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/clamp.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/col2im.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/concat.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/constant.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/convert.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/convert_color_nv12.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/convolution.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/convolution_backprop_data.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/copy.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/cos.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/cosh.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/ctc_greedy_decoder.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/ctc_greedy_decoder_seq_len.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/ctc_loss.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/cum_sum.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/deformable_convolution.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/deformable_psroi_pooling.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/depth_to_space.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/detection_output.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/divide.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/einsum.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/elu.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/embedding_bag_offsets.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/embedding_bag_packed.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/embedding_segments_sum.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/equal.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/erf.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/erfinv.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/exp.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/experimental_detectron_detection_output.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/experimental_detectron_prior_grid_generator.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/experimental_detectron_proposal_single_image.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/experimental_detectron_roi_feature_extractor.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/experimental_detectron_topk_rois.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/extract_image_patches.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/eye.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/fake_convert.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/fake_quantize.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/fft.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/floor.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/floor_mod.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/function.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/gated_delta_net.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/gather.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/gather_elements.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/gather_nd.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/gather_tree.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/gelu.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/generate_proposal.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/greater.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/greater_eq.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/grid_sample.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/grn.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/group_convolution.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/group_convolution_backprop_data.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/group_normalization.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/grouped_matmul.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/gru_cell.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/hard_sigmoid.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/helpers.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/hsigmoid.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/hswish.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/identity.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/if.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/interpolate.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/interpolate_pil.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/inverse.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/irdft.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/is_finite.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/is_inf.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/is_nan.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/istft.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/less.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/less_eq.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/log.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/log_softmax.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/logical_not.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/logical_reduction.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/loop.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/lrn.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/lstm_cell.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/matmul.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/matrix_nms.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/max_pool.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/maximum.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/minimum.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/mish.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/mod.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/multiclass_nms.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/multinomial.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/multiply.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/mvn.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/negate.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/nms_rotated.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/nms_rotated_util.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/non_max_suppression.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/non_zero.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/normalize_l2.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/not_equal.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/one_hot.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/or.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/pad.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/paged_attention.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/power.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/prelu.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/prior_box.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/prior_box_clustered.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/proposal.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/psroi_pooling.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/random_uniform.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/range.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/rdft.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reduce_l1.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reduce_l2.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reduce_max.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reduce_mean.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reduce_min.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reduce_prod.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reduce_sum.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/region_yolo.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/relu.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reorg_yolo.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reshape.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/result.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reverse.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/reverse_sequence.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/rms_norm.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/rnn_cell.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/roi_align.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/roi_pooling.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/roll.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/round.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/rounding_guard.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/scaled_dot_product_attention.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/scatter_elements_update.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/scatter_nd_update.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/scatter_update.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/search_sorted.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/segment_max.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/select.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/selu.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/sequences.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/shape_of.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/shuffle_channels.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/sigmoid.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/sign.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/sin.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/sinh.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/slice.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/slice_scatter.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/softmax.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/softplus.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/softsign.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/space_to_depth.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/sparse_fill_empty_rows.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/split.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/sqrt.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/squared_difference.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/stft.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/strided_slice.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/string_tensor_pack.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/string_tensor_unpack.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/subtract.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/swish.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/tan.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/tanh.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/tensor_iterator.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/tile.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/topk.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/transpose.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/unique.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/convert_util.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/convert_x86_intrinsics.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/coordinate_index.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/coordinate_range.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/coordinate_transform.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/fft_common.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/jit_generator.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/nms_common.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/paged_cache_manager.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/paged_cache_manager_helper.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/philox_converter.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/philox_generator.hpp
    ${REF_IMPL_INCLUDE_DIR}/openvino/reference/utils/registers_pool.hpp
)

add_library(openvino::reference ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME reference)

if(ENABLE_AVX2)
    ov_avx2_optimization_flags(avx2_flags)

    set(OV_REFERENCE_X86_AVX2_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/op/convert_x86_intrinsics.cpp)
    set_source_files_properties(${OV_REFERENCE_X86_AVX2_SRC} PROPERTIES COMPILE_OPTIONS "${avx2_flags}"
                                                                        SKIP_UNITY_BUILD_INCLUSION ON
                                                                        SKIP_PRECOMPILE_HEADERS ON)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_AVX2)
endif()

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH_HEADER "src/precomp.hpp")

ov_set_threading_interface_for(${TARGET_NAME})

target_compile_definitions(${TARGET_NAME} PRIVATE XBYAK_NO_OP_NAMES XBYAK64)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${REF_IMPL_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${OV_CORE_DEV_API_PATH}>
    $<BUILD_INTERFACE:${OV_CORE_INCLUDE_PATH}>)

find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads openvino::core::dev)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET openvino::reference
                                    INSTALL_INCLUDE_DIRECTORIES "${REF_IMPL_INCLUDE_DIR}/")
