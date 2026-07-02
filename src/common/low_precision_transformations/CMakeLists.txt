# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "openvino_lp_transformations")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(LIBRARY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/align_quantization_intervals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/align_quantization_parameters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/assign_and_read_value.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/avg_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/base_matcher_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/batch_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clamp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cleanup_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/convert_subtract_constant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/convolution_backprop_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/depth_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/eliminate_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/eltwise_base_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fake_quantize_decomposition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fake_quantize_dequantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fold_convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fold_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fuse_convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fuse_elementwise_to_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fuse_multiply_to_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fuse_subtract_to_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/group_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/interpolate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kv_cache_concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/layer_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/low_precision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/markup_avg_pool_precision_preserved.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/markup_bias.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/markup_can_be_quantized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/markup_precisions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/markup_quantization_granularity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/max_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/move_fake_convert_up_through_kv_cache_concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/move_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multiply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multiply_partial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multiply_to_group_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mvn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network_helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/normalize_l2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/prelu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/propagate_precisions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pull_reshape_through_dequantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pull_transpose_through_dequantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qdq_stripping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization_details.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/recurrent_cell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reduce_base_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reduce_max.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reduce_mean.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reduce_min.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reduce_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/relu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resolve_precision_attribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt_info/avg_pool_precision_preserved_attribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt_info/bias_attribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt_info/intervals_alignment_attribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt_info/precision_preserved_attribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt_info/precisions_attribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt_info/quantization_alignment_attribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt_info/quantization_granularity_attribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt_info/quantization_mode_attribute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shuffle_channels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/space_to_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/squeeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/strided_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transparent_base_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/unsqueeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/variadic_split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/weightable_layer_transformation.cpp
)

set(PUBLIC_HEADERS
    ${PUBLIC_HEADERS_DIR}/low_precision/add.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/align_quantization_intervals.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/align_quantization_parameters.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/assign_and_read_value.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/avg_pool.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/base_matcher_pass.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/batch_to_space.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/broadcast.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/clamp.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/cleanup_transformation.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/common/fake_quantize_dequantization.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/common/ie_lpt_exception.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/common/port_quantization_granularity_restriction.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/common/precisions_restriction.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/common/quantization_granularity_restriction.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/concat.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/convert_subtract_constant.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/convolution_backprop_data.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/convolution.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/create_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/create_precisions_dependent_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/depth_to_space.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/eliminate_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/eltwise_base_transformation.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/fake_quantize_decomposition.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/fold_convert.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/fold_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/fuse_convert.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/fuse_elementwise_to_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/fuse_multiply_to_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/fuse_subtract_to_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/gather.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/group_convolution.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/interpolate.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/kv_cache_concat.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/layer_transformation.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/low_precision.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/lpt_itt.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/lpt_visibility.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/markup_avg_pool_precision_preserved.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/markup_bias.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/markup_can_be_quantized.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/markup_precisions.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/markup_quantization_granularity.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/mat_mul.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/max_pool.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/move_fake_convert_up_through_kv_cache_concat.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/move_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/multiply.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/multiply_partial.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/multiply_to_group_convolution.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/mvn.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/network_helper.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/normalize_l2.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/pad.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/prelu.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/propagate_precisions.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/propagate_shared_value.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/propagate_through_precision_preserved.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/propagate_to_input.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/pull_reshape_through_dequantization.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/pull_transpose_through_dequantization.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/qdq_stripping.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/quantization_details.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/recurrent_cell.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/reduce_base_transformation.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/reduce_max.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/reduce_mean.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/reduce_min.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/reduce_sum.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/relu.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/reshape.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/resolve_precision_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/attribute_parameters.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/bias_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/disable_cleanup_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/intervals_alignment_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/precision_preserved_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/precisions_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/quantization_alignment_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/quantization_granularity_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/quantization_mode_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/rt_info/shared_value_attribute.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/shuffle_channels.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/slice.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/space_to_batch.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/split.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/squeeze.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/strided_slice.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/transparent_base_transformation.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/transpose.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/unsqueeze.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/update_shared_precision_preserved.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/variadic_split.hpp
    ${PUBLIC_HEADERS_DIR}/low_precision/weightable_layer_transformation.hpp
)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create library
add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_OPENVINO_API)

ov_build_target_faster(${TARGET_NAME}_obj UNITY PCH)

target_link_libraries(${TARGET_NAME}_obj PRIVATE openvino::itt openvino::core::dev)

target_include_directories(${TARGET_NAME}_obj PRIVATE $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>)

ov_mark_target_as_cc(${TARGET_NAME}_obj)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME}_obj PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

# LTO

set_target_properties(${TARGET_NAME}_obj PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
