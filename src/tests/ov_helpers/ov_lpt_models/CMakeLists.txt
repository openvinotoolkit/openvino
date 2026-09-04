# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_lpt_models)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/align_concat_quantization_parameters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/assign_and_read_value.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/avg_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/batch_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/clamp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/builders.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/constant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/dequantization_operations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/fake_quantize_on_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/fake_quantize_on_weights.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/multiply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compose_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/convolution_backprop_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/depth_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise_with_multi_parent_dequantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fake_quantize_and_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fake_quantize_and_two_output_branches_with_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fake_quantize_on_weights_and_unsupported_child.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fake_quantize_precision_selection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fold_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fuse_convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fuse_fake_quantize_and_scale_shift.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fuse_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fuse_multiply_to_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fuse_subtract_to_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/get_dequantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/group_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/interpolate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/markup_avg_pool_precisions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/markup_bias.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat_mul_with_optimized_constant_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/max_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/move_dequantization_after.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/move_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multiply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multiply_partial_function.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multiply_to_group_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multiply_with_one_parent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mvn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/normalize_dequantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/normalize_l2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/precision_propagation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/prelu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/qdq_stripping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/recurrent_cell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/relu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/round.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shuffle_channels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/space_to_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/squeeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/strided_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subtract_multiply_to_multiply_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transformations_after_split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transpose_after_mat_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/unsqueeze.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/variadic_split.cpp
)

set(PUBLIC_HEADERS
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/add.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/align_concat_quantization_parameters.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/assign_and_read_value.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/avg_pool.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/batch_to_space.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/broadcast.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/clamp.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/add.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/builders.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/constant.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/convolution.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/dequantization_operations.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/fake_quantize_on_data.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/fake_quantize_on_weights.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/multiply.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/reshape.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/common/transpose.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/compose_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/concat.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/convolution_backprop_data.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/convolution.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/depth_to_space.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/elementwise.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/elementwise_with_multi_parent_dequantization.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fake_quantize_and_convolution.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fake_quantize_and_two_output_branches_with_convolution.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fake_quantize_on_weights_and_unsupported_child.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fake_quantize_precision_selection.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fold_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fuse_convert.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fuse_fake_quantize_and_scale_shift.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fuse_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fuse_multiply_to_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/fuse_subtract_to_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/gather.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/get_dequantization.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/group_convolution.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/interpolate.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/markup_avg_pool_precisions.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/markup_bias.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/mat_mul.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/mat_mul_with_optimized_constant_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/max_pool.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/move_dequantization_after.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/move_fake_quantize.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/mul_add_to_scaleshift_or_power.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/multiply.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/multiply_partial_function.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/multiply_to_group_convolution.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/multiply_with_one_parent.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/mvn.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/normalize_dequantization.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/normalize_l2.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/pad.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/precision_propagation.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/prelu.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/qdq_stripping.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/recurrent_cell.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/reduce.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/relu.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/reshape.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/round.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/shuffle_channels.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/slice.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/space_to_batch.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/split.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/squeeze.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/strided_slice.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/subtract_multiply_to_multiply_add.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/transformations_after_split.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/transpose_after_mat_mul.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/transpose.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/unsqueeze.hpp
    ${PUBLIC_HEADERS_DIR}/ov_lpt_models/variadic_split.hpp
)

set(PRIVATE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp)

source_group("src"     FILES ${SOURCES})
source_group("include" FILES ${PUBLIC_HEADERS})

add_library(${TARGET_NAME} STATIC ${SOURCES} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC
    "$<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>")

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime::dev
    common_test_utils
)

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH_HEADER "src/precomp.hpp"
)

# install & export

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${PUBLIC_HEADERS_DIR}/")
