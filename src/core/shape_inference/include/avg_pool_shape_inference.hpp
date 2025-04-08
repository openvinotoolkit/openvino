// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "max_pool_shape_inference.hpp"
#include "openvino/op/max_pool.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace pooling {

template <>
inline void valid_dilated_kernel_with_padding(const v1::AvgPool* op,
                                              const size_t kernel,
                                              const size_t pad_begin,
                                              const size_t pad_end,
                                              const size_t axis) {
    NODE_VALIDATION_CHECK(op,
                          !op->get_exclude_pad() || ((kernel > pad_begin) && (kernel > pad_end)),
                          "Kernel after dilation is sometimes entirely in the padding area for axis ",
                          axis,
                          " (dilated kernel dimension: ",
                          kernel,
                          ", padding below dimension: ",
                          pad_begin,
                          ", padding above dimension: ",
                          pad_end,
                          ") and this is not ",
                          "allowed.");
}

template <>
inline void valid_dilated_kernel_with_padding(const v14::AvgPool* op,
                                              const size_t kernel,
                                              const size_t pad_begin,
                                              const size_t pad_end,
                                              const size_t axis) {
    NODE_VALIDATION_CHECK(op,
                          !op->get_exclude_pad() || ((kernel > pad_begin) && (kernel > pad_end)),
                          "Kernel after dilation is sometimes entirely in the padding area for axis ",
                          axis,
                          " (dilated kernel dimension: ",
                          kernel,
                          ", padding below dimension: ",
                          pad_begin,
                          ", padding above dimension: ",
                          pad_end,
                          ") and this is not allowed.");
}

template <class TOp, class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> avg_pool_shape_infer_util(const TOp* op,
                                               const std::vector<TShape>& input_shapes,
                                               TContainer& pads_begin,
                                               TContainer& pads_end) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    const auto& data_shape = input_shapes[0];
    const auto dilations = Strides(op->get_kernel().size(), 1);

    auto num_spatial = dilations.size();
    pooling::resize_empty_padding(num_spatial, pads_begin, pads_end);
    pooling::validate::padding(op, pads_begin, pads_end);
    pooling::validate::attributes(op, data_shape, dilations);
    pooling::apply_padding(op, data_shape, dilations, pads_begin, pads_end);

    return {pooling::out_shape_infer(op, data_shape, pads_begin, pads_end, dilations)};
}
}  // namespace pooling

namespace v1 {
template <class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const AvgPool* op,
                                 const std::vector<TShape>& input_shapes,
                                 TContainer& pads_begin,
                                 TContainer& pads_end) {
    return pooling::avg_pool_shape_infer_util(op, input_shapes, pads_begin, pads_end);
}
}  // namespace v1

namespace v14 {
template <class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const AvgPool* op,
                                 const std::vector<TShape>& input_shapes,
                                 TContainer& pads_begin,
                                 TContainer& pads_end) {
    return pooling::avg_pool_shape_infer_util(op, input_shapes, pads_begin, pads_end);
}
}  // namespace v14
}  // namespace op
}  // namespace ov
