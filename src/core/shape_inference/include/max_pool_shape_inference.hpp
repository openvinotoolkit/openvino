// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "pooling_shape_inference_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace pooling {

inline void resize_dilations(Strides& dilations, const size_t num_spatial) {
    if (dilations.empty()) {
        dilations.resize(num_spatial, 1);
    }
}

template <class TOp, class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> max_pool_shape_infer_util(const TOp* op,
                                               const std::vector<TShape>& input_shapes,
                                               TContainer& pads_begin,
                                               TContainer& pads_end) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    const auto& data_shape = input_shapes[0];

    auto dilations = op->get_dilations();
    auto num_spatial = op->get_kernel().size();
    resize_dilations(dilations, num_spatial);
    pooling::resize_empty_padding(num_spatial, pads_begin, pads_end);
    pooling::validate::padding(op, pads_begin, pads_end);
    pooling::validate::attributes(op, data_shape, dilations);
    pooling::apply_padding(op, data_shape, dilations, pads_begin, pads_end);

    return {2, pooling::out_shape_infer(op, data_shape, pads_begin, pads_end, dilations)};
}
}  // namespace pooling

namespace v1 {
template <class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const MaxPool* op,
                                 const std::vector<TShape>& input_shapes,
                                 TContainer& pads_begin,
                                 TContainer& pads_end) {
    const auto& data_shape = input_shapes[0];
    const auto dilations = Strides(op->get_kernel().size(), 1);

    auto num_spatial = dilations.size();
    pooling::resize_empty_padding(num_spatial, pads_begin, pads_end);
    pooling::validate::padding(op, pads_begin, pads_end);
    pooling::validate::attributes(op, data_shape, dilations);
    pooling::apply_padding(op, data_shape, dilations, pads_begin, pads_end);

    return {pooling::out_shape_infer(op, data_shape, pads_begin, pads_end, dilations)};
}
}  // namespace v1

namespace v8 {
template <class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const MaxPool* op,
                                 const std::vector<TShape>& input_shapes,
                                 TContainer& pads_begin,
                                 TContainer& pads_end) {
    return pooling::max_pool_shape_infer_util(op, input_shapes, pads_begin, pads_end);
}
}  // namespace v8

namespace v14 {
template <class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const MaxPool* op,
                                 const std::vector<TShape>& input_shapes,
                                 TContainer& pads_begin,
                                 TContainer& pads_end) {
    return pooling::max_pool_shape_infer_util(op, input_shapes, pads_begin, pads_end);
}
}  // namespace v14
}  // namespace op
}  // namespace ov
