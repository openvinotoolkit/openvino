// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/adaptive_avg_pool.hpp"
#include "pooling_shape_inference_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v8 {

template <class TShape>
std::vector<TShape> shape_infer(const AdaptiveAvgPool* op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    return {pooling::out_shape_infer(op, input_shapes, constant_data)};
}

template <class TShape>
void shape_infer(const AdaptiveAvgPool* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    output_shapes = shape_infer(op, input_shapes, constant_data);
}
}  // namespace v8
}  // namespace op
}  // namespace ov
