// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "prior_box_shape_inference_util.hpp"

namespace ov {
namespace op {
namespace v0 {
template <class TShape>
std::vector<TShape> shape_infer(const PriorBoxClustered* const op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    return prior_box::shape_infer(op, input_shapes, constant_data);
}

template <class TShape>
void shape_infer(const PriorBoxClustered* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    output_shapes = prior_box::shape_infer(op, input_shapes, constant_data);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
