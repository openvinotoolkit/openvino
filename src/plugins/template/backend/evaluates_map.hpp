// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/tensor.hpp"

std::vector<float> get_floats(const ov::Tensor& input, const ov::Shape& shape);

std::vector<int64_t> get_integers(const ov::Tensor& input, const ov::Shape& shape);

std::vector<int64_t> get_signal_size(const ov::TensorVector& inputs, size_t num_of_axes);

namespace ov {
namespace runtime {
namespace interpreter {
using EvaluatorsMap = std::map<
    ov::NodeTypeInfo,
    std::function<
        bool(const std::shared_ptr<ov::Node>& node, ov::TensorVector& outputs, const ov::TensorVector& inputs)>>;
EvaluatorsMap& get_evaluators_map();
}  // namespace interpreter
}  // namespace runtime
}  // namespace ov
