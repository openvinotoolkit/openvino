// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/core/model.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace reference {
using custom_evaluate_function = std::function<
    void(const std::shared_ptr<Model>& function, const ov::TensorVector& inputs, ov::TensorVector& outputs)>;
void tensor_iterator(uint64_t num_iterations,
                     const std::shared_ptr<Model>& body,
                     const op::util::OutputDescriptionVector& out_descs,
                     const op::util::InputDescriptionVector& input_descs,
                     ov::TensorVector& out,
                     const ov::TensorVector& args,
                     const custom_evaluate_function& evaluate = nullptr);
}  // namespace reference
}  // namespace ov
