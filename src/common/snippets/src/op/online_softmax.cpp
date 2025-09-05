// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/online_softmax.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::op {

OnlineSoftmax::OnlineSoftmax(const Output<Node>& x, const int64_t axis) : Softmax(x, axis) {}

void OnlineSoftmax::validate_and_infer_types() {
    const auto& rank = get_input_partial_shape(0).size();
    OPENVINO_ASSERT(utils::any_of(get_axis(), static_cast<int64_t>(rank - 1), -1),
                    "Online softmax only support innermost axis");
    Softmax::validate_and_infer_types();
}

}  // namespace ov::snippets::op
