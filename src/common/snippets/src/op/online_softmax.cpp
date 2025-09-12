// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/online_softmax.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/op.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

OnlineSoftmax::OnlineSoftmax(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> OnlineSoftmax::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(OnlineSoftmax);
    check_new_args_count(this, new_args);
    return std::make_shared<OnlineSoftmax>(new_args.at(0));
}

void OnlineSoftmax::validate_and_infer_types() {
    auto input_shape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), input_shape);
    const auto& rank = input_shape.size();
    input_shape[rank - 1] = 1;
    set_output_type(1, get_input_element_type(0), input_shape);
}

}  // namespace ov::snippets::op
