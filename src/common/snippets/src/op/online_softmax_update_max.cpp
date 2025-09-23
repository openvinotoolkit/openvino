// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/online_softmax_update_max.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/op.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

OnlineSoftmaxUpdateMax::OnlineSoftmaxUpdateMax(const Output<Node>& max_local) : Op({max_local}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> OnlineSoftmaxUpdateMax::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(OnlineSoftmaxUpdateMax);
    check_new_args_count(this, new_args);
    return std::make_shared<OnlineSoftmaxUpdateMax>(new_args.at(0));
}

void OnlineSoftmaxUpdateMax::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    set_output_type(1, get_input_element_type(0), get_input_partial_shape(0));
}

}  // namespace ov::snippets::op
