// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/online_softmax_update_sum.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/op.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

OnlineSoftmaxUpdateSum::OnlineSoftmaxUpdateSum(const Output<Node>& A, const Output<Node>& B) : Op({A, B}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> OnlineSoftmaxUpdateSum::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(OnlineSoftmaxUpdateSum);
    check_new_args_count(this, new_args);
    return std::make_shared<OnlineSoftmaxUpdateSum>(new_args.at(0), new_args.at(1));
}

void OnlineSoftmaxUpdateSum::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    set_output_type(1, get_input_element_type(0), get_input_partial_shape(0));
}

}  // namespace ov::snippets::op
