// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/tensorarray_write.hpp"

#include <algorithm>

#include "openvino/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

op::internal::TensorArrayWrite::TensorArrayWrite(const Output<Node>& input, const Output<Node>& index)
    : Op({input, index}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::TensorArrayWrite::clone_with_new_inputs(const OutputVector& new_args) const {
    return make_shared<TensorArrayWrite>(new_args[0], new_args[1]);
}

bool op::internal::TensorArrayWrite::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

// tensorarray_write will be transformed and replaced.
// Here we simply make it to be an internal dynamic node, to make sure
// all its offsprings will validate and infer shapes from an dynamic input.
void op::internal::TensorArrayWrite::validate_and_infer_types() {
    auto ps = get_input_partial_shape(0);
    if (ps.rank().is_static() && ps.rank().get_length() >= 1) {
        ps.insert(ps.begin(), 1);  // unsqueeze in order to handyfully slice a tensorarray

        // will use concat to implement tensor_write and a different dimension is enough for
        // a zero-dimension const input
        if (ps[1].is_static()) {
            ps[1] += 1;
        }
    }
    set_output_type(0, get_input_element_type(0), ps);
}
