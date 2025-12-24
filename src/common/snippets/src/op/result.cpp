// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/result.hpp"

#include <algorithm>
#include <memory>

#include "itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

Result::Result(const OutputVector& arguments) {
    set_arguments(arguments);
    constructor_validate_and_infer_types();
}

void Result::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Result_validate_and_infer_types);

    OPENVINO_ASSERT(get_input_size() != 0, "Snippets Result must have inputs");
    const auto inputs = input_values();
    const auto& inshape = get_input_partial_shape(0);
    const auto& intype = get_input_element_type(0);
    OPENVINO_ASSERT(std::all_of(inputs.cbegin() + 1,
                                inputs.cend(),
                                [&](const ov::Output<ov::Node>& in) {
                                    return in.get_partial_shape() == inshape && in.get_element_type() == intype;
                                }),
                    "All inputs of Snippets Result must have the same shape and element type");
    set_output_type(0, intype, inshape);
}

std::shared_ptr<Node> Result::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Result_clone_with_new_inputs);

    return std::make_shared<Result>(new_args);
}

}  // namespace ov::snippets::op
