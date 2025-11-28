// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/result.hpp"

#include <memory>
#include <typeindex>
#include <typeinfo>

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ov::snippets::op {

Result::Result(const OutputVector& arguments) : Op(arguments) {
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

    // Result shares input tensor but can have specific properties which are added/removed to input.
    descriptor::set_shared_tensor(get_output_descriptor(0),
                                  get_input_descriptor(0),
                                  ov::op::util::is_parameter(get_input_node_ptr(0)));
}

std::shared_ptr<Node> Result::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Result_clone_with_new_inputs);
    // check_new_args_count(this, new_args);

    return std::make_shared<Result>(new_args);
}

}  // namespace ov::snippets::op
