// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/result.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/util/op_types.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

Result::Result(const OutputVector& arguments) {
    set_arguments(arguments);
    constructor_validate_and_infer_types();
}

void Result::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(snippets_result_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_size() >= 1,
                          "Argument has ",
                          get_input_size(),
                          " outputs (expect at least 1) in snippets result.");

    descriptor::set_shared_tensor(get_output_descriptor(0),
                                  get_input_descriptor(0),
                                  ov::op::util::is_parameter(get_input_node_ptr(0)));
}

std::shared_ptr<Node> Result::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Result_clone_with_new_inputs);

    return std::make_shared<Result>(new_args);
}

}  // namespace ov::snippets::op
