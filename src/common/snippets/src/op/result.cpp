// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/result.hpp"

#include <cstddef>
#include <memory>

#include "itt.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/tensor.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

Result::Result(const Output<Node>& main, const OutputVector& nodes) : ov::op::v0::Result(main) {
    if (!nodes.empty()) {
        size_t i = 1;
        for (const auto& node : nodes) {
            set_argument(i++, node);
        }
    }
    constructor_validate_and_infer_types();
}

void Result::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(snippets_Result_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_size() >= 1,
                          "Argument has ",
                          get_input_size(),
                          " outputs (expect at least 1) in snippets result.");

    descriptor::set_shared_tensor(get_output_descriptor(0),
                                  get_input_descriptor(0),
                                  ov::op::util::is_parameter(get_input_node_ptr(0)));

    if (get_input_size() > 1) {
        const auto& main_input_desc = get_input_descriptor(0);
        const auto& main_type = main_input_desc.get_element_type();
        const auto& main_shape = main_input_desc.get_partial_shape();
        for (size_t i = 1; i < get_input_size(); ++i) {
            const auto& input_desc = get_input_descriptor(i);
            NODE_VALIDATION_CHECK(
                this,
                input_desc.get_element_type() == main_type,
                "All inputs of Snippets Result node must have the same element type. Mismatch at input ",
                i,
                ": expected ",
                main_type,
                ", got ",
                input_desc.get_element_type());
            NODE_VALIDATION_CHECK(this,
                                  input_desc.get_partial_shape().same_scheme(main_shape),
                                  "All inputs of Snippets Result node must have compatible shapes. Mismatch at input ",
                                  i,
                                  ": expected compatible with ",
                                  main_shape,
                                  ", got ",
                                  input_desc.get_partial_shape());
        }
    }
}

std::shared_ptr<Node> Result::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(snippet_Result_clone_with_new_inputs);

    NODE_VALIDATION_CHECK(this,
                          !new_args.empty(),
                          "Incorrect number of new arguments for Snippets Result, at least one argument is expected.");

    if (new_args.size() == 1) {
        return std::make_shared<Result>(new_args.at(0));
    }
    const OutputVector nodes{new_args.begin() + 1, new_args.end()};
    return std::make_shared<Result>(new_args.at(0), nodes);
}

bool Result::has_evaluate() const {
    INTERNAL_OP_SCOPE(snippets_Result_has_evaluate);
    return false;
}

bool Result::evaluate([[maybe_unused]] TensorVector& outputs, [[maybe_unused]] const TensorVector& inputs) const {
    return false;
}

}  // namespace ov::snippets::op
