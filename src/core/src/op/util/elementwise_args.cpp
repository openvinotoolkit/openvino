// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/elementwise_args.hpp"

#include "eltwise_shape_inference.hpp"
#include "openvino/op/convert.hpp"

std::tuple<ov::element::Type, ov::PartialShape> ov::op::util::validate_and_infer_elementwise_args(Node* node) {
    OPENVINO_ASSERT(node != nullptr, "Node is empty! Cannot validate eltwise arguments.");
    constexpr size_t valid_inputs_count = 2;
    NODE_VALIDATION_CHECK(node,
                          node->get_input_size() == valid_inputs_count,
                          "Incorrect number of inputs. Required: ",
                          valid_inputs_count);

    element::Type result_et;
    if (!element::Type::merge(result_et, node->get_input_element_type(0), node->get_input_element_type(1))) {
        const auto& a = node->get_input_element_type(0);
        const auto& b = node->get_input_element_type(1);
        if (a.is_static() && b.is_static() && a.is_real() && b.is_real()) {
            element::Type promote = element::bf16;
            for (size_t i = 0; i < 2; ++i) {
                if (node->get_input_element_type(i) != promote) {
                    auto src = node->input_value(i);
                    auto cvt = std::make_shared<ov::op::v0::Convert>(src, promote);
                    node->input(i).replace_source_output(cvt->output(0));
                }
            }
            result_et = promote;
        } else {
            NODE_VALIDATION_CHECK(node, false,
                "Arguments do not have the same element type (arg0 element type: ",
                node->get_input_element_type(0),
                ", arg1 element type: ",
                node->get_input_element_type(1),
                ").");
        }
    }

    const auto& A_shape = node->get_input_partial_shape(0);
    const auto& B_shape = node->get_input_partial_shape(1);
    std::vector<ov::PartialShape> input_shapes = {A_shape, B_shape};
    const auto output_shapes = ov::op::eltwise_shape_infer(node, input_shapes);

    return std::make_tuple(result_et, output_shapes[0]);
}
