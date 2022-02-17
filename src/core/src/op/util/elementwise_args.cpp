// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/elementwise_args.hpp"

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

std::tuple<ov::element::Type, ov::PartialShape> ov::op::util::validate_and_infer_elementwise_args(
    Node* node,
    const op::AutoBroadcastSpec& autob) {
    NGRAPH_CHECK(node != nullptr, "nGraph node is empty! Cannot validate eltwise arguments.");
    element::Type element_type = node->get_input_element_type(0);
    PartialShape pshape = node->get_input_partial_shape(0);

    if (node->get_input_size() > 1) {
        for (size_t i = 1; i < node->get_input_size(); ++i) {
            NODE_VALIDATION_CHECK(node,
                                  element::Type::merge(element_type, element_type, node->get_input_element_type(i)),
                                  "Argument element types are inconsistent.");

            if (autob.m_type == op::AutoBroadcastType::NONE) {
                NODE_VALIDATION_CHECK(node,
                                      PartialShape::merge_into(pshape, node->get_input_partial_shape(i)),
                                      "Argument shapes are inconsistent.");
            } else if (autob.m_type == op::AutoBroadcastType::NUMPY || autob.m_type == op::AutoBroadcastType::PDPD) {
                NODE_VALIDATION_CHECK(
                    node,
                    PartialShape::broadcast_merge_into(pshape, node->get_input_partial_shape(i), autob),
                    "Argument shapes are inconsistent.");
            } else {
                NODE_VALIDATION_CHECK(node, false, "Unsupported auto broadcast specification");
            }
        }
    }

    return std::make_tuple(element_type, pshape);
}
