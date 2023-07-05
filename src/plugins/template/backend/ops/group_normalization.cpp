// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "evaluate_node.hpp"

#include "ngraph/runtime/reference/group_normalization.hpp"

#include "openvino/op/group_normalization.hpp"
// clang-format on

using namespace ov;

template <element::Type_t DATA_ET>
bool evaluate(const std::shared_ptr<ov::op::v12::GroupNormalization>& node,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    ngraph::runtime::reference::group_normalization(inputs[0]->get_data_ptr<DATA_ET>(),
                                                    inputs[1]->get_data_ptr<DATA_ET>(),
                                                    inputs[2]->get_data_ptr<DATA_ET>(),
                                                    outputs[0]->get_data_ptr<DATA_ET>(),
                                                    inputs[0]->get_shape(),
                                                    static_cast<size_t>(node->get_num_groups()),
                                                    node->get_epsilon());
    return true;
}

template <>
bool evaluate_node<op::v12::GroupNormalization>(std::shared_ptr<ov::Node> node,
                                                const ov::HostTensorVector& outputs,
                                                const ov::HostTensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case element::Type_t::bf16:
        return evaluate<element::Type_t::bf16>(as_type_ptr<op::v12::GroupNormalization>(node), outputs, inputs);
    case element::Type_t::f16:
        return evaluate<element::Type_t::f16>(as_type_ptr<op::v12::GroupNormalization>(node), outputs, inputs);
    case element::Type_t::f64:
        return evaluate<element::Type_t::f64>(as_type_ptr<op::v12::GroupNormalization>(node), outputs, inputs);
    case element::Type_t::f32:
        return evaluate<element::Type_t::f32>(as_type_ptr<op::v12::GroupNormalization>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
