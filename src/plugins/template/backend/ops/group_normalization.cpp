// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/group_normalization.hpp"

#include "evaluate_node.hpp"
#include "openvino/op/group_normalization.hpp"

using namespace ov;

template <element::Type_t DATA_ET>
bool evaluate(const std::shared_ptr<ov::op::v12::GroupNormalization>& node,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    outputs[0]->set_shape(inputs[0]->get_shape());
    ov::reference::group_normalization(inputs[0]->get_data_ptr<DATA_ET>(),
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
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), "in evaluate_node()");
    }
}
