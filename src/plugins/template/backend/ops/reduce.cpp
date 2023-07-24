// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluate_node.hpp"
#include <ngraph/runtime/reference/reduce_l1.hpp>
#include <ngraph/runtime/reference/reduce_l2.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/reduce_l1.hpp>
#include <openvino/op/reduce_l2.hpp>
#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v4::ReduceL1>& op, const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const auto axes_vector = host_tensor_2_vector<int64_t>(inputs[1]);
    const auto normalized_axes = ov::normalize_axes(op->get_friendly_name(), axes_vector, inputs[0]->get_partial_shape().rank());
    const auto reduction_axes = ov::AxisSet{normalized_axes};

    ngraph::runtime::reference::reduce_l1<T>(inputs[0]->get_data_ptr<T>(),
                                             outputs[0]->get_data_ptr<T>(),
                                             inputs[0]->get_shape(),
                                             reduction_axes);
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v4::ReduceL2>& op, const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const auto axes_vector = host_tensor_2_vector<int64_t>(inputs[1]);
    const auto normalized_axes = ov::normalize_axes(op->get_friendly_name(), axes_vector, inputs[0]->get_partial_shape().rank());
    const auto reduction_axes = ov::AxisSet{normalized_axes};

    ngraph::runtime::reference::reduce_l2<T>(inputs[0]->get_data_ptr<T>(),
                                             outputs[0]->get_data_ptr<T>(),
                                             inputs[0]->get_shape(),
                                             reduction_axes);
    return true;
}

template <>
bool evaluate_node<ov::op::v4::ReduceL1>(std::shared_ptr<ov::Node> node,
                                         const ov::HostTensorVector& outputs,
                                         const ov::HostTensorVector& inputs) {
    const ov::element::Type_t element_type = node->get_output_element_type(0);
    auto reduce_node = ov::as_type_ptr<ov::op::v4::ReduceL1>(node);

    switch (element_type) {
    case ov::element::Type_t::i64:
        return evaluate<ov::element::Type_t::i64>(reduce_node, outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

template <>
bool evaluate_node<ov::op::v4::ReduceL2>(std::shared_ptr<ov::Node> node,
                                         const ov::HostTensorVector& outputs,
                                         const ov::HostTensorVector& inputs) {
    const ov::element::Type_t element_type = node->get_output_element_type(0);
    auto reduce_node = ov::as_type_ptr<ov::op::v4::ReduceL2>(node);

    switch (element_type) {
        case ov::element::Type_t::i64:
            return evaluate<ov::element::Type_t::i64>(reduce_node, outputs, inputs);
        default:
            OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                           std::string("in evaluate_node()"));
    }
}
