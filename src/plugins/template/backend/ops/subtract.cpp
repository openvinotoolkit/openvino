// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/subtract.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/subtract.hpp"

namespace {
template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::Subtract>& op, ov::TensorVector& outputs, const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ov::element::i8>::value_type;
    auto in0_shape = inputs[0].get_shape();
    auto in1_shape = inputs[1].get_shape();

    auto iter0 = ov::element::iterator<ov::element::u2>(inputs[0].data());
    auto in0_data = std::vector<int8_t>(iter0, iter0 + inputs[0].get_size());
    ov::Tensor converted_in0(ov::element::i8, in0_shape, in0_data.data());

    auto iter1 = ov::element::iterator<ov::element::u2>(inputs[1].data());
    auto in1_data = std::vector<int8_t>(iter1, iter1 + inputs[1].get_size());
    ov::Tensor converted_in1(ov::element::i8, in1_shape, in1_data.data());

    ov::Tensor converted_out(ov::element::i8, outputs[0].get_shape());

    ov::reference::subtract<T>(converted_in0.data<T>(), converted_in1.data<T>(), converted_out.data<T>(), in0_shape, in1_shape, op->get_autob());

    auto iter_out = ov::element::iterator<ov::element::u2>(outputs[0].data());
    auto out_data_ptr = reinterpret_cast<int8_t*>(converted_out.data());
    std::copy(out_data_ptr, out_data_ptr + converted_out.get_size(), iter_out);

    return true;
}
}  // namespace

template <>
bool evaluate_node<ov::op::v1::Subtract>(std::shared_ptr<ov::Node> node,
                                    ov::TensorVector& outputs,
                                    const ov::TensorVector& inputs) {
    switch (node->get_output_element_type(0)) {
    case ov::element::u2:
        return evaluate<ov::element::u2>(ov::as_type_ptr<ov::op::v1::Subtract>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
