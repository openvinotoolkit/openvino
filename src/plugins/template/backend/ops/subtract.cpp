// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/subtract.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/reference/convert.hpp"

namespace {
template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::Subtract>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    return false;
}

template <>
bool evaluate<ov::element::u2>(const std::shared_ptr<ov::op::v1::Subtract>& op,
                               ov::TensorVector& outputs,
                               const ov::TensorVector& inputs) {
    constexpr const auto conversion_type = ov::element::i8;
    using T = typename ov::element_type_traits<conversion_type>::value_type;
    const auto& in0_shape = inputs[0].get_shape();
    const auto& in1_shape = inputs[1].get_shape();
    const auto& out_shape = outputs[0].get_shape();

    ov::Tensor converted_in0(conversion_type, in0_shape);
    ov::reference::convert(ov::element::iterator<ov::element::u2>(inputs[0].data()),
                           ov::element::iterator<conversion_type>(converted_in0.data()),
                           shape_size(in0_shape));

    ov::Tensor converted_in1(conversion_type, in1_shape);
    ov::reference::convert(ov::element::iterator<ov::element::u2>(inputs[1].data()),
                           ov::element::iterator<conversion_type>(converted_in1.data()),
                           shape_size(in1_shape));

    ov::Tensor converted_out(conversion_type, out_shape);

    ov::reference::subtract<T>(converted_in0.data<T>(),
                               converted_in1.data<T>(),
                               converted_out.data<T>(),
                               in0_shape,
                               in1_shape,
                               op->get_autob());

    ov::reference::convert(ov::element::iterator<conversion_type>(converted_out.data()),
                           ov::element::iterator<ov::element::u2>(outputs[0].data()),
                           shape_size(out_shape));

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
        OPENVINO_THROW("Unhandled data type ", node->get_element_type(), " in evaluate_node()");
    }
}
