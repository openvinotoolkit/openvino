// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/gated_delta_net.hpp"

#include "evaluate_node.hpp"
#include "gated_delta_net_shape_inference.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/gated_delta_net.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::internal::GatedDeltaNet>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shapes = ov::op::internal::shape_infer(op.get(), input_shapes);
    outputs[0].set_shape(output_shapes[0].to_shape());
    outputs[1].set_shape(output_shapes[1].to_shape());

    ov::reference::gated_delta_net<T>(inputs[0].data<const T>(),
                                      inputs[1].data<const T>(),
                                      inputs[2].data<const T>(),
                                      inputs[3].data<const T>(),
                                      inputs[4].data<const T>(),
                                      inputs[5].data<const T>(),
                                      outputs[0].data<T>(),
                                      outputs[1].data<T>(),
                                      inputs[0].get_shape(),
                                      inputs[2].get_shape(),
                                      op->get_fuse_qk_l2norm(),
                                      static_cast<T>(op->get_q_l2_norm_eps()),
                                      static_cast<T>(op->get_k_l2_norm_eps()));
    return true;
}

template <>
bool evaluate_node<ov::op::internal::GatedDeltaNet>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs) {
    const auto& element_type = node->get_input_element_type(0);

    switch (element_type) {
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
}
