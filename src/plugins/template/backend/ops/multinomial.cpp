// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/multinomial.hpp"

#include "evaluate_node.hpp"
#include "multinomial_shape_inference.hpp"

template <ov::element::Type_t INPUT_T, ov::element::Type_t SAMPLES_T, ov::element::Type_t OUTPUT_T>
inline void evaluate_output_t(const std::shared_ptr<ov::op::v13::Multinomial>& op,
                              ov::TensorVector& outputs,
                              const ov::TensorVector& inputs) {
    using T1 = typename ov::element_type_traits<INPUT_T>::value_type;
    using T2 = typename ov::element_type_traits<SAMPLES_T>::value_type;
    using T3 = typename ov::element_type_traits<OUTPUT_T>::value_type;

    const auto tensor_acc = make_tensor_accessor(inputs);
    const std::vector<ov::PartialShape> input_shapes{op->get_input_shape(0), op->get_input_shape(1)};
    const auto out_shape = ov::op::v13::shape_infer(op.get(), input_shapes, tensor_acc).front().to_shape();
    outputs[0].set_shape(out_shape);

    ov::reference::multinomial::multinomial<T1, T2, T3>(inputs[0].data<const T1>(),
                                                        op->get_input_shape(0),
                                                        inputs[1].data<const T2>(),
                                                        op->get_input_shape(1),
                                                        outputs[0].data<T3>(),
                                                        out_shape,
                                                        op->get_with_replacement(),
                                                        op->get_log_probs(),
                                                        op->get_global_seed(),
                                                        op->get_op_seed());
}

template <ov::element::Type_t INPUT_T, ov::element::Type_t SAMPLES_T>
inline void evaluate_samples_t(const std::shared_ptr<ov::op::v13::Multinomial>& op,
                               ov::TensorVector& outputs,
                               const ov::TensorVector& inputs) {
    switch (op->get_convert_type()) {
    case ov::element::Type_t::i32:
        evaluate_output_t<INPUT_T, SAMPLES_T, ov::element::Type_t::i32>(op, outputs, inputs);
        return;
    case ov::element::Type_t::i64:
        evaluate_output_t<INPUT_T, SAMPLES_T, ov::element::Type_t::i64>(op, outputs, inputs);
        return;
    default:
        OPENVINO_THROW(std::string("Unhandled convert data type '") +
                       ov::element::Type(op->get_convert_type()).get_type_name() +
                       std::string("' in evaluate_node(). Use either i32 or i64 and apply conversion manually."));
    }
}

template <ov::element::Type_t INPUT_T>
bool evaluate_input_t(const std::shared_ptr<ov::op::v13::Multinomial>& op,
                      ov::TensorVector& outputs,
                      const ov::TensorVector& inputs) {
    switch (inputs[1].get_element_type()) {
    case ov::element::Type_t::i64:
        evaluate_samples_t<INPUT_T, ov::element::Type_t::i64>(op, outputs, inputs);
        break;
    default:
        evaluate_samples_t<INPUT_T, ov::element::Type_t::i32>(op, outputs, inputs);
        break;
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v13::Multinomial>(std::shared_ptr<ov::Node> node,
                                             ov::TensorVector& outputs,
                                             const ov::TensorVector& inputs) {
    switch (node->get_input_element_type(0)) {
    case ov::element::Type_t::f16:
        return evaluate_input_t<ov::element::Type_t::f16>(ov::as_type_ptr<ov::op::v13::Multinomial>(node),
                                                          outputs,
                                                          inputs);
    case ov::element::Type_t::f32:
        return evaluate_input_t<ov::element::Type_t::f32>(ov::as_type_ptr<ov::op::v13::Multinomial>(node),
                                                          outputs,
                                                          inputs);
    case ov::element::Type_t::f64:
        return evaluate_input_t<ov::element::Type_t::f64>(ov::as_type_ptr<ov::op::v13::Multinomial>(node),
                                                          outputs,
                                                          inputs);
    case ov::element::Type_t::bf16:
        return evaluate_input_t<ov::element::Type_t::bf16>(ov::as_type_ptr<ov::op::v13::Multinomial>(node),
                                                           outputs,
                                                           inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled input data type ") + node->get_input_element_type(0).get_type_name() +
                       std::string(" in evaluate_node()."));
    }
}
