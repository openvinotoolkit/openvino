// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/multinomial.hpp"
#include "multinomial_shape_inference.hpp"
#include "evaluate_node.hpp"

namespace multinomial {

template <ov::element::Type_t INPUT_T, ov::element::Type_t SAMPLES_T, ov::element::Type_t OUTPUT_T>
inline void evaluate_internal(const std::shared_ptr<ov::op::v13::Multinomial>& op,
                              const ov::TensorVector& outputs,
                              const ov::TensorVector& inputs) {
    using T1 = typename ov::element_type_traits<INPUT_T>::value_type;
    using T2 = typename ov::element_type_traits<SAMPLES_T>::value_type;
    using T3 = typename ov::element_type_traits<OUTPUT_T>::value_type;
    ov::reference::multinomial::multinomial<T1, T2, T3>(inputs[0]->get_data_ptr<const T1>(),
                                                        op->get_input_shape(0),
                                                        inputs[1]->get_data_ptr<const T2>(),
                                                        op->get_input_shape(1),
                                                        outputs[0]->get_data_ptr<T3>(),
                                                        op->get_output_shape(0),
                                                        op->get_with_replacement(),
                                                        op->get_log_probs(),
                                                        op->get_global_seed(),
                                                        op->get_op_seed());
    const auto tensor_acc = make_tensor_accessor(inputs);
    std::vector<ov::Shape> input_shapes{op->get_input_shape(0), op->get_input_shape(1)};
    const auto out_shape = shape_infer(this, input_shapes, tensor_acc).front().to_shape();
    outputs[0].set_shape(out_shape);
}

template <ov::element::Type_t INPUT_T, ov::element::Type_t SAMPLES_T>
inline void evaluate(const std::shared_ptr<ov::op::v13::Multinomial>& op,
                     const ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    switch (op->get_convert_type()) {
    case ov::element::Type_t::i32:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::i32>(op, outputs, inputs);
        return;
    case ov::element::Type_t::i64:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::i64>(op, outputs, inputs);
        return;
    default:
        OPENVINO_THROW(std::string("Unhandled output data type ") +
                       ov::element::Type(op->get_output_type()).get_type_name() + std::string("in evaluate_node(). Use either i32 or i64 and apply conversion manually."));
    }
}
}  // namespace multinomial

template <ov::element::Type_t INPUT_T>
bool evaluate(const std::shared_ptr<ov::op::v13::Multinomial>& op,
              const ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ov::element::Type_t::i64:
        multinomial::evaluate<INPUT_T, ov::element::Type_t::i64>(op, outputs, inputs);
        break;
    default:
        multinomial::evaluate<INPUT_T, ov::element::Type_t::i32>(op, outputs, inputs);
        break;
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v13::Multinomial>(std::shared_ptr<ov::Node> node,
                                             const ov::TensorVector& outputs,
                                             const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::Type_t::f16:
        return evaluate<ov::element::Type_t::f16>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::f32:
        return evaluate<ov::element::Type_t::f32>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::f64:
        return evaluate<ov::element::Type_t::f64>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::bf16:
        return evaluate<ov::element::Type_t::f16>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
