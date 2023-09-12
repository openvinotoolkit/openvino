// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/multinomial.hpp"

#include "evaluate_node.hpp"

namespace multinomial_internal {

template <ngraph::element::Type_t INPUT_T, ngraph::element::Type_t SAMPLES_T, ngraph::element::Type_t OUTPUT_T>
inline void evaluate_internal(const std::shared_ptr<ngraph::op::v13::Multinomial>& op,
                              const ngraph::HostTensorVector& outputs,
                              const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<INPUT_T>::value_type;
    using T2 = typename ngraph::element_type_traits<SAMPLES_T>::value_type;
    using T3 = typename ngraph::element_type_traits<OUTPUT_T>::value_type;
    ov::reference::multinomial<T1, T2, T3>(inputs[0]->get_data_ptr<const T1>(),
                                           op->get_input_shape(0),
                                           inputs[1]->get_data_ptr<const T2>(),
                                           op->get_input_shape(1),
                                           outputs[0]->get_data_ptr<const T3>(),
                                           op->get_output_shape(0),
                                           op->get_with_replacement(),
                                           op->get_log_probs(),
                                           op->get_global_seed(),
                                           op->get_op_seed(),
                                           false);
}

template <ngraph::element::Type_t INPUT_T, ngraph::element::Type_t SAMPLES_T>
inline void evaluate(const std::shared_ptr<ngraph::op::v13::Multinomial>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    switch (op->get_output_type()) {
    case ngraph::element::Type_t::boolean:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::boolean>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::bf16>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f16:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::f16>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f64:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::f64>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f32:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::f32>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i4:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::i4>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i8:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::i8>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i16:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::i16>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i32:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::i32>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i64:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::i64>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u1:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::u1>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u4:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::u4>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u8:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::u8>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u16:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::u16>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u32:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::u32>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u64:
        return evaluate_internal<INPUT_T, SAMPLES_T, ngraph::element::Type_t::u64>(
            ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
            outputs,
            inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled output data type ") + op->get_output_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
}  // namespace multinomial_internal

template <ngraph::element::Type_t INPUT_T>
bool evaluate(const std::shared_ptr<ngraph::op::v13::Multinomial>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ngraph::element::Type_t::i64:
        multinomial_internal::evaluate<INPUT_T, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    default:
        multinomial_internal::evaluate<INPUT_T, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    }
    return true;
}

template <>
bool evaluate_node<ngraph::op::v13::Multinomial>(std::shared_ptr<ngraph::Node> node,
                                                 const ngraph::HostTensorVector& outputs,
                                                 const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v13::Multinomial>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}