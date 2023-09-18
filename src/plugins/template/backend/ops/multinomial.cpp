// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/multinomial.hpp"

#include "evaluate_node.hpp"

namespace multinomial {

template <ov::element::Type_t INPUT_T, ov::element::Type_t SAMPLES_T, ov::element::Type_t OUTPUT_T>
inline void evaluate_internal(const std::shared_ptr<ov::op::v13::Multinomial>& op,
                              const ov::HostTensorVector& outputs,
                              const ov::HostTensorVector& inputs) {
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
                                                        op->get_op_seed(),
                                                        false);
}

template <ov::element::Type_t INPUT_T, ov::element::Type_t SAMPLES_T>
inline void evaluate(const std::shared_ptr<ov::op::v13::Multinomial>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    switch (op->get_output_type()) {
    case ov::element::Type_t::boolean:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::boolean>(op, outputs, inputs);
        return;
    case ov::element::Type_t::bf16:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::bf16>(op, outputs, inputs);
        return;
    case ov::element::Type_t::f16:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::f16>(op, outputs, inputs);
        return;
    case ov::element::Type_t::f64:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::f64>(op, outputs, inputs);
        return;
    case ov::element::Type_t::f32:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::f32>(op, outputs, inputs);
        return;
    case ov::element::Type_t::i4:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::i4>(op, outputs, inputs);
        return;
    case ov::element::Type_t::i8:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::i8>(op, outputs, inputs);
        return;
    case ov::element::Type_t::i16:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::i16>(op, outputs, inputs);
        return;
    case ov::element::Type_t::i32:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::i32>(op, outputs, inputs);
        return;
    case ov::element::Type_t::i64:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::i64>(op, outputs, inputs);
        return;
    case ov::element::Type_t::u1:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::u1>(op, outputs, inputs);
        return;
    case ov::element::Type_t::u4:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::u4>(op, outputs, inputs);
        return;
    case ov::element::Type_t::u8:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::u8>(op, outputs, inputs);
        return;
    case ov::element::Type_t::u16:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::u16>(op, outputs, inputs);
        return;
    case ov::element::Type_t::u32:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::u32>(op, outputs, inputs);
        return;
    case ov::element::Type_t::u64:
        evaluate_internal<INPUT_T, SAMPLES_T, ov::element::Type_t::u64>(op, outputs, inputs);
        return;
    default:
        OPENVINO_THROW(std::string("Unhandled output data type ") +
                       ov::element::Type(op->get_output_type()).get_type_name() + std::string("in evaluate_node()"));
    }
}
}  // namespace multinomial

template <ov::element::Type_t INPUT_T>
bool evaluate(const std::shared_ptr<ov::op::v13::Multinomial>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
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
                                             const ov::HostTensorVector& outputs,
                                             const ov::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::Type_t::boolean:
        return evaluate<ov::element::Type_t::boolean>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::bf16:
        return evaluate<ov::element::Type_t::bf16>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::f16:
        return evaluate<ov::element::Type_t::f16>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::f64:
        return evaluate<ov::element::Type_t::f64>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::f32:
        return evaluate<ov::element::Type_t::f32>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::i4:
        return evaluate<ov::element::Type_t::i4>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::i8:
        return evaluate<ov::element::Type_t::i8>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::i16:
        return evaluate<ov::element::Type_t::i16>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::i32:
        return evaluate<ov::element::Type_t::i32>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::i64:
        return evaluate<ov::element::Type_t::i64>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::u1:
        return evaluate<ov::element::Type_t::u1>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::u4:
        return evaluate<ov::element::Type_t::u4>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::u8:
        return evaluate<ov::element::Type_t::u8>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::u16:
        return evaluate<ov::element::Type_t::u16>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::u32:
        return evaluate<ov::element::Type_t::u32>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    case ov::element::Type_t::u64:
        return evaluate<ov::element::Type_t::u64>(ov::as_type_ptr<ov::op::v13::Multinomial>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
