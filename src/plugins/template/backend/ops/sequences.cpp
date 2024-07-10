// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "evaluate_node.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "openvino/reference/sequences.hpp"
// clang-format on

namespace rnn_seq_v5 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v5::RNNSequence>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ov::reference::rnn_sequence<T1, T2>(static_cast<char*>(inputs[0].data()),
                                        inputs[0].get_shape(),
                                        static_cast<char*>(inputs[1].data()),
                                        inputs[1].get_shape(),
                                        static_cast<char*>(inputs[2].data()),
                                        inputs[2].get_shape(),
                                        static_cast<char*>(inputs[3].data()),
                                        inputs[3].get_shape(),
                                        static_cast<char*>(inputs[4].data()),
                                        inputs[4].get_shape(),
                                        static_cast<char*>(inputs[5].data()),
                                        inputs[5].get_shape(),
                                        static_cast<char*>(outputs[0].data()),
                                        static_cast<char*>(outputs[1].data()),
                                        op->get_activations()[0],
                                        op->get_clip(),
                                        op->get_direction());
}
}  // namespace rnn_seq_v5

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v5::RNNSequence>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    switch (inputs[2].get_element_type()) {
    case ov::element::i64:
    case ov::element::u64:
        rnn_seq_v5::evaluate<ET, ov::element::i64>(op, outputs, inputs);
        break;
    case ov::element::i32:
    case ov::element::u32:
        rnn_seq_v5::evaluate<ET, ov::element::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace lstm_seq_v1 {
OPENVINO_SUPPRESS_DEPRECATED_START
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v0::LSTMSequence>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    OPENVINO_SUPPRESS_DEPRECATED_END
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ov::reference::lstm_sequence_v1<T1, T2>(static_cast<char*>(inputs[0].data()),
                                            inputs[0].get_shape(),
                                            static_cast<char*>(inputs[1].data()),
                                            inputs[1].get_shape(),
                                            static_cast<char*>(inputs[2].data()),
                                            inputs[2].get_shape(),
                                            static_cast<char*>(inputs[3].data()),
                                            inputs[3].get_shape(),
                                            static_cast<char*>(inputs[4].data()),
                                            inputs[4].get_shape(),
                                            static_cast<char*>(inputs[5].data()),
                                            inputs[5].get_shape(),
                                            static_cast<char*>(inputs[6].data()),
                                            inputs[6].get_shape(),
                                            static_cast<char*>(inputs[7].data()),
                                            inputs[7].get_shape(),
                                            static_cast<char*>(outputs[0].data()),
                                            static_cast<char*>(outputs[1].data()),
                                            static_cast<char*>(outputs[2].data()),
                                            op->get_activations()[0],
                                            op->get_activations()[1],
                                            op->get_activations()[2],
                                            op->get_clip_threshold(),
                                            op->get_weights_format(),
                                            op->get_input_forget(),
                                            op->get_direction());
}
}  // namespace lstm_seq_v1

namespace lstm_seq_v5 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v5::LSTMSequence>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ov::reference::lstm_sequence<T1, T2>(static_cast<char*>(inputs[0].data()),
                                         inputs[0].get_shape(),
                                         static_cast<char*>(inputs[1].data()),
                                         inputs[1].get_shape(),
                                         static_cast<char*>(inputs[2].data()),
                                         inputs[2].get_shape(),
                                         static_cast<char*>(inputs[3].data()),
                                         inputs[3].get_shape(),
                                         static_cast<char*>(inputs[4].data()),
                                         inputs[4].get_shape(),
                                         static_cast<char*>(inputs[5].data()),
                                         inputs[5].get_shape(),
                                         static_cast<char*>(inputs[6].data()),
                                         inputs[6].get_shape(),
                                         static_cast<char*>(outputs[0].data()),
                                         static_cast<char*>(outputs[1].data()),
                                         static_cast<char*>(outputs[2].data()),
                                         op->get_activations()[0],
                                         op->get_activations()[1],
                                         op->get_activations()[2],
                                         op->get_clip(),
                                         op->get_direction());
}
}  // namespace lstm_seq_v5

OPENVINO_SUPPRESS_DEPRECATED_START
template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::LSTMSequence>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    OPENVINO_SUPPRESS_DEPRECATED_END
    switch (inputs[3].get_element_type()) {
    case ov::element::i64:
    case ov::element::u64:
        lstm_seq_v1::evaluate<ET, ov::element::i64>(op, outputs, inputs);
        break;
    case ov::element::i32:
    case ov::element::u32:
        lstm_seq_v1::evaluate<ET, ov::element::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v5::LSTMSequence>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    switch (inputs[3].get_element_type()) {
    case ov::element::i64:
    case ov::element::u64:
        lstm_seq_v5::evaluate<ET, ov::element::i64>(op, outputs, inputs);
        break;
    case ov::element::i32:
    case ov::element::u32:
        lstm_seq_v5::evaluate<ET, ov::element::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace gru_seq_v5 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v5::GRUSequence>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ov::reference::gru_sequence<T1, T2>(static_cast<char*>(inputs[0].data()),
                                        inputs[0].get_shape(),
                                        static_cast<char*>(inputs[1].data()),
                                        inputs[1].get_shape(),
                                        static_cast<char*>(inputs[2].data()),
                                        inputs[2].get_shape(),
                                        static_cast<char*>(inputs[3].data()),
                                        inputs[3].get_shape(),
                                        static_cast<char*>(inputs[4].data()),
                                        inputs[4].get_shape(),
                                        static_cast<char*>(inputs[5].data()),
                                        inputs[5].get_shape(),
                                        static_cast<char*>(outputs[0].data()),
                                        static_cast<char*>(outputs[1].data()),
                                        op->get_activations()[0],
                                        op->get_activations()[1],
                                        op->get_clip(),
                                        op->get_direction(),
                                        op->get_linear_before_reset());
}
}  // namespace gru_seq_v5

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v5::GRUSequence>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    switch (inputs[2].get_element_type()) {
    case ov::element::i64:
    case ov::element::u64:
        gru_seq_v5::evaluate<ET, ov::element::i64>(op, outputs, inputs);
        break;
    case ov::element::i32:
    case ov::element::u32:
        gru_seq_v5::evaluate<ET, ov::element::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace augru_seq {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::internal::AUGRUSequence>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ov::reference::gru_sequence<T1, T2>(static_cast<char*>(inputs[0].data()),
                                        inputs[0].get_shape(),
                                        static_cast<char*>(inputs[1].data()),
                                        inputs[1].get_shape(),
                                        static_cast<char*>(inputs[2].data()),
                                        inputs[2].get_shape(),
                                        static_cast<char*>(inputs[3].data()),
                                        inputs[3].get_shape(),
                                        static_cast<char*>(inputs[4].data()),
                                        inputs[4].get_shape(),
                                        static_cast<char*>(inputs[5].data()),
                                        inputs[5].get_shape(),
                                        static_cast<char*>(outputs[0].data()),
                                        static_cast<char*>(outputs[1].data()),
                                        op->get_activations()[0],
                                        op->get_activations()[1],
                                        op->get_clip(),
                                        op->get_direction(),
                                        op->get_linear_before_reset(),
                                        static_cast<char*>(inputs[6].data()));
}
}  // namespace augru_seq

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::internal::AUGRUSequence>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    switch (inputs[2].get_element_type()) {
    case ov::element::i64:
    case ov::element::u64:
        augru_seq::evaluate<ET, ov::element::i64>(op, outputs, inputs);
        break;
    case ov::element::i32:
    case ov::element::u32:
        augru_seq::evaluate<ET, ov::element::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ov::op::internal::AUGRUSequence>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::internal::AUGRUSequence>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

template <>
bool evaluate_node<ov::op::v5::GRUSequence>(std::shared_ptr<ov::Node> node,
                                            ov::TensorVector& outputs,
                                            const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v5::GRUSequence>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

template <>
bool evaluate_node<ov::op::v5::LSTMSequence>(std::shared_ptr<ov::Node> node,
                                             ov::TensorVector& outputs,
                                             const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v5::LSTMSequence>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

OPENVINO_SUPPRESS_DEPRECATED_START
template <>
bool evaluate_node<ov::op::v0::LSTMSequence>(std::shared_ptr<ov::Node> node,
                                             ov::TensorVector& outputs,
                                             const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v0::LSTMSequence>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
OPENVINO_SUPPRESS_DEPRECATED_END

template <>
bool evaluate_node<ov::op::v5::RNNSequence>(std::shared_ptr<ov::Node> node,
                                            ov::TensorVector& outputs,
                                            const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v5::RNNSequence>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
