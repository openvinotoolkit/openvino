// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/sequences.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reverse_sequence.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"

namespace gru_seq_v5 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v5::GRUSequence>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::gru_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                     inputs[0]->get_shape(),
                                                     inputs[1]->get_data_ptr<char>(),
                                                     inputs[1]->get_shape(),
                                                     inputs[2]->get_data_ptr<char>(),
                                                     inputs[2]->get_shape(),
                                                     inputs[3]->get_data_ptr<char>(),
                                                     inputs[3]->get_shape(),
                                                     inputs[4]->get_data_ptr<char>(),
                                                     inputs[4]->get_shape(),
                                                     inputs[5]->get_data_ptr<char>(),
                                                     inputs[5]->get_shape(),
                                                     outputs[0]->get_data_ptr<char>(),
                                                     outputs[1]->get_data_ptr<char>(),
                                                     op->get_activations()[0],
                                                     op->get_activations()[1],
                                                     op->get_clip(),
                                                     op->get_direction(),
                                                     op->get_linear_before_reset());
}
}  // namespace gru_seq_v5

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v5::GRUSequence>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    switch (inputs[2]->get_element_type()) {
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u64:
        gru_seq_v5::evaluate<ET, ov::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i32:
    case ov::element::Type_t::u32:
        gru_seq_v5::evaluate<ET, ov::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace augru_seq {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::internal::AUGRUSequence>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::gru_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                     inputs[0]->get_shape(),
                                                     inputs[1]->get_data_ptr<char>(),
                                                     inputs[1]->get_shape(),
                                                     inputs[2]->get_data_ptr<char>(),
                                                     inputs[2]->get_shape(),
                                                     inputs[3]->get_data_ptr<char>(),
                                                     inputs[3]->get_shape(),
                                                     inputs[4]->get_data_ptr<char>(),
                                                     inputs[4]->get_shape(),
                                                     inputs[5]->get_data_ptr<char>(),
                                                     inputs[5]->get_shape(),
                                                     outputs[0]->get_data_ptr<char>(),
                                                     outputs[1]->get_data_ptr<char>(),
                                                     op->get_activations()[0],
                                                     op->get_activations()[1],
                                                     op->get_clip(),
                                                     op->get_direction(),
                                                     op->get_linear_before_reset(),
                                                     inputs[6]->get_data_ptr<char>());
}
}  // namespace augru_seq

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::internal::AUGRUSequence>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    switch (inputs[2]->get_element_type()) {
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u64:
        augru_seq::evaluate<ET, ov::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i32:
    case ov::element::Type_t::u32:
        augru_seq::evaluate<ET, ov::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::Relu>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::relu<T>(inputs[0]->get_data_ptr<T>(),
                                        outputs[0]->get_data_ptr<T>(),
                                        ov::shape_size(inputs[0]->get_shape()));
    return true;
}

namespace reverse_sequence_v0 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v0::ReverseSequence>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::reverse_sequence<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                                         outputs[0]->get_data_ptr<T1>(),
                                                         inputs[0]->get_shape(),
                                                         op->get_batch_axis(),
                                                         op->get_sequence_axis(),
                                                         inputs[1]->get_data_ptr<T2>());
}
}  // namespace reverse_sequence_v0

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::ReverseSequence>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ov::element::Type_t::boolean:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::boolean>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i8:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::i8>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i16:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::i16>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i32:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::i32>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i64:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u8:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::u8>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u16:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::u16>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u32:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::u32>(op, outputs, inputs);
        break;
    case ov::element::Type_t::u64:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::u64>(op, outputs, inputs);
        break;
    case ov::element::Type_t::f16:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::f16>(op, outputs, inputs);
        break;
    case ov::element::Type_t::f32:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::f32>(op, outputs, inputs);
        break;
    case ov::element::Type_t::f64:
        reverse_sequence_v0::evaluate<ET, ov::element::Type_t::f64>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::RNNCell>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::rnn_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                            inputs[0]->get_shape(),
                                            inputs[1]->get_data_ptr<ET>(),
                                            inputs[1]->get_shape(),
                                            inputs[2]->get_data_ptr<ET>(),
                                            inputs[2]->get_shape(),
                                            inputs[3]->get_data_ptr<ET>(),
                                            inputs[3]->get_shape(),
                                            inputs[4]->get_data_ptr<ET>(),
                                            inputs[4]->get_shape(),
                                            outputs[0]->get_data_ptr<ET>(),
                                            op->get_activations().front(),
                                            op->get_clip());
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::LSTMCell>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::lstm_cell_v1<T>(inputs[0]->get_data_ptr<ET>(),
                                                inputs[0]->get_shape(),
                                                inputs[1]->get_data_ptr<ET>(),
                                                inputs[1]->get_shape(),
                                                inputs[2]->get_data_ptr<ET>(),
                                                inputs[2]->get_shape(),
                                                inputs[3]->get_data_ptr<ET>(),
                                                inputs[3]->get_shape(),
                                                inputs[4]->get_data_ptr<ET>(),
                                                inputs[4]->get_shape(),
                                                inputs[5]->get_data_ptr<ET>(),
                                                inputs[5]->get_shape(),
                                                inputs[6]->get_data_ptr<ET>(),
                                                inputs[6]->get_shape(),
                                                outputs[0]->get_data_ptr<ET>(),
                                                outputs[1]->get_data_ptr<ET>(),
                                                op->get_activations()[0],
                                                op->get_activations()[1],
                                                op->get_activations()[2],
                                                op->get_clip(),
                                                op->get_weights_format(),
                                                op->get_input_forget());
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v4::LSTMCell>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::lstm_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                             inputs[0]->get_shape(),
                                             inputs[1]->get_data_ptr<ET>(),
                                             inputs[1]->get_shape(),
                                             inputs[2]->get_data_ptr<ET>(),
                                             inputs[2]->get_shape(),
                                             inputs[3]->get_data_ptr<ET>(),
                                             inputs[3]->get_shape(),
                                             inputs[4]->get_data_ptr<ET>(),
                                             inputs[4]->get_shape(),
                                             inputs[5]->get_data_ptr<ET>(),
                                             inputs[5]->get_shape(),
                                             outputs[0]->get_data_ptr<ET>(),
                                             outputs[1]->get_data_ptr<ET>(),
                                             op->get_activations()[0],
                                             op->get_activations()[1],
                                             op->get_activations()[2],
                                             op->get_clip());
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v3::GRUCell>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::gru_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                            inputs[0]->get_shape(),
                                            inputs[1]->get_data_ptr<ET>(),
                                            inputs[1]->get_shape(),
                                            inputs[2]->get_data_ptr<ET>(),
                                            inputs[2]->get_shape(),
                                            inputs[3]->get_data_ptr<ET>(),
                                            inputs[3]->get_shape(),
                                            inputs[4]->get_data_ptr<ET>(),
                                            inputs[4]->get_shape(),
                                            outputs[0]->get_data_ptr<ET>(),
                                            op->get_activations()[0],
                                            op->get_activations()[1],
                                            op->get_clip(),
                                            op->get_linear_before_reset());
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::internal::AUGRUCell>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::gru_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                            inputs[0]->get_shape(),
                                            inputs[1]->get_data_ptr<ET>(),
                                            inputs[1]->get_shape(),
                                            inputs[2]->get_data_ptr<ET>(),
                                            inputs[2]->get_shape(),
                                            inputs[3]->get_data_ptr<ET>(),
                                            inputs[3]->get_shape(),
                                            inputs[4]->get_data_ptr<ET>(),
                                            inputs[4]->get_shape(),
                                            outputs[0]->get_data_ptr<ET>(),
                                            op->get_activations()[0],
                                            op->get_activations()[1],
                                            op->get_clip(),
                                            op->get_linear_before_reset(),
                                            inputs[5]->get_data_ptr<ET>());
    return true;
}

namespace rnn_seq_v5 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v5::RNNSequence>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::rnn_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                     inputs[0]->get_shape(),
                                                     inputs[1]->get_data_ptr<char>(),
                                                     inputs[1]->get_shape(),
                                                     inputs[2]->get_data_ptr<char>(),
                                                     inputs[2]->get_shape(),
                                                     inputs[3]->get_data_ptr<char>(),
                                                     inputs[3]->get_shape(),
                                                     inputs[4]->get_data_ptr<char>(),
                                                     inputs[4]->get_shape(),
                                                     inputs[5]->get_data_ptr<char>(),
                                                     inputs[5]->get_shape(),
                                                     outputs[0]->get_data_ptr<char>(),
                                                     outputs[1]->get_data_ptr<char>(),
                                                     op->get_activations()[0],
                                                     op->get_clip(),
                                                     op->get_direction());
}
}  // namespace rnn_seq_v5

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v5::RNNSequence>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    switch (inputs[2]->get_element_type()) {
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u64:
        rnn_seq_v5::evaluate<ET, ov::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i32:
    case ov::element::Type_t::u32:
        rnn_seq_v5::evaluate<ET, ov::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace lstm_seq_v1 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v0::LSTMSequence>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::lstm_sequence_v1<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                         inputs[0]->get_shape(),
                                                         inputs[1]->get_data_ptr<char>(),
                                                         inputs[1]->get_shape(),
                                                         inputs[2]->get_data_ptr<char>(),
                                                         inputs[2]->get_shape(),
                                                         inputs[3]->get_data_ptr<char>(),
                                                         inputs[3]->get_shape(),
                                                         inputs[4]->get_data_ptr<char>(),
                                                         inputs[4]->get_shape(),
                                                         inputs[5]->get_data_ptr<char>(),
                                                         inputs[5]->get_shape(),
                                                         inputs[6]->get_data_ptr<char>(),
                                                         inputs[6]->get_shape(),
                                                         inputs[7]->get_data_ptr<char>(),
                                                         inputs[7]->get_shape(),
                                                         outputs[0]->get_data_ptr<char>(),
                                                         outputs[1]->get_data_ptr<char>(),
                                                         outputs[2]->get_data_ptr<char>(),
                                                         op->get_activations()[0],
                                                         op->get_activations()[1],
                                                         op->get_activations()[2],
                                                         op->get_clip_threshold(),
                                                         op->get_weights_format(),
                                                         op->get_input_forget(),
                                                         op->get_direction());
}
}  // namespace lstm_seq_v1

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::LSTMSequence>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    switch (inputs[3]->get_element_type()) {
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u64:
        lstm_seq_v1::evaluate<ET, ov::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i32:
    case ov::element::Type_t::u32:
        lstm_seq_v1::evaluate<ET, ov::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace lstm_seq_v5 {
template <ov::element::Type_t t1, ov::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::v5::LSTMSequence>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    using T1 = typename ov::element_type_traits<t1>::value_type;
    using T2 = typename ov::element_type_traits<t2>::value_type;
    ngraph::runtime::reference::lstm_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                      inputs[0]->get_shape(),
                                                      inputs[1]->get_data_ptr<char>(),
                                                      inputs[1]->get_shape(),
                                                      inputs[2]->get_data_ptr<char>(),
                                                      inputs[2]->get_shape(),
                                                      inputs[3]->get_data_ptr<char>(),
                                                      inputs[3]->get_shape(),
                                                      inputs[4]->get_data_ptr<char>(),
                                                      inputs[4]->get_shape(),
                                                      inputs[5]->get_data_ptr<char>(),
                                                      inputs[5]->get_shape(),
                                                      inputs[6]->get_data_ptr<char>(),
                                                      inputs[6]->get_shape(),
                                                      outputs[0]->get_data_ptr<char>(),
                                                      outputs[1]->get_data_ptr<char>(),
                                                      outputs[2]->get_data_ptr<char>(),
                                                      op->get_activations()[0],
                                                      op->get_activations()[1],
                                                      op->get_activations()[2],
                                                      op->get_clip(),
                                                      op->get_direction());
}
}  // namespace lstm_seq_v5

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v5::LSTMSequence>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    switch (inputs[3]->get_element_type()) {
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u64:
        lstm_seq_v5::evaluate<ET, ov::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ov::element::Type_t::i32:
    case ov::element::Type_t::u32:
        lstm_seq_v5::evaluate<ET, ov::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}