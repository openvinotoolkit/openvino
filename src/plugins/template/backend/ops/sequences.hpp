// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/sequences.hpp"
#include "ov_ops/augru_sequence.hpp"

namespace rnn_seq_v5 {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ngraph::op::v5::RNNSequence>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
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

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v5::RNNSequence>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[2]->get_element_type()) {
    case ngraph::element::Type_t::i64:
    case ngraph::element::Type_t::u64:
        rnn_seq_v5::evaluate<ET, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i32:
    case ngraph::element::Type_t::u32:
        rnn_seq_v5::evaluate<ET, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace lstm_seq_v1 {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ngraph::op::v0::LSTMSequence>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
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

namespace lstm_seq_v5 {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ngraph::op::v5::LSTMSequence>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
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

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::LSTMSequence>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[3]->get_element_type()) {
    case ngraph::element::Type_t::i64:
    case ngraph::element::Type_t::u64:
        lstm_seq_v1::evaluate<ET, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i32:
    case ngraph::element::Type_t::u32:
        lstm_seq_v1::evaluate<ET, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v5::LSTMSequence>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[3]->get_element_type()) {
    case ngraph::element::Type_t::i64:
    case ngraph::element::Type_t::u64:
        lstm_seq_v5::evaluate<ET, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i32:
    case ngraph::element::Type_t::u32:
        lstm_seq_v5::evaluate<ET, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace gru_seq_v5 {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ngraph::op::v5::GRUSequence>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
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

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v5::GRUSequence>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[2]->get_element_type()) {
    case ngraph::element::Type_t::i64:
    case ngraph::element::Type_t::u64:
        gru_seq_v5::evaluate<ET, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i32:
    case ngraph::element::Type_t::u32:
        gru_seq_v5::evaluate<ET, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace augru_seq {
template <ngraph::element::Type_t t1, ngraph::element::Type_t t2>
inline void evaluate(const std::shared_ptr<ov::op::internal::AUGRUSequence>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T1 = typename ngraph::element_type_traits<t1>::value_type;
    using T2 = typename ngraph::element_type_traits<t2>::value_type;
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

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::internal::AUGRUSequence>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[2]->get_element_type()) {
    case ngraph::element::Type_t::i64:
    case ngraph::element::Type_t::u64:
        augru_seq::evaluate<ET, ngraph::element::Type_t::i64>(op, outputs, inputs);
        break;
    case ngraph::element::Type_t::i32:
    case ngraph::element::Type_t::u32:
        augru_seq::evaluate<ET, ngraph::element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}
