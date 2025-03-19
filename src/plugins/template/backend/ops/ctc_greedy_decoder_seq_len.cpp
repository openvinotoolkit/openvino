// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/ctc_greedy_decoder_seq_len.hpp"

#include "evaluate_node.hpp"

namespace ctc_greedy_decoder_v6 {
template <ov::element::Type_t T1, ov::element::Type_t T2, ov::element::Type_t TOUT>
inline void evaluate(const std::shared_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using TF = typename ov::element_type_traits<T1>::value_type;
    using TI = typename ov::element_type_traits<T2>::value_type;
    using TIND1 = typename ov::element_type_traits<TOUT>::value_type;
    TI blank_index_val = static_cast<TI>(inputs[0].get_shape().back() - 1);
    const TI* blank_index = &blank_index_val;
    if (inputs.size() == 3) {
        blank_index = inputs[2].data<const TI>();
    }
    if (op->get_sequence_length_type() == ov::element::i32) {
        ov::reference::ctc_greedy_decoder_seq_len<TF>(inputs[0].data<const TF>(),
                                                      inputs[1].data<const TI>(),
                                                      blank_index,
                                                      outputs[0].data<TIND1>(),
                                                      outputs[1].data<int32_t>(),
                                                      inputs[0].get_shape(),
                                                      outputs[0].get_shape(),
                                                      op->get_merge_repeated());
    } else if (op->get_sequence_length_type() == ov::element::i64) {
        ov::reference::ctc_greedy_decoder_seq_len<TF>(inputs[0].data<const TF>(),
                                                      inputs[1].data<const TI>(),
                                                      blank_index,
                                                      outputs[0].data<TIND1>(),
                                                      outputs[1].data<int64_t>(),
                                                      inputs[0].get_shape(),
                                                      outputs[0].get_shape(),
                                                      op->get_merge_repeated());
    }
}
}  // namespace ctc_greedy_decoder_v6
template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    const auto& dataType = inputs[0].get_element_type();
    const auto& seqLenType = inputs[1].get_element_type();
    if (dataType == ov::element::f16 && seqLenType == ov::element::i32) {
        ctc_greedy_decoder_v6::evaluate<ov::element::f16, ov::element::i32, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::f32 && seqLenType == ov::element::i32) {
        ctc_greedy_decoder_v6::evaluate<ov::element::f32, ov::element::i32, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::f64 && seqLenType == ov::element::i32) {
        ctc_greedy_decoder_v6::evaluate<ov::element::f64, ov::element::i32, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::f16 && seqLenType == ov::element::i64) {
        ctc_greedy_decoder_v6::evaluate<ov::element::f16, ov::element::i64, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::f32 && seqLenType == ov::element::i64) {
        ctc_greedy_decoder_v6::evaluate<ov::element::f32, ov::element::i64, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::f64 && seqLenType == ov::element::i64) {
        ctc_greedy_decoder_v6::evaluate<ov::element::f64, ov::element::i64, ET>(op, outputs, inputs);
    } else {
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v6::CTCGreedyDecoderSeqLen>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node),
                                              outputs,
                                              inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
