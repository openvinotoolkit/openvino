// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/ctc_greedy_decoder_seq_len.hpp"

#include "evaluate_node.hpp"

namespace ctc_greedy_decoder_v6 {
template <ngraph::element::Type_t T1, ngraph::element::Type_t T2, ngraph::element::Type_t TOUT>
inline void evaluate(const std::shared_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using TF = typename ngraph::element_type_traits<T1>::value_type;
    using TI = typename ngraph::element_type_traits<T2>::value_type;
    using TIND1 = typename ngraph::element_type_traits<TOUT>::value_type;
    TI blank_index_val = static_cast<TI>(inputs[0]->get_shape().back() - 1);
    const TI* blank_index = &blank_index_val;
    if (inputs.size() == 3) {
        blank_index = inputs[2]->get_data_ptr<const TI>();
    }
    if (op->get_sequence_length_type() == ngraph::element::i32) {
        ngraph::reference::ctc_greedy_decoder_seq_len<TF>(inputs[0]->get_data_ptr<const TF>(),
                                                          inputs[1]->get_data_ptr<const TI>(),
                                                          blank_index,
                                                          outputs[0]->get_data_ptr<TIND1>(),
                                                          outputs[1]->get_data_ptr<int32_t>(),
                                                          inputs[0]->get_shape(),
                                                          outputs[0]->get_shape(),
                                                          op->get_merge_repeated());
    } else if (op->get_sequence_length_type() == ngraph::element::i64) {
        ngraph::reference::ctc_greedy_decoder_seq_len<TF>(inputs[0]->get_data_ptr<const TF>(),
                                                          inputs[1]->get_data_ptr<const TI>(),
                                                          blank_index,
                                                          outputs[0]->get_data_ptr<TIND1>(),
                                                          outputs[1]->get_data_ptr<int64_t>(),
                                                          inputs[0]->get_shape(),
                                                          outputs[0]->get_shape(),
                                                          op->get_merge_repeated());
    }
}
}  // namespace ctc_greedy_decoder_v6
template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto& dataType = inputs[0]->get_element_type();
    const auto& seqLenType = inputs[1]->get_element_type();
    if (dataType == ngraph::element::Type_t::f16 && seqLenType == ngraph::element::Type_t::i32) {
        ctc_greedy_decoder_v6::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::i32, ET>(op,
                                                                                                        outputs,
                                                                                                        inputs);
    } else if (dataType == ngraph::element::Type_t::f32 && seqLenType == ngraph::element::Type_t::i32) {
        ctc_greedy_decoder_v6::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::i32, ET>(op,
                                                                                                        outputs,
                                                                                                        inputs);
    } else if (dataType == ngraph::element::Type_t::f64 && seqLenType == ngraph::element::Type_t::i32) {
        ctc_greedy_decoder_v6::evaluate<ngraph::element::Type_t::f64, ngraph::element::Type_t::i32, ET>(op,
                                                                                                        outputs,
                                                                                                        inputs);
    } else if (dataType == ngraph::element::Type_t::f16 && seqLenType == ngraph::element::Type_t::i64) {
        ctc_greedy_decoder_v6::evaluate<ngraph::element::Type_t::f16, ngraph::element::Type_t::i64, ET>(op,
                                                                                                        outputs,
                                                                                                        inputs);
    } else if (dataType == ngraph::element::Type_t::f32 && seqLenType == ngraph::element::Type_t::i64) {
        ctc_greedy_decoder_v6::evaluate<ngraph::element::Type_t::f32, ngraph::element::Type_t::i64, ET>(op,
                                                                                                        outputs,
                                                                                                        inputs);
    } else if (dataType == ngraph::element::Type_t::f64 && seqLenType == ngraph::element::Type_t::i64) {
        ctc_greedy_decoder_v6::evaluate<ngraph::element::Type_t::f64, ngraph::element::Type_t::i64, ET>(op,
                                                                                                        outputs,
                                                                                                        inputs);
    } else {
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ngraph::op::v6::CTCGreedyDecoderSeqLen>(std::shared_ptr<ngraph::Node> node,
                                                           const ngraph::HostTensorVector& outputs,
                                                           const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                       outputs,
                                                       inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                     outputs,
                                                     inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                      outputs,
                                                      inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>(node),
                                                      outputs,
                                                      inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
