// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

namespace ctc_greedy_decoder_v6 {
template <ov::element::Type_t T1, ov::element::Type_t T2, ov::element::Type_t TOUT>
inline void evaluate(const std::shared_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    using TF = typename ov::element_type_traits<T1>::value_type;
    using TI = typename ov::element_type_traits<T2>::value_type;
    using TIND1 = typename ov::element_type_traits<TOUT>::value_type;
    TI blank_index_val = static_cast<TI>(inputs[0]->get_shape().back() - 1);
    const TI* blank_index = &blank_index_val;
    if (inputs.size() == 3) {
        blank_index = inputs[2]->get_data_ptr<const TI>();
    }
    if (op->get_sequence_length_type() == ov::element::i32) {
        ngraph::runtime::reference::ctc_greedy_decoder_seq_len<TF>(inputs[0]->get_data_ptr<const TF>(),
                                                                   inputs[1]->get_data_ptr<const TI>(),
                                                                   blank_index,
                                                                   outputs[0]->get_data_ptr<TIND1>(),
                                                                   outputs[1]->get_data_ptr<int32_t>(),
                                                                   inputs[0]->get_shape(),
                                                                   outputs[0]->get_shape(),
                                                                   op->get_merge_repeated());
    } else if (op->get_sequence_length_type() == ov::element::i64) {
        ngraph::runtime::reference::ctc_greedy_decoder_seq_len<TF>(inputs[0]->get_data_ptr<const TF>(),
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

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    const auto& dataType = inputs[0]->get_element_type();
    const auto& seqLenType = inputs[1]->get_element_type();
    if (dataType == ov::element::Type_t::f16 && seqLenType == ov::element::Type_t::i32) {
        ctc_greedy_decoder_v6::evaluate<ov::element::Type_t::f16, ov::element::Type_t::i32, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::Type_t::f32 && seqLenType == ov::element::Type_t::i32) {
        ctc_greedy_decoder_v6::evaluate<ov::element::Type_t::f32, ov::element::Type_t::i32, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::Type_t::f64 && seqLenType == ov::element::Type_t::i32) {
        ctc_greedy_decoder_v6::evaluate<ov::element::Type_t::f64, ov::element::Type_t::i32, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::Type_t::f16 && seqLenType == ov::element::Type_t::i64) {
        ctc_greedy_decoder_v6::evaluate<ov::element::Type_t::f16, ov::element::Type_t::i64, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::Type_t::f32 && seqLenType == ov::element::Type_t::i64) {
        ctc_greedy_decoder_v6::evaluate<ov::element::Type_t::f32, ov::element::Type_t::i64, ET>(op, outputs, inputs);
    } else if (dataType == ov::element::Type_t::f64 && seqLenType == ov::element::Type_t::i64) {
        ctc_greedy_decoder_v6::evaluate<ov::element::Type_t::f64, ov::element::Type_t::i64, ET>(op, outputs, inputs);
    } else {
        return false;
    }
    return true;
}