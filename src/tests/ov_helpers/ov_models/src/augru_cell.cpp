// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/augru_cell.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"
#include "ov_ops/augru_sequence.hpp"

namespace ngraph {
namespace builder {

/**
 * There are 2 options to paramter "in" when "make_sequence" is true.
 * 0          1               2           3
 * X   init_hidden_state  attention    seq_length
 * or,
 * 0          1               2
 * X   init_hidden_state   attention
 *
 */
std::shared_ptr<ov::Node> makeAUGRU(const OutputVector& in,
                                    const std::vector<ov::Shape>& constants,
                                    std::size_t hidden_size,
                                    bool make_sequence,
                                    ov::op::RecurrentSequenceDirection direction,
                                    ov::test::utils::SequenceTestsMode mode) {
    std::vector<float> empty;
    auto W = ngraph::builder::makeConstant(in[0].get_element_type(), constants[0], empty, true);
    W->set_friendly_name("augru_w");
    auto R = ngraph::builder::makeConstant(in[0].get_element_type(), constants[1], empty, true);
    R->set_friendly_name("augru_r");
    auto B = ngraph::builder::makeConstant(in[0].get_element_type(), constants[2], empty, true);
    B->set_friendly_name("augru_b");
    if (!make_sequence) {
        return std::make_shared<ov::op::internal::AUGRUCell>(in[0], in[1], W, R, B, in[2], hidden_size);
    } else {
        if (in.size() > 3 && in[3].get_partial_shape().is_dynamic()) {
            return std::make_shared<ov::op::internal::AUGRUSequence>(in[0], in[1], in[3], W, R, B, in[2], hidden_size);
        } else {
            std::shared_ptr<Node> seq_lengths;
            switch (mode) {
            case ov::test::utils::SequenceTestsMode::PURE_SEQ:
            case ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST: {
                std::vector<float> lengths(in[0].get_partial_shape()[0].get_min_length(),
                                           in[0].get_partial_shape()[1].get_min_length());
                seq_lengths = ngraph::builder::makeConstant(element::i64, constants[3], lengths, false);
                break;
            }
            case ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST:
            case ov::test::utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST: {
                for (size_t i = 0; i <= in[0].get_shape().at(0); ++i) {
                    std::vector<float> lengths;
                    seq_lengths = ngraph::builder::makeConstant(element::i64,
                                                                constants[3],
                                                                lengths,
                                                                true,
                                                                static_cast<float>(in[0].get_shape()[1]),
                                                                0.f);
                }
                break;
            }
            case ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM:
            case ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM:
            case ov::test::utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM: {
                // Seq_lengths should be as a Parameter node for these two modes
                if (in.size() < 4)
                    throw std::runtime_error("Incorrect number of inputs for creation of Sequence operation");
                seq_lengths = in.at(3).get_node_shared_ptr();
                break;
            }
            default:
                throw std::runtime_error("Incorrect mode for creation of Sequence operation");
            }
            return std::make_shared<ov::op::internal::AUGRUSequence>(in[0],
                                                                     in[1],
                                                                     seq_lengths,
                                                                     W,
                                                                     R,
                                                                     B,
                                                                     in[2],
                                                                     hidden_size);
        }
    }
}
}  // namespace builder
}  // namespace ngraph
