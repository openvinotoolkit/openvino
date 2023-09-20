// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "openvino/op/lstm_sequence.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeLSTM(const std::vector<ov::Output<Node>>& in,
                                   const std::vector<ov::Shape>& constants,
                                   std::size_t hidden_size,
                                   const std::vector<std::string>& activations,
                                   const std::vector<float>& activations_alpha,
                                   const std::vector<float>& activations_beta,
                                   float clip,
                                   bool make_sequence,
                                   ov::op::RecurrentSequenceDirection direction,
                                   ov::test::utils::SequenceTestsMode mode,
                                   float WRB_range) {
    std::vector<float> empty;
    auto W = ngraph::builder::makeConstant(in[0].get_element_type(), constants[0], empty, true);
    auto R = ngraph::builder::makeConstant(in[0].get_element_type(), constants[1], empty, true);
    auto B = ngraph::builder::makeConstant(in[0].get_element_type(), constants[2], empty, true);

    if (WRB_range > 0) {
        W = ngraph::builder::makeConstant(in[0].get_element_type(), constants[0], empty, true, -WRB_range, WRB_range);
        R = ngraph::builder::makeConstant(in[0].get_element_type(), constants[1], empty, true, -WRB_range, WRB_range);
        B = ngraph::builder::makeConstant(in[0].get_element_type(), constants[2], empty, true, -WRB_range, WRB_range);
    }
    if (!make_sequence) {
        return std::make_shared<ov::op::v4::LSTMCell>(in[0],
                                                      in[1],
                                                      in[2],
                                                      W,
                                                      R,
                                                      B,
                                                      hidden_size,
                                                      activations,
                                                      activations_alpha,
                                                      activations_beta,
                                                      clip);
    } else {
        if (in.size() > 3 && in[3].get_partial_shape().is_dynamic()) {
            return std::make_shared<ov::op::v5::LSTMSequence>(in[0],
                                                              in[1],
                                                              in[2],
                                                              in[3],
                                                              W,
                                                              R,
                                                              B,
                                                              hidden_size,
                                                              direction,
                                                              activations_alpha,
                                                              activations_beta,
                                                              activations,
                                                              clip);
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
            case ov::test::utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM:
            case ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM:
            case ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM: {
                // Seq_lengths should be as a Parameter node for these two modes
                seq_lengths = in.at(3).get_node_shared_ptr();
                break;
            }
            default:
                throw std::runtime_error("Incorrect mode for creation of Sequence operation");
            }
            return std::make_shared<ov::op::v5::LSTMSequence>(in[0],
                                                              in[1],
                                                              in[2],
                                                              seq_lengths,
                                                              W,
                                                              R,
                                                              B,
                                                              hidden_size,
                                                              direction,
                                                              activations_alpha,
                                                              activations_beta,
                                                              activations,
                                                              clip);
        }
    }
}
}  // namespace builder
}  // namespace ngraph
