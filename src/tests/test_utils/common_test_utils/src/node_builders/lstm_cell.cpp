// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/lstm_cell.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_lstm(const std::vector<ov::Output<Node>>& in,
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
    auto w_tensor = ov::test::utils::create_and_fill_tensor(in[0].get_element_type(), constants[0]);
    auto W = std::make_shared<ov::op::v0::Constant>(w_tensor);
    auto r_tensor = ov::test::utils::create_and_fill_tensor(in[0].get_element_type(), constants[1]);
    auto R = std::make_shared<ov::op::v0::Constant>(r_tensor);
    auto b_tensor = ov::test::utils::create_and_fill_tensor(in[0].get_element_type(), constants[2]);
    auto B = std::make_shared<ov::op::v0::Constant>(b_tensor);

    if (WRB_range > 0) {
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -WRB_range;
        in_data.range = 2 * WRB_range;
        w_tensor = ov::test::utils::create_and_fill_tensor(in[0].get_element_type(), constants[0], in_data);
        W = std::make_shared<ov::op::v0::Constant>(w_tensor);

        r_tensor = ov::test::utils::create_and_fill_tensor(in[0].get_element_type(), constants[1], in_data);
        R = std::make_shared<ov::op::v0::Constant>(r_tensor);

        b_tensor = ov::test::utils::create_and_fill_tensor(in[0].get_element_type(), constants[2], in_data);
        B = std::make_shared<ov::op::v0::Constant>(b_tensor);
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
                seq_lengths = std::make_shared<ov::op::v0::Constant>(element::i64, constants[3], lengths);
                break;
            }
            case ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST:
            case ov::test::utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST: {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = in[0].get_shape()[1];
                auto seq_lengths_tensor =
                    ov::test::utils::create_and_fill_tensor(ov::element::i64, constants[3], in_data);
                seq_lengths = std::make_shared<ov::op::v0::Constant>(seq_lengths_tensor);
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
}  // namespace utils
}  // namespace test
}  // namespace ov
