// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/augru_cell.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"

namespace ov {
namespace test {
namespace utils {
/**
 * There are 2 options to paramter "in" when "make_sequence" is true.
 * 0          1               2           3
 * X   init_hidden_state  attention    seq_length
 * or,
 * 0          1               2
 * X   init_hidden_state   attention
 *
 */
std::shared_ptr<ov::Node> make_augru(const OutputVector& in,
                                     const std::vector<ov::Shape>& constants,
                                     std::size_t hidden_size,
                                     bool make_sequence,
                                     ov::op::RecurrentSequenceDirection direction,
                                     ov::test::utils::SequenceTestsMode mode) {
    auto w_tensor = ov::test::utils::create_and_fill_tensor(in[0].get_element_type(), constants[0]);
    auto W = std::make_shared<ov::op::v0::Constant>(w_tensor);
    W->set_friendly_name("augru_w");
    auto r_tensor = ov::test::utils::create_and_fill_tensor(in[0].get_element_type(), constants[1]);
    auto R = std::make_shared<ov::op::v0::Constant>(r_tensor);
    R->set_friendly_name("augru_r");
    auto b_tensor = ov::test::utils::create_and_fill_tensor(in[0].get_element_type(), constants[2]);
    auto B = std::make_shared<ov::op::v0::Constant>(b_tensor);
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
}  // namespace utils
}  // namespace test
}  // namespace ov
