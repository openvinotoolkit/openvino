// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/lstm_sequence.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LSTMSequenceToMultiLSTMSequenceFusion;

}  // namespace pass
}  // namespace ov
bool is_equal_cells(const std::shared_ptr<ov::op::v5::LSTMSequence>&, const std::shared_ptr<ov::op::v5::LSTMSequence>&);
std::shared_ptr<ov::op::v5::LSTMSequence> find_lstm_chain(ov::pass::NodeRegistry&,
                                                          ov::pass::NodeRegistry&,
                                                          const std::shared_ptr<ov::op::v5::LSTMSequence>&,
                                                          ov::OutputVector&,
                                                          ov::OutputVector&,
                                                          ov::OutputVector&,
                                                          std::map<int, ov::Output<ov::Node>>&,
                                                          int&,
                                                          const std::shared_ptr<ov::Node>&);
bool create_sequence(ov::pass::NodeRegistry&,
                     const std::shared_ptr<ov::op::v5::LSTMSequence>&,
                     const std::shared_ptr<ov::op::v5::LSTMSequence>&,
                     const ov::OutputVector&,
                     const ov::OutputVector&,
                     const ov::OutputVector&,
                     const std::map<int, ov::Output<ov::Node>>&,
                     int,
                     const std::shared_ptr<ov::Node>&,
                     const std::shared_ptr<ov::Node>&);
/**
 * @ingroup ie_transformation_common_api
 * @brief LSTMSequenceToMultiLSTMSequenceFusion transformation replaces a sequence of
 * LSTMSequence operations with a MultiLSTMSequence operator.
 */
class ov::pass::LSTMSequenceToMultiLSTMSequenceFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("LSTMSequenceToMultiLSTMSequenceFusion", "0");
    LSTMSequenceToMultiLSTMSequenceFusion();
};
