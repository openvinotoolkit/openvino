// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LSTMSequenceToMultiLSTMSequenceFusion;

}  // namespace pass
}  // namespace ov

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
