// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <string>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertLSTMSequenceMatcher;
class ConvertGRUSequenceMatcher;
class ConvertRNNSequenceMatcher;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts LSTMSequence to legacy LSTMSequenceIE.
 * SequenceIE op doesn't use seq_length input and num_direction (direction) attribute.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ngraph::pass::ConvertLSTMSequenceMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertLSTMSequenceMatcher", "0");
    ConvertLSTMSequenceMatcher();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts GRUSequence to legacy GRUSequenceIE.
 * SequenceIE op doesn't use seq_length input and num_direction (direction) attribute.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ngraph::pass::ConvertGRUSequenceMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGRUSequenceMatcher", "0");
    ConvertGRUSequenceMatcher();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts RNNSequence to legacy RNNSequenceIE.
 * SequenceIE op doesn't use seq_length input and num_direction (direction) attribute.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ngraph::pass::ConvertRNNSequenceMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertRNNSequenceMatcher", "0");
    ConvertRNNSequenceMatcher();
};
