// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertLSTMSequenceMatcher;
class TRANSFORMATIONS_API ConvertGRUSequenceMatcher;
class TRANSFORMATIONS_API ConvertRNNSequenceMatcher;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts opset4::(GRU/RNN/LSTM)Sequence to legacy (GRU/RNN/LSTM)SequenceIE.
 * SequenceIE op doesn't take seq_length input and doesn't use num_direction (direction) variable.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ngraph::pass::ConvertLSTMSequenceMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertLSTMSequenceMatcher();
};

class ngraph::pass::ConvertGRUSequenceMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertGRUSequenceMatcher();
};

class ngraph::pass::ConvertRNNSequenceMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertRNNSequenceMatcher();
};
