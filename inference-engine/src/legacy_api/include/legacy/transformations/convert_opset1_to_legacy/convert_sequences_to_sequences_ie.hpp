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
 * @brief Converts LSTMSequence to legacy LSTMSequenceIE.
 * SequenceIE op doesn't use seq_length input and num_direction (direction) attribute.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ngraph::pass::ConvertLSTMSequenceMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertLSTMSequenceMatcher();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts GRUSequence to legacy GRUSequenceIE.
 * SequenceIE op doesn't use seq_length input and num_direction (direction) attribute.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ngraph::pass::ConvertGRUSequenceMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertGRUSequenceMatcher();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts RNNSequence to legacy RNNSequenceIE.
 * SequenceIE op doesn't use seq_length input and num_direction (direction) attribute.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ngraph::pass::ConvertRNNSequenceMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertRNNSequenceMatcher();
};
