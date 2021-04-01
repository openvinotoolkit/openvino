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

class TRANSFORMATIONS_API BidirectionalSequenceDecomposition;

class TRANSFORMATIONS_API BidirectionalLSTMSequenceDecomposition;
class TRANSFORMATIONS_API BidirectionalGRUSequenceDecomposition;
class TRANSFORMATIONS_API BidirectionalRNNSequenceDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose LSTMSequence to forward and reverse LSTMSequence.
 *
 */

class ngraph::pass::BidirectionalLSTMSequenceDecomposition : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BidirectionalLSTMSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose GRUSequence to forward and reverse GRUSequence.
 *
 */

class ngraph::pass::BidirectionalGRUSequenceDecomposition : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BidirectionalGRUSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose RNNSequence to forward and reverse RNNSequence.
 *
 */

class ngraph::pass::BidirectionalRNNSequenceDecomposition : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BidirectionalRNNSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Container for all types of sequences decomposition.
 *
 */

class ngraph::pass::BidirectionalSequenceDecomposition : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    BidirectionalSequenceDecomposition() {
        add_matcher<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
        add_matcher<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
        add_matcher<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
    }
};
