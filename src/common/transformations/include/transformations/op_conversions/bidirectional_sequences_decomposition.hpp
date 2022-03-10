// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <string>
#include <transformations_visibility.hpp>
#include <vector>

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
    OPENVINO_RTTI("BidirectionalLSTMSequenceDecomposition", "0");
    BidirectionalLSTMSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose GRUSequence to forward and reverse GRUSequence.
 *
 */

class ngraph::pass::BidirectionalGRUSequenceDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("BidirectionalGRUSequenceDecomposition", "0");
    BidirectionalGRUSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose RNNSequence to forward and reverse RNNSequence.
 *
 */

class ngraph::pass::BidirectionalRNNSequenceDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("BidirectionalRNNSequenceDecomposition", "0");
    BidirectionalRNNSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Container for all types of sequences decomposition.
 *
 */

class ngraph::pass::BidirectionalSequenceDecomposition : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("BidirectionalSequenceDecomposition", "0");
    BidirectionalSequenceDecomposition() {
        add_matcher<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
        add_matcher<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
        add_matcher<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
    }
};
