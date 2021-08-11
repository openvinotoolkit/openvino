// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BidirectionalSequenceDecomposition;

class TRANSFORMATIONS_API BidirectionalLSTMSequenceDecomposition;
class TRANSFORMATIONS_API BidirectionalGRUSequenceDecomposition;
class TRANSFORMATIONS_API BidirectionalRNNSequenceDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose LSTMSequence to forward and reverse LSTMSequence.
 *
 */

class ov::pass::BidirectionalLSTMSequenceDecomposition : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BidirectionalLSTMSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose GRUSequence to forward and reverse GRUSequence.
 *
 */

class ov::pass::BidirectionalGRUSequenceDecomposition : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BidirectionalGRUSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose RNNSequence to forward and reverse RNNSequence.
 *
 */

class ov::pass::BidirectionalRNNSequenceDecomposition : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BidirectionalRNNSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Container for all types of sequences decomposition.
 *
 */

class ov::pass::BidirectionalSequenceDecomposition : public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    BidirectionalSequenceDecomposition() {
        add_matcher<ov::pass::BidirectionalLSTMSequenceDecomposition>();
        add_matcher<ov::pass::BidirectionalGRUSequenceDecomposition>();
        add_matcher<ov::pass::BidirectionalRNNSequenceDecomposition>();
    }
};
