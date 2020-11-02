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
    BidirectionalLSTMSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose GRUSequence to forward and reverse GRUSequence.
 *
 */

class ngraph::pass::BidirectionalGRUSequenceDecomposition : public ngraph::pass::MatcherPass {
public:
    BidirectionalGRUSequenceDecomposition();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Decompose RNNSequence to forward and reverse RNNSequence.
 *
 */

class ngraph::pass::BidirectionalRNNSequenceDecomposition : public ngraph::pass::MatcherPass {
public:
    BidirectionalRNNSequenceDecomposition();
};
