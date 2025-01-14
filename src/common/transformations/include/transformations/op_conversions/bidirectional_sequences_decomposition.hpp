// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BidirectionalSequenceDecomposition;

class TRANSFORMATIONS_API BidirectionalLSTMSequenceDecomposition;
class TRANSFORMATIONS_API BidirectionalGRUSequenceDecomposition;
class TRANSFORMATIONS_API BidirectionalRNNSequenceDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Decompose LSTMSequence to forward and reverse LSTMSequence.
 *
 */

class ov::pass::BidirectionalLSTMSequenceDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BidirectionalLSTMSequenceDecomposition");
    BidirectionalLSTMSequenceDecomposition();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Decompose GRUSequence to forward and reverse GRUSequence.
 *
 */

class ov::pass::BidirectionalGRUSequenceDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BidirectionalGRUSequenceDecomposition");
    BidirectionalGRUSequenceDecomposition();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Decompose RNNSequence to forward and reverse RNNSequence.
 *
 */

class ov::pass::BidirectionalRNNSequenceDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BidirectionalRNNSequenceDecomposition");
    BidirectionalRNNSequenceDecomposition();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Container for all types of sequences decomposition.
 *
 */

class ov::pass::BidirectionalSequenceDecomposition : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("BidirectionalSequenceDecomposition");
    BidirectionalSequenceDecomposition() {
        add_matcher<ov::pass::BidirectionalLSTMSequenceDecomposition>();
        add_matcher<ov::pass::BidirectionalGRUSequenceDecomposition>();
        add_matcher<ov::pass::BidirectionalRNNSequenceDecomposition>();
    }
};
