// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertTensorIteratorToLSTMSequence;
class TRANSFORMATIONS_API ConvertTensorIteratorToRNNSequence;
class TRANSFORMATIONS_API ConvertTensorIteratorToGRUSequence;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Finds all TensorIterator layers, detects the pattern Squeeze->LSTMCell->Unsqueeze in the TensorIterator body,
 * converts this pattern to LSTMSequence layer and replaces them TensorIterator.
 */

class ov::pass::ConvertTensorIteratorToLSTMSequence: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertTensorIteratorToLSTMSequence();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Finds all TensorIterator layers, detects the pattern Squeeze->RNNCell->Unsqueeze in the TensorIterator body,
 * converts this pattern to RNNSequence layer and replaces them TensorIterator.
 */

class ov::pass::ConvertTensorIteratorToRNNSequence: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertTensorIteratorToRNNSequence();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Finds all TensorIterator layers, detects the pattern Squeeze->GRUCell->Unsqueeze in the TensorIterator body,
 * converts this pattern to GRUSequence layer and replaces them TensorIterator.
 */

class ov::pass::ConvertTensorIteratorToGRUSequence: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertTensorIteratorToGRUSequence();
};
