// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertTensorIteratorToLSTMSequence;
class TRANSFORMATIONS_API ConvertTensorIteratorToRNNSequence;
class TRANSFORMATIONS_API ConvertTensorIteratorToGRUSequence;
class TRANSFORMATIONS_API ConvertTensorIteratorToSequence;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Finds all TensorIterator layers, detects the pattern Squeeze->LSTMCell->Unsqueeze in the TensorIterator body,
 * converts this pattern to LSTMSequence layer and replaces them TensorIterator.
 */

class ngraph::pass::ConvertTensorIteratorToLSTMSequence : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTensorIteratorToLSTMSequence", "0");
    ConvertTensorIteratorToLSTMSequence();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Finds all TensorIterator layers, detects the pattern Squeeze->RNNCell->Unsqueeze in the TensorIterator body,
 * converts this pattern to RNNSequence layer and replaces them TensorIterator.
 */

class ngraph::pass::ConvertTensorIteratorToRNNSequence : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTensorIteratorToRNNSequence", "0");
    ConvertTensorIteratorToRNNSequence();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Finds all TensorIterator layers, detects the pattern Squeeze->GRUCell->Unsqueeze in the TensorIterator body,
 * converts this pattern to GRUSequence layer and replaces them TensorIterator.
 */

class ngraph::pass::ConvertTensorIteratorToGRUSequence : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTensorIteratorToGRUSequence", "0");
    ConvertTensorIteratorToGRUSequence();
};

class ngraph::pass::ConvertTensorIteratorToSequence : public GraphRewrite {
public:
    OPENVINO_RTTI("ConvertTensorIteratorToSequence", "0");
    ConvertTensorIteratorToSequence();
};
