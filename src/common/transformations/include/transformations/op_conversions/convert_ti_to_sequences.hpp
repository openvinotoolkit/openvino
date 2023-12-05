// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertTensorIteratorToLSTMSequence;
class TRANSFORMATIONS_API ConvertTensorIteratorToRNNSequence;
class TRANSFORMATIONS_API ConvertTensorIteratorToGRUSequence;
class TRANSFORMATIONS_API ConvertTensorIteratorToSequence;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Finds all TensorIterator layers, detects the pattern Squeeze->LSTMCell->Unsqueeze in the TensorIterator body,
 * converts this pattern to LSTMSequence layer and replaces them TensorIterator.
 */

class ov::pass::ConvertTensorIteratorToLSTMSequence : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTensorIteratorToLSTMSequence", "0");
    ConvertTensorIteratorToLSTMSequence();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Finds all TensorIterator layers, detects the pattern Squeeze->RNNCell->Unsqueeze in the TensorIterator body,
 * converts this pattern to RNNSequence layer and replaces them TensorIterator.
 */

class ov::pass::ConvertTensorIteratorToRNNSequence : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTensorIteratorToRNNSequence", "0");
    ConvertTensorIteratorToRNNSequence();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Finds all TensorIterator layers, detects the pattern Squeeze->GRUCell->Unsqueeze in the TensorIterator body,
 * converts this pattern to GRUSequence layer and replaces them TensorIterator.
 */

class ov::pass::ConvertTensorIteratorToGRUSequence : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertTensorIteratorToGRUSequence", "0");
    ConvertTensorIteratorToGRUSequence();
};

class ov::pass::ConvertTensorIteratorToSequence : public GraphRewrite {
public:
    OPENVINO_RTTI("ConvertTensorIteratorToSequence", "0");
    ConvertTensorIteratorToSequence();
};
