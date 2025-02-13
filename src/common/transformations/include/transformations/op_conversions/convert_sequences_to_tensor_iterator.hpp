// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertRNNSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertGRUSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertLSTMSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertSequenceToTensorIterator;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertRNNSequenceToTensorIterator transformation converts RNNSequence layer to TensorIterator
 * *
 */

class ov::pass::ConvertRNNSequenceToTensorIterator : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertRNNSequenceToTensorIterator");
    ConvertRNNSequenceToTensorIterator();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertGRUSequenceToTensorIterator transformation converts GRUSequence layer to TensorIterator
 * *
 */

class ov::pass::ConvertGRUSequenceToTensorIterator : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGRUSequenceToTensorIterator");
    ConvertGRUSequenceToTensorIterator();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertLSTMSequenceToTensorIterator transformation converts LSTMSequence layer to TensorIterator
 * *
 */

class ov::pass::ConvertLSTMSequenceToTensorIterator : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertLSTMSequenceToTensorIterator");
    ConvertLSTMSequenceToTensorIterator();
};

class ov::pass::ConvertSequenceToTensorIterator : public GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertSequenceToTensorIterator");
    ConvertSequenceToTensorIterator();
};
