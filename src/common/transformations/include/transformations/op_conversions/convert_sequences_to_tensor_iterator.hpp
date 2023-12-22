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

class TRANSFORMATIONS_API ConvertRNNSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertGRUSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertLSTMSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertSequenceToTensorIterator;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertRNNSequenceToTensorIterator transformation converts RNNSequence layer to TensorIterator
 * *
 */

class ov::pass::ConvertRNNSequenceToTensorIterator : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertRNNSequenceToTensorIterator", "0");
    ConvertRNNSequenceToTensorIterator();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGRUSequenceToTensorIterator transformation converts GRUSequence layer to TensorIterator
 * *
 */

class ov::pass::ConvertGRUSequenceToTensorIterator : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGRUSequenceToTensorIterator", "0");
    ConvertGRUSequenceToTensorIterator();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertLSTMSequenceToTensorIterator transformation converts LSTMSequence layer to TensorIterator
 * *
 */

class ov::pass::ConvertLSTMSequenceToTensorIterator : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertLSTMSequenceToTensorIterator", "0");
    ConvertLSTMSequenceToTensorIterator();
};

class ov::pass::ConvertSequenceToTensorIterator : public GraphRewrite {
public:
    OPENVINO_RTTI("ConvertSequenceToTensorIterator", "0");
    ConvertSequenceToTensorIterator();
};
