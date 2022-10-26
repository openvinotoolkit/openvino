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

class TRANSFORMATIONS_API ConvertRNNSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertGRUSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertLSTMSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertSequenceToTensorIterator;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertRNNSequenceToTensorIterator transformation converts RNNSequence layer to TensorIterator
 * *
 */

class ngraph::pass::ConvertRNNSequenceToTensorIterator : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertRNNSequenceToTensorIterator", "0");
    ConvertRNNSequenceToTensorIterator();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGRUSequenceToTensorIterator transformation converts GRUSequence layer to TensorIterator
 * *
 */

class ngraph::pass::ConvertGRUSequenceToTensorIterator : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGRUSequenceToTensorIterator", "0");
    ConvertGRUSequenceToTensorIterator();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertLSTMSequenceToTensorIterator transformation converts LSTMSequence layer to TensorIterator
 * *
 */

class ngraph::pass::ConvertLSTMSequenceToTensorIterator : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertLSTMSequenceToTensorIterator", "0");
    ConvertLSTMSequenceToTensorIterator();
};

class ngraph::pass::ConvertSequenceToTensorIterator : public GraphRewrite {
public:
    OPENVINO_RTTI("ConvertSequenceToTensorIterator", "0");
    ConvertSequenceToTensorIterator();
};
