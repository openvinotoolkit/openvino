// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <openvino/core/visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API ConvertRNNSequenceToTensorIterator;
class OPENVINO_API ConvertGRUSequenceToTensorIterator;
class OPENVINO_API ConvertLSTMSequenceToTensorIterator;
class OPENVINO_API ConvertSequenceToTensorIterator;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertRNNSequenceToTensorIterator transformation converts RNNSequence layer to TensorIterator
 * *
 */

class ngraph::pass::ConvertRNNSequenceToTensorIterator: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertRNNSequenceToTensorIterator();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGRUSequenceToTensorIterator transformation converts GRUSequence layer to TensorIterator
 * *
 */

class ngraph::pass::ConvertGRUSequenceToTensorIterator: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGRUSequenceToTensorIterator();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertLSTMSequenceToTensorIterator transformation converts LSTMSequence layer to TensorIterator
 * *
 */

class ngraph::pass::ConvertLSTMSequenceToTensorIterator: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLSTMSequenceToTensorIterator();
};

class ngraph::pass::ConvertSequenceToTensorIterator : public GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertSequenceToTensorIterator();
};
