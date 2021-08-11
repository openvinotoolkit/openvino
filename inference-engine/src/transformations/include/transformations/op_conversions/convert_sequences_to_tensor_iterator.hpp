// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertRNNSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertGRUSequenceToTensorIterator;
class TRANSFORMATIONS_API ConvertLSTMSequenceToTensorIterator;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertRNNSequenceToTensorIterator transformation converts RNNSequence layer to TensorIterator
 * *
 */

class ov::pass::ConvertRNNSequenceToTensorIterator: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertRNNSequenceToTensorIterator();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGRUSequenceToTensorIterator transformation converts GRUSequence layer to TensorIterator
 * *
 */

class ov::pass::ConvertGRUSequenceToTensorIterator: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGRUSequenceToTensorIterator();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertLSTMSequenceToTensorIterator transformation converts LSTMSequence layer to TensorIterator
 * *
 */

class ov::pass::ConvertLSTMSequenceToTensorIterator: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLSTMSequenceToTensorIterator();
};
