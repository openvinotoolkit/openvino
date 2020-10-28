// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API LowLatency;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief The transformation finds all TensorIterator layers in the network, processes all back edges in the TensorIterator
 *  body, and inserts ReadValue layer between Parameter and the next layers after this Parameter and Assign layer between
 *  the Result layer and the previous layers before this Result layer.
 *
 *  [] - TensorIterator body
 *  () - new layer
 *  back edge -> [Parameter -> (ReadValue layer) -> some layers ... -> (Assign layer) -> Result] -> back edge
 *
 *  It is recommended to use this transformation in conjunction with the UnrollTI transformation.
 *  After applying both of these transformations, you will have a network that allows you to infer step by step,
 *  the state will save between inferences.
 * *
 */

class ngraph::pass::LowLatency: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    LowLatency();
};
