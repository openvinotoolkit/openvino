// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph
{
    namespace pass
    {
        class TRANSFORMATIONS_API LowLatency;

    } // namespace pass
} // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief The transformation finds all TensorIterator layers in the network, processes all back
 * edges that describe a connection between Result and Parameter of the TensorIterator body,
 * and inserts ReadValue layer between Parameter and the next layers after this Parameter,
 * and Assign layer after the layers before the Result layer.
 *
 *  The example below describes the changes to the inner part (body, back edges) of the Tensor
 * Iterator layer.
 *  [] - TensorIterator body
 *  () - new layer
 *
 *  before applying the transformation:
 *  back_edge_1 -> [Parameter -> some layers ... -> Result ] -> back_edge_1
 *
 *  after applying the transformation:
 *  back_edge_1 -> [Parameter -> (ReadValue layer) -> some layers ... -> (Assign layer) ]
 *                                                              \
 *                                                               -> Result ] -> back_edge_1
 *
 *  It is recommended to use this transformation in conjunction with the UnrollTI transformation.
 *  After applying both of these transformations, you will have a network that allows you to infer
 *  step by step, the state will store between inferences.
 * *
 */

class ngraph::pass::LowLatency : public ngraph::pass::MatcherPass
{
public:
    NGRAPH_RTTI_DECLARATION;
    LowLatency();
};
