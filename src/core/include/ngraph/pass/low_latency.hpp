// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <memory>
#include <vector>

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/pass.hpp"
#include "openvino/pass/low_latency.hpp"

namespace ngraph {
namespace pass {
/**
 * @brief The transformation finds all TensorIterator/Loop layers in the network,
 * processes all back edges that describe a connection between Result and Parameter
 * of the TensorIterator body,and inserts ReadValue layer between Parameter
 * and the next layers after this Parameter, and Assign layer after the layers
 * before the Result layer. Supported platforms: CPU, GNA.
 *
 * The example below describes the changes to the inner part (body, back edges) of the
 * Tensor Iterator layer.
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
 * It is recommended to use this transformation in conjunction with the Reshape feature to
 * set  sequence dimension to 1 and with the UnrollTensorIterator transformation.
 * For convenience, we have already enabled the unconditional execution of the
 * UnrollTensorIterator transformation when using the LowLatency transformation for
 * CPU, GNA plugins, no action is required here.
 * After applying both of these transformations, the resulting network can be inferred step
 * by step, the states will store between inferences.
 */

class NGRAPH_API_DEPRECATED NGRAPH_API LowLatency : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    LowLatency();
};

using ov::pass::LowLatency2;
}  // namespace pass
}  // namespace ngraph
