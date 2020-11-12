// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(PatternBeforeTopKFusion);

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *      Fusion of the subgraph
 *          ShapeOf -> Gather -> Unsqueeze -> Concat -> Convert -> ReduceMin -> Convert -> Unsqueeze
 *      when
 *          1) input of ShapeOf has a static output shape;
 *          2) all other nodes of the subgraph have constants in input ports with numbers greater than 0.
 *
 *      Such subgraph appears in the ONNX SSD model. This model can be found in
 *      https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd.
 */

class ngraph::pass::PatternBeforeTopKFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PatternBeforeTopKFusion();
};

