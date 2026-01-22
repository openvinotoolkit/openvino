// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

/// \brief RetinaNetNmsToDetectionOutput transformation replaces TensorFlow RetinaNet
/// NMS post-processing subgraph with the optimized DetectionOutput operation.
///
/// The transformation looks for the following pattern:
/// - Input nodes: regression/concat, classification/concat, anchors/concat
/// - Output nodes: filtered_detections/map/TensorArrayStack/TensorArrayGatherV3 (3 outputs)
///
/// The subgraph is replaced with DetectionOutput operation that performs:
/// - Prior boxes scaling and variance application
/// - Box coordinate regression
/// - Non-maximum suppression
///
/// This transformation is equivalent to RetinaNetFilteredDetectionsReplacement from Model Optimizer.
class RetinaNetNmsToDetectionOutput : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::tensorflow::pass::RetinaNetNmsToDetectionOutput");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
