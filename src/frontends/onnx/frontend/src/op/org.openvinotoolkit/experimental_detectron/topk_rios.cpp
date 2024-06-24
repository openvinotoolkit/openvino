// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/experimental_detectron_topkrois.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
ov::OutputVector experimental_detectron_topk_rois(const ov::frontend::onnx::Node& node) {
    using TopKROIs = v6::ExperimentalDetectronTopKROIs;

    auto inputs = node.get_ov_inputs();
    auto input_rois = inputs[0];
    auto rois_probs = inputs[1];
    auto max_rois = static_cast<std::size_t>(node.get_attribute_value<std::int64_t>("max_rois", 1000));

    return {std::make_shared<TopKROIs>(input_rois, rois_probs, max_rois)};
}

ONNX_OP("ExperimentalDetectronTopKROIs",
        OPSET_SINCE(1),
        org_openvinotoolkit::opset_1::experimental_detectron_topk_rois,
        OPENVINO_ONNX_DOMAIN);
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
