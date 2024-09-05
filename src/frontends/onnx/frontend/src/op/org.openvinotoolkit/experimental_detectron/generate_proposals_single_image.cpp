// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/experimental_detectron_generate_proposals.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
ov::OutputVector experimental_detectron_generate_proposals(const ov::frontend::onnx::Node& node) {
    using GenerateProposalsSingleImage = v6::ExperimentalDetectronGenerateProposalsSingleImage;

    const auto inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() == 4,
                            "ExperimentalDetectronGenerateProposalsSingleImage expects 4 "
                            "inputs, received: ",
                            inputs.size());

    auto im_info = inputs[0];
    auto anchors = inputs[1];
    auto deltas = inputs[2];
    auto scores = inputs[3];

    GenerateProposalsSingleImage::Attributes attrs{};
    attrs.min_size = node.get_attribute_value<float>("min_size", 0.0f);
    attrs.nms_threshold = node.get_attribute_value<float>("nms_threshold", 0.7f);
    attrs.post_nms_count = node.get_attribute_value<std::int64_t>("post_nms_count", 1000);
    attrs.pre_nms_count = node.get_attribute_value<std::int64_t>("pre_nms_count", 1000);
    auto generate_proposals_single_image =
        std::make_shared<GenerateProposalsSingleImage>(im_info, anchors, deltas, scores, attrs);
    return {generate_proposals_single_image->output(0), generate_proposals_single_image->output(1)};
}

ONNX_OP("ExperimentalDetectronGenerateProposalsSingleImage",
        OPSET_SINCE(1),
        org_openvinotoolkit::opset_1::experimental_detectron_generate_proposals,
        OPENVINO_ONNX_DOMAIN);
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
