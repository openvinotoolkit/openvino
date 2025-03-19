// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/experimental_detectron_prior_grid_generator.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
ov::OutputVector experimental_detectron_prior_grid_generator(const ov::frontend::onnx::Node& node) {
    using PriorGridGenerator = v6::ExperimentalDetectronPriorGridGenerator;

    auto inputs = node.get_ov_inputs();
    auto priors = inputs[0];
    auto feature_map = inputs[1];
    auto im_data = inputs[2];

    PriorGridGenerator::Attributes attrs{};
    attrs.flatten = static_cast<bool>(node.get_attribute_value<int64_t>("flatten", 1));
    attrs.h = node.get_attribute_value<int64_t>("h", 0);
    attrs.w = node.get_attribute_value<int64_t>("w", 0);
    attrs.stride_x = node.get_attribute_value<float>("stride_x", 0.0f);
    attrs.stride_y = node.get_attribute_value<float>("stride_y", 0.0f);

    return {std::make_shared<PriorGridGenerator>(priors, feature_map, im_data, attrs)};
}
ONNX_OP("ExperimentalDetectronPriorGridGenerator",
        OPSET_SINCE(1),
        org_openvinotoolkit::opset_1::experimental_detectron_prior_grid_generator,
        OPENVINO_ONNX_DOMAIN);
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
