// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/org.openvinotoolkit/experimental_detectron/prior_grid_generator.hpp"

#include "onnx_import/core/node.hpp"
#include "openvino/op/experimental_detectron_prior_grid_generator.hpp"

using namespace ov::op;

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector experimental_detectron_prior_grid_generator(const Node& node) {
    using PriorGridGenerator = v6::ExperimentalDetectronPriorGridGenerator;

    auto inputs = node.get_ng_inputs();
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
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
