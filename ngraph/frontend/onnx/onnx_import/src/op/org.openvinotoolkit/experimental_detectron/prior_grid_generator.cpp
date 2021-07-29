// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector experimental_detectron_prior_grid_generator(const Node& node)
                {
                    using PriorGridGenerator =
                        ngraph::op::v6::ExperimentalDetectronPriorGridGenerator;

                    auto inputs = node.get_ng_inputs();
                    auto priors = inputs[0];
                    auto feature_map = inputs[1];
                    auto im_data = inputs[2];

                    PriorGridGenerator::Attributes attrs{};
                    attrs.flatten =
                        static_cast<bool>(node.get_attribute_value<int64_t>("flatten", 1));
                    attrs.h = node.get_attribute_value<int64_t>("h", 0);
                    attrs.w = node.get_attribute_value<int64_t>("w", 0);
                    attrs.stride_x = node.get_attribute_value<float>("stride_x", 0.0f);
                    attrs.stride_x = node.get_attribute_value<float>("stride_y", 0.0f);

                    return {
                        std::make_shared<PriorGridGenerator>(priors, feature_map, im_data, attrs)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
