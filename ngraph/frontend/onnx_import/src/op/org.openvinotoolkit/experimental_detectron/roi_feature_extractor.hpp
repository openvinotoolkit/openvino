//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/experimental_detectron_roi_feature.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector experimental_detectron_roi_feature_extractor(const Node& node)
                {
                    using ROIFeatureExtractor =
                        ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor;

                    auto inputs = node.get_ng_inputs();

                    ROIFeatureExtractor::Attributes attrs{};
                    attrs.output_size = node.get_attribute_value<std::int64_t>("output_size", 7);
                    attrs.sampling_ratio = node.get_attribute_value<std::int64_t>("sampling_ratio", 2);
                    attrs.aligned = static_cast<bool>(node.get_attribute_value<std::int64_t>("aligned", 0));
                    attrs.num_classes = node.get_attribute_value<std::int64_t>("num_classes", 81);
                    attrs.post_nms_count = node.get_attribute_value<std::int64_t>("post_nms_count", 2000);
                    attrs.score_threshold = node.get_attribute_value<float>("score_threshold", 0.05f);
                    attrs.pyramid_scales = node.get_attribute_value<std::vector<std::int64_t>>("pyramid_scales", {4, 8, 16, 32, 64});
                    return {std::make_shared<ROIFeatureExtractor>(inputs, attrs)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
