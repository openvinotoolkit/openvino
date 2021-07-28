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
                OutputVector experimental_detectron_roi_feature_extractor(const Node& node)
                {
                    using ROIFeatureExtractor =
                        ngraph::op::v6::ExperimentalDetectronROIFeatureExtractor;

                    auto inputs = node.get_ng_inputs();

                    ROIFeatureExtractor::Attributes attrs{};
                    attrs.output_size = node.get_attribute_value<std::int64_t>("output_size", 7);
                    attrs.sampling_ratio =
                        node.get_attribute_value<std::int64_t>("sampling_ratio", 2);
                    attrs.aligned =
                        static_cast<bool>(node.get_attribute_value<std::int64_t>("aligned", 0));
                    attrs.pyramid_scales = node.get_attribute_value<std::vector<std::int64_t>>(
                        "pyramid_scales", {4, 8, 16, 32, 64});
                    auto roi_feature_extractor =
                        std::make_shared<ROIFeatureExtractor>(inputs, attrs);
                    return {roi_feature_extractor->output(0), roi_feature_extractor->output(1)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
