// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "ngraph/opsets/opset3.hpp"
#include "op/roi_align.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector roi_align(const Node& node)
                {
                    const auto inputs = node.get_ng_inputs();

                    NGRAPH_CHECK(inputs.size() == 3,
                                 "The RoiAlign operator expects 3 inputs. Got: ",
                                 inputs.size());

                    const auto& data = inputs[0];
                    const auto& rois = inputs[1];
                    const auto& num_rois = inputs[2];

                    const auto pooled_h = node.get_attribute_value<int64_t>("output_height", 1);
                    const auto pooled_w = node.get_attribute_value<int64_t>("output_width", 1);
                    const auto sampling_ratio =
                        node.get_attribute_value<int64_t>("sampling_ratio", 1);
                    const auto spatial_scale =
                        node.get_attribute_value<float>("spatial_scale", 1.0f);
                    const auto mode = node.get_attribute_value<std::string>("mode", "avg");

                    return {std::make_shared<ngraph::opset3::ROIAlign>(data,
                                                                       rois,
                                                                       num_rois,
                                                                       pooled_h,
                                                                       pooled_w,
                                                                       sampling_ratio,
                                                                       spatial_scale,
                                                                       mode)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
