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

#include "ngraph/op/prior_box_clustered.hpp"
#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"
#include "op/org.openvinotoolkit/prior_box_clustered.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector prior_box_clustered(const Node& node)
                {
                    using PriorBoxClustered = default_opset::PriorBoxClustered;

                    auto inputs = node.get_ng_inputs();
                    auto layer_shape = inputs[0];
                    auto img_shape = inputs[1];

                    ngraph::op::PriorBoxClusteredAttrs attrs{};
                    attrs.widths =
                        node.get_attribute_value<std::vector<float>>("width", {1.0});
                    attrs.heights =
                        node.get_attribute_value<std::vector<float>>("height", {1.0});
                    // attrs.flip =
                    //     node.get_attribute_value<int64_t>("flip", 0);   
                    attrs.clip =
                        node.get_attribute_value<int64_t>("clip", 0); 
                    attrs.variances =
                        node.get_attribute_value<std::vector<float>>("variance", {0.1f});
                    // attrs.img_size =
                    //     node.get_attribute_value<int64_t>("img_size", 0);
                    // attrs.img_h =
                    //     node.get_attribute_value<int64_t>("img_h", 0);
                    // attrs.img_w =
                    //     node.get_attribute_value<int64_t>("img_w", 0);
                    // attrs.step =
                    //     node.get_attribute_value<float>("step", 0.0f);
                    attrs.step_heights =
                        node.get_attribute_value<float>("step_h", 0.0f);
                    attrs.step_widths =
                        node.get_attribute_value<float>("step_w", 0.0f);
                    attrs.offset =
                        node.get_attribute_value<float>("offset", 0.0f);

                    return {std::make_shared<PriorBoxClustered>(layer_shape, img_shape, attrs)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
