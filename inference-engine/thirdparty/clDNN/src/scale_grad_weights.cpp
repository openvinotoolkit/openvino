/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "scale_grad_weights_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id scale_grad_weights_type_id()
{
    static primitive_type_base<scale_grad_weights> instance;
    return &instance;
}

layout scale_grad_weights_inst::calc_output_layout(scale_grad_weights_node const& node)
{
    //output buffer will not be used in this primitive
    auto input_grad_layout_size = node.input().get_output_layout();
    return{ input_grad_layout_size.data_type, input_grad_layout_size.format,{ 1, 1, 1, 1 } };
}

std::string scale_grad_weights_inst::to_string(scale_grad_weights_node const& node)
{
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();
    auto& scale_input = node.weights();
    auto& input_grad = node.input_grad();

    std::stringstream primitive_description;

    json_composite scale_grad_weights_info;
    scale_grad_weights_info.add("input", input.id());
    scale_grad_weights_info.add("scale input", scale_input.id());
    scale_grad_weights_info.add("input grad", input_grad.id());
    if (node.bias_term())
        scale_grad_weights_info.add("bias", node.bias().id());
        
    node_info.add("scale_grad_weights info", scale_grad_weights_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

scale_grad_weights_inst::typed_primitive_inst(network_impl& network, scale_grad_weights_node const& node)
    :parent(network, node)
{
    auto scale_layout = node.weights().get_output_layout();
    auto scale_format = scale_layout.format;

    auto scale_sizes = scale_layout.size;
    auto scale_feature_size = scale_layout.size.feature[0];

    auto input_layout = node.input().get_output_layout();
    auto input_feature_size = input_layout.size.feature[0];

    CLDNN_ERROR_NOT_EQUAL(node.id(), "Scale feature size", scale_feature_size, "input feature size", input_feature_size, "");

    if (scale_sizes.spatial[0] != 1 || scale_sizes.spatial[1] != 1 || scale_sizes.batch[0] != 1) //Remove if support for other scale sizes will be added.
    {
        CLDNN_ERROR_MESSAGE(node.id(), "All sizes in scale_input except feature should be 1.");
    }

    if (node.use_momentum())
    {
        CLDNN_ERROR_LAYOUT_MISMATCH(node.id(), "Scale memory", node.weights().get_output_layout(), "previous scale grad memory", node.prev_scale_grad().get_output_layout(), "");
        CLDNN_ERROR_LAYOUT_MISMATCH(node.id(), "Bias memory", node.bias().get_output_layout(), "previous bias grad memory", node.prev_bias_grad().get_output_layout(), "");
    }

    if (node.bias_term()) 
    {
        auto bias_layout = node.bias().get_output_layout();
        auto bias_format = bias_layout.format;
        auto bias_raw_sizes = bias_layout.size.raw;

        CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Scale format", scale_format.value, "bias format", bias_format);

        for (size_t i = 0; i < bias_layout.size.raw.size(); ++i)
        {
            if (scale_layout.size.raw[i] != bias_raw_sizes[i])
                CLDNN_ERROR_MESSAGE(node.id(), "Scale input size do not match bias size! Size index:" + std::to_string(i));
        }
    }
}
}
