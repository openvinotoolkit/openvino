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

#include "scale_grad_input_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
    primitive_type_id scale_grad_input_type_id()
    {
        static primitive_type_base<scale_grad_input> instance;
        return &instance;
    }

    layout scale_grad_input_inst::calc_output_layout(scale_grad_input_node const& node)
    {
        auto result = node.input().get_non_padded_output_layout();

        auto scale_in_sizes = node.scale_in().get_non_padded_output_layout().size;
        auto input_sizes = result.size;

        auto scale_in_x_size = scale_in_sizes.spatial[0];
        auto scale_in_y_size = scale_in_sizes.spatial[1];

        auto input_x_size = input_sizes.spatial[0];
        auto input_y_size = input_sizes.spatial[1];

        if (scale_in_x_size != 1)
        {
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Scale x size", scale_in_x_size, "input x size", input_x_size, "");
        }
        if (scale_in_y_size != 1)
        {
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Scale y size", scale_in_y_size, "input y size", input_y_size, "");
        }

        return result;
    }

    std::string scale_grad_input_inst::to_string(scale_grad_input_node const& node)
    {
        auto desc = node.get_primitive();
        auto node_info = node.desc_to_json();
        auto& input = node.input();
        auto& scale_input = node.scale_in();

        std::stringstream primitive_description;

        json_composite scale_grad_input_info;
        scale_grad_input_info.add("input", input.id());
        scale_grad_input_info.add("scale input", scale_input.id());

        node_info.add("scale_grad_input info", scale_grad_input_info);
        node_info.dump(primitive_description);

        return primitive_description.str();
    }

    scale_grad_input_inst::typed_primitive_inst(network_impl& network, scale_grad_input_node const& node)
        :parent(network, node)
    {
        auto scale_input_layout = node.scale_in().get_output_layout();
        auto scale_input_batch_size = scale_input_layout.size.batch[0];
        auto scale_input_feature_size = scale_input_layout.size.feature[0];

        auto input_layout = node.input().get_output_layout();
        auto input_batch_size = input_layout.size.batch[0];
        auto input_feature_size = input_layout.size.feature[0];

        if (scale_input_batch_size != 1)
        {
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Scale batch size", scale_input_batch_size, "input batch size", input_batch_size, "");
        }

        if (scale_input_feature_size != 1)
        {
            CLDNN_ERROR_NOT_EQUAL(node.id(), "Scale feature size", scale_input_feature_size, "input feature size", input_feature_size, "");
        }
    }
}
