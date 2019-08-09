/*
// Copyright (c) 2019 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "deformable_convolution_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id deformable_conv_type_id() {
    static primitive_type_base<deformable_conv> instance;
    return &instance;
}

layout deformable_conv_inst::calc_output_layout(deformable_conv_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();

    auto input_type = input_layout.data_type;
    auto output_type = node.get_primitive()->output_data_type ? *node.get_primitive()->output_data_type : input_type;

    tensor output_size(input_layout.size.batch[0],
                       desc->output_size.feature[0],
                       desc->output_size.spatial[0],
                       desc->output_size.spatial[1],
                       desc->output_size.spatial[2]);

    return {output_type, input_layout.format, output_size};
}

std::string deformable_conv_inst::to_string(deformable_conv_node const& node) {
    auto desc = node.get_primitive();
    auto split = node.get_split();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite conv_info;
    conv_info.add("split", split);
    conv_info.add("groups", desc->groups);

    json_composite ud_out_size_info;
    ud_out_size_info.add("size", desc->output_size.to_string());
    conv_info.add("with user defined output size", ud_out_size_info);

    node_info->add("deformable_convolution info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

deformable_conv_inst::typed_primitive_inst(network_impl& network, deformable_conv_node const& node) : parent(network, node) {
}


primitive_type_id deformable_interp_type_id() {
    static primitive_type_base<deformable_interp> instance;
    return &instance;
}

layout deformable_interp_inst::calc_output_layout(deformable_interp_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();

    auto kernel_size = desc->kernel_size;
    auto input_type = input_layout.data_type;
    auto output_type = node.get_primitive()->output_data_type ? *node.get_primitive()->output_data_type : input_type;

    tensor output_size(input_layout.size.batch[0],
                       input_layout.size.feature[0]*kernel_size.spatial[0]*kernel_size.spatial[1],
                       desc->output_size.spatial[0],
                       desc->output_size.spatial[1],
                       desc->output_size.spatial[2]);

    return {output_type, input_layout.format, output_size};
}

std::string deformable_interp_inst::to_string(deformable_interp_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto split = node.get_split();
    auto dilation = desc->dilation;
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite interp_info;
    interp_info.add("stride", strd.to_string());
    interp_info.add("input offset", desc->input_offset.to_string());
    interp_info.add("split", split);
    interp_info.add("dilation", dilation.to_string());
    interp_info.add("deformable_groups", desc->deformable_groups);
    interp_info.add("groups", desc->groups);

    json_composite ud_out_size_info;
    ud_out_size_info.add("size", desc->output_size.to_string());
    interp_info.add("with user defined output size", ud_out_size_info);

    node_info->add("deformable_interpolation info", interp_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

deformable_interp_inst::typed_primitive_inst(network_impl& network, deformable_interp_node const& node) : parent(network, node) {
}

}  // namespace cldnn
