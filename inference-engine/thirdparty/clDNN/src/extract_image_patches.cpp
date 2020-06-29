/*
// Copyright (c) 2020 Intel Corporation
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

#include "extract_image_patches_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id extract_image_patches::type_id() {
    static primitive_type_base<extract_image_patches> instance;
    return &instance;
}

layout extract_image_patches_inst::calc_output_layout(extract_image_patches_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;

    auto output_shape = desc->output_shape;
    return layout(input_layout.data_type, input_format, output_shape);
}

std::string extract_image_patches_inst::to_string(extract_image_patches_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    std::stringstream sizes, strides, rates;
    sizes << desc->sizes[0] << "," << desc->sizes[1];
    strides << desc->strides[0] << "," << desc->strides[1];
    rates << desc->rates[0] << "," << desc->rates[1];

    json_composite extract_image_patches_info;
    extract_image_patches_info.add("input id", input.id());
    extract_image_patches_info.add("input shape", input.get_output_layout().size.to_string());
    extract_image_patches_info.add("sizes", sizes.str());
    extract_image_patches_info.add("strides", strides.str());
    extract_image_patches_info.add("rates", rates.str());
    extract_image_patches_info.add("auto_pad", desc->auto_pad);
    extract_image_patches_info.add("output shape", input.calc_output_layout().size.to_string());

    node_info->add("extract_image_patches info", extract_image_patches_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

extract_image_patches_inst::typed_primitive_inst(network_impl& network, extract_image_patches_node const& node) : parent(network, node) {}

}  // namespace cldnn
