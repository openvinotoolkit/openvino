/*
// Copyright (c) 2021 Intel Corporation
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

#include "gather_elements_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id gather_elements::type_id() {
    static primitive_type_base<gather_elements> instance;
    return &instance;
}

layout gather_elements_inst::calc_output_layout(gather_elements_node const& node) {
    auto op = node.get_primitive();

    auto input_layout_origin = node.input(0).get_output_layout();
    auto indices_layout_origin = node.input(1).get_output_layout();

    auto input_layout = input_layout_origin.size.sizes(input_layout_origin.format);
    auto indices_layout = indices_layout_origin.size.sizes(indices_layout_origin.format);

    // const size_t input_dims = input_layout.size();

    // const auto indices_rank = op->indices_rank;
    const auto axis = op->axis;

    // calculate initial output shape
    std::vector<tensor::value_type> output_sizes;

    // for (uint8_t x = 0; x < indices_rank - 1; x++) {
    //     output_sizes.push_back(indices_layout[x]);
    // }

    // const size_t indices_last_dim = indices_layout[indices_rank - 1];
    // for (size_t x = static_cast<size_t>(axis + indices_last_dim); x < input_dims; x++) {
    //     output_sizes.push_back(input_layout[x]);
    // }

    // // calculate batch_size by axis
    // int batch_size = 1;
    // for (uint8_t x = 0; x < axis; x++) {
    //     batch_size *= output_sizes[x];
    // }

    // create final output shape by axis
    std::vector<tensor::value_type> final_output_sizes;

    // if (axis > 0) {
    //     final_output_sizes.push_back(batch_size);
    // }

    for (size_t x = static_cast<size_t>(axis); x < output_sizes.size(); x++) {
        final_output_sizes.push_back(output_sizes[x]);
    }

    auto output_format = cldnn::format::bfyx;
    if (final_output_sizes.size() >= 6) {
        output_format = cldnn::format::bfwzyx;
    } else if (final_output_sizes.size() == 5) {
        output_format = cldnn::format::bfzyx;
    }

    auto output_sizes_tensor = tensor(tensor(final_output_sizes).sizes(output_format));
    auto padding = op->output_padding;


    // if (node.has_fused_primitives()) {
    //     input_layout_origin.data_type = node.get_fused_output_layout().data_type;
    // }

    return layout(input_layout_origin.data_type, output_format, output_sizes_tensor, padding);
}

std::string gather_elements_inst::to_string(gather_elements_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_elements_info;
    gather_elements_info.add("input id", input.id());
    gather_elements_info.add("input shape", node.input(0).get_output_layout().size.to_string());
    gather_elements_info.add("indices shape", node.input(1).get_output_layout().size.to_string());
    // gather_elements_info.add("indices rank", desc->indices_rank);
    gather_elements_info.add("axis", desc->axis);
    // gather_elements_info.add("output shape", calc_output_layout(node).size.to_string());

    node_info->add("gather_elements info", gather_elements_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_elements_inst::typed_primitive_inst(network_impl& network, gather_elements_node const& node) : parent(network, node) {}

}  // namespace cldnn
