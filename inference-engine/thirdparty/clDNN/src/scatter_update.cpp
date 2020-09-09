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

#include "scatter_update_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id scatter_update::type_id() {
    static primitive_type_base<scatter_update> instance;
    return &instance;
}

static size_t GetNonEmptyDimsNumber(const layout& layout) {
    if (layout.size.count() != 1) {
        // Count the number of "one size" dimensions starting with X to Batch
        size_t one_size_dims = 0;
        std::vector<int32_t> dims;
        if (layout.format == cldnn::format::bfwzyx)
            dims = layout.size.sizes(format::bfwzyx);
        else if (layout.format == cldnn::format::bfzyx)
            dims = layout.size.sizes(format::bfzyx);
        else
            dims = layout.size.sizes(format::bfyx);
        for (size_t i = 0; i < dims.size(); i++) {
            if (dims[dims.size() - 1 - i] == 1)
                one_size_dims++;
            else
                break;
        }
        return dims.size() - one_size_dims;
    } else {
        return 1;
    }
}

layout scatter_update_inst::calc_output_layout(scatter_update_node const& node) {
    auto desc = node.get_primitive();

    const int32_t axis = desc->axis;
    const size_t indices_size = node.input(1).get_output_layout().size.count();
    const size_t input_number_of_dims = node.input(0).get_output_layout().size.sizes().size();
    const size_t updates_number_of_dims = node.input(2).get_output_layout().size.sizes().size();
    const size_t nonempty_indices_dims = GetNonEmptyDimsNumber(node.input(1).get_output_layout());

    auto input_layout = node.input(0).get_output_layout();
    
    auto output_shape = input_layout.size;
    auto input_format = input_layout.format;
    auto output_type = input_layout.data_type;

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    if (static_cast<size_t>(axis) < 0 || static_cast<size_t>(axis) >= input_number_of_dims)
        CLDNN_ERROR_MESSAGE(node.id(), "Incorrect axis value for ScatterUpdate: Axis must be positive and less than the input tensor dimension.");

    if (indices_size > static_cast<size_t>(output_shape.sizes()[axis])) {
        CLDNN_ERROR_MESSAGE(node.id(),
            "Undefined behavior ScatterUpdate: indices size must not be larger than the output size along the Axis.");
    }

    if (nonempty_indices_dims + static_cast<size_t>(axis) > updates_number_of_dims) {
        CLDNN_ERROR_MESSAGE(node.id(),
            "Undefined behavior ScatterUpdate: indices dimention must not be larger than the updates[:Axis] dimentional size.");
    }

    return layout{output_type, input_format, output_shape};
}

std::string scatter_update_inst::to_string(scatter_update_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scatter_update_info;
    scatter_update_info.add("input id", input.id());
    scatter_update_info.add("axis", desc->axis);
    scatter_update_info.add("output shape", node.input(0).get_output_layout().size.to_string());

    node_info->add("scatter_update info", scatter_update_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scatter_update_inst::typed_primitive_inst(network_impl& network, scatter_update_node const& node) : parent(network, node) {}

}  // namespace cldnn
