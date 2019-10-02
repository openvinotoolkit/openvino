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

#include "border_inst.h"

#include "error_handler.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <algorithm>

namespace cldnn {
primitive_type_id border::type_id() {
    static primitive_type_base<border> instance;
    return &instance;
}

layout border_inst::calc_output_layout(border_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for border_node!");
    auto input_layout = node.input().get_output_layout();
    auto desc = node.get_primitive();

    auto&& new_size = input_layout.size;
    new_size += desc->left_top_sizes.sub(tensor(0));
    new_size += desc->right_bottom_sizes.sub(tensor(0));

    auto ret_data_t = input_layout.data_type;
    auto ret_format = input_layout.format;

    if (ret_format == format::bfwzyx) {
        return layout{ ret_data_t, ret_format, tensor(batch(new_size.batch[0]), feature(new_size.feature[0]),
            spatial(new_size.spatial[0], new_size.spatial[1], new_size.spatial[2], new_size.spatial[3])) };
    } else if (ret_format == format::bfzyx) {
        return layout{ ret_data_t, ret_format, tensor(batch(new_size.batch[0]), feature(new_size.feature[0]),
            spatial(new_size.spatial[0], new_size.spatial[1], new_size.spatial[2])) };
    }
    return layout{ ret_data_t, ret_format, tensor(batch(new_size.batch[0]), feature(new_size.feature[0]),
        spatial(new_size.spatial[0], new_size.spatial[1])) };
}

std::string border_inst::to_string(border_node const& node) {
    auto desc = node.get_primitive();

    const auto& left_top_sizes = desc->left_top_sizes.sub({0, 0, 0, 0});
    const auto& right_bottom_sizes = desc->right_bottom_sizes.sub({0, 0, 0, 0});
    const auto& border_value = std::to_string(desc->border_value);

    const char* border_type_str = "unknown";
    switch (desc->type) {
        case border_type::zero:
            border_type_str = "zero";
            break;
        case border_type::constant:
            border_type_str = "constant";
            break;
        case border_type::edge:
            border_type_str = "edge";
            break;
        case border_type::mirror:
            border_type_str = "mirror";
            break;
        case border_type::mirror_101:
            border_type_str = "mirror-101";
            break;
        default:
            border_type_str = "unknown";
            break;
    }

    auto node_info = node.desc_to_json();

    json_composite border_info;
    border_info.add("left/top sizes", left_top_sizes.to_string());
    border_info.add("right/bottom sizes", right_bottom_sizes.to_string());
    border_info.add("border type", border_type_str);
    border_info.add("border value", border_value);

    node_info->add("border info", border_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

border_inst::typed_primitive_inst(network_impl& network, border_node const& node) : parent(network, node) {
    auto input_layout = node.input().get_output_layout();

    const auto input_format = input_layout.format;
    const auto& input_sizes = input_layout.size;

    auto lt_sizes = argument.left_top_sizes.sub(tensor(0));
    auto rb_sizes = argument.right_bottom_sizes.sub(tensor(0));
    auto b_type = argument.type;

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "Input format",
                                  input_format.value,
                                  "supported border primitive input formats",
                                  format::bfyx,
                                  format::yxfb,
                                  format::byxf,
                                  format::bfzyx,
                                  format::bfwzyx);

    tensor null_tensor = tensor(0);

    // Check if sizes of border are in proper range.
    CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                       "Left/Top border sizes",
                                       lt_sizes,
                                       "0 value",
                                       null_tensor,
                                       "Invalid border size: negative value");
    CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                       "Right/Bottom border sizes",
                                       rb_sizes,
                                       "0 value",
                                       null_tensor,
                                       "Invalid border size: negative value");

    if (b_type == border_type::mirror) {
        CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                              "Left/Top border sizes",
                                              lt_sizes,
                                              "input_sizes",
                                              input_sizes,
                                              "Not enough data in input to create mirror border of specified size");
        CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                              "Right/Bottom border sizes",
                                              rb_sizes,
                                              "input_sizes",
                                              input_sizes,
                                              "Not enough data in input to create mirror border of specified size");
    } else if (b_type == border_type::mirror_101) {
        auto reduced_input_sizes = input_sizes;
        reduced_input_sizes -= tensor(1);
        reduced_input_sizes = tensor::max(reduced_input_sizes, tensor());

        CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                              "Left/Top border sizes",
                                              lt_sizes,
                                              "input_sizes - 1",
                                              reduced_input_sizes,
                                              "Not enough data in input to create mirror-101 border of specified size");
        CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                              "Right/Bottom border sizes",
                                              rb_sizes,
                                              "input_sizes - 1",
                                              reduced_input_sizes,
                                              "Not enough data in input to create mirror-101 border of specified size");
    }
}
}  // namespace cldnn
