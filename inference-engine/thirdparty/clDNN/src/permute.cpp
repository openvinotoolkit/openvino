/*
// Copyright (c) 2016 Intel Corporation
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
#include "permute_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

#include <algorithm>

namespace cldnn
{

primitive_type_id permute_type_id()
{
    static primitive_type_base<permute> instance;
    return &instance;
}

static std::vector<uint16_t> get_permute_order(permute_node const& node, format::type fmt)
{

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "node format", fmt, "byxf, yxfb, bfyx, fyxb", format::byxf, format::yxfb, format::bfyx, format::fyxb);
    switch (fmt)
    {
        // For input formats:
        // 0 - batch (b), 1 - feature (f), 2, 3 - spatial (x -> 2, y -> 3)
    case format::byxf:
        return{ 0, 3, 2, 1 };

    case format::yxfb:
        return{ 3, 2, 1, 0 };

    case format::bfyx:
        return{ 0, 1, 3, 2 };

    case format::fyxb:
        return{ 1, 3, 2, 0 };

    default:
        throw std::invalid_argument("This format is not supported in GPU permute_inst");
    }
}

layout permute_inst::calc_output_layout(permute_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    auto permute_order = node.get_primitive()->permute_order;
    auto input_sizes_ordered = input_layout.size.sizes(input_layout.format);

    const auto& fmt_2_bfxy = get_permute_order(node, input_layout.format);
    std::vector<tensor::value_type> output_sizes;
    for (auto i : fmt_2_bfxy)
    {
        output_sizes.push_back(input_sizes_ordered[permute_order[i]]);
    }

    auto input_size = tensor(output_sizes);
    auto op = node.get_primitive()->output_padding;

    return layout(input_layout.data_type, input_layout.format, input_size, op);
}

std::string permute_inst::to_string(permute_node const& node)
{
    auto desc          = node.get_primitive();
    auto node_info     = node.desc_to_json();
    auto permute_order = desc->permute_order;
    auto& input        = node.input();
    
    std::stringstream primitive_description;
    std::stringstream ss_permute_order;

    for (size_t i = 0; i < permute_order.size(); ++i)
    {
        ss_permute_order << permute_order.at(i);
        i != (permute_order.size() - 1) ? ss_permute_order << ", " : ss_permute_order << "";
    }

    json_composite permute_info;
    permute_info.add("input id", input.id());
    permute_info.add("permute order", ss_permute_order.str());
    
    node_info.add("permute info", permute_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

permute_inst::typed_primitive_inst(network_impl& network, permute_node const& node)
    : parent(network, node)
{
    auto permute_order = argument.permute_order;

    CLDNN_ERROR_NOT_EQUAL(node.id(), "Permute order size", permute_order.size(), "expected order size", 4, "Permute order size needs to be 4.");

    std::vector<uint16_t> required_order_values = { 0, 1, 2, 3 };
    auto required_order_values_size = required_order_values.size();

    for (decltype(required_order_values_size) i = 0; i < required_order_values_size; i++)
    {
        if (!(std::find(permute_order.begin(), permute_order.end(), required_order_values[i]) != permute_order.end()))
            CLDNN_ERROR_MESSAGE(node.id(), "Permute order does not contain all of required values.");
    }
}
}
