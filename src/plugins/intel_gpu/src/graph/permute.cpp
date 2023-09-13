// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "permute_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"

#include <algorithm>
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(permute)

layout permute_inst::calc_output_layout(permute_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<permute>();
    auto input_layout = impl_param.get_input_layout();
    auto permute_order = desc->permute_order;
    std::vector<tensor::value_type> output_shape;

    auto input_shape = input_layout.get_dims();

    for (size_t x = 0; x < permute_order.size(); x++) {
        output_shape.push_back(input_shape[permute_order[x]]);
    }

    for (size_t i = output_shape.size(); i < 4; i++) {
        output_shape.push_back(1);
    }

    auto output_size = tensor(format::get_default_format(input_layout.get_rank()), output_shape);
    auto op = desc->output_paddings[0];

    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    // Adjust output format for optimizing out of transpose related to acdb format.
    auto out_fmt = input_layout.format;
    if (node.get_preferred_output_fmt() != format::any) {
        out_fmt = node.get_preferred_output_fmt();
    }

    return layout(output_type, out_fmt, output_size, op);
}

template<typename ShapeType>
std::vector<layout> permute_inst::calc_output_layouts(permute_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<permute>();
    auto input_layout = impl_param.get_input_layout();

    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    ShapeType input_shape = input_layout.get<ShapeType>();
    ShapeType output_shape;
    ov::Rank input_rank = input_shape.rank();

    if (input_rank.is_dynamic()) {
        output_shape = ShapeType::dynamic(desc->permute_order.size());
        return { layout{output_shape, output_type, input_layout.format} };
    }

    int64_t input_static_rank = input_rank.get_length();
    auto permute_order = desc->permute_order;
    if (permute_order.empty()) {
        for (int64_t i = 1; i <= input_static_rank; ++i) {
            permute_order.emplace_back(input_static_rank - i);
        }
    }

    for (int64_t i = 0; i < input_static_rank; ++i) {
        output_shape.push_back(input_shape[permute_order[i]]);
    }

    return { layout{output_shape, output_type, input_layout.format, desc->output_paddings[0]} };
}

template std::vector<layout> permute_inst::calc_output_layouts<ov::PartialShape>(permute_node const& node, const kernel_impl_params& impl_param);

std::string permute_inst::to_string(permute_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto permute_order = desc->permute_order;
    auto& input = node.input();

    std::stringstream primitive_description;
    std::stringstream ss_permute_order;

    for (size_t i = 0; i < permute_order.size(); ++i) {
        ss_permute_order << permute_order.at(i);
        i != (permute_order.size() - 1) ? ss_permute_order << ", " : ss_permute_order << "";
    }

    json_composite permute_info;
    permute_info.add("input id", input.id());
    permute_info.add("permute order", ss_permute_order.str());

    node_info->add("permute info", permute_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

permute_inst::typed_primitive_inst(network& network, permute_node const& node) :
        parent(network, node, !node.can_be_optimized()
                              && (node.get_output_layout().is_static() || node.get_output_layout().has_upper_bound())) {
    auto permute_order = argument->permute_order;

    auto required_order_values_size = static_cast<uint32_t>(permute_order.size());

    for (decltype(required_order_values_size) i = 0; i < required_order_values_size; i++) {
        if (!(std::find(permute_order.begin(), permute_order.end(), i) != permute_order.end()))
            CLDNN_ERROR_MESSAGE(node.id(), "Permute order does not contain all of required values.");
    }

    if (node.can_be_optimized()) {
        reuse_input();
    }
}

void permute_inst::on_execute() {
    if (can_be_optimized())
        reuse_input();
}

void permute_inst::reuse_input() {
    update_output_memory();
}

void permute_inst::update_output_memory() {
    if (!can_be_optimized())
        return;

    if (_outputs.size() > 0 && static_cast<bool>(_outputs[0])
        && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    if (_node != nullptr)
        build_deps();

    _outputs = {_network.get_engine().reinterpret_buffer(input_memory(), _impl_params->get_output_layout())};
    _mem_allocated = false;
}



}  // namespace cldnn
