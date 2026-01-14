// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_mask_gen_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_mask_gen)

layout moe_mask_gen_inst::calc_output_layout(moe_mask_gen_node const& node, kernel_impl_params const& impl_param) {
    OPENVINO_THROW("moe_mask_gen has multiple outputs so only supports allow_new_shape_infer = true.");
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[1];
}

template<typename ShapeType>
std::vector<layout> moe_mask_gen_inst::calc_output_layouts(moe_mask_gen_node const& /*node*/, const kernel_impl_params& impl_param) {
    std::vector<layout> output_layouts;
    const auto& num_total_experts = impl_param.typed_desc<moe_mask_gen>()->num_total_experts;
    const auto& num_experts_per_token = impl_param.typed_desc<moe_mask_gen>()->num_experts_per_token;

    if (impl_param.get_input_layout(0).is_dynamic()) {
        // out0: tokens_per_expert
        auto tokens_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
        output_layouts.emplace_back(tokens_per_expert_shape, data_types::i32, format::bfyx);
    } else {
        const auto num_tokens = impl_param.get_input_layout(0).get_shape()[0];
        // out0: tokens_per_expert
        auto tokens_per_expert_shape = ov::Shape{num_tokens * num_experts_per_token};
        output_layouts.emplace_back(tokens_per_expert_shape, data_types::i32, format::bfyx);
    }
    // out1: experts_info_start_idx
    auto experts_info_start_idx_shape = ov::Shape{static_cast<size_t>(num_total_experts)};
     output_layouts.emplace_back(experts_info_start_idx_shape, data_types::i32, format::bfyx);
    // out2: experts_id
    auto experts_ids = ov::Shape{static_cast<size_t>(num_total_experts)};
    output_layouts.emplace_back(experts_ids, data_types::i32, format::bfyx);
    // out3: tokens_lens_per_expert
    auto tokens_lens_per_expert = ov::Shape{static_cast<size_t>(num_total_experts)};
    output_layouts.emplace_back(tokens_lens_per_expert, data_types::i32, format::bfyx);
    // out4: num_actual_expert
    auto num_actual_used_experts_shape = ov::Shape{static_cast<size_t>(1)};
    output_layouts.emplace_back(num_actual_used_experts_shape, data_types::i32, format::bfyx);
    return output_layouts;
}

template std::vector<layout> moe_mask_gen_inst::calc_output_layouts<ov::PartialShape>(moe_mask_gen_node const& node, const kernel_impl_params& impl_param);

std::string moe_mask_gen_inst::to_string(moe_mask_gen_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    std::stringstream primitive_description;

    json_composite moe_mask_gen_info;
    for (auto o : desc->output_data_types) {
        if (o.has_value())
            moe_mask_gen_info.add("out dt: ", dt_to_str(*o));
    }

    node_info->dump(primitive_description);

    return primitive_description.str();
}

moe_mask_gen_inst::typed_primitive_inst(network& network, moe_mask_gen_node const& node) : parent(network, node) { }

GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_mask_gen_reshape)
layout moe_mask_gen_reshape_inst::calc_output_layout(moe_mask_gen_reshape_node const& node, kernel_impl_params const& impl_param) {
    OPENVINO_THROW("moe_mask_gen has multiple outputs so only supports allow_new_shape_infer = true.");
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[1];
}

template<typename ShapeType>
std::vector<layout> moe_mask_gen_reshape_inst::calc_output_layouts(moe_mask_gen_reshape_node const& /*node*/, const kernel_impl_params& impl_param) {
    std::vector<layout> output_layouts;
    if (!impl_param.memory_deps.count(4)) {
        auto out_shape = ov::PartialShape{ov::Dimension::dynamic()};
        output_layouts.emplace_back(out_shape, data_types::i32, format::bfyx);
        output_layouts.emplace_back(out_shape, data_types::i32, format::bfyx);
        output_layouts.emplace_back(out_shape, data_types::i32, format::bfyx);
        output_layouts.emplace_back(out_shape, data_types::i32, format::bfyx);
        return output_layouts;
    }
    auto num_actually_used_experts =
        read_vector<int32_t>(impl_param.memory_deps.at(moe_mask_gen::MoEMaskGenOutputIdx::NUM_ACTUALLY_USED_EXPERTS), impl_param.get_stream())[0];
    // tokens_per_expert
    output_layouts.emplace_back(impl_param.get_input_layout(0));
    // experts_info_start_idx
    auto experts_info_start_idx_shape = ov::Shape{static_cast<size_t>(num_actually_used_experts)};
    output_layouts.emplace_back(experts_info_start_idx_shape, data_types::i32, format::bfyx);
    // experts_id
    auto experts_ids_shape = ov::Shape{static_cast<size_t>(num_actually_used_experts)};
    output_layouts.emplace_back(experts_ids_shape, data_types::i32, format::bfyx);
    // tokens_lens_per_expert
    auto tokens_lens_per_expert_shape = ov::Shape{static_cast<size_t>(num_actually_used_experts)};
    output_layouts.emplace_back(tokens_lens_per_expert_shape, data_types::i32, format::bfyx);
    return output_layouts;
}

template std::vector<layout> moe_mask_gen_reshape_inst::calc_output_layouts<ov::PartialShape>
                        (moe_mask_gen_reshape_node const& node, const kernel_impl_params& impl_param);

std::string moe_mask_gen_reshape_inst::to_string(moe_mask_gen_reshape_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    std::stringstream primitive_description;

    json_composite moe_mask_gen_reshape_info;
    for (auto o : desc->output_data_types) {
        if (o.has_value())
            moe_mask_gen_reshape_info.add("out dt: ", dt_to_str(*o));
    }
    node_info->dump(primitive_description);

    return primitive_description.str();
}

moe_mask_gen_reshape_inst::typed_primitive_inst(network& network, moe_mask_gen_reshape_node const& node) : parent(network, node, false) {
    update_output_memory();
}

void moe_mask_gen_reshape_inst::on_execute() {
    update_output_memory();
}

void moe_mask_gen_reshape_inst::update_output_memory() {
    if (!can_be_optimized())
        return;
    if (_node != nullptr)
        build_deps();

    _mem_allocated = false;
    if (_impl_params->get_input_layout(0).is_dynamic())
        return;
    for (size_t i = 0; i < _outputs.size(); ++i) {
        if (static_cast<bool>(_outputs[i]) && get_node().get_program().get_config().get_enable_memory_pool()) {
            _network.get_memory_pool().release_memory(_outputs[i].get(), get_node().get_unique_id(), get_node().id(), _network.get_id());
        }
    }

    _outputs = {
        _network.get_engine().reinterpret_buffer(input_memory(0), _impl_params->get_output_layout(0)),
        _network.get_engine().reinterpret_buffer(input_memory(1), _impl_params->get_output_layout(1)),
        _network.get_engine().reinterpret_buffer(input_memory(2), _impl_params->get_output_layout(2)),
        _network.get_engine().reinterpret_buffer(input_memory(3), _impl_params->get_output_layout(3))
    };
}
}  // namespace cldnn
