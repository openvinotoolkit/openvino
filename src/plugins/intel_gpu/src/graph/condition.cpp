// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "condition_inst.h"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(condition)


static layout get_output_layout_from_inner_program(kernel_impl_params const& impl_param, size_t branch_idx) {
    std::string branch_name = (branch_idx == 0) ? "true" : "false";
    auto& outputs  = impl_param.inner_progs[branch_idx]->get_outputs();
    auto& io_output_map  = impl_param.io_output_maps[branch_idx];

    CLDNN_ERROR_NOT_EQUAL(impl_param.desc->id,
                        "Count of branch " + branch_name + " outputs",
                        io_output_map.size(),
                        "expected outputs size",
                        1,
                        "Branch " + branch_name + " should have one output.");

    auto inner_prim_id = io_output_map.at(0);
    for (size_t idx = 0; idx < outputs.size(); idx++) {
        if (outputs[idx]->id() == inner_prim_id) {
            return outputs.at(idx)->get_output_layout();
        }
    }
    OPENVINO_THROW("Not found output with prim_id: ", inner_prim_id);
}

static layout get_output_layout_from_inner_network(kernel_impl_params const& impl_param, size_t branch_idx) {
    std::string branch_name = (branch_idx == 0) ? "true" : "false";
    auto& outputs  = impl_param.inner_nets[branch_idx]->get_outputs();
    auto& io_output_map  = impl_param.io_output_maps[branch_idx];

    CLDNN_ERROR_NOT_EQUAL(impl_param.desc->id,
                        "Count of branch " + branch_name + " outputs",
                        io_output_map.size(),
                        "expected outputs size",
                        1,
                        "Branch " + branch_name + " should have one output.");

    auto inner_prim_id = io_output_map.at(0);
    for (size_t idx = 0; idx < outputs.size(); idx++) {
        if (outputs[idx]->id() == inner_prim_id) {
            return outputs.at(idx)->get_output_layout();
        }
    }
    OPENVINO_THROW("Not found output with prim_id: ", inner_prim_id);
}

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout condition_inst::calc_output_layout(condition_node const& /* node */, kernel_impl_params const& impl_param) {
    OPENVINO_ASSERT(static_cast<bool>(impl_param.desc->output_data_types[0]) == false, "Output data type forcing is not supported for condition_node!");
    OPENVINO_ASSERT(impl_param.get_input_layout(0).count() == 1, "layout of compare_data of condition should be {1,1,1,1}");

    OPENVINO_ASSERT(impl_param.inner_progs.size() == 2, "If(Condition) contains incorrect number of inner programs ", impl_param.inner_progs.size());
    OPENVINO_ASSERT(impl_param.io_output_maps.size() == 2, "If(Condition) contains incorrect number of io output maps ", impl_param.io_output_maps.size());

    auto layout_true  = get_output_layout_from_inner_program(impl_param, 0);
    auto layout_false = get_output_layout_from_inner_program(impl_param, 1);

    CLDNN_ERROR_LAYOUT_MISMATCH(impl_param.desc->id,
                                "Branch true output layout",
                                layout_true,
                                "branch false output layout",
                                layout_false,
                                "Layout of the branches should be the same.");

    return layout_true;
}

template <class T>
bool compare_data(memory::ptr mem, stream& stream) {
    mem_lock<T, mem_lock_type::read> lock_compare_data{mem, stream};
    return (static_cast<float>(*lock_compare_data.data()) != 0.f);
}

static bool get_compare_data(memory::ptr mem, stream& stream) {
    auto mem_dt = mem->get_layout().data_type;
    switch (mem_dt) {
        case cldnn::data_types::f32:
            return compare_data<float>(mem, stream);
        case cldnn::data_types::f16:
            return compare_data<half_t>(mem, stream);
        case cldnn::data_types::i64:
            return compare_data<int64_t>(mem, stream);
        case cldnn::data_types::i32:
            return compare_data<int32_t>(mem, stream);
        case cldnn::data_types::i8:
            return compare_data<int8_t>(mem, stream);
        case cldnn::data_types::u8:
            return compare_data<uint8_t>(mem, stream);
        case cldnn::data_types::bin:
        default:
            return compare_data<uint32_t>(mem, stream);
    }
}

template<typename ShapeType>
std::vector<layout> condition_inst::calc_output_layouts(condition_node const& node, kernel_impl_params const& impl_param) {
    // TODO : Add the case for cond is constant

    // In the case of parameter / equal
    if (impl_param.inner_nets.empty()) {
        OPENVINO_ASSERT(impl_param.inner_progs.empty() == false, "The count of inner programs should not be zero");
        auto layout_true  = get_output_layout_from_inner_program(impl_param, 0);
        auto layout_false = get_output_layout_from_inner_program(impl_param, 1);
        OPENVINO_ASSERT(layout_true.get_rank() == layout_false.get_rank(), "dynamic rank is not supported");
        return {layout{ov::PartialShape::dynamic(layout_true.get_rank()), layout_true.data_type, layout_true.format }};
    } else {
        auto layout_true = get_output_layout_from_inner_network(impl_param, 0);
        auto layout_false = get_output_layout_from_inner_network(impl_param, 1);

        auto& memory_deps = impl_param.memory_deps;
        OPENVINO_ASSERT(memory_deps.count(0) > 0, "");
        auto mem_ptr = memory_deps.at(0);
        auto compare_data = get_compare_data(mem_ptr, impl_param.get_stream());
        if (compare_data) {
            return {layout_true};
        } else {
            return {layout_false};
        }
    }
}

template std::vector<layout> condition_inst::calc_output_layouts<ov::PartialShape>(condition_node const& node, const kernel_impl_params& impl_param);

std::string condition_inst::to_string(condition_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite condition_info;

    node_info->add("condition info", condition_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
Condition primitive is resuing memory with the input.
*/
condition_inst::typed_primitive_inst(network& network, condition_node const& node)
    : parent(network, node),
      _net_true(network::allocate_network(node.get_program().get_engine(), node.get_branch_true().inner_program, true)),
      _net_false(network::allocate_network(node.get_program().get_engine(), node.get_branch_false().inner_program, true)) {
    this->set_inner_networks({_net_true, _net_false});
}

network::ptr condition_inst::get_inner_networks(bool is_net_true) {
    const std::string name_str = (is_net_true ? "branch_true" : "branch_false");
    auto net = is_net_true? _net_true : _net_false;
    const auto& branch = is_net_true? node->get_branch_true() : node->get_branch_false();

    std::cout << "==========================================================" << std::endl;
    std::cout << name_str << " => " << branch.id << std::endl;
    for (auto& b : branch.input_map) {
        std::cout << "- " << b.first << " : " << b.second << std::endl;
    }

    // set input memory
    for (size_t mem_idx = 0; mem_idx < inputs_memory_count(); mem_idx++) {
        const primitive_id& input_external_id = dependencies().at(mem_idx).first->id();
        std::cout << "Try to input memory match " << input_external_id << " for " << name_str  << std::endl;
        auto iter = branch.input_map.find(input_external_id);
        if (iter != branch.input_map.end()) {
            const primitive_id& input_internal_id = iter->second;
            auto mem_ptr = input_memory_ptr(mem_idx);
            net->set_input_data(input_internal_id, mem_ptr);
            std::cout << "Match input for " << name_str << " : " << input_internal_id;
            std::cout << " vs " << input_external_id << std::endl;
        }
    }

    // set output memory
    for (auto out_mem_map : branch.output_map) {
        auto idx = out_mem_map.first;
        auto out_internal_id = out_mem_map.second;
        auto mem_ptr = output_memory_ptr(idx);
        net->set_output_memory(out_internal_id, mem_ptr);
    }

    std::cout << "==========================================================" << std::endl;
    return net;
}

}  // namespace cldnn
