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

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout condition_inst::calc_output_layout(condition_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for condition_node!");

    OPENVINO_ASSERT(node.get_dependency(0).get_output_layout().count() == 1,
                    "layout of compare_data of condition should be {1,1,1,1}");

    auto branch_true_output = node.get_branch_true()->get_outputs();
    auto branch_false_output = node.get_branch_false()->get_outputs();

    CLDNN_ERROR_NOT_EQUAL(impl_param.desc->id,
                          "Count of branch true outputs",
                          branch_true_output.size(),
                          "expected outputs size",
                          1,
                          "Branch true should have one output.");
    CLDNN_ERROR_NOT_EQUAL(impl_param.desc->id,
                          "Count of branch false outputs",
                          branch_false_output.size(),
                          "expected outputs size",
                          1,
                          "Branch false should have one output.");

    auto layout_true = branch_true_output.at(0)->get_output_layout();
    auto layout_false = branch_false_output.at(0)->get_output_layout();

    CLDNN_ERROR_LAYOUT_MISMATCH(impl_param.desc->id,
                                "Branch true output layout",
                                layout_true,
                                "branch false output layout",
                                layout_false,
                                "Layout of the branches should be the same.");

    return layout_true;
}

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
      _net_true(network::allocate_network(node.get_program().get_engine(), node.get_branch_true(), true)),
      _net_false(network::allocate_network(node.get_program().get_engine(), node.get_branch_false(), true)) {
}

network::ptr condition_inst::get_inner_networks(bool is_net_true) {
    auto net = is_net_true? _net_true : _net_false;
    auto& branch = is_net_true? node->get_primitive()->branch_true : node->get_primitive()->branch_false;

    // set input memory
    for (size_t mem_idx = 0; mem_idx < inputs_memory_count(); mem_idx++) {
        const primitive_id& input_external_id = dependencies().at(mem_idx).first->id();
        auto iter = branch.input_map.find(input_external_id);
        if (iter != branch.input_map.end()) {
            const primitive_id& input_internal_id = iter->second;
            auto mem_ptr = input_memory_ptr(mem_idx);
            net->set_input_data(input_internal_id, mem_ptr);
        }
    }

    // set output memory
    for (auto out_mem_map : branch.output_map) {
        auto idx = out_mem_map.first;
        auto out_internal_id = out_mem_map.second;
        auto mem_ptr = output_memory_ptr(idx);
        net->set_output_memory(out_internal_id, mem_ptr);
    }

    return net;
}

}  // namespace cldnn
