// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "impls/ocl/primitive_base.hpp"
#include "fully_connected_inst.h"
#include "fully_connected/fully_connected_params.h"
#include <memory>
#include <stdexcept>

using namespace cldnn;

/*
This pass checks if if primitive's input format matches implementation's input format
If not than required reorder is added to the network.
*/

/*
Add a reorder in between node and usr with reorder_layout as layout
*/
program_node& post_input_reorder::add_reorder(program& p,
                                              program_node* node,
                                              program_node* usr,
                                              const layout& reorder_layout) {
    auto new_reorder = std::make_shared<reorder>(node->id() + "_reorder_" + usr->id(), node->id(), reorder_layout);
    auto& new_reorder_node = p.get_or_create(new_reorder);

    // ToDo: add a method to program class which adds an intermediate node given a node and its user
    auto it = std::find_if(usr->get_dependencies().begin(), usr->get_dependencies().end(),
    [&](const std::pair<program_node*, int32_t>& dep) {
        return node == dep.first;
    });
    if (it == usr->get_dependencies().end()) {
        throw std::runtime_error("Inconcistency in topology description: user of a node is not present among its dependecies.");
    }
    auto idx = it - usr->get_dependencies().begin();
    if (idx < 0 || (size_t)idx >= usr->get_dependencies().size()) {
        throw std::runtime_error("Internal Error: container index out of range exception.");
    }
    p.add_intermediate(new_reorder_node, *usr, idx);
    return new_reorder_node;
}

void post_input_reorder::run(program& p) {
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = *node_itr++;
        const auto impl = node->get_selected_impl();
        // add a reorder if primitive's input format doesn't match implementation's input format
        if (node->is_type<fully_connected>()) {
            const auto fc_impl = dynamic_cast<ocl::typed_primitive_impl_ocl<fully_connected>*>(impl);
            if (!fc_impl || node->can_be_optimized())
                continue;
            const auto& fc_params =
                *static_cast<kernel_selector::fully_connected_params*>(fc_impl->_kernel_data.params.get());

            auto layout_format = from_data_layout(fc_params.inputs[0].GetLayout());
            auto& input = node->get_dependencies()[0].first;
            auto input_layout = input->get_output_layout();

            if (input_layout.format != layout_format) {
                auto previous_layout = node->get_output_layout();
                layout current_layout(input_layout.get_partial_shape(),
                                      input_layout.data_type,
                                      layout_format,
                                      input_layout.data_padding);
                auto& reorder = add_reorder(p, input, node, current_layout);
                reorder.set_unique_id();
                reorder.get_output_layout(false);
                node->set_output_layout(previous_layout, false);
                reorder.set_selected_impl(reorder.type()->create_impl(reorder));
                if (auto impl = reorder.get_selected_impl()) {
                    auto params = reorder.get_kernel_impl_params();
                    p.get_kernels_cache().add_kernels_source(*params, impl->get_kernels_source());
                }
            }
        }
    }
}
