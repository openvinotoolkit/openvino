// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "program_helpers.h"

#include "group_normalization_inst.h"
#include "convolution_inst.h"

using namespace cldnn;

void conv_groupnorm_stat::run(program& p) {
    bool update_processing_order = false;
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = *(itr++);
        if (!node->is_dynamic() && node->is_type<group_normalization>()) {
            auto &gn_node = node->as<group_normalization>();
            auto gn_prim = gn_node.get_primitive();
            auto parent = gn_node.get_dependencies()[0].first;
            if (!parent->is_dynamic() && parent->is_type<convolution>()) {
                auto conv_prim = parent->as<convolution>().get_primitive();
                auto weights_nodes_id = conv_prim->weights;
                auto biases_nodes_id = conv_prim->bias;
                auto sumx_connection = input_info(gn_prim->get_dependency(0).pid, 1);
                auto sumxsq_connection = input_info(gn_prim->get_dependency(0).pid, 2);
                auto new_gn_prim = std::make_shared<group_normalization>(gn_node.id() + "_tmp",
                                                           gn_prim->get_dependency(0),
                                                           gn_prim->get_dependency(1),
                                                           gn_prim->get_dependency(2),
                                                           sumx_connection,
                                                           sumxsq_connection,
                                                           gn_prim->num_groups,
                                                           gn_prim->epsilon);
                program_node& new_gn_node = p.get_or_create(new_gn_prim);
                p.replace(gn_node, new_gn_node);
                p.add_connection(*parent, new_gn_node, 1);
                p.add_connection(*parent, new_gn_node, 2);
                auto batch_size = parent->get_output_pshape(0)[0];
                auto new_conv_prim = std::make_shared<convolution>(parent->id() + "_tmp",
                                                           conv_prim->get_dependency(0),
                                                           weights_nodes_id.pid,
                                                           biases_nodes_id.is_valid() ? biases_nodes_id.pid : "",
                                                           conv_prim->groups,
                                                           conv_prim->stride,
                                                           conv_prim->dilation,
                                                           conv_prim->padding_begin,
                                                           conv_prim->padding_end,
                                                           conv_prim->grouped_weights_shape,
                                                           ov::op::PadType::EXPLICIT,
                                                           gn_prim->num_groups);
                program_node& new_conv_node = p.get_or_create(new_conv_prim);
                p.replace(*parent, new_conv_node);
                auto stat_layout = layout({batch_size, gn_prim->num_groups, 1, 1}, data_types::f32, format::bfyx);
                new_conv_node.set_output_layout(stat_layout, false, 1);
                new_conv_node.set_output_layout(stat_layout, false, 2);
                update_processing_order = true;
            }
        }
    }
    if (update_processing_order) {
        p.get_processing_order().calc_processing_order(p);
    }
}
