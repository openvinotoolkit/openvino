// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "reorder_inst.h"
#include "permute_inst.h"
#include "program_helpers.h"

using namespace cldnn;

void reorder_transfer::run(program& p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;

        if (!node->is_type<reorder>())
            continue;

        auto& reorder_node = node->as<reorder>();

        bool is_simple_reorder = reorder_node.is_constant() &&
                                 !reorder_node.is_output() &&
                                 reorder_node.get_users().size() == 1 &&
                                 reorder_node.get_dependencies().size() == 1 &&
                                 !reorder_node.has_mean() &&
                                 reorder_node.get_primitive()->subtract_per_feature.empty();
        if (!is_simple_reorder)
            continue;

        layout input_layout = reorder_node.get_input_layout();
        layout output_layout = reorder_node.get_output_layout();
        bool only_precision_changed = input_layout.get_partial_shape() == output_layout.get_partial_shape() &&
                                      input_layout.format == output_layout.format &&
                                      input_layout.data_type != output_layout.data_type;
        if (!only_precision_changed)
            continue;

        auto transfer_through_node = [](cldnn::program_node* node) -> bool { // Conditions can be extended to other ops
            return node->is_type<permute>() &&
                   node->get_users().size() == 1 &&
                   node->get_dependencies().size() == 1;
        };

        auto change_output_dtype = [](cldnn::program_node* node, cldnn::data_types dtype) {
            layout new_layout = node->get_output_layout();
            new_layout.data_type = dtype;
            node->set_output_layout(new_layout, false);
        };

        auto* supposed_new_prev = reorder_node.get_users().front();
        cldnn::program_node* new_prev = nullptr;
        while (transfer_through_node(supposed_new_prev)) {
            change_output_dtype(supposed_new_prev, reorder_node.get_input_layout().data_type);
            new_prev = supposed_new_prev;
            supposed_new_prev = supposed_new_prev->get_users().front();
        }

        if (new_prev != nullptr) {
            auto& new_next = new_prev->get_users().front();
            p.move_node(reorder_node, *new_prev, *new_next);
            reorder_node.recalc_output_layout(false);
        }
    }
}
