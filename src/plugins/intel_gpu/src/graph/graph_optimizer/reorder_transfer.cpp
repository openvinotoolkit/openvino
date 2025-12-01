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

        bool is_simple_type_conversion_reorder = !reorder_node.is_output() &&
                                                 reorder_node.get_users().size() == 1 &&
                                                 reorder_node.get_dependencies().size() == 1 &&
                                                 reorder_node.is_type_conversion_only();
        if (!is_simple_type_conversion_reorder)
            continue;

        auto type_rank = [](ov::element::Type_t t) -> int {
            switch (t) {
                case ov::element::Type_t::boolean: return 0;
                case ov::element::Type_t::i8:      return 1;
                case ov::element::Type_t::i16:     return 2;
                case ov::element::Type_t::i32:     return 4;
                case ov::element::Type_t::i64:     return 8;
                case ov::element::Type_t::u8:      return 1;
                case ov::element::Type_t::u16:     return 2;
                case ov::element::Type_t::u32:     return 4;
                case ov::element::Type_t::u64:     return 8;
                case ov::element::Type_t::bf16:    return 2;
                case ov::element::Type_t::f16:     return 2;
                case ov::element::Type_t::f32:     return 4;
                case ov::element::Type_t::f64:     return 8;
                default:                           return -1;
            }
        };

        int input_rank  = type_rank(reorder_node.get_input_layout().data_type),
            output_rank = type_rank(reorder_node.get_output_layout().data_type);
        bool is_lower_to_higher = input_rank < output_rank;

        if (!is_lower_to_higher || input_rank == -1 || output_rank == -1)
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
