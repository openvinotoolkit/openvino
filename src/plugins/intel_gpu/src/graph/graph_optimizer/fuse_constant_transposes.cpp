// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"

#include "permute_inst.h"
#include "fully_connected_inst.h"

using namespace cldnn;

format fuse_constant_transposes::convert_weights_format_by_order(format fmt, const std::vector<uint16_t>& order) const {
    format weights_fmt = from_weights_layout(to_weights_layout(fmt, false));
    const auto& old_order = weights_fmt.dims_order();
    auto new_order = old_order;

    for (size_t i = 0; i < order.size(); ++i) {
        new_order[i] = old_order[order[i]];
    }

    return format::find_format(new_order, fmt.block_sizes(), true);
}

void fuse_constant_transposes::run(program& p) {
    std::function<bool(const cldnn::program_node*)> is_matmul_weights_path =
        [&is_matmul_weights_path](const cldnn::program_node* node) {
        if (node->get_users().empty())
            return false;

        const auto* next_node = node->get_users().front();

        if (next_node->is_type<fully_connected>()) {
            size_t weights_offset = next_node->get_primitive()->input_size();
            return &next_node->get_dependency(weights_offset) == node;
        }

        if (node->is_constant() && node->get_users().size() == 1)
            return is_matmul_weights_path(next_node);

        return false;
    };

    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;

        if (!node->is_type<permute>())
            continue;

        auto& permute_node = node->as<permute>();

        if (!is_matmul_weights_path(&permute_node) || !permute_node.get_dependency(0).is_type<data>())
            continue;

        auto& prev_const = permute_node.get_dependency(0).as<data>();
        const auto& permute_order = permute_node.get_primitive()->permute_order;

        format new_fmt = convert_weights_format_by_order(prev_const.get_output_layout().format, permute_order);

        layout updated_const_layout = prev_const.get_output_layout();
        updated_const_layout.format = new_fmt;
        updated_const_layout.set_partial_shape(permute_node.get_output_pshape());

        p.extract_and_remove(permute_node);

        const auto& new_mem = p.get_engine().reinterpret_buffer(prev_const.get_attached_memory(), updated_const_layout);

        auto new_const_prim = std::make_shared<data>(prev_const.id() + "_fused_with_transpose", new_mem);
        auto& new_const_node = p.get_or_create(new_const_prim);

        p.replace(prev_const, new_const_node);
        new_const_node.recalc_output_layout(false);
    }
}
