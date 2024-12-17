// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "pass_manager.h"
#include "permute_inst.h"
#include "program_helpers.h"
#include "reorder_inst.h"
#include "reshape_inst.h"

using namespace cldnn;

void reshape_transfer::run(program& p) {
    // (reorder) + reshape + transpose
    // sink reshape for further possible optimization
    auto is_suitable_permute = [](cldnn::program_node* node) {
        return node->get_users().size() == 1 && node->is_dynamic() == false &&
               node->get_output_layout().get_rank() == 4;
    };

    auto is_suitable_reshape = [](cldnn::program_node* node) -> bool {
        if (node->get_users().size() != 1 || node->is_dynamic())
            return false;
        auto& input_lay = node->get_input_layout(0);
        auto& output_lay = node->get_output_layout();
        if (input_lay.compatible(output_lay))
            return true;
        return false;
    };

    std::function<bool(cldnn::program_node*)> is_suitable_parent;
    is_suitable_parent = [&is_suitable_parent](const cldnn::program_node* node) -> bool {
        if (node->get_users().size() != 1 || node->is_dynamic())
            return false;
        for (size_t idx = 0; idx < node->get_dependencies().size(); idx++) {
            auto& input = node->get_dependency(idx);
            if (!input.is_in_data_flow() || input.is_constant())
                continue;
            if (node->is_type<convolution>() || input.is_type<convolution>()) {
                return true;
            } else if (input.is_type<eltwise>() && input.get_dependency(1).is_constant()) {
                return is_suitable_parent(&input);
            } else if (input.is_type<activation>()) {
                return is_suitable_parent(&input);
            }
            return false;
        }
        return false;
    };

    auto update_order = [](std::vector<uint16_t> original_order, cldnn::program_node* reshape) {
        if (!reshape)
            return original_order;
        // Example. For this sequence, there is Reshape node which merges 2 consecutive dims into one
        // order must be updated like permute is done before reshape
        // [1,3,4,6] -> Reshape[1,3,24,1]-> permute(0,2,1) -> [1,24,3,1]
        // updated order must be (0,2,3,1):
        // dim with index=2 is split into 2 parts: 2 and 3
        const auto& reshape_in_shape = reshape->get_input_layout().get_dims();
        const auto& reshape_out_shape = reshape->get_output_layout().get_dims();
        auto transformed_order = original_order;
        ov::Shape new_shape(transformed_order.size());
        const uint16_t merge_dim_idx = [&]() {
            for (uint16_t i = 0; i < reshape_in_shape.size(); ++i) {
                if (reshape_in_shape[i] != reshape_out_shape[i])
                    return i;
            }
            OPENVINO_THROW("same input/output for reshape node");
        }();
        auto insertIt = transformed_order.end();
        for (auto it = transformed_order.begin(); it != transformed_order.end(); ++it) {
            auto& elem = *it;
            if (elem > merge_dim_idx) {
                elem++;
            } else if (elem == merge_dim_idx) {
                insertIt = it + 1;
            }
        }
        transformed_order.insert(insertIt, merge_dim_idx + 1);
        // remove invalid orders
        if (transformed_order.size() > reshape_out_shape.size()) {
            transformed_order.erase(
                std::remove_if(transformed_order.begin(), transformed_order.end(), [&](uint16_t& order) {
                    return order >= reshape_out_shape.size();
                }));
        }
        return transformed_order;
    };

    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;
        if (!node->is_type<permute>())
            continue;
        auto& transpose_node = node->as<permute>();
        if (!is_suitable_permute(&transpose_node))
            continue;
        auto& child_node = transpose_node;
        auto parent_node = child_node.get_dependency_with_port(0).first;
        cldnn::program_node* inter_node;
        if (parent_node->is_type<reshape>()) {
            inter_node = parent_node;
            if (!is_suitable_reshape(inter_node)) {
                continue;
            }
            parent_node = inter_node->get_dependency_with_port(0).first;
        } else {
            continue;
        }

        if (!is_suitable_parent(parent_node)) {
            continue;
        }
        reshape_node* reshape_node = nullptr;
        if (inter_node && inter_node->is_type<reshape>())
            reshape_node = &(inter_node->as<reshape>());

        auto transpose_order = update_order(transpose_node.get_permute_order(), reshape_node);
        auto new_permute =
            std::make_shared<permute>(transpose_node.id() + "_reordered", parent_node->id(), transpose_order);
        auto& new_permute_node = p.get_or_create(new_permute);
        if (new_permute_node.as<permute>().is_rotating_except_batch()) {
            auto next_node = transpose_node.get_users().front();
            auto new_reshape_tensor = transpose_node.get_output_layout().get_tensor();
            p.move_node(*reshape_node, *node, *next_node);
            // replace the permute node and reshape node
            auto new_reshape =
                std::make_shared<reshape>(reshape_node->id() + "_sinked", new_permute_node.id(), new_reshape_tensor);
            auto& new_reshape_node = p.get_or_create(new_reshape);

            p.replace(transpose_node, new_permute_node);
            p.replace(*reshape_node, new_reshape_node);
            new_permute_node.recalc_output_layout(false);
            new_reshape_node.recalc_output_layout(false);
        } else {
            p.remove_if_dangling(new_permute_node);
        }
    }
}
