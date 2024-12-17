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
bool reorder_reshape_transpose_fuse::check_order(const std::vector<uint16_t>& transpose_order,
                                                 const std::vector<size_t>& layout_order,
                                                 const std::vector<size_t>& reorder_inorder,
                                                 const std::vector<size_t>& reorder_outorder) {
    if (transpose_order.size() != layout_order.size() || layout_order.size() != reorder_inorder.size() ||
        reorder_inorder.size() != reorder_outorder.size()) {
        return false;
    }
    auto rev_layout_order = std::vector<size_t>(layout_order.size());
    for (size_t i = 0; i < rev_layout_order.size(); i++) {
        rev_layout_order[layout_order[i]] = i;
    }

    auto new_transpose_order = std::vector<uint16_t>(transpose_order.size());
    for (size_t i = 0; i < new_transpose_order.size(); i++) {
        new_transpose_order[i] = layout_order[transpose_order[rev_layout_order[i]]];
    }

    auto reorder_order = std::vector<size_t>(reorder_outorder.size());
    for (size_t i = 0; i < reorder_order.size(); i++) {
        for (size_t j = 0; j < reorder_order.size(); j++) {
            if (reorder_outorder[i] == reorder_inorder[j]) {
                reorder_order[i] = j;
                continue;
            }
        }
    }

    auto summary_order = std::vector<uint16_t>(transpose_order.size());
    for (size_t i = 0; i < summary_order.size(); i++) {
        summary_order[i] = reorder_order[new_transpose_order[i]];
    }

    for (size_t i = 0; i < summary_order.size(); i++) {
        if (summary_order[i] != i) {
            return false;
        }
    }
    return true;
}

void reorder_reshape_transpose_fuse::run(program& p) {
    bool update_processing_order = false;
    // temp code to validate reorder + reshape + permute opt
    // other patterns to consider: permute + (reshape) + reorder?
    auto is_suitable_reorder = [](cldnn::program_node* node) {
        return node->get_users().size() == 1 && node->is_dynamic() == false;
    };
    auto is_suitable_reshape = [](cldnn::program_node* node) {
        if (node->get_users().size() != 1 || node->is_dynamic())
            return false;
        const auto& in_shape = node->get_input_layout(0).get_dims();
        const auto& out_shape = node->get_output_layout().get_dims();
        return in_shape.size() == out_shape.size();
    };
    auto is_suitable_transpose = [](cldnn::program_node* node) {
        return node->get_users().size() == 1 && node->is_dynamic() == false;
    };
    auto update_order = [](std::vector<uint16_t> original_order, cldnn::program_node* reshape) {
        if (!reshape)
            return original_order;
        // Example. For this sequence:
        // [1,3,4,6] -> Reshape[1,3,24,1]-> [1,24,3,1]
        // org order as (0,2,1)
        // first reshape to [1,3,24] ->transpose(0,2,1) -> [1,24,3]
        // updated order must be (0,2,3,1):
        // dim with index=2 is split into 2 parts: 2 and 3
        const auto& reshape_in_shape = reshape->get_input_layout().get_dims();
        const auto& reshape_out_dim = reshape->get_output_layout().get_dims();
        auto reshape_out_shape = reshape_out_dim;
        auto transformed_order = original_order;
        ov::Shape new_shape(transformed_order.size());
        if (original_order.size() < reshape_out_dim.size() && reshape_out_dim.size() == 4) {
            // if order dims is less than reshape dims, means reshape shape has been converted to upper dims
            // merge spatial dims
            reshape_out_shape.resize(original_order.size());
            for (size_t i = 0; i < reshape_out_dim.size(); ++i) {
                if (i < 2) {
                    reshape_out_shape[i] = reshape_out_dim[i];
                } else {
                    reshape_out_shape[2] *= reshape_out_dim[i];
                }
            }
            const size_t merge_dim_idx = [&]() {
                for (size_t i = 0; i < reshape_in_shape.size(); ++i) {
                    if (reshape_in_shape[i] != reshape_out_shape[i])
                        return i;
                }
                OPENVINO_THROW("merged_dim_idx can not be found");
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
        } else {
            auto reorder_orders = [](std::vector<uint16_t>& order, std::vector<uint16_t> place_order) {
                // for all elements to put in place
                for (size_t i = 0; i < order.size() - 1; ++i) {
                    while (i != place_order[i]) {
                        // swap it with the element at its final place
                        auto alt = place_order[i];
                        std::swap(order[i], order[alt]);
                        std::swap(place_order[i], place_order[alt]);
                    }
                }
            };
            reorder_orders(transformed_order, std::vector<uint16_t>({0, 1, 3, 2}));
        }
        return transformed_order;
    };

    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr++;
        if (!node->is_type<permute>())
            continue;
        auto& transpose_node = node->as<permute>();
        if (transpose_node.id() == "transpose:/detect/Transpose")
            std::cout << "break" << std::endl;
        if (!is_suitable_transpose(&transpose_node))
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
            continue; // to be matched further
        }

        if (!is_suitable_reorder(parent_node)) {
            continue;
        }
        auto& reshape_node = inter_node->as<reshape>();  // TODO: check null reshape node

        auto transpose_order = update_order(transpose_node.get_permute_order(), &reshape_node);
        auto next_node = transpose_node.get_users().front();
        auto next_layout = next_node->get_input_layout();
        auto order_after_transpose = next_node->get_output_layout().get_dims_order();
        auto reorder_in_dims_order = parent_node->get_input_layout().get_dims_order();
        auto reorder_out_dims_order = parent_node->get_output_layout().get_dims_order();

        if (check_order(transpose_order, order_after_transpose, reorder_in_dims_order, reorder_out_dims_order)) {
            std::cout << "debug: " << transpose_node.id() << std::endl;
            // qualified for merge
            // making new reorder
            const auto& prev_node = parent_node->get_dependency_with_port(0).first;
            auto new_reorder = std::make_shared<reorder>(parent_node->id() + reshape_node.id() + transpose_node.id(),
                                                         prev_node->id(),
                                                         parent_node->get_output_layout());
            std::vector<int> permute_order(transpose_order.size());
            std::copy_n(transpose_order.begin(), transpose_order.size(), permute_order.begin());
            new_reorder->set_src_permutation(permute_order);
            auto& new_reorder_node = p.get_or_create(new_reorder);
            p.remove_all_connections(transpose_node);
            p.remove_all_connections(reshape_node);
            p.remove_all_connections(*parent_node);
            p.remove_if_dangling(transpose_node);
            p.remove_if_dangling(reshape_node);
            p.remove_if_dangling(*parent_node);
            p.add_connection(*prev_node, *next_node);
            p.add_intermediate(new_reorder_node, *next_node, *prev_node);
            new_reorder_node.recalc_output_layouts(false);
            new_reorder_node.can_be_optimized(true);
            update_processing_order = true;

            // if shapes don't match, another reshape must be inserted to perform shape alignment with next node
            if (next_layout.get_dims() != new_reorder_node.get_output_layout().get_dims()) {
                auto new_reshape = std::make_shared<reshape>(parent_node->id() + reshape_node.id() + transpose_node.id() + "fake_reshape",
                                                         next_node->id(),
                                                         next_layout.get_tensor());
                auto& new_reshape_node = p.get_or_create(new_reshape);
                p.add_intermediate(new_reshape_node, *next_node, new_reorder_node);
            }
        }
    }
    if (update_processing_order) {
        p.get_processing_order().calc_processing_order(p);
    }
}
