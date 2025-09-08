// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "pooling_inst.h"
#include "convolution_inst.h"
#include "fully_connected_inst.h"
#include "data_inst.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/graph/program.hpp"

#include <vector>
#include <tuple>

using namespace cldnn;

namespace {

using shuffle_range = std::pair<int32_t, int32_t>;

bool can_shuffle_features(program_node& node, program_node& concat_node, stream& stream) {
    if (node.is_type<convolution>()) {
        auto& conv_node = node.as<convolution>();
        auto& wei_node = conv_node.weights();
        if (ov::element::Type(wei_node.get_output_layout().data_type).bitwidth() < 8)
            return false;

        return conv_node.get_groups() == 1 && node.get_dependency_index(concat_node) == 0 &&
            conv_node.get_deformable_groups() == 1 && !conv_node.get_transposed() &&
            !conv_node.activations_zero_points_term() &&
            wei_node.is_type<data>() && wei_node.is_constant() && !wei_node.is_output();
    }
    if (node.is_type<fully_connected>()) {
        auto& fc_node = node.as<fully_connected>();
        auto& wei_node = fc_node.weights();
        if (ov::element::Type(wei_node.get_output_layout().data_type).bitwidth() < 8)
            return false;

        return node.get_dependency_index(concat_node) == 0 && wei_node.is_type<data>() && wei_node.is_constant() && !wei_node.is_output();
    }

    bool pass_through = false;
    pass_through |= node.is_type<activation>();
    pass_through |= node.is_type<pooling>();
    // General conditions for pass-through layers
    pass_through &= !node.is_output() && node.get_dependencies().size() == 1 && !node.has_fused_primitives();
    if (pass_through) {
        // Primitives that are feature order invariant, pass-through shuffled features to users
        for (auto& user : node.get_users()) {
            if (!can_shuffle_features(*user, node, stream))
                return false;
        }
        return true;
    }

    return false;
}

void shuffle_weights(data_node& node, const std::vector<shuffle_range>& ranges, stream& stream) {
    // Correct for shuffled features by shuffling input feature dimension in weights.
    // This allows to restore correct feature order on output and only changes calculation order.
    auto wei_layout = node.get_output_layout();
    auto old_weights_memory = node.get_attached_memory_ptr();
    bool need_reset = static_cast<bool>(wei_layout.data_padding) || wei_layout.format.is_blocked();
    auto new_weights_memory = old_weights_memory->get_engine()->allocate_memory(wei_layout, old_weights_memory->get_allocation_type(), need_reset);

    auto bytes_per_elem = data_type_traits::size_of(wei_layout.data_type);
    mem_lock<uint8_t, mem_lock_type::read> old_weights_memory_lock{old_weights_memory, stream};
    mem_lock<uint8_t, mem_lock_type::write> new_weights_memory_lock{new_weights_memory, stream};
    auto old_ptr = old_weights_memory_lock.data();
    auto new_ptr = new_weights_memory_lock.data();
    for (int32_t ofi = 0; ofi < wei_layout.batch(); ++ofi) {
        int32_t new_ifi = 0;
        for (auto& range : ranges) {
            for (int32_t ifi = range.first; ifi < range.second; ++ifi, ++new_ifi) {
                for (int32_t wi = 0; wi < wei_layout.spatial(3); ++wi) {
                    for (int32_t zi = 0; zi < wei_layout.spatial(2); ++zi) {
                        for (int32_t yi = 0; yi < wei_layout.spatial(1); ++yi) {
                            for (int32_t xi = 0; xi < wei_layout.spatial(0); ++xi) {
                                auto old_coords = tensor(batch(ofi), feature(ifi), spatial(xi, yi, zi, wi));
                                auto new_coords = tensor(batch(ofi), feature(new_ifi), spatial(xi, yi, zi, wi));
                                auto old_offset = wei_layout.get_linear_offset(old_coords);
                                auto new_offset = wei_layout.get_linear_offset(new_coords);
                                for (size_t byte = 0; byte < bytes_per_elem; ++byte) {
                                    new_ptr[new_offset * bytes_per_elem + byte] = old_ptr[old_offset * bytes_per_elem + byte];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    node.attach_memory(new_weights_memory, false);
}

void shuffle_features(program_node& node, const std::vector<shuffle_range>& ranges, stream& stream) {
    if (node.is_type<convolution>()) {
        auto& conv = node.as<convolution>();
        shuffle_weights(conv.weights().as<data>(), ranges, stream);
    } else if (node.is_type<fully_connected>()) {
        auto& fc = node.as<fully_connected>();
        shuffle_weights(fc.weights().as<data>(), ranges, stream);
    } else {
        // General case for pass-through layers
        for (auto& user : node.get_users()) {
            shuffle_features(*user, ranges, stream);
        }
    }
}

}  // namespace

void concat_input_order::run(program& p) {
    for (auto node : p.get_processing_order()) {
        // Check that optimization can be performed:
        // 1. Not an output
        // 2. Concatenation along features
        // 3. Currently only fsv16 format on input/output
        // 4. Not already aligned
        // 5. Users can accept shuffled features
        // 6. No fused primitives
        if (!node->is_type<concatenation>() || node->is_output() || node->is_dynamic())
            continue;

        auto& concat_node = node->as<concatenation>();
        auto prim = concat_node.get_primitive();

        bool along_f = prim->axis == 1;
        size_t inputs_count = prim->input_size();
        bool no_fusing = !concat_node.has_fused_primitives() && concat_node.get_dependencies().size() == inputs_count;

        auto out_format = concat_node.get_output_layout().format;
        bool correct_format = (out_format == format::b_fs_yx_fsv16) || (out_format == format::b_fs_yx_fsv32);
        tensor::value_type alignment = 1;
        if (out_format == format::b_fs_yx_fsv16)
            alignment = 16;
        else if (out_format == format::b_fs_yx_fsv32)
            alignment = 32;

        bool single_format = true;
        std::vector<tensor::value_type> feature_sizes;
        feature_sizes.reserve(inputs_count);
        for (size_t input_idx = 0; input_idx < inputs_count; ++input_idx) {
            auto& dep = concat_node.get_dependency(input_idx);
            auto dep_layout = dep.get_output_layout();
            single_format &= dep_layout.format == out_format;
            feature_sizes.push_back(dep_layout.feature());
        }

        // Alignment is not optimal if aligned input follows unaligned one
        bool already_aligned = true;
        for (size_t i = 1; i < feature_sizes.size(); ++i) {
            bool current_aligned = feature_sizes[i] % alignment == 0;
            bool previous_aligned = feature_sizes[i - 1] % alignment == 0;
            already_aligned &= previous_aligned || !current_aligned;
        }
        // Check that we can fuse shuffling to users
        bool can_shuffle_users = true;
        for (auto user : concat_node.get_users()) {
            can_shuffle_users &= can_shuffle_features(*user, concat_node, p.get_stream());
        }

        if (!along_f || !no_fusing || !correct_format || !single_format || already_aligned || !can_shuffle_users)
            continue;

        // Perform the optimization
        // Calculate new input order - first inputs preserving alignment, then rest
        std::vector<size_t> new_order;
        new_order.reserve(inputs_count);
        for (size_t i = 0; i < feature_sizes.size(); ++i) {
            if (feature_sizes[i] % alignment == 0)
                new_order.push_back(i);
        }
        for (size_t i = 0; i < feature_sizes.size(); ++i) {
            if (feature_sizes[i] % alignment != 0)
                new_order.push_back(i);
        }
        // Calculate new ranges
        int32_t current_offset = 0;
        std::vector<shuffle_range> original_ranges;
        original_ranges.reserve(inputs_count);
        for (auto& feature_size : feature_sizes) {
            original_ranges.emplace_back(current_offset, current_offset + feature_size);
            current_offset += feature_size;
        }
        std::vector<shuffle_range> shuffled_ranges;
        shuffled_ranges.reserve(inputs_count);
        for (auto& ord : new_order) {
            shuffled_ranges.push_back(original_ranges[ord]);
        }
        // Change input order
        std::vector<std::pair<program_node*, int32_t>> new_dependencies = {};
        new_dependencies.reserve(inputs_count);
        for (auto& ord : new_order) {
            new_dependencies.push_back({concat_node.get_dependency_with_port(ord).first, concat_node.get_dependency_with_port(ord).second});
        }
        // Update in place with const cast instead of replacing
        auto& dependencies = concat_node.get_dependencies();
        auto& mutable_dependencies = const_cast<std::vector<std::pair<program_node*, int32_t>>&>(dependencies);
        for (size_t i = 0; i < new_dependencies.size(); ++i) {
            mutable_dependencies[i] = new_dependencies[i];
        }
        std::vector<input_info> new_input_info;
        new_input_info.reserve(inputs_count);
        for (auto& ord : new_order) {
            new_input_info.push_back(input_info(prim->input[ord].pid, prim->input[ord].idx));
        }
        auto mutable_prim = std::const_pointer_cast<concatenation>(prim);
        mutable_prim->input = new_input_info;
        // Correct users for shuffled features
        for (auto& user : concat_node.get_users()) {
            shuffle_features(*user, shuffled_ranges, p.get_stream());
        }
    }
}
