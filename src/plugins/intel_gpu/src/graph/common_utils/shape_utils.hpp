// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/shape_util.hpp"

namespace cldnn {

inline bool shapes_are_broadcastable(const ov::PartialShape& source_shape,
                                     const ov::PartialShape& target_shape,
                                     bool require_equal_rank,
                                     bool source_to_target_only = false) {
    if (source_shape.is_dynamic() || target_shape.is_dynamic()) {
        return false;
    }
    if (source_to_target_only) {
        if (source_shape.size() > target_shape.size()) {
            return false;
        }
    } else if (source_shape.size() != target_shape.size() && require_equal_rank) {
        return false;
    }

    const auto dimensions_to_check = std::min(source_shape.size(), target_shape.size());
    for (size_t index = 0; index < dimensions_to_check; ++index) {
        if (source_shape[index] != 1 && (source_to_target_only || target_shape[index] != 1) && source_shape[index] != target_shape[index]) {
            return false;
        }
    }
    return true;
}

inline ov::PartialShape extend_shape_to_rank_from_end(ov::PartialShape shape, size_t rank = 4) {
    if (shape.size() >= rank) {
        return shape;
    }
    shape.insert(shape.end(), rank - shape.size(), ov::Dimension(1));
    return shape;
}

inline ov::PartialShape extend_shape_to_rank_from_begin(const ov::PartialShape& shape, size_t rank = 4) {
    if (shape.size() >= rank) {
        return shape;
    }
    ov::PartialShape extended_shape(std::vector<int64_t>(rank - shape.size(), 1));
    extended_shape.insert(extended_shape.end(), shape.begin(), shape.end());
    return extended_shape;
}

// Computes a lower-rank representation of a fused eltwise peer when collapsing its leading spatial
// axes is an order-preserving reshape and the result broadcasts directly to the host layout.
inline std::optional<ov::PartialShape> fold_higher_rank_fused_peer(const layout& peer_layout, const layout& host_layout) {
    const auto& peer_shape = peer_layout.get_partial_shape();
    const auto& host_shape = host_layout.get_partial_shape();

    const size_t peer_rank = peer_shape.size();
    const size_t host_rank = host_shape.size();
    if (peer_rank <= host_rank || host_rank < 3)
        return std::nullopt;
    if (peer_shape.is_dynamic() || host_shape.is_dynamic())
        return std::nullopt;
    if (peer_layout.data_padding || host_layout.data_padding)
        return std::nullopt;

    const auto& peer_format = peer_layout.format;
    const auto& host_format = host_layout.format;
    if (!format::is_default_format(peer_format) || !format::is_default_format(host_format))
        return std::nullopt;
    OPENVINO_ASSERT(format::adjust_to_rank(peer_format, host_rank) == host_format, "Default format rank adjustment must match the host's default format");

    const auto peer_dims = peer_shape.to_shape();
    const auto host_dims = host_shape.to_shape();
    const size_t fold_count = peer_rank - host_rank + 1;
    ov::Shape folded_dims;
    folded_dims.reserve(host_rank);
    folded_dims.push_back(peer_dims[0]);
    folded_dims.push_back(peer_dims[1]);

    size_t grouped = 1;
    for (size_t index = 2; index < 2 + fold_count; ++index) {
        const auto grouped_size = ov::util::shape_size_safe({grouped, peer_dims[index]});
        if (!grouped_size.has_value())
            return std::nullopt;
        grouped = grouped_size.value();
    }
    folded_dims.push_back(grouped);
    folded_dims.insert(folded_dims.end(), peer_dims.begin() + 2 + fold_count, peer_dims.end());

    const auto peer_total = ov::util::shape_size_safe(peer_dims);
    const auto folded_total = ov::util::shape_size_safe(folded_dims);
    if (!peer_total.has_value() || !folded_total.has_value())
        return std::nullopt;
    OPENVINO_ASSERT(peer_total.value() == folded_total.value(), "Peer folding must preserve the total element count");
    if (folded_dims.size() != host_dims.size())
        return std::nullopt;

    for (size_t index = 0; index < folded_dims.size(); ++index) {
        if (folded_dims[index] != 1 && folded_dims[index] != host_dims[index])
            return std::nullopt;
    }

    return ov::PartialShape(folded_dims);
}

inline kernel_impl_params canonicalize_fused_shapes(const kernel_impl_params& impl_params) {
    auto updated_impl_params = impl_params;
    const bool require_equal_rank = impl_params.prog->is_new_shape_infer();

    for (auto& descriptor : updated_impl_params.fused_desc) {
        if (descriptor.is_type<eltwise>() && descriptor.total_num_deps == 2 && descriptor.has_outer_dep()) {
            if (updated_impl_params.input_layouts.size() > static_cast<size_t>(descriptor.outer_dep_start_idx)) {
                const auto& output_layout = updated_impl_params.output_layouts[0];
                const auto& output_shape = output_layout.get_partial_shape();
                auto& dependency_layout = updated_impl_params.input_layouts[descriptor.outer_dep_start_idx];
                const auto& dependency_shape = dependency_layout.get_partial_shape();

                if (dependency_shape.size() > output_shape.size()) {
                    auto folded_shape = fold_higher_rank_fused_peer(dependency_layout, output_layout);
                    OPENVINO_ASSERT(folded_shape.has_value(),
                                    "Unfoldable higher-rank fused eltwise peer reached canonicalization; "
                                    "can_fuse_reorder_to_prev guard was expected to prevent this.");
                    dependency_layout.set_partial_shape(*folded_shape);
                    dependency_layout.format = format::adjust_to_rank(dependency_layout.format, output_shape.size());
                } else if (!shapes_are_broadcastable(dependency_shape, output_shape, require_equal_rank)) {
                    dependency_layout.set_partial_shape(extend_shape_to_rank_from_begin(dependency_shape, output_shape.size()));
                }
            }
        }
    }
    return updated_impl_params;
}

}  // namespace cldnn
