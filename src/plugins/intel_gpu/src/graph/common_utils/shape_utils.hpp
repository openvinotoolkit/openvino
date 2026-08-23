// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "openvino/core/partial_shape.hpp"

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

inline kernel_impl_params canonicalize_fused_shapes(const kernel_impl_params& impl_params) {
    auto updated_impl_params = impl_params;
    const bool require_equal_rank = impl_params.prog->is_new_shape_infer();

    for (auto& descriptor : updated_impl_params.fused_desc) {
        if (descriptor.is_type<eltwise>() && descriptor.total_num_deps == 2 && descriptor.has_outer_dep()) {
            if (updated_impl_params.input_layouts.size() > static_cast<size_t>(descriptor.outer_dep_start_idx)) {
                const auto& output_shape = updated_impl_params.output_layouts[0].get_partial_shape();
                auto& dependency_layout = updated_impl_params.input_layouts[descriptor.outer_dep_start_idx];
                const auto& dependency_shape = dependency_layout.get_partial_shape();

                if (!shapes_are_broadcastable(dependency_shape, output_shape, require_equal_rank)) {
                    dependency_layout.set_partial_shape(extend_shape_to_rank_from_begin(dependency_shape, output_shape.size()));
                }
            }
        }
    }
    return updated_impl_params;
}

}  // namespace cldnn
