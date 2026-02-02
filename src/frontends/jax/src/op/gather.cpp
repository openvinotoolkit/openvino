// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

namespace {

// --- Helper Functions ---

std::shared_ptr<Node> make_transpose(const Output<Node>& input, const std::vector<int64_t>& permutation) {
    auto perm_node = std::make_shared<v0::Constant>(element::i64, Shape{permutation.size()}, permutation);
    return std::make_shared<v1::Transpose>(input, perm_node);
}

// Reorders the operand dimensions based on start_index_map to align with GatherND requirements.
// Moves dimensions referenced in start_index_map to the front.
std::vector<int64_t> compute_gather_input_permutation(int64_t operand_rank,
                                                      const std::vector<int64_t>& start_index_map) {
    std::vector<int64_t> permutation;
    permutation.reserve(operand_rank);
    std::vector<bool> seen(operand_rank, false);

    for (auto dim : start_index_map) {
        if (dim >= 0 && dim < operand_rank) {
            permutation.push_back(dim);
            seen[dim] = true;
        }
    }

    for (int64_t i = 0; i < operand_rank; i++) {
        if (!seen[i]) {
            permutation.push_back(i);
        }
    }
    return permutation;
}

// Reorders a vector (e.g., slice_sizes) according to a permutation.
std::vector<int64_t> reorder_vector(const std::vector<int64_t>& original, const std::vector<int64_t>& permutation) {
    std::vector<int64_t> reordered(original.size());
    for (size_t i = 0; i < original.size(); i++) {
        if (i < reordered.size() && i < permutation.size()) {
            reordered[i] = original[permutation[i]];
        }
    }
    return reordered;
}

// Computes the output permutation required to satisfy offset_dims.
std::vector<int64_t> compute_output_permutation(int64_t total_rank, const std::vector<int64_t>& offset_dims) {
    std::vector<int64_t> perm(total_rank, -1);
    std::vector<int64_t> batch_dims;
    
    // Identify batch dims (dimensions NOT in offset_dims)
    for (int64_t i = 0; i < total_rank; ++i) {
        bool is_offset = false;
        for (auto off : offset_dims) {
            if (off == i) {
                is_offset = true;
                break;
            }
        }
        if (!is_offset) {
            batch_dims.push_back(i);
        }
    }

    // Map batch_dims first, then offset_dims (which correspond to slice_dims)
    int64_t idx = 0;
    for (auto d : batch_dims) {
        perm[d] = idx++;
    }
    for (auto d : offset_dims) {
        perm[d] = idx++;
    }

    return perm;
}

// Normalizes start_indices so the index vector dimension is always the last dimension.
Output<Node> normalize_start_indices(const Output<Node>& indices, int64_t index_vector_dim) {
    auto pshape = indices.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(pshape.rank().is_static(),
                                  "Dynamic rank for start_indices is not supported yet in normalize_start_indices");

    int64_t rank = pshape.rank().get_length();
    if (index_vector_dim < 0) index_vector_dim += rank;

    if (index_vector_dim == rank - 1) {
        return indices;
    }

    std::vector<int64_t> permutation;
    permutation.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
        if (i != index_vector_dim) {
            permutation.push_back(i);
        }
    }
    permutation.push_back(index_vector_dim);

    return make_transpose(indices, permutation);
}

// Applies masking logic for FILL_OR_DROP mode.
// It checks if original indices are within bounds defined by (operand_dim - window_size).
Output<Node> apply_fill_or_drop(const Output<Node>& gathered,
                                const Output<Node>& indices,
                                const Output<Node>& operand,
                                const std::vector<int64_t>& start_index_map,
                                const std::vector<int64_t>& slice_sizes_reordered,
                                int64_t fill_value_scalar = 0) {
    
    auto operand_pshape = operand.get_partial_shape();
    std::vector<int64_t> upper_bounds_val;

    for (size_t i = 0; i < start_index_map.size(); ++i) {
        int64_t dim_idx = start_index_map[i];
        
        FRONT_END_OP_CONVERSION_CHECK(operand_pshape[dim_idx].is_static(),
                                      "Dynamic operand shape not supported for FILL_OR_DROP check yet.");

        int64_t dim_size = operand_pshape[dim_idx].get_length();
        int64_t window_size = slice_sizes_reordered[i];
        
        // JAX semantics: window must fully fit inside the operand
        int64_t valid_limit = dim_size - window_size;
        if (valid_limit < 0) valid_limit = 0;
        
        upper_bounds_val.push_back(valid_limit);
    }

    // Prepare constants for broadcasting
    int64_t indices_rank = indices.get_partial_shape().rank().get_length();
    Shape limits_shape(indices_rank, 1);
    limits_shape.back() = upper_bounds_val.size(); // Broadcast matches the last dim (index depth)
    
    auto upper_bounds_const = v0::Constant::create(element::i64, limits_shape, upper_bounds_val);
    auto zero_const = v0::Constant::create(element::i64, Shape{1}, {0});

    // Mask generation: 0 <= index <= (dim - window)
    auto mask_ge = std::make_shared<v1::GreaterEqual>(indices, zero_const);
    auto mask_le = std::make_shared<v1::LessEqual>(indices, upper_bounds_const);
    auto is_in_bounds = std::make_shared<v1::LogicalAnd>(mask_ge, mask_le);

    // Reduce mask along the index vector dimension (last dim)
    // If any coordinate is OOB, the whole gather slice is OOB.
    auto reduce_axis = v0::Constant::create(element::i64, Shape{1}, {indices_rank - 1});
    auto valid_mask = std::make_shared<v1::ReduceLogicalAnd>(is_in_bounds, reduce_axis, false); 

    // Apply Fill Value
    auto fill_const = v0::Constant::create(element::i64, Shape{1}, {fill_value_scalar});
    auto fill_value_converted = std::make_shared<v0::Convert>(fill_const, gathered.get_element_type());

    return std::make_shared<v1::Select>(valid_mask, gathered, fill_value_converted);
}

} // namespace

// --- Main Translation Function ---

OutputVector translate_gather(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto operand = context.get_input(0);
    auto start_indices = context.get_input(1);

    // Retrieve JAX parameters
    auto collapsed_slice_dims = context.const_named_param<std::vector<int64_t>>("collapsed_slice_dims");
    auto offset_dims = context.const_named_param<std::vector<int64_t>>("offset_dims");
    auto start_index_map = context.const_named_param<std::vector<int64_t>>("start_index_map");
    auto slice_sizes = context.const_named_param<std::vector<int64_t>>("slice_sizes");
    
    // JAX GatherScatterMode: 0=PROMISE_IN_BOUNDS, 1=CLIP, 2=FILL_OR_DROP
    int64_t mode = 0;
    if (context.has_param("mode")) {
        mode = context.const_named_param<int64_t>("mode");
    }

    // Rank Validation
    auto operand_pshape = operand.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(operand_pshape.rank().is_static(), 
                                  "Dynamic rank for gather operand is not supported yet.");
    int64_t operand_rank = operand_pshape.rank().get_length();

    // 1. Prepare Operand (Transpose to align with start_index_map)
    auto input_perm = compute_gather_input_permutation(operand_rank, start_index_map);
    auto operand_reordered = make_transpose(operand, input_perm);
    auto slice_sizes_reordered = reorder_vector(slice_sizes, input_perm);

    // 2. Prepare Indices (Normalize layout)
    int64_t index_depth = start_index_map.size();
    int64_t index_vector_axis = -1;
    if (context.has_param("index_vector_dim")) {
        index_vector_axis = context.const_named_param<int64_t>("index_vector_dim");
    } else {
        auto indices_rank = start_indices.get_partial_shape().rank();
        if (indices_rank.is_static()) {
            index_vector_axis = indices_rank.get_length() - 1;
        }
    }

    auto normalized_indices = normalize_start_indices(start_indices, index_vector_axis);
    
    // Validate indices rank for subsequent logic
    auto indices_pshape_norm = normalized_indices.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(indices_pshape_norm.rank().is_static(),
                                  "Dynamic rank for start_indices is not supported yet.");
    int64_t indices_rank_val = indices_pshape_norm.rank().get_length();

    // Preserve raw indices for FILL_OR_DROP mask check logic
    Output<Node> indices_for_mask = normalized_indices;

    // 3. Mode Handling: Clamping
    // We apply clamping for both CLIP (1) and FILL_OR_DROP (2).
    // For mode 2, we clamp to prevent Segfaults during GatherND, then mask later.
    if (mode == 1 || mode == 2) {
        auto zero_const = v0::Constant::create(element::i64, Shape{1}, {0});

        // Clamp Lower Bound: Max(indices, 0)
        normalized_indices = std::make_shared<v1::Maximum>(normalized_indices, zero_const);

        // Clamp Upper Bound: Min(indices, dim - window_size)
        std::vector<int64_t> upper_bounds_val;
        for (size_t i = 0; i < start_index_map.size(); ++i) {
            int64_t dim_idx = start_index_map[i];

            if (operand_pshape[dim_idx].is_dynamic()) {
                FRONT_END_OP_CONVERSION_CHECK(false,
                                              "Dynamic Dimension in operand is not supported for 'clip' mode yet.");
            }

            int64_t dim_size = operand_pshape[dim_idx].get_length();
            int64_t window_size = slice_sizes_reordered[i];

            int64_t valid_limit = dim_size - window_size;
            if (valid_limit < 0) valid_limit = 0;

            upper_bounds_val.push_back(valid_limit);
        }

        // Create Limit Constant with broadcasting shape [1, ..., Depth]
        Shape limits_shape(indices_rank_val, 1);
        limits_shape.back() = upper_bounds_val.size(); 
        auto upper_limits_const = v0::Constant::create(element::i64, limits_shape, upper_bounds_val);

        normalized_indices = std::make_shared<v1::Minimum>(normalized_indices, upper_limits_const);

    } else if (mode == 0) {
        // PROMISE_IN_BOUNDS: No clamping needed.
    } else {
        FRONT_END_NOT_IMPLEMENTED("Gather mode is not supported yet.");
    }

    // 4. Execution: GatherND
    // At this point, normalized_indices are guaranteed to be safe (if clamped).
    bool all_slice_one = std::all_of(slice_sizes_reordered.begin(), slice_sizes_reordered.end(), [](int64_t s){ return s == 1; });
    if (!all_slice_one) {
        FRONT_END_NOT_IMPLEMENTED("Slice gathering (slice_sizes > 1) is not fully implemented yet.");
    }


    int64_t batch_dims = indices_rank_val - 1;
    Output<Node> gathered = std::make_shared<v8::GatherND>(operand_reordered, normalized_indices, batch_dims);

    // 5. Mode Handling: Masking (FILL_OR_DROP)
    if (mode == 2) {
        gathered = apply_fill_or_drop(gathered,
                                      indices_for_mask, // Use original indices for check
                                      operand,          // Use original operand for shape check
                                      start_index_map,
                                      slice_sizes_reordered);
    }

    // 6. Post-processing: Collapse Slices & Transpose Output
    auto current_pshape = gathered.get_partial_shape();
    int64_t num_slice_dims = offset_dims.size();

    // Handle collapsed_slice_dims (Squeeze)
    if (all_slice_one && !collapsed_slice_dims.empty()) {
        FRONT_END_OP_CONVERSION_CHECK(static_cast<int64_t>(collapsed_slice_dims.size()) == num_slice_dims,
                                      "collapsed_slice_dims size must match number of slice dims when slice_sizes == 1");
        
        auto gathered_pshape = gathered.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(gathered_pshape.rank().is_static(),
                                      "Dynamic rank after GatherND is not supported for squeeze.");

        int64_t out_rank = gathered_pshape.rank().get_length();
        std::vector<int64_t> squeeze_axes;
        // Since we reordered dims to front, slices are at the end.
        for (int64_t i = 0; i < num_slice_dims; i++) {
            squeeze_axes.push_back(out_rank - 1 - i);
        }

        auto axes_const = v0::Constant::create(element::i64, Shape{squeeze_axes.size()}, squeeze_axes);
        gathered = std::make_shared<v0::Squeeze>(gathered, axes_const);
        current_pshape = gathered.get_partial_shape();
    }

    // Handle offset_dims (Transpose)
    if (current_pshape.rank().is_static()) {
        int64_t current_rank = current_pshape.rank().get_length();

        // Validate offset_dims properties
        FRONT_END_OP_CONVERSION_CHECK(static_cast<int64_t>(offset_dims.size()) == num_slice_dims,
                                      "offset_dims must match number of slice dims");

        bool is_identity = true;
        int64_t expected_start = current_rank - num_slice_dims;

        // Check if dimensions are already in the correct place
        for (size_t i = 0; i < offset_dims.size(); ++i) {
            if (offset_dims[i] != expected_start + static_cast<int64_t>(i)) {
                is_identity = false;
            }
            if (i > 0) {
                 FRONT_END_OP_CONVERSION_CHECK(offset_dims[i] == offset_dims[i - 1] + 1,
                                               "Only contiguous offset_dims are supported.");
            }
        }

        if (!is_identity) {
            auto perm = compute_output_permutation(current_rank, offset_dims);
            gathered = make_transpose(gathered, perm);
        }
    } else {
        FRONT_END_NOT_IMPLEMENTED("Dynamic Rank output is not supported.");
    }

    return {gathered};
};

} // namespace op
} // namespace jax
} // namespace frontend
} // namespace ov
