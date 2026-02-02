// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

namespace {

    std::shared_ptr<Node> make_transpose(const Output<Node>& input, const std::vector<int64_t>& permutation) {
        auto perm_node = std::make_shared<v0::Constant>(element::i64, Shape{permutation.size()}, permutation);
        return std::make_shared<v1::Transpose>(input, perm_node);
    }

    // helper reordering operand berdasarkan start_index_map
    std::vector<int64_t> compute_gather_input_permutation(
                                int64_t operand_rank,
                                const std::vector<int64_t>& start_index_map) {
        std::vector<int64_t> permutation;
        permutation.reserve(operand_rank);
        std::vector<bool> seen(operand_rank, false);

        for (auto dim: start_index_map) {
            permutation.push_back(dim);
            if (dim >= 0 && dim < operand_rank) seen[dim] = true;
        }

        for (int64_t i = 0; i < operand_rank; i++) {
            if (!seen[i]) {
                permutation.push_back(i);
            }
        }
        return permutation;
    }

    std::vector<int64_t> reorder_vector(const std::vector<int64_t>& original,
                                        const std::vector<int64_t>& permutation) {
        std::vector<int64_t> reordered(original.size());
        for (size_t i = 0; i < original.size(); i++) {
            if (i < reordered.size() && i < permutation.size()) {
                reordered[i] = original[permutation[i]];
            }
        }
        return reordered;
    }

    // output permutation logic (offset_dims)
    std::vector<int64_t> compute_output_permutation(int64_t total_rank, 
                                                    const std::vector<int64_t>& offset_dims) {
        std::vector<int64_t> permutation(total_rank);

        int64_t num_slice_dims = offset_dims.size();
        int64_t num_batch_dims = total_rank - num_slice_dims;

        int64_t batch_counter = 0;
        int64_t slice_counter = num_batch_dims;

        for (int64_t i = 0; i< total_rank; i++) {
            bool is_slice_dim = false;
            for (auto off: offset_dims) {
                if (off == i) {
                    is_slice_dim = true;
                    break;
                }
            }

            if (is_slice_dim) {
                permutation[i] = slice_counter++;
            } else {
                permutation[i] = batch_counter++;
            }
        }

        return permutation;
    }

    Output<Node> normalize_start_indices(const Output<Node>& indices, int64_t index_vector_dim) {
        auto pshape = indices.get_partial_shape();

        FRONT_END_OP_CONVERSION_CHECK(pshape.rank().is_static(),
            "Dynamic rank for start_indices is not supported yet in normalize_start_indices");

        int64_t rank = pshape.rank().get_length();

        if (index_vector_dim < 0) {
            index_vector_dim += rank;
        }

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
} // namespace

OutputVector translate_gather(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto operand = context.get_input(0);
    auto start_indices = context.get_input(1);

    auto collapsed_slice_dims = context.const_named_param<std::vector<int64_t>>("collapsed_slice_dims");
    auto offset_dims = context.const_named_param<std::vector<int64_t>>("offset_dims");
    auto start_index_map = context.const_named_param<std::vector<int64_t>>("start_index_map");
    auto slice_sizes = context.const_named_param<std::vector<int64_t>>("slice_sizes");
    int64_t mode = 0;
    if (context.has_param("mode")) {
        mode = context.const_named_param<int64_t>("mode");
    }

    auto operand_pshape = operand.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(operand_pshape.rank().is_static(), 
        "Dynamic rank for gather operand is not supported yet.");
    int64_t operand_rank = operand_pshape.rank().get_length();

    auto input_perm = compute_gather_input_permutation(operand_rank, start_index_map);
    
    auto operand_reordered = make_transpose(operand, input_perm);
    
    auto slice_sizes_reordered = reorder_vector(slice_sizes, input_perm);

    int64_t index_depth = start_index_map.size();
    auto indices_pshape = start_indices.get_partial_shape();
    
    if (indices_pshape.rank().is_static()) {
        auto rank_val = indices_pshape.rank().get_length();
        auto last_dim = indices_pshape[rank_val - 1];

        if (last_dim.is_static()) {
            int64_t current_idx_width = last_dim.get_length();
            FRONT_END_OP_CONVERSION_CHECK(
                current_idx_width == index_depth,
                "Last Dimension of start_indices must equal length of start_index_map. "
                "Got: ", current_idx_width, ", expected: ", index_depth
            );
        }
    }

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

    auto indices_pshape_norm = normalized_indices.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(
        indices_pshape_norm.rank().is_static(),
        "Dynamic rank for start_indices is not supported yet."
    );
    int64_t indices_rank_val = indices_pshape_norm.rank().get_length();

    // clip mode
    if (mode == 1) {
        auto zero_const = v0::Constant::create(element::i64, Shape{1}, {0});

        normalized_indices = std::make_shared<v1::Maximum>(normalized_indices, zero_const);

        std::vector<int64_t>  upper_bounds_val;

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

        Shape limits_shape(indices_rank_val, 1);
        limits_shape.back() = upper_bounds_val.size();

        auto upper_limits_const = v0::Constant::create(element::i64, limits_shape, upper_bounds_val);

        normalized_indices = std::make_shared<v1::Minimum>(normalized_indices, upper_limits_const);
    } else if (mode == 0) {
        // PROMISE_IN_BOUNDS -> do Nothing
    } else {
        FRONT_END_NOT_IMPLEMENTED("other Gather mode is not supported yet.");
    }
    
    bool all_slice_one = true;
    for (const auto& s: slice_sizes_reordered) {
        if (s != 1) {
            all_slice_one = false;
            break;
        } 
    }
    
    if (!all_slice_one) {
        FRONT_END_NOT_IMPLEMENTED("Slice gathering (slice_sizes > 1) is not fully implemented yet.");
    }

    // batch dims
    int64_t batch_dims = indices_pshape_norm.rank().get_length() - 1;

    Output<Node> gathered = std::make_shared<v8::GatherND>(operand_reordered, normalized_indices, batch_dims);

    auto current_pshape = gathered.get_partial_shape();
    int64_t num_slice_dims = offset_dims.size();

    // collapse_slize_dims handling (minimal scalar gather support)
    if (all_slice_one) {
        FRONT_END_OP_CONVERSION_CHECK(
            static_cast<int64_t>(collapsed_slice_dims.size()) == num_slice_dims,
            "collapsed_slice_dims size must match number of slice dims when slice_sizes == 1"
        );

        auto gathered_pshape = gathered.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(
            gathered_pshape.rank().is_static(),
            "Dynamic rank after GatherND is not supported for squeeze."
        );

        int64_t out_rank = gathered_pshape.rank().get_length();

        std::vector<int64_t> squeeze_axes;
        for (int64_t i = 0; i < num_slice_dims; i++) {
            squeeze_axes.push_back(out_rank - 1 - i);
        }

        auto axes_const = v0::Constant::create(
            element::i64, Shape{squeeze_axes.size()}, squeeze_axes);

        gathered = std::make_shared<v0::Squeeze>(gathered, axes_const);
        auto current_pshape = gathered.get_partial_shape();
    }

    if (current_pshape.rank().is_static()) {
        int64_t current_rank = current_pshape.rank().get_length();

        bool is_identity = true;
        int64_t check_start = current_rank - num_slice_dims;

        for (size_t i = 0; i < offset_dims.size(); i++) {
            if (i > 0) {
                FRONT_END_OP_CONVERSION_CHECK(
                    offset_dims[i] == offset_dims[i - 1] + 1,
                    "Only contiguous offset_dims are supported."
                );
            }
            if (offset_dims[i] != (check_start + static_cast<int64_t>(i))) {
                is_identity = false;
            }
        }

        // offset dims validation
        FRONT_END_OP_CONVERSION_CHECK(
            static_cast<int64_t>(offset_dims.size()) == num_slice_dims,
            "offset_dims must match number of slice dims"
        );

        if (!is_identity) {
            // note: this permutation logic assumes contiguous offset_dims and scalar gather.
            auto perm_vector = compute_output_permutation(current_rank, offset_dims);
            gathered = make_transpose(gathered, perm_vector);
        }
    } else {
        FRONT_END_NOT_IMPLEMENTED("Dynamic Rank output haven't been supported.");
    }

    return {gathered};
};

} // namespace op
} // namespace jax
} // namespace frontend
} // namespace ov
