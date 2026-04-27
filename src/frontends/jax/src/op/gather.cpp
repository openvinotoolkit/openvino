// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/gather.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

namespace {
std::shared_ptr<Node> make_transpose(const Output<Node>& input, const std::vector<int64_t>& permutation) {
    auto perm_node = std::make_shared<v0::Constant>(element::i64, Shape{permutation.size()}, permutation);
    return std::make_shared<v1::Transpose>(input, perm_node);
}

Output<Node> normalize_start_indices(const Output<Node>& indices, int64_t index_vector_dim) {
    auto pshape = indices.get_partial_shape();
    int64_t rank = pshape.rank().get_length();
    JAX_OP_CONVERSION_CHECK(pshape.rank().is_static(),
                            "Dynamic rank for start_indices is not supported yet in normalize_start_indices");
    if (index_vector_dim < 0)
        index_vector_dim += rank;

    JAX_OP_CONVERSION_CHECK(index_vector_dim >= 0 && index_vector_dim < rank, 
            "normalize_start_indices: index_vector_dim must be in range [0, ", 
            rank,
            "], but get ",
            index_vector_dim);
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
}  // namespace

OutputVector translate_gather(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto operand = context.get_input(0);
    auto start_indices = context.get_input(1);

    auto start_index_map = context.const_named_param<std::vector<int64_t>>("start_index_map");
    auto slice_sizes = context.const_named_param<std::vector<int64_t>>("slice_sizes");

    int64_t mode = 0;
    int64_t batch_dims = 0;

    if (context.has_param("operand_batching_dims")) {
        batch_dims = context.const_named_param<std::vector<int64_t>>("operand_batching_dims").size();
    } else if (context.has_param("start_indices_batching_dims")) {
        batch_dims = context.const_named_param<std::vector<int64_t>>("start_indices_batching_dims").size();
    }

    if (context.has_param("mode")) {
        mode = context.const_named_param<int64_t>("mode");
    }

    auto operand_pshape = operand.get_partial_shape();
    JAX_OP_CONVERSION_CHECK(operand_pshape.rank().is_static(), "Dynamic rank for gather operand is not supported yet.");

    auto operand_reordered = operand;
    auto operand_rank = operand_pshape.rank().get_length();
    auto slice_sizes_reordered = slice_sizes;

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

    bool all_slice_one = std::all_of(slice_sizes_reordered.begin(), slice_sizes_reordered.end(), [](int64_t s) {
        return s == 1;
    });

    // validate start_index_map is identity
    for (size_t i = 0; i < start_index_map.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(
            start_index_map[i] == static_cast<int64_t>(i),
            "Non-identity start_index_map is not supported yet.");
    }

    FRONT_END_OP_CONVERSION_CHECK(
        all_slice_one,
        "OpenVINO JAX Frontend currently only supports scalar point gathering (slice_sizes == 1).");

    FRONT_END_OP_CONVERSION_CHECK(mode == 0 || mode == 1,
                                  "Only PROMISE_IN_BOUNDS (0) and CLIP (1) modes are supported.");

    // mode 1: clamping
    if (mode == 1) {
        auto indices_i64 = std::make_shared<v0::Convert>(normalized_indices, element::i64);
        auto zero_const = v0::Constant::create(element::i64, Shape{}, {0});
	      auto clamped_min = std::make_shared<v1::Maximum>(indices_i64, zero_const);
	
        std::vector<int64_t> upper_bounds_val;
        for (size_t i = 0; i < start_index_map.size(); ++i) {
            int64_t dim_idx = start_index_map[i];

            JAX_OP_CONVERSION_CHECK(dim_idx >= 0 && dim_idx < operand_rank,
                    "start_index_map contains out-of-range dimension index.");
            JAX_OP_CONVERSION_CHECK(operand_pshape[dim_idx].is_static(), 
                    "Dynamic operand dimension not supported in CLIP Mode.");
            JAX_OP_CONVERSION_CHECK(dim_idx < static_cast<int64_t>(slice_sizes.size()),
                    "slice_sizes does not cover dimension referenced by start_index_map.");
            
            int64_t dim_size = operand_pshape[dim_idx].get_length();
            int64_t window_size = slice_sizes[dim_idx];
	          upper_bounds_val.push_back(std::max<int64_t>(0, dim_size - window_size));
        }
	      int64_t indices_rank_val = normalized_indices.get_partial_shape().rank().get_length();
	      Shape limits_shape(indices_rank_val, 1);
        limits_shape.back() = upper_bounds_val.size();

        auto upper_limits_const = v0::Constant::create(element::i64, limits_shape, upper_bounds_val);
        normalized_indices = std::make_shared<v1::Minimum>(clamped_min, upper_limits_const);
    }

    Output<Node> gathered = std::make_shared<v8::GatherND>(operand_reordered, normalized_indices, batch_dims);
    return {gathered};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov
