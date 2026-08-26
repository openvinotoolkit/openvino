// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_flatten(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    auto x_pshape = x.get_partial_shape();

    auto zero = v0::Constant::create(element::i32, Shape{1}, {0});
    auto one = v0::Constant::create(element::i32, Shape{1}, {1});
    auto int_max = v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
    auto neg_1_const = v0::Constant::create(element::i32, Shape{1}, {-1});

    // Fast-path: when the input rank is statically known and the start/end
    // dimension arguments are compile-time constants we can build the reshape
    // target shape with constant Slice indices.  This guarantees that the
    // output rank is always statically known, which prevents downstream shape-
    // inference failures (e.g. opset1::Transpose checking that its permutation
    // vector length equals the input rank).
    if (x_pshape.rank().is_static()) {
        const int64_t rank = x_pshape.rank().get_length();

        int64_t start_dim = 0;
        int64_t end_dim = rank - 1;
        bool start_known = false;
        bool end_known = false;

        if (!context.input_is_none(1)) {
            if (const auto c = ov::util::get_constant_from_source(context.get_input(1))) {
                if (ov::shape_size(c->get_shape()) == 1) {
                    start_dim = c->cast_vector<int64_t>()[0];
                    start_known = true;
                }
            }
        } else {
            start_known = true;  // default 0
        }

        if (!context.input_is_none(2)) {
            if (const auto c = ov::util::get_constant_from_source(context.get_input(2))) {
                if (ov::shape_size(c->get_shape()) == 1) {
                    end_dim = c->cast_vector<int64_t>()[0];
                    end_known = true;
                }
            }
        } else {
            end_dim = -1;  // default -1 (last dim)
            end_known = true;
        }

        if (start_known && end_known) {
            // A scalar (rank-0) input is always flattened to a 1-D tensor.
            if (rank == 0) {
                auto neg1 = context.mark_node(neg_1_const);
                return {context.mark_node(std::make_shared<v1::Reshape>(x, neg1, false))};
            }
            // Normalise negative values.
            start_dim = ov::util::normalize(start_dim, rank);
            end_dim = ov::util::normalize(end_dim, rank);
            PYTORCH_OP_CONVERSION_CHECK(
                start_dim >= 0 && start_dim < rank && end_dim >= 0 && end_dim < rank && start_dim <= end_dim,
                "aten::flatten: start_dim and end_dim are out of the valid range for the input rank.");

            // Build the target shape using constant-index Slice operations so
            // that the Concat output has a statically known length (and therefore
            // the Reshape output has a statically known rank).
            auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i32));
            auto step1 = context.mark_node(one);

            OutputVector parts;

            // Dimensions before start_dim (unchanged).
            if (start_dim > 0) {
                auto s = context.mark_node(zero);
                auto e = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {start_dim}));
                parts.push_back(context.mark_node(std::make_shared<v8::Slice>(shape, s, e, step1)));
            }

            // Collapse the flattened range explicitly. ShapeOf values are real
            // dimensions, so special_zero must not reinterpret them as copies.
            auto flattened_begin = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {start_dim}));
            auto flattened_end = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {end_dim + 1}));
            auto flattened_dims = context.mark_node(
                std::make_shared<v8::Slice>(shape, flattened_begin, flattened_end, step1));
            auto flattened_size = context.mark_node(std::make_shared<v1::ReduceProd>(flattened_dims, zero, false));
            parts.push_back(context.mark_node(std::make_shared<v0::Unsqueeze>(flattened_size, zero)));

            // Dimensions after end_dim (unchanged).
            if (end_dim + 1 < rank) {
                auto s = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {end_dim + 1}));
                auto e = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {rank}));
                parts.push_back(context.mark_node(std::make_shared<v8::Slice>(shape, s, e, step1)));
            }

            Output<Node> new_shape;
            if (parts.size() == 1) {
                new_shape = parts[0];
            } else {
                new_shape = context.mark_node(std::make_shared<v0::Concat>(parts, 0));
            }
            return {context.mark_node(std::make_shared<v1::Reshape>(x, new_shape, false))};
        }
    }

    // Dynamic fallback path (original implementation) for cases where the rank
    // or the dimension arguments are not statically known.
    Output<Node> shape;
    Output<Node> rank;
    std::tie(shape, rank) = get_shape_rank(context, x, true);
    // Use opset::If for dim normalization. For now we only have flatten with constant start and end
    Output<Node> start_dim_node;
    Output<Node> end_dim_node;
    if (!context.input_is_none(1)) {
        start_dim_node = get_input_as_i32(context, 1);
    } else {
        start_dim_node = v0::Constant::create(element::i32, Shape{}, {0});
    }
    if (!context.input_is_none(2)) {
        end_dim_node = get_input_as_i32(context, 2);
    } else {
        end_dim_node = v0::Constant::create(element::i32, Shape{}, {-1});
    }
    start_dim_node = normalize_axis(context, start_dim_node, rank);
    end_dim_node = normalize_axis(context, end_dim_node, rank);
    // Slice shape from begin and end, then concatenate the product of the
    // flattened dimensions. ShapeOf values are literal dimensions, not copy
    // markers for Reshape.
    auto start_dim_u = std::make_shared<v0::Unsqueeze>(start_dim_node, zero);
    auto slice_begin = std::make_shared<v8::Slice>(shape, zero, start_dim_u, one);
    auto end_dim_u = std::make_shared<v0::Unsqueeze>(end_dim_node, zero);
    auto end_dim_next = std::make_shared<v1::Add>(end_dim_u, one);
    auto flattened_dims = std::make_shared<v8::Slice>(shape, start_dim_u, end_dim_next, one);
    auto flattened_size = std::make_shared<v1::ReduceProd>(flattened_dims, zero, false);
    auto flattened_size_1d = std::make_shared<v0::Unsqueeze>(flattened_size, zero);
    auto slice_end = std::make_shared<v8::Slice>(shape, end_dim_next, int_max, one);
    auto new_shape = std::make_shared<v0::Concat>(OutputVector{slice_begin, flattened_size_1d, slice_end}, 0);

    context.mark_nodes(
        {zero, one, int_max, start_dim_u, end_dim_u, slice_begin, flattened_dims, flattened_size, flattened_size_1d,
         slice_end, new_shape});

    return {context.mark_node(std::make_shared<v1::Reshape>(x, new_shape, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov