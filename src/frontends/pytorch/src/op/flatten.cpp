// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
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

    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto neg_1_const = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));

    // Fast-path: when the input rank is static and start/end dims are constants,
    // build the reshape target from constant-index slices so the output rank is
    // statically known. Otherwise a dynamic-length target shape yields a
    // dynamic-rank output that breaks rank-sensitive consumers (e.g. Transpose).
    const auto& x_pshape = x.get_partial_shape();
    if (x_pshape.rank().is_static()) {
        const int64_t rank_len = x_pshape.rank().get_length();
        int64_t start_dim = 0;
        int64_t end_dim = -1;
        bool dims_known = true;
        if (!context.input_is_none(1)) {
            const auto c = ov::util::get_constant_from_source(context.get_input(1));
            if (c && ov::shape_size(c->get_shape()) == 1)
                start_dim = c->cast_vector<int64_t>()[0];
            else
                dims_known = false;
        }
        if (!context.input_is_none(2)) {
            const auto c = ov::util::get_constant_from_source(context.get_input(2));
            if (c && ov::shape_size(c->get_shape()) == 1)
                end_dim = c->cast_vector<int64_t>()[0];
            else
                dims_known = false;
        }

        const int64_t min_dim = rank_len == 0 ? -1 : -rank_len;
        const int64_t max_dim = rank_len == 0 ? 0 : rank_len - 1;
        PYTORCH_OP_CONVERSION_CHECK(min_dim <= start_dim && start_dim <= max_dim,
                                    "aten::flatten: start_dim is out of range.");
        PYTORCH_OP_CONVERSION_CHECK(min_dim <= end_dim && end_dim <= max_dim,
                                    "aten::flatten: end_dim is out of range.");

        if (dims_known && rank_len == 0) {
            return {context.mark_node(std::make_shared<v1::Reshape>(x, neg_1_const, false))};
        }
        if (dims_known) {
            start_dim = ov::util::normalize(start_dim, rank_len);
            end_dim = ov::util::normalize(end_dim, rank_len);
            PYTORCH_OP_CONVERSION_CHECK(start_dim <= end_dim,
                                        "aten::flatten: start_dim must not be greater than end_dim.");
            auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i32));
            OutputVector parts;
            if (start_dim > 0) {
                auto e = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {start_dim}));
                parts.push_back(context.mark_node(std::make_shared<v8::Slice>(shape, zero, e, one)));
            }
            parts.push_back(neg_1_const);
            if (end_dim + 1 < rank_len) {
                auto s = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {end_dim + 1}));
                auto e = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {rank_len}));
                parts.push_back(context.mark_node(std::make_shared<v8::Slice>(shape, s, e, one)));
            }
            Output<Node> new_shape =
                parts.size() == 1 ? parts[0] : context.mark_node(std::make_shared<v0::Concat>(parts, 0));
            return {context.mark_node(std::make_shared<v1::Reshape>(x, new_shape, true))};
        }
    }

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
    // Slice shape from begin and end, then concat with -1, if slice return empty tensor concat should still be able to
    // work with it
    auto int_max = v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
    auto start_dim_u = std::make_shared<v0::Unsqueeze>(start_dim_node, zero);
    auto slice_begin = std::make_shared<v8::Slice>(shape, zero, start_dim_u, one);
    auto end_dim_u = std::make_shared<v0::Unsqueeze>(end_dim_node, zero);
    auto end_dim_next = std::make_shared<v1::Add>(end_dim_u, one);
    auto slice_end = std::make_shared<v8::Slice>(shape, end_dim_next, int_max, one);
    auto new_shape = std::make_shared<v0::Concat>(OutputVector{slice_begin, neg_1_const, slice_end}, 0);

    context.mark_nodes({zero, one, int_max, start_dim_u, end_dim_u, slice_begin, slice_end, neg_1_const, new_shape});

    return {context.mark_node(std::make_shared<v1::Reshape>(x, new_shape, true))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
