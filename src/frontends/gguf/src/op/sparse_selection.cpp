// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// ggml sparse-attention selection operations (DeepSeek and Qwen sparse attention indexers).
// Shapes are OpenVINO order, the reverse of ggml's ne.

#include <limits>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

using namespace ov::op;

namespace {

std::shared_ptr<v0::Constant> i64s(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, {values.size()}, values);
}

int64_t static_dim(const ov::Output<ov::Node>& value, int64_t axis, const char* what) {
    const auto& shape = value.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(shape.rank().is_static(), what, " requires a static rank");
    const auto rank = shape.rank().get_length();
    const auto dim = shape[axis < 0 ? rank + axis : axis];
    FRONT_END_OP_CONVERSION_CHECK(dim.is_static(), what, " requires a static dimension ", axis);
    return dim.get_length();
}

ov::Output<ov::Node> f32(const ov::Output<ov::Node>& value) {
    return value.get_element_type() == ov::element::f32
               ? value
               : std::make_shared<v0::Convert>(value, ov::element::f32)->output(0);
}

}  // namespace

// ggml_lightning_indexer: res[s, 0, t, c] = sum_h relu(q[s, t, h] . k[s, c]) * w[s, 0, t, h] + mask[s', 0, t, c]
// q [S, T, H, E], k [S, KV, 1, E], weights [S, 1, T, H], mask [S', 1, T, KV].
OutputVector translate_lightning_indexer(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    auto q = f32(context.get_input(0));
    const int64_t width = static_dim(q, -1, "LIGHTNING_INDEXER query");
    const int64_t heads = static_dim(q, -2, "LIGHTNING_INDEXER query");
    auto keys = std::make_shared<v1::Reshape>(f32(context.get_input(1)), i64s({0, -1, width}), true);
    auto queries = std::make_shared<v1::Reshape>(q, i64s({0, -1, width}), true);
    auto scores = std::make_shared<v0::MatMul>(queries, keys, false, true);  // [S, T*H, KV]
    auto zero = v0::Constant::create(ov::element::i64, {}, {0});
    auto head_shape = std::make_shared<v0::Concat>(
        ov::OutputVector{
            std::make_shared<v8::Slice>(std::make_shared<v3::ShapeOf>(q, ov::element::i64),
                                        i64s({0}),
                                        i64s({3}),
                                        i64s({1})),
            std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(scores, ov::element::i64), i64s({2}), zero)},
        0);
    auto per_head = std::make_shared<v1::Reshape>(scores, head_shape, false);  // [S, T, H, KV]
    auto weights = std::make_shared<v1::Reshape>(f32(context.get_input(2)), i64s({0, -1, heads, 1}), true);
    auto weighted = std::make_shared<v1::Multiply>(std::make_shared<v0::Relu>(per_head), weights);
    auto summed = std::make_shared<v1::ReduceSum>(weighted, i64s({2}), false);  // [S, T, KV]
    auto result =
        std::make_shared<v1::Add>(std::make_shared<v0::Unsqueeze>(summed, i64s({1})), f32(context.get_input(3)));
    return rename_outputs_with_suffix({result}, context.get_name());
}

// ggml_top_k_mask: keep mask[.., c] where c is a cell of a live pick of the row or an extra cell,
// -inf elsewhere. mask [m3, m2, m1, KV] broadcasts to the rows of top_k [ne3, ne2, ne1, n_top].
// Optional inputs follow in this order when the attributes "has_score", "has_blocks" and
// "has_extra" (default: present when all five inputs are) say so: score [.., n_pick] (a pick is
// live when its score is above -inf), blocks [1, 1, n_pick, n_blk] (the member cells of each pick)
// and extra [.., n_extra]. Picks and cells out of range are skipped.
OutputVector translate_top_k_mask(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    const bool all = context.get_input_size() == 5;
    const bool has_score = context.get_attribute<bool>("has_score", all);
    const bool has_blocks = context.get_attribute<bool>("has_blocks", all);
    const bool has_extra = context.get_attribute<bool>("has_extra", all);
    FRONT_END_OP_CONVERSION_CHECK(2 + has_score + has_blocks + has_extra == context.get_input_size(),
                                  "TOP_K_MASK optional-input attributes do not match its inputs");
    int index = 2;
    const auto mask = context.get_input(0);
    const auto top_k = std::make_shared<v0::Convert>(context.get_input(1), ov::element::i64);
    const int64_t n_top = static_dim(top_k, -1, "TOP_K_MASK top_k");
    auto picks = std::make_shared<v1::Reshape>(top_k, i64s({-1, n_top}), false);  // [R, n_top]
    auto zero_i = v0::Constant::create(ov::element::i64, {}, {0});
    auto rows_shape = std::make_shared<v3::ShapeOf>(top_k, ov::element::i64);
    auto kv = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(mask, ov::element::i64), i64s({-1}), zero_i);

    ov::Output<ov::Node> live;
    if (has_score) {
        auto score = f32(context.get_input(index++));
        const int64_t n_pick = static_dim(score, -1, "TOP_K_MASK score");
        auto scores = std::make_shared<v1::Reshape>(score, i64s({-1, n_pick}), false);
        auto in_range = std::make_shared<v1::LogicalAnd>(
            std::make_shared<v1::GreaterEqual>(picks, zero_i),
            std::make_shared<v1::Less>(picks, v0::Constant::create(ov::element::i64, {}, {n_pick})));
        auto safe = std::make_shared<v1::Select>(in_range, picks, zero_i);
        auto picked = std::make_shared<v8::Gather>(scores, safe, v0::Constant::create(ov::element::i64, {}, {1}), 1);
        live = std::make_shared<v1::Greater>(
            picked,
            v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<float>::infinity()}));
    }
    ov::Output<ov::Node> cells = picks;
    if (has_blocks) {
        auto blocks = std::make_shared<v0::Convert>(context.get_input(index++), ov::element::i64);
        const int64_t n_blk = static_dim(blocks, -1, "TOP_K_MASK blocks");
        const int64_t n_pick = static_dim(blocks, -2, "TOP_K_MASK blocks");
        auto table = std::make_shared<v1::Reshape>(blocks, i64s({n_pick, n_blk}), false);
        auto in_range = std::make_shared<v1::LogicalAnd>(
            std::make_shared<v1::GreaterEqual>(picks, zero_i),
            std::make_shared<v1::Less>(picks, v0::Constant::create(ov::element::i64, {}, {n_pick})));
        auto safe = std::make_shared<v1::Select>(in_range, picks, zero_i);
        auto members = std::make_shared<v8::Gather>(table, safe, zero_i);  // [R, n_top, n_blk]
        ov::Output<ov::Node> pick_live = in_range;
        if (live.get_node())
            pick_live = std::make_shared<v1::LogicalAnd>(live, in_range);
        auto member_live = std::make_shared<v3::Broadcast>(std::make_shared<v0::Unsqueeze>(pick_live, i64s({-1})),
                                                           std::make_shared<v3::ShapeOf>(members, ov::element::i64));
        cells = std::make_shared<v1::Reshape>(members, i64s({0, -1}), true);
        live = std::make_shared<v1::Reshape>(member_live, i64s({0, -1}), true);
    }
    if (!live.get_node())
        live = std::make_shared<v3::Broadcast>(v0::Constant::create(ov::element::boolean, {}, {true}),
                                               std::make_shared<v3::ShapeOf>(cells, ov::element::i64));
    if (has_extra) {
        auto extra = std::make_shared<v0::Convert>(context.get_input(index++), ov::element::i64);
        const int64_t n_extra = static_dim(extra, -1, "TOP_K_MASK extra");
        auto extra_cells = std::make_shared<v1::Reshape>(extra, i64s({-1, n_extra}), false);
        cells = std::make_shared<v0::Concat>(ov::OutputVector{cells, extra_cells}, 1);
        live = std::make_shared<v0::Concat>(
            ov::OutputVector{live,
                             std::make_shared<v3::Broadcast>(v0::Constant::create(ov::element::boolean, {}, {true}),
                                                             std::make_shared<v3::ShapeOf>(extra_cells))},
            1);
    }
    // Dead or out-of-range cells write a spare column KV, dropped below.
    auto valid = std::make_shared<v1::LogicalAnd>(
        live,
        std::make_shared<v1::LogicalAnd>(std::make_shared<v1::GreaterEqual>(cells, zero_i),
                                         std::make_shared<v1::Less>(cells, kv)));
    auto targets = std::make_shared<v1::Select>(valid, cells, kv);
    auto rows = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(picks, ov::element::i64), i64s({0}), zero_i);
    auto one = v0::Constant::create(ov::element::i64, {1}, {1});
    auto keep = std::make_shared<v3::ScatterElementsUpdate>(
        std::make_shared<v3::Broadcast>(
            v0::Constant::create(ov::element::u8, {}, {0}),
            std::make_shared<v0::Concat>(ov::OutputVector{rows, std::make_shared<v1::Add>(kv, one)}, 0)),
        targets,
        std::make_shared<v3::Broadcast>(v0::Constant::create(ov::element::u8, {}, {1}),
                                        std::make_shared<v3::ShapeOf>(targets, ov::element::i64)),
        v0::Constant::create(ov::element::i64, {}, {1}));
    auto kept = std::make_shared<v8::Slice>(keep, i64s({0}), kv, i64s({1}), i64s({1}));
    auto out_shape = std::make_shared<v0::Concat>(
        ov::OutputVector{std::make_shared<v8::Slice>(rows_shape, i64s({0}), i64s({-1}), i64s({1})), kv},
        0);
    auto keep_rows = std::make_shared<v1::Reshape>(kept, out_shape, false);
    auto result = std::make_shared<v1::Select>(
        std::make_shared<v1::NotEqual>(keep_rows, v0::Constant::create(ov::element::u8, {}, {0})),
        mask,
        v0::Constant::create(mask.get_element_type(), {}, {-std::numeric_limits<float>::infinity()}));
    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
