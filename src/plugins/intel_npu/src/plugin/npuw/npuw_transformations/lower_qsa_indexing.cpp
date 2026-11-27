// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lower_qsa_indexing.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;
namespace v11 = ov::op::v11;
namespace v12 = ov::op::v12;
namespace v17 = ov::op::v17;

struct Builder {
    NodeVector nodes;

    template <typename T, typename... Args>
    std::shared_ptr<T> make(Args&&... args) {
        auto node = std::make_shared<T>(std::forward<Args>(args)...);
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<v0::Constant> i32(int32_t value) {
        auto node = v0::Constant::create(element::i32, Shape{}, {value});
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<v0::Constant> i64(std::vector<int64_t> values) {
        auto node = v0::Constant::create(element::i64, Shape{values.size()}, values);
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<v0::Constant> i32_array(const Shape& shape, const std::vector<int32_t>& values) {
        auto node = v0::Constant::create(element::i32, shape, values);
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<v0::Constant> f32(float value) {
        auto node = v0::Constant::create(element::f32, Shape{}, {value});
        nodes.push_back(node);
        return node;
    }

    Output<Node> slice(const Output<Node>& value, int64_t start, int64_t end, int64_t axis) {
        return make<v8::Slice>(value, i64({start}), i64({end}), i64({1}), i64({axis}));
    }

    Output<Node> clamp(const Output<Node>& value, int32_t upper) {
        return make<v1::Minimum>(make<v1::Maximum>(value, i32(0)), i32(upper));
    }
};

void lower_prepare(const std::shared_ptr<v17::QSAPrepare>& op) {
    const auto raw = op->input_value(0);
    const auto blocks = op->input_value(1);
    const auto cos = op->input_value(3);
    const auto sin = op->input_value(4);
    const auto weights = op->input_value(5);
    const auto incomplete = op->input_value(6);
    const auto incomplete_rope = op->input_value(7);
    const auto raw_shape = raw.get_partial_shape();
    const auto block_shape = blocks.get_partial_shape();
    const auto cos_shape = cos.get_partial_shape();
    const auto sin_shape = sin.get_partial_shape();
    const auto pending_shape = incomplete.get_partial_shape();
    const auto rope_shape = incomplete_rope.get_partial_shape();
    const auto weight_shape = weights.get_partial_shape();
    const auto length_shape = op->get_input_partial_shape(2);
    const auto block_size = op->get_key_block_size();
    OPENVINO_ASSERT(raw_shape.is_static() && raw_shape.size() == 3 && raw_shape[0].get_length() == 1 &&
                        block_shape.is_static() && block_shape.size() == 3 && block_shape[0].get_length() == 1 &&
                        cos_shape.is_static() && cos_shape.size() == 3 && cos_shape[0].get_length() == 1 &&
                        sin_shape == cos_shape && pending_shape.is_static() && pending_shape.size() == 3 &&
                        rope_shape.is_static() && rope_shape.size() == 2 && weight_shape.is_static() &&
                        weight_shape.size() == 1 && length_shape == (PartialShape{1}),
                    "NPUW QSAPrepare requires static batch-1 inputs, RoPE tables and normalization weights: ",
                    op->get_friendly_name());
    const auto length = raw_shape[1].get_length();
    const auto dim = raw_shape[2].get_length();
    const auto capacity = block_shape[1].get_length();
    const auto rotary = cos_shape[2].get_length();
    OPENVINO_ASSERT(block_size > 1 && length > 0 && capacity > 0 && dim > 0 && rotary > 0 && rotary % 2 == 0 &&
                        rotary <= dim && cos_shape[1] == raw_shape[1] && block_shape[2] == raw_shape[2] &&
                        pending_shape == (PartialShape{1, block_size - 1, dim}) &&
                        rope_shape == (PartialShape{1, rotary * 2}) && weight_shape == (PartialShape{dim}) &&
                        block_size <= std::numeric_limits<int32_t>::max() &&
                        length <= std::numeric_limits<int32_t>::max() - block_size &&
                        capacity < std::numeric_limits<int32_t>::max() &&
                        capacity <= std::numeric_limits<int32_t>::max() / block_size,
                    "Unsupported NPUW QSAPrepare dimensions: ",
                    op->get_friendly_name());
    for (const auto& value : {raw, blocks, cos, sin, weights, incomplete, incomplete_rope}) {
        OPENVINO_ASSERT(value.get_element_type() == element::f32,
                        "NPUW QSAPrepare currently requires f32 inputs: ",
                        op->get_friendly_name());
    }
    const auto new_block_capacity = (length + block_size - 1) / block_size;
    Builder b;
    const auto zero = b.i32(0);
    const auto r = b.i32(static_cast<int32_t>(block_size));
    // Pack the valid pending keys and left-padded new keys in logical sequence order.
    const auto key_length = b.make<v0::Convert>(b.make<v0::Squeeze>(op->input_value(2), b.i64({0})), element::i32);
    const auto past = b.make<v1::Maximum>(b.make<v1::Subtract>(key_length, b.i32(static_cast<int32_t>(length))), zero);
    const auto cached = b.make<v1::Divide>(past, r);
    const auto pending = b.make<v1::Subtract>(past, b.make<v1::Multiply>(cached, r));
    const auto complete = b.make<v1::Divide>(key_length, r);
    const auto new_blocks = b.make<v1::Subtract>(complete, cached);
    const auto padding =
        b.make<v1::Subtract>(b.i32(static_cast<int32_t>(length)), b.make<v1::Subtract>(key_length, past));
    const auto all_keys = b.make<v0::Concat>(OutputVector{incomplete, raw}, 1);
    auto key_indices = [&](const Output<Node>& logical) -> Output<Node> {
        const auto current = b.make<v1::Add>(b.i32(static_cast<int32_t>(block_size - 1)),
                                             b.make<v1::Add>(padding, b.make<v1::Subtract>(logical, pending)));
        return b.clamp(b.make<v1::Select>(b.make<v1::Less>(logical, pending), logical, current),
                       static_cast<int32_t>(block_size - 2 + length));
    };
    auto rope_rows = [&](const Output<Node>& logical) -> std::pair<Output<Node>, Output<Node>> {
        Output<Node> from_pending = b.make<v1::Less>(logical, pending);
        const auto current =
            b.clamp(b.make<v1::Add>(padding, b.make<v1::Subtract>(logical, pending)), static_cast<int32_t>(length - 1));
        auto current_cos = b.make<v8::Gather>(cos, current, b.i32(1));
        auto current_sin = b.make<v8::Gather>(sin, current, b.i32(1));
        auto old_cos = b.slice(incomplete_rope, 0, rotary, 1);
        auto old_sin = b.slice(incomplete_rope, rotary, rotary * 2, 1);
        if (logical.get_partial_shape().rank().get_length() == 1) {
            from_pending = b.make<v0::Unsqueeze>(from_pending, b.i64({0, 2}));
            old_cos = b.make<v0::Unsqueeze>(old_cos, b.i64({1}));
            old_sin = b.make<v0::Unsqueeze>(old_sin, b.i64({1}));
        }
        return {b.make<v1::Select>(from_pending, old_cos, current_cos),
                b.make<v1::Select>(from_pending, old_sin, current_sin)};
    };

    std::vector<int32_t> raw_positions(static_cast<size_t>(new_block_capacity * block_size));
    std::iota(raw_positions.begin(), raw_positions.end(), 0);
    auto all_positions = b.i32_array(Shape{raw_positions.size()}, raw_positions);
    auto gathered = b.make<v8::Gather>(all_keys, key_indices(all_positions), b.i32(1));
    auto grouped = b.make<v1::Reshape>(gathered, b.i64({1, new_block_capacity, block_size, dim}), false);
    auto pooled = b.make<v1::ReduceMean>(grouped, b.i64({2}), false);
    auto squared = b.make<v1::Multiply>(pooled, pooled);
    auto mean_square = b.make<v1::ReduceMean>(squared, b.i64({-1}), true);
    auto normalized = b.make<v1::Multiply>(
        b.make<v1::Divide>(pooled, b.make<v0::Sqrt>(b.make<v1::Add>(mean_square, b.f32(op->get_epsilon())))),
        weights);

    std::vector<int32_t> starts(static_cast<size_t>(new_block_capacity));
    for (size_t i = 0; i < starts.size(); ++i) {
        starts[i] = static_cast<int32_t>(i * block_size);
    }
    auto first = b.i32_array(Shape{starts.size()}, starts);
    auto rows = rope_rows(first);
    auto x = b.slice(normalized, 0, rotary, 2);
    auto first_half = b.slice(x, 0, rotary / 2, 2);
    auto second_half = b.slice(x, rotary / 2, rotary, 2);
    auto rotate_half = b.make<v0::Concat>(OutputVector{b.make<v0::Negative>(second_half), first_half}, 2);
    auto rotated = b.make<v1::Add>(b.make<v1::Multiply>(x, rows.first), b.make<v1::Multiply>(rotate_half, rows.second));
    Output<Node> new_keys = rotated;
    if (rotary < dim) {
        new_keys = b.make<v0::Concat>(OutputVector{rotated, b.slice(normalized, rotary, dim, 2)}, 2);
    }

    std::vector<int32_t> cache_positions(static_cast<size_t>(capacity + 1));
    std::iota(cache_positions.begin(), cache_positions.end(), 0);
    auto cache_range = b.i32_array(Shape{1, static_cast<size_t>(capacity + 1), 1}, cache_positions);
    auto empty_slot = v0::Constant::create(element::f32, Shape{1, 1, static_cast<size_t>(dim)}, {0.f});
    b.nodes.push_back(empty_slot);
    auto cache = b.make<v0::Concat>(OutputVector{blocks, empty_slot}, 1);
    auto valid_cache = b.make<v1::Select>(b.make<v1::Less>(cache_range, cached), cache, b.f32(0.f));
    // The extra slot absorbs candidates that do not complete a block.
    std::vector<int32_t> candidate_positions(static_cast<size_t>(new_block_capacity));
    std::iota(candidate_positions.begin(), candidate_positions.end(), 0);
    auto candidate_range = b.i32_array(Shape{candidate_positions.size()}, candidate_positions);
    auto destinations = b.make<v1::Select>(b.make<v1::Less>(candidate_range, new_blocks),
                                           b.make<v1::Add>(cached, candidate_range),
                                           b.i32(static_cast<int32_t>(capacity)));
    auto indices =
        b.make<v3::Broadcast>(b.make<v0::Unsqueeze>(destinations, b.i64({0, 2})), b.i64({1, new_block_capacity, dim}));
    auto updated = b.make<v12::ScatterElementsUpdate>(valid_cache, indices, new_keys, b.i32(1));
    auto result_blocks = b.slice(updated, 0, capacity, 1);

    auto rest = b.make<v1::Subtract>(key_length, b.make<v1::Multiply>(complete, r));
    std::vector<int32_t> tail_positions(static_cast<size_t>(block_size - 1));
    std::iota(tail_positions.begin(), tail_positions.end(), 0);
    auto tail_range = b.i32_array(Shape{tail_positions.size()}, tail_positions);
    auto tail_start = b.make<v1::Multiply>(new_blocks, r);
    auto tail_keys = b.make<v8::Gather>(all_keys, key_indices(b.make<v1::Add>(tail_start, tail_range)), b.i32(1));
    auto tail_mask = b.make<v0::Unsqueeze>(b.make<v1::Less>(tail_range, rest), b.i64({0, 2}));
    auto result_incomplete = b.make<v1::Select>(tail_mask, tail_keys, b.f32(0.f));
    auto tail_rope = rope_rows(tail_start);
    auto rope_pair = b.make<v0::Concat>(OutputVector{tail_rope.first, tail_rope.second}, 1);
    auto result_rope = b.make<v1::Select>(b.make<v1::Greater>(rest, zero), rope_pair, b.f32(0.f));

    ov::copy_runtime_info(op, b.nodes);
    ov::replace_node(op, OutputVector{result_blocks, result_incomplete, result_rope});
}

void lower_indexer(const std::shared_ptr<v17::SparseAttentionIndexer>& op) {
    const auto query_shape = op->get_input_partial_shape(0);
    const auto blocks_shape = op->get_input_partial_shape(1);
    const auto length_shape = op->get_input_partial_shape(2);
    const auto r = op->get_key_block_size();
    const auto budget = op->get_token_budget();
    OPENVINO_ASSERT(op->get_input_size() == 3 && op->get_num_selection_heads() == 1 && query_shape.is_static() &&
                        query_shape.size() == 4 && query_shape[0].get_length() == 1 && blocks_shape.is_static() &&
                        blocks_shape.size() == 3 && blocks_shape[0].get_length() == 1 &&
                        length_shape == (PartialShape{1}) && r > 0 && r <= std::numeric_limits<int32_t>::max() &&
                        budget > 0 && budget % r == 0,
                    "Unsupported NPUW SparseAttentionIndexer inputs: ",
                    op->get_friendly_name());
    const auto length = query_shape[2].get_length();
    const auto capacity = blocks_shape[1].get_length();
    const auto width = budget / r;
    OPENVINO_ASSERT(length > 0 && capacity > 0 && width > 0 && query_shape[3] == blocks_shape[2] &&
                        length <= std::numeric_limits<int32_t>::max() &&
                        capacity <= std::numeric_limits<int32_t>::max() / r &&
                        width <= std::numeric_limits<int32_t>::max() && op->get_input_element_type(0) == element::f32 &&
                        op->get_input_element_type(1) == element::f32,
                    "Unsupported NPUW SparseAttentionIndexer dimensions or precision: ",
                    op->get_friendly_name());
    const auto topk = std::min(capacity, width);
    Builder b;
    const auto key_length = b.make<v0::Convert>(b.make<v0::Squeeze>(op->input_value(2), b.i64({0})), element::i32);
    auto keys = b.make<v0::Unsqueeze>(op->input_value(1), b.i64({1}));
    auto scores = b.make<v0::MatMul>(op->input_value(0), keys, false, true);
    auto relu = b.make<v0::Relu>(scores);
    auto reduced = b.make<v1::ReduceSum>(relu, b.i64({1}), true);
    std::vector<int32_t> query_positions(static_cast<size_t>(length));
    std::iota(query_positions.begin(), query_positions.end(), 0);
    auto queries = b.i32_array(Shape{1, 1, static_cast<size_t>(length), 1}, query_positions);
    auto visible = b.make<v1::Divide>(
        b.make<v1::Maximum>(b.make<v1::Add>(b.make<v1::Subtract>(key_length, b.i32(static_cast<int32_t>(length))),
                                            b.make<v1::Add>(queries, b.i32(1))),
                            b.i32(0)),
        b.i32(static_cast<int32_t>(r)));
    std::vector<int32_t> block_positions(static_cast<size_t>(capacity));
    std::iota(block_positions.begin(), block_positions.end(), 0);
    auto block_range = b.i32_array(Shape{1, 1, 1, static_cast<size_t>(capacity)}, block_positions);
    auto masked = b.make<v1::Select>(b.make<v1::Less>(block_range, visible),
                                     reduced,
                                     b.f32(-std::numeric_limits<float>::infinity()));
    auto top = b.make<v11::TopK>(masked,
                                 b.i32(static_cast<int32_t>(topk)),
                                 -1,
                                 ov::op::TopKMode::MAX,
                                 ov::op::TopKSortType::SORT_VALUES,
                                 element::i32,
                                 true);
    Output<Node> padded = top->output(1);
    if (topk < width) {
        padded = b.make<v1::Pad>(padded,
                                 b.i64({0, 0, 0, 0}),
                                 b.i64({0, 0, 0, width - topk}),
                                 b.i32(-1),
                                 ov::op::PadMode::CONSTANT);
    }
    auto counts4 = b.make<v1::Minimum>(visible, b.i32(static_cast<int32_t>(width)));
    auto counts = b.make<v0::Squeeze>(counts4, b.i64({3}));
    std::vector<int32_t> slots(static_cast<size_t>(width));
    std::iota(slots.begin(), slots.end(), 0);
    auto slot_range = b.i32_array(Shape{1, 1, 1, static_cast<size_t>(width)}, slots);
    auto indices = b.make<v1::Select>(b.make<v1::Less>(slot_range, counts4), padded, b.i32(-1));
    ov::copy_runtime_info(op, b.nodes);
    ov::replace_node(op, OutputVector{indices, counts});
}

}  // namespace

bool ov::npuw::LowerQSAPrepare::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto op = ov::as_type_ptr<ov::op::v17::QSAPrepare>(node)) {
            lower_prepare(op);
            changed = true;
        }
    }
    return changed;
}

bool ov::npuw::LowerSparseAttentionIndexer::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto op = ov::as_type_ptr<ov::op::v17::SparseAttentionIndexer>(node)) {
            lower_indexer(op);
            changed = true;
        }
    }
    return changed;
}
