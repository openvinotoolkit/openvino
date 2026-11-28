// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_sparse_attention_to_dense.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/node_util.hpp"
#include "transformations/utils/utils.hpp"

namespace {

namespace op = ov::op;
namespace util = ov::npuw::util;
using Out = ov::Output<ov::Node>;

constexpr int64_t kEnd = std::numeric_limits<int64_t>::max();

// The QSA state whose block dimension is resized (one of util::get_qsa_state_kinds())
constexpr const char* kBlockIndexKey = "block_index_key";

// Graph building helpers. Every new node is collected to copy the runtime info of the replaced operation.
class Builder {
public:
    // Constant subgraphs are folded right away. ShapeOf is kept as is: the model is reshaped to static afterwards.
    template <typename T, typename... Args>
    Out make(Args&&... args) {
        std::shared_ptr<ov::Node> node;
        if constexpr (std::is_same_v<T, op::v3::ShapeOf>) {
            node = std::make_shared<T>(std::forward<Args>(args)...);
        } else {
            node = ov::op::util::make_try_fold<T>(std::forward<Args>(args)...);
        }
        m_nodes.push_back(node);
        return node->output(0);
    }

    template <typename T, typename... Args>
    std::shared_ptr<T> make_node(Args&&... args) {
        auto node = std::make_shared<T>(std::forward<Args>(args)...);
        m_nodes.push_back(node);
        return node;
    }

    Out i64(int64_t v) {
        return make<op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{v});
    }
    Out i64v(const std::vector<int64_t>& v) {
        return make<op::v0::Constant>(ov::element::i64, ov::Shape{v.size()}, v);
    }
    Out f32(float v) {
        return make<op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{v});
    }
    Out scalar_like(float v, const ov::element::Type& type) {
        return make<op::v0::Constant>(type, ov::Shape{}, std::vector<float>{v});
    }

    // scalar i64 size of an axis
    Out dim(const Out& x, int64_t axis) {
        return make<op::v8::Gather>(make<op::v3::ShapeOf>(x, ov::element::i64), i64(axis), i64(0));
    }
    // 1D shape from i64 scalars
    Out vec(const std::vector<Out>& scalars) {
        ov::OutputVector parts;
        for (const auto& s : scalars) {
            parts.push_back(unsq(s, 0));
        }
        return make<op::v0::Concat>(parts, 0);
    }

    Out add(const Out& a, const Out& b) {
        return make<op::v1::Add>(a, b);
    }
    Out sub(const Out& a, const Out& b) {
        return make<op::v1::Subtract>(a, b);
    }
    Out mul(const Out& a, const Out& b) {
        return make<op::v1::Multiply>(a, b);
    }
    Out div(const Out& a, const Out& b) {  // floor division for integers
        return make<op::v1::Divide>(a, b);
    }
    Out min(const Out& a, const Out& b) {
        return make<op::v1::Minimum>(a, b);
    }
    Out max(const Out& a, const Out& b) {
        return make<op::v1::Maximum>(a, b);
    }
    Out less(const Out& a, const Out& b) {
        return make<op::v1::Less>(a, b);
    }
    Out greater(const Out& a, const Out& b) {
        return make<op::v1::Greater>(a, b);
    }
    Out ge(const Out& a, const Out& b) {
        return make<op::v1::GreaterEqual>(a, b);
    }
    Out le(const Out& a, const Out& b) {
        return make<op::v1::LessEqual>(a, b);
    }
    Out land(const Out& a, const Out& b) {
        return make<op::v1::LogicalAnd>(a, b);
    }
    Out lor(const Out& a, const Out& b) {
        return make<op::v1::LogicalOr>(a, b);
    }
    Out select(const Out& c, const Out& a, const Out& b) {
        return make<op::v1::Select>(c, a, b);
    }
    Out unsq(const Out& x, int64_t axis) {
        return make<op::v0::Unsqueeze>(x, i64(axis));
    }
    Out convert(const Out& x, const ov::element::Type& type) {
        return x.get_element_type() == type ? x : make<op::v0::Convert>(x, type);
    }
    Out range(const Out& start, const Out& stop) {
        return make<op::v4::Range>(start, stop, i64(1), ov::element::i64);
    }
    Out slice(const Out& x, const Out& start, const Out& stop, int64_t axis) {
        return make<op::v8::Slice>(x, start, stop, i64v({1}), i64v({axis}));
    }
    Out gather(const Out& x, const Out& indices, int64_t axis, int64_t batch_dims) {
        return make<op::v8::Gather>(x, indices, i64(axis), batch_dims);
    }
    Out concat(const ov::OutputVector& parts, int64_t axis) {
        return make<op::v0::Concat>(parts, axis);
    }
    Out broadcast(const Out& x, const Out& shape) {
        return make<op::v3::Broadcast>(x, shape);
    }
    Out reduce_sum(const Out& x, const Out& axes) {
        return make<op::v1::ReduceSum>(x, axes, false);
    }
    Out reduce_mean(const Out& x, const Out& axes, bool keep_dims) {
        return make<op::v1::ReduceMean>(x, axes, keep_dims);
    }
    Out shape_of(const Out& x) {
        return make<op::v3::ShapeOf>(x, ov::element::i64);
    }
    Out reshape(const Out& x, const Out& shape) {
        return make<op::v1::Reshape>(x, shape, false);
    }

    const ov::NodeVector& nodes() const {
        return m_nodes;
    }

private:
    ov::NodeVector m_nodes;
};

bool is_empty(const Out& x) {
    const auto& shape = x.get_partial_shape();
    return shape.is_static() && ov::shape_size(shape.to_shape()) == 0;
}

// attention_mask derived values shared by all the layers
struct MaskInfo {
    Out valid;      // [N, S] i64, 1 for the valid key slots
    Out positions;  // [N, S] i64, the logical position of every valid key slot, -1 for the masked ones
    Out length;     // S, scalar i64
};

MaskInfo build_mask_info(Builder& b, const std::shared_ptr<op::v0::Parameter>& attention_mask) {
    const auto mask = attention_mask->output(0);
    const auto valid_bool = b.greater(mask, b.scalar_like(0.0f, mask.get_element_type()));
    const auto valid = b.convert(valid_bool, ov::element::i64);
    const auto count = b.make<op::v0::CumSum>(valid, b.i64(1));
    const auto positions = b.select(valid_bool, b.sub(count, b.i64(1)), b.i64(-1));
    positions.get_node()->set_friendly_name("npuw/qsa_sequence_positions");
    return MaskInfo{valid, positions, b.dim(mask, 1)};
}

// [N, L] logical positions of the query tokens: the last L key slots, since the valid keys of an
// NPUW sequence are right-aligned. The positions themselves come from the mask and are
// layout-independent; this last-L slice is one of the places to revisit if NPUW moves to right padding.
Out query_positions(Builder& b, const MaskInfo& m, const Out& query_length) {
    return b.slice(m.positions, b.unsq(b.sub(m.length, query_length), 0), b.i64v({kEnd}), 1);
}

// Block index keys output -> the same keys with one extra row, [N, B + 1, Di] (see decompose_indexer)
using PaddedKeys = std::map<Out, Out>;

// Indexer block_indices output -> the 0/1 mask of the selected visible blocks, [N, Hs, L, B + 1]. The attention reads
// the mask instead of scattering the indices again.
using SelectedBlocks = std::map<Out, Out>;

// QSAPrepare, see its specification. The past/current split is counted from attention_mask, since a
// partial NPUW prefill chunk is left-padded after the past tokens, and the number of blocks is static.
// That right alignment is also what offset = M - current below relies on: the valid new tokens are the
// last rows of raw. Like query_positions, it must be revisited if NPUW moves to right padding.
ov::OutputVector decompose_qsa_prepare(Builder& b,
                                       const std::shared_ptr<op::v17::QSAPrepare>& node,
                                       const MaskInfo& m,
                                       int64_t num_blocks,
                                       PaddedKeys& padded_keys) {
    const int64_t r = node->get_key_block_size();
    const auto raw = node->input_value(0);
    const auto past_keys = node->input_value(1);
    const auto rope_cos = node->input_value(3);
    const auto rope_sin = node->input_value(4);
    const auto k_norm_weight = node->input_value(5);
    const auto past_raw = node->input_value(6);
    const auto past_rope = node->input_value(7);
    const auto& data_type = raw.get_element_type();

    const auto& past_shape = past_keys.get_partial_shape();
    OPENVINO_ASSERT(past_shape.rank().is_static() && past_shape.size() == 3 && past_shape[1] == num_blocks,
                    "ConvertSparseAttentionToDense: ",
                    node->get_friendly_name(),
                    " past block index keys have shape ",
                    past_shape,
                    ", expected ",
                    num_blocks,
                    " blocks");

    const auto zero = b.i64(0), one = b.i64(1), ratio = b.i64(r);
    const auto batch = b.dim(raw, 0), seq_len = b.dim(raw, 1), head_dim = b.dim(raw, 2);

    const auto split = b.unsq(b.sub(m.length, seq_len), 0);
    // Only full-width reductions: a [0, split) slice is empty when a prefill chunk covers the whole
    // prompt (split == 0), and the NPU cannot execute empty tensors. Deriving past as total - current
    // keeps both reduction inputs non-empty ([N, S] and [N, L] with L >= 1).
    const auto total = b.reduce_sum(m.valid, b.i64v({1}));
    const auto current = b.reduce_sum(b.slice(m.valid, split, b.i64v({kEnd}), 1), b.i64v({1}));
    const auto past = b.sub(total, current);
    const auto num_cached = b.div(past, ratio);
    const auto pending_past = b.sub(past, b.mul(num_cached, ratio));  // past tokens of the incomplete block
    const auto pending = b.add(pending_past, current);
    const auto num_new = b.div(pending, ratio);
    const auto rest = b.sub(pending, b.mul(num_new, ratio));
    const auto num_tokens = b.add(seq_len, b.i64(r - 1));                  // M
    const auto num_steps = b.div(b.add(num_tokens, b.i64(r - 1)), ratio);  // J >= 1
    const auto offset = b.unsq(b.sub(num_tokens, current), 1);             // [N, 1] row of the first new token
    const auto pending_past_col = b.unsq(pending_past, 1);

    // pending token index -> row of a table holding the past entries at 0.. and the new ones from base
    const auto source_rows = [&](const Out& token, const Out& base, const Out& last) {
        const auto rows = b.select(b.less(token, pending_past_col), token, b.add(b.sub(token, pending_past_col), base));
        return b.min(b.max(rows, zero), last);
    };

    // 1. the pending tokens: the incomplete block of the previous calls, then the valid new tokens
    const auto tokens = b.concat({b.convert(past_raw, data_type), raw}, 1);
    const auto t = b.unsq(b.range(zero, b.mul(num_steps, ratio)), 0);                              // [1, J * r]
    const auto gathered = b.gather(tokens, source_rows(t, offset, b.sub(num_tokens, one)), 1, 1);  // [N, J * r, Di]

    // 2. mean-pool and RMS-normalize every block that can be completed
    auto keys = b.reshape(gathered, b.vec({batch, num_steps, ratio, head_dim}));
    keys = b.reduce_mean(b.convert(keys, ov::element::f32), b.i64(2), false);  // [N, J, Di]
    if (!is_empty(k_norm_weight)) {
        const auto mean_square = b.reduce_mean(b.mul(keys, keys), b.i64(-1), true);
        const auto inv_rms = b.div(b.f32(1.0f), b.make<op::v0::Sqrt>(b.add(mean_square, b.f32(node->get_epsilon()))));
        keys = b.mul(b.mul(keys, inv_rms), b.convert(k_norm_weight, ov::element::f32));
    }

    // 3. rotate-half RoPE at the first token of every block: a past one from past_incomplete_rope
    const bool use_rope = !is_empty(rope_cos);
    std::function<std::pair<Out, Out>(const Out&)> rope_at;
    if (use_rope) {
        const auto rotary_dim = b.dim(rope_cos, 2);
        const auto rotary_dim_1d = b.unsq(rotary_dim, 0);
        const auto past_rope_rows = b.unsq(b.convert(past_rope, ov::element::f32), 1);  // [N, 1, 2 * Dr]
        const auto past_cos = b.slice(past_rope_rows, b.i64v({0}), rotary_dim_1d, -1);
        const auto past_sin = b.slice(past_rope_rows, rotary_dim_1d, b.i64v({kEnd}), -1);
        const auto table = [&](const Out& past_rows, const Out& current_rows) {  // [N, 1 + L, Dr]
            const auto rows =
                b.broadcast(b.convert(current_rows, ov::element::f32), b.vec({batch, seq_len, rotary_dim}));
            return b.concat({past_rows, rows}, 1);
        };
        const auto cos_table = table(past_cos, rope_cos);
        const auto sin_table = table(past_sin, rope_sin);
        const auto rope_offset = b.sub(offset, b.i64(r - 2));  // 1 + row of the first valid new token
        rope_at = [&b, &source_rows, rope_offset, seq_len, cos_table, sin_table](const Out& token) {
            const auto rows = source_rows(token, rope_offset, seq_len);
            return std::make_pair(b.gather(cos_table, rows, 1, 1), b.gather(sin_table, rows, 1, 1));
        };

        const auto first = b.unsq(b.mul(b.range(zero, num_steps), ratio), 0);  // [1, J]
        const auto [first_cos, first_sin] = rope_at(first);
        const auto half_1d = b.unsq(b.div(rotary_dim, b.i64(2)), 0);
        const auto x_rope = b.slice(keys, b.i64v({0}), rotary_dim_1d, -1);
        const auto x_pass = b.slice(keys, rotary_dim_1d, b.i64v({kEnd}), -1);
        const auto x1 = b.slice(x_rope, b.i64v({0}), half_1d, -1);
        const auto x2 = b.slice(x_rope, half_1d, b.i64v({kEnd}), -1);
        const auto rotated = b.concat({b.make<op::v0::Negative>(x2), x1}, -1);
        keys = b.concat({b.add(b.mul(x_rope, first_cos), b.mul(rotated, first_sin)), x_pass}, -1);
    }

    // 4. the cached blocks of every sequence, then the computed ones; the extra column B absorbs the invalid ones
    const auto zero_value = b.scalar_like(0.0f, data_type);
    auto output = b.concat({b.convert(past_keys, data_type), b.broadcast(zero_value, b.vec({batch, one, head_dim}))},
                           1);  // [N, B + 1, Di]
    const auto columns = b.unsq(b.range(zero, b.i64(num_blocks + 1)), 0);
    const auto cached = b.less(columns, b.unsq(num_cached, 1));  // [N, B + 1]
    output = b.select(b.unsq(cached, -1), output, zero_value);
    const auto steps = b.unsq(b.range(zero, num_steps), 0);  // [1, J]
    auto targets = b.select(b.less(steps, b.unsq(num_new, 1)), b.add(b.unsq(num_cached, 1), steps), b.i64(num_blocks));
    targets = b.broadcast(b.unsq(targets, -1), b.vec({batch, num_steps, head_dim}));
    const auto padded = b.make<op::v3::ScatterElementsUpdate>(output, targets, b.convert(keys, data_type), b.i64(1));
    output = b.slice(padded, b.i64v({0}), b.i64v({num_blocks}), 1);  // [N, B, Di]
    padded_keys[output] = padded;

    // 5. the incomplete block after this call
    const auto slots = b.unsq(b.range(zero, b.i64(r - 1)), 0);  // [1, r - 1]
    const auto valid = b.less(slots, b.unsq(rest, 1));          // [N, r - 1]
    auto idx = b.add(b.unsq(b.mul(num_new, ratio), 1), slots);
    idx = b.min(idx, b.sub(b.mul(num_steps, ratio), one));
    auto incomplete_raw = b.gather(gathered, idx, 1, 1);  // [N, r - 1, Di]
    incomplete_raw = b.select(b.unsq(valid, -1), incomplete_raw, zero_value);

    Out incomplete_rope;
    if (use_rope) {
        const auto [rest_cos, rest_sin] = rope_at(b.unsq(b.mul(num_new, ratio), 1));                 // [N, 1, Dr]
        incomplete_rope = b.make<op::v0::Squeeze>(b.concat({rest_cos, rest_sin}, -1), b.i64v({1}));  // [N, 2 * Dr]
        const auto has_rest = b.unsq(b.greater(rest, zero), 1);
        incomplete_rope = b.select(has_rest, incomplete_rope, b.f32(0.0f));
    } else {
        incomplete_rope = b.broadcast(b.f32(0.0f), b.vec({batch, zero}));
    }
    return {output, b.convert(incomplete_raw, data_type), b.convert(incomplete_rope, data_type)};
}

// The token budget covers every key block, so the indexer selects all the visible blocks whatever the scores are
// (the order of the selected blocks does not matter): the scoring and the top-k are not needed.
ov::OutputVector select_all_visible_blocks(Builder& b,
                                           const std::shared_ptr<op::v17::SparseAttentionIndexer>& node,
                                           const MaskInfo& m,
                                           int64_t block_topk) {
    const auto zero = b.i64(0), one = b.i64(1), ratio = b.i64(node->get_key_block_size());
    const auto index_query = node->input_value(0);
    const auto batch = b.dim(index_query, 0);
    const auto query_length = b.dim(index_query, 2);
    const auto num_selection_heads = b.i64(node->get_num_selection_heads());

    const auto positions = query_positions(b, m, query_length);  // [N, L]
    const auto num_visible = b.max(b.div(b.add(positions, one), ratio), zero);
    const auto block_count = b.min(num_visible, b.i64(block_topk));  // [N, L]
    const auto slots = b.range(zero, b.i64(block_topk));
    const auto valid = b.less(slots, b.unsq(block_count, -1));  // [N, L, KB]
    const auto minus_one = b.make<op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{-1});
    const auto ids = b.broadcast(b.convert(slots, ov::element::i32),
                                 b.vec({batch, num_selection_heads, query_length, b.i64(block_topk)}));
    const auto block_indices = b.select(b.unsq(valid, 1), ids, minus_one);
    const auto count = b.broadcast(b.convert(b.unsq(block_count, 1), ov::element::i32),
                                   b.vec({batch, num_selection_heads, query_length}));
    return {b.convert(block_indices, node->get_output_element_type(0)),
            b.convert(count, node->get_output_element_type(1))};
}

// SparseAttentionIndexer, see its specification. The query positions come from attention_mask.
ov::OutputVector decompose_indexer(Builder& b,
                                   const std::shared_ptr<op::v17::SparseAttentionIndexer>& node,
                                   const MaskInfo& m,
                                   const PaddedKeys& padded_keys,
                                   SelectedBlocks& selected_blocks) {
    const int64_t r = node->get_key_block_size();
    const int64_t block_topk = node->get_token_budget() / r;
    const int64_t num_selection_heads = node->get_num_selection_heads();
    const auto index_query = node->input_value(0);
    const auto block_index_keys = node->input_value(1);

    const auto& keys_shape = block_index_keys.get_partial_shape();
    OPENVINO_ASSERT(keys_shape.rank().is_static() && keys_shape.size() == 3 && keys_shape[1].is_static(),
                    "ConvertSparseAttentionToDense: ",
                    node->get_friendly_name(),
                    " needs a static number of key blocks, got ",
                    keys_shape);
    const int64_t num_blocks = keys_shape[1].get_length();
    if (block_topk >= num_blocks) {
        return select_all_visible_blocks(b, node, m, block_topk);
    }

    const auto zero = b.i64(0), one = b.i64(1), ratio = b.i64(r);
    const auto batch = b.dim(index_query, 0);
    const auto num_index_heads = b.dim(index_query, 1);
    const auto query_length = b.dim(index_query, 2);
    const auto head_dim = b.dim(block_index_keys, 2);

    // One extra never-visible key keeps every tensor non-empty when there is no complete block yet.
    // A query sees the blocks below (p + 1) / r <= B, so the value of the row B does not matter: the
    // QSAPrepare one is read as is. This also leaves the present block keys with the sole Result reader,
    // otherwise a cross-subgraph f32 model output would be linked to an F16IC-lowered parameter.
    Out keys;
    if (const auto it = padded_keys.find(block_index_keys); it != padded_keys.end()) {
        keys = b.convert(it->second, ov::element::f32);
    } else {
        keys = b.concat(
            {b.convert(block_index_keys, ov::element::f32), b.broadcast(b.f32(0.0f), b.vec({batch, one, head_dim}))},
            1);
    }
    auto scores = b.make<op::v0::Relu>(
        b.make<op::v0::MatMul>(b.convert(index_query, ov::element::f32), b.unsq(keys, 1), false, true));
    if (node->get_input_size() > 3) {
        const auto weights =
            b.make<op::v1::Transpose>(b.convert(node->input_value(3), ov::element::f32), b.i64v({0, 2, 1}));
        scores = b.mul(scores, b.unsq(weights, -1));
    }
    const auto group_size = b.div(num_index_heads, b.i64(num_selection_heads));
    scores =
        b.reshape(scores, b.vec({batch, b.i64(num_selection_heads), group_size, query_length, b.i64(num_blocks + 1)}));
    scores = b.reduce_sum(scores, b.i64(2));  // [N, Hs, L, B + 1]

    // a query at the position p sees the blocks b < (p + 1) / r; a padding query (p = -1) sees none
    const auto positions = query_positions(b, m, query_length);  // [N, L]
    const auto num_visible = b.max(b.div(b.add(positions, one), ratio), zero);
    const auto block_ids = b.range(zero, b.i64(num_blocks + 1));
    const auto visible = b.less(block_ids, b.unsq(num_visible, -1));  // [N, L, B + 1]
    scores = b.select(b.unsq(visible, 1), scores, b.f32(-std::numeric_limits<float>::infinity()));

    // stable top-k: equal scores -> smaller block index first
    // block_topk < num_blocks here (see the early return above), so the num_blocks + 1 score columns
    // always cover the request: TopK never truncates and no padding of the selection is needed.
    const auto topk = b.make_node<op::v11::TopK>(scores,
                                                 b.i64(block_topk),
                                                 -1,
                                                 op::TopKMode::MAX,
                                                 op::TopKSortType::SORT_VALUES,
                                                 ov::element::i32,
                                                 true);
    Out block_indices = topk->output(1);
    const auto minus_one = b.make<op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{-1});

    const auto block_count = b.min(num_visible, b.i64(block_topk));  // [N, L]
    const auto slots = b.range(zero, b.i64(block_topk));
    const auto valid = b.less(slots, b.unsq(block_count, -1));  // [N, L, KB]
    block_indices = b.select(b.unsq(valid, 1), block_indices, minus_one);
    const auto count = b.broadcast(b.convert(b.unsq(block_count, 1), ov::element::i32),
                                   b.vec({batch, b.i64(num_selection_heads), query_length}));

    // the mask of the selected visible blocks: a top-k that has fewer visible blocks than k also returns invisible ones
    const auto topk_ids = b.convert(topk->output(1), ov::element::i64);
    const auto no_blocks = b.broadcast(b.f32(0.0f), b.shape_of(scores));
    const auto marks = b.broadcast(b.f32(1.0f), b.shape_of(topk_ids));
    auto selected = b.make<op::v3::ScatterElementsUpdate>(no_blocks, topk_ids, marks, b.i64(-1));
    selected = b.select(b.unsq(visible, 1), selected, b.f32(0.0f));

    ov::OutputVector outputs{b.convert(block_indices, node->get_output_element_type(0)),
                             b.convert(count, node->get_output_element_type(1))};
    selected_blocks[outputs[0]] = selected;
    return outputs;
}

// GQA: [N, G, rest...] -> [N, H, rest...], every group repeated H / G times
Out repeat_heads(Builder& b,
                 const Out& x,
                 const Out& batch,
                 const Out& num_heads,
                 const Out& num_groups,
                 const std::vector<Out>& rest) {
    std::vector<Out> shape{batch, num_groups, b.div(num_heads, num_groups)};
    shape.insert(shape.end(), rest.begin(), rest.end());
    const auto expanded = b.broadcast(b.unsq(x, 2), b.vec(shape));
    std::vector<Out> out_shape{batch, num_heads};
    out_shape.insert(out_shape.end(), rest.begin(), rest.end());
    return b.reshape(expanded, b.vec(out_shape));
}

// SparseAttention as a dense SDPA: the mask allows the tokens of the selected blocks and the causal tail.
Out decompose_sparse_attention(Builder& b,
                               const std::shared_ptr<op::v17::SparseAttention>& node,
                               const MaskInfo& m,
                               const SelectedBlocks& selected_blocks) {
    OPENVINO_ASSERT(node->get_input_size() <= 7,
                    "ConvertSparseAttentionToDense: attention sinks of ",
                    node->get_friendly_name(),
                    " are not supported");
    OPENVINO_ASSERT(node->get_query_block_size() == 1,
                    "ConvertSparseAttentionToDense: ",
                    node->get_friendly_name(),
                    " has query_block_size ",
                    node->get_query_block_size(),
                    ", only 1 is supported");
    const int64_t r = node->get_key_block_size();
    const auto query = node->input_value(0);
    const auto& query_type = query.get_element_type();
    const auto key = b.convert(node->input_value(1), query_type);
    const auto value = b.convert(node->input_value(2), query_type);
    auto block_indices = b.convert(node->input_value(3), ov::element::i64);
    auto block_count = b.convert(node->input_value(4), ov::element::i64);

    const auto zero = b.i64(0), one = b.i64(1), ratio = b.i64(r);
    const auto batch = b.dim(query, 0), num_heads = b.dim(query, 1), query_length = b.dim(query, 2);
    const auto num_kv_heads = b.dim(key, 1), num_positions = b.dim(key, 2);
    const auto num_selection_heads = b.dim(block_indices, 1), block_topk = b.dim(block_indices, 3);

    // block mask [N, Hs, L, nB + 2]: the invalid entries -> column nB, the masked keys -> column nB + 1 (never set)
    const auto num_key_blocks = b.div(b.add(num_positions, b.i64(r - 1)), ratio);
    Out block_mask;
    const auto selected = selected_blocks.find(node->input_value(3));
    if (selected != selected_blocks.end()) {
        // the indexer has the mask of the selected blocks, [N, Hs, L, nB + 1]: add the column nB + 1
        block_mask = b.make<op::v12::Pad>(selected->second,
                                          b.i64v({0, 0, 0, 0}),
                                          b.i64v({0, 0, 0, 1}),
                                          b.f32(0.0f),
                                          op::PadMode::CONSTANT);
    } else {
        const auto valid = b.less(b.range(zero, block_topk), b.unsq(block_count, -1));
        block_indices = b.select(valid, block_indices, num_key_blocks);
        block_mask = b.broadcast(b.f32(0.0f),
                                 b.vec({batch, num_selection_heads, query_length, b.add(num_key_blocks, b.i64(2))}));
        block_mask = b.make<op::v3::ScatterElementsUpdate>(block_mask,
                                                           block_indices,
                                                           b.broadcast(b.f32(1.0f), b.shape_of(block_indices)),
                                                           b.i64(-1));
    }

    // token mask [N, Hs, L, S]: the block of every key slot within its sequence
    const auto& positions = m.positions;  // [N, S]
    const auto is_valid_key = b.ge(positions, zero);
    auto key_blocks = b.select(is_valid_key, b.div(b.max(positions, zero), ratio), b.add(num_key_blocks, one));
    key_blocks = b.broadcast(b.make<op::v0::Unsqueeze>(key_blocks, b.i64v({1, 2})),
                             b.vec({batch, num_selection_heads, query_length, num_positions}));
    auto mask = b.greater(b.make<op::v6::GatherElements>(block_mask, key_blocks, 3), b.f32(0.0f));
    if (node->get_causal()) {
        const auto query_pos = query_positions(b, m, query_length);  // [N, L]
        const auto tail_begin = b.mul(b.div(b.add(query_pos, one), ratio), ratio);
        const auto key_pos = b.unsq(positions, 1);  // [N, 1, S]
        const auto tail = b.land(b.land(b.ge(key_pos, b.unsq(tail_begin, -1)), b.le(key_pos, b.unsq(query_pos, -1))),
                                 b.ge(key_pos, zero));  // [N, L, S]
        mask = b.lor(mask, b.unsq(tail, 1));
    }

    // a single selection head broadcasts over the query heads in the SDPA
    const auto& hs = node->input_value(3).get_partial_shape();
    const bool single_selection = hs.rank().is_static() && hs[1].is_static() && hs[1].get_length() == 1;
    if (!single_selection) {
        mask = repeat_heads(b, mask, batch, num_heads, num_selection_heads, {query_length, num_positions});
    }
    const auto head_dim = b.dim(query, 3);
    const auto key_rep = repeat_heads(b, key, batch, num_heads, num_kv_heads, {num_positions, head_dim});
    const auto value_rep = repeat_heads(b, value, batch, num_heads, num_kv_heads, {num_positions, b.dim(value, 3)});

    Out scale;
    if (node->get_input_size() > 6) {
        scale = b.convert(node->input_value(6), query_type);
    } else {
        const auto& qs = query.get_partial_shape();
        OPENVINO_ASSERT(qs.rank().is_static() && qs[3].is_static(),
                        "ConvertSparseAttentionToDense: ",
                        node->get_friendly_name(),
                        " needs a static head size");
        scale = b.scalar_like(1.0f / std::sqrt(static_cast<float>(qs[3].get_length())), query_type);
    }
    auto sdpa = b.make_node<op::v13::ScaledDotProductAttention>(query, key_rep, value_rep, mask, scale, false);
    sdpa->set_friendly_name(node->get_friendly_name() + "/npuw_dense_sdpa");

    // a query without attended keys (a padding query) produces zeros
    const auto has_keys = b.make<op::v1::ReduceLogicalOr>(mask, b.i64(-1), true);
    return b.select(has_keys, sdpa->output(0), b.scalar_like(0.0f, query_type));
}

struct QSAState {
    std::string kind;   // block_index_key | incomplete_index_key | incomplete_index_rope
    std::string index;  // layer index
};

// StatefulToStateless keeps the QSA variables unrecognized, as
//     {input|output}_restored.cache_params.past.<kind>.N cache_params.present.<kind>.N
std::optional<QSAState> qsa_state(const ov::descriptor::Tensor& tensor, const std::string& io) {
    static const std::regex pattern("^(input|output)_restored\\.cache_params\\.past\\.(" +
                                    util::qsa_state_kinds_alternation() +
                                    ")\\.(\\d+)cache_params\\.present\\.\\2\\.\\3$");
    for (const auto& name : tensor.get_names()) {
        std::smatch match;
        if (std::regex_match(name, match, pattern) && match[1].str() == io) {
            return QSAState{match[2].str(), match[3].str()};
        }
    }
    return std::nullopt;
}

void replace(const std::shared_ptr<ov::Node>& node, const ov::OutputVector& outputs, const Builder& b) {
    OPENVINO_ASSERT(outputs.size() == node->get_output_size());
    ov::copy_runtime_info(node, b.nodes());
    const auto& name = node->get_friendly_name();
    for (size_t i = 0; i < outputs.size(); ++i) {
        OPENVINO_ASSERT(outputs[i].get_element_type() == node->get_output_element_type(i),
                        "ConvertSparseAttentionToDense: ",
                        name,
                        " output ",
                        i,
                        " type mismatch");
        // the new producer of every output takes the name of the replaced operation, suffixed for a multi-output one
        const auto producer = outputs[i].get_node_shared_ptr();
        if (std::find(b.nodes().begin(), b.nodes().end(), producer) != b.nodes().end()) {
            producer->set_friendly_name(outputs.size() == 1 ? name : name + "." + std::to_string(i));
        }
        node->output(i).replace(outputs[i]);
    }
}

}  // namespace

bool ov::npuw::ConvertSparseAttentionToDense::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::Node>> qsa_ops;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type_any_of<op::v17::QSAPrepare, op::v17::SparseAttentionIndexer, op::v17::SparseAttention>(node)) {
            qsa_ops.push_back(node);
        }
    }
    if (qsa_ops.empty()) {
        return false;
    }

    std::shared_ptr<op::v0::Parameter> attention_mask;
    std::vector<std::pair<QSAState, std::shared_ptr<op::v0::Parameter>>> past;
    std::vector<std::pair<QSAState, std::shared_ptr<op::v0::Result>>> present;
    for (const auto& param : model->get_parameters()) {
        if (auto state = qsa_state(param->get_output_tensor(0), "input")) {
            past.emplace_back(std::move(*state), param);
        } else if (param->get_output_tensor(0).get_names().count("attention_mask")) {
            attention_mask = param;
        }
    }
    for (const auto& result : model->get_results()) {
        if (auto state = qsa_state(result->get_output_tensor(0), "output")) {
            present.emplace_back(std::move(*state), result);
        }
    }
    OPENVINO_ASSERT(attention_mask && attention_mask->get_partial_shape().rank().is_static() &&
                        attention_mask->get_partial_shape().size() == 2,
                    "ConvertSparseAttentionToDense: a 2D attention_mask input is required");
    OPENVINO_ASSERT(past.size() == present.size(),
                    "ConvertSparseAttentionToDense: ",
                    past.size(),
                    " past QSA states but ",
                    present.size(),
                    " present ones");

    // 1. the static number of key blocks sizes the block_index_key state
    int64_t ratio = 0;
    for (const auto& node : qsa_ops) {
        if (const auto prepare = ov::as_type_ptr<op::v17::QSAPrepare>(node)) {
            OPENVINO_ASSERT(ratio == 0 || ratio == prepare->get_key_block_size(),
                            "ConvertSparseAttentionToDense: the QSA layers disagree on the key block size");
            ratio = prepare->get_key_block_size();
        }
    }
    OPENVINO_ASSERT(ratio > 0, "ConvertSparseAttentionToDense: no QSAPrepare operation");
    OPENVINO_ASSERT(static_cast<int64_t>(m_max_kv_len) % ratio == 0,
                    "ConvertSparseAttentionToDense: the KV cache length ",
                    m_max_kv_len,
                    " must be a multiple of the QSA key block size ",
                    ratio,
                    "; a non-zero remainder would leave the tail tokens without a key block");
    const int64_t num_blocks = static_cast<int64_t>(m_max_kv_len) / ratio;
    OPENVINO_ASSERT(num_blocks > 0, "ConvertSparseAttentionToDense: the KV cache is shorter than one key block");
    for (const auto& [state, param] : past) {
        if (state.kind == kBlockIndexKey) {
            auto shape = param->get_partial_shape();
            OPENVINO_ASSERT(shape.rank().is_static() && shape.size() == 3,
                            "ConvertSparseAttentionToDense: block_index_key.",
                            state.index,
                            " is expected to be rank-3");
            shape[1] = num_blocks;
            param->set_partial_shape(shape);
        }
    }
    model->validate_nodes_and_infer_types();

    // 2. the operations, in the topological order: the producers are replaced before their consumers
    Builder shared;
    const auto mask_info = build_mask_info(shared, attention_mask);
    ov::copy_runtime_info(attention_mask, shared.nodes());
    PaddedKeys padded_keys;
    SelectedBlocks selected_blocks;
    size_t num_prepare = 0, num_indexer = 0, num_attention = 0;
    for (const auto& node : qsa_ops) {
        Builder b;
        if (const auto prepare = ov::as_type_ptr<op::v17::QSAPrepare>(node)) {
            replace(node, decompose_qsa_prepare(b, prepare, mask_info, num_blocks, padded_keys), b);
            ++num_prepare;
        } else if (const auto indexer = ov::as_type_ptr<op::v17::SparseAttentionIndexer>(node)) {
            replace(node, decompose_indexer(b, indexer, mask_info, padded_keys, selected_blocks), b);
            ++num_indexer;
        } else if (const auto attention = ov::as_type_ptr<op::v17::SparseAttention>(node)) {
            replace(node, {decompose_sparse_attention(b, attention, mask_info, selected_blocks)}, b);
            ++num_attention;
        }
    }

    // 3. lincache-like names of the QSA states
    for (const auto& [state, param] : past) {
        ov::op::util::set_name(*param, "cache_params.past." + state.kind + "." + state.index);
    }
    for (const auto& [state, result] : present) {
        ov::op::util::set_name(*result, "cache_params.present." + state.kind + "." + state.index);
    }
    model->validate_nodes_and_infer_types();

    for (const auto& [state, result] : present) {
        if (state.kind == kBlockIndexKey) {
            const auto& shape = result->get_input_partial_shape(0);
            OPENVINO_ASSERT(shape.rank().is_static() && shape.size() == 3 && shape[1] == num_blocks,
                            "ConvertSparseAttentionToDense: present block_index_key.",
                            state.index,
                            " got shape ",
                            shape,
                            ", expected ",
                            num_blocks,
                            " blocks");
        }
    }

    LOG_INFO("ConvertSparseAttentionToDense: " << num_prepare << " QSAPrepare, " << num_indexer
                                               << " SparseAttentionIndexer, " << num_attention
                                               << " SparseAttention -> dense SDPA, " << past.size() << " QSA states, "
                                               << num_blocks << " key blocks of " << ratio << " tokens");
    return true;
}
