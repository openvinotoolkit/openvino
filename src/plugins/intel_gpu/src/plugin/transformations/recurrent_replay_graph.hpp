// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "openvino/opsets/opset13.hpp"
#include "openvino/core/model.hpp"
#include <limits>
#include <cmath>

namespace ov::recurrent_replay {
namespace ops = ov::opset13;
using Value = ov::Output<ov::Node>;

inline Value scalar(int64_t n) { return ops::Constant::create(element::i64, {}, {n}); }
inline Value vec(std::vector<int64_t> v) { return ops::Constant::create(element::i64, {v.size()}, v); }
inline Value fp(float x) { return ops::Constant::create(element::f32, {}, {x}); }
inline Value f32(Value x) {
    return x.get_element_type() == element::f32 ? x : Value(std::make_shared<ops::Convert>(x, element::f32));
}
inline Value gather(Value x, Value i, int axis, int batch = 0) {
    return std::make_shared<ops::Gather>(x, i, scalar(axis), batch);
}
inline Value unsqueeze(Value x, int axis) { return std::make_shared<ops::Unsqueeze>(x, scalar(axis)); }
inline Value squeeze(Value x, int axis) { return std::make_shared<ops::Squeeze>(x, scalar(axis)); }
inline Value reshape(Value x, std::vector<int64_t> dims) { return std::make_shared<ops::Reshape>(x, vec(dims), true); }
inline Value dim(Value x, int axis) { return gather(std::make_shared<ops::ShapeOf>(x, element::i64), scalar(axis), 0); }
inline Value cat(OutputVector xs, int axis) { return std::make_shared<ops::Concat>(xs, axis); }
inline Value slice(Value x, Value begin, Value end, int axis) {
    return std::make_shared<ops::Slice>(x, unsqueeze(begin, 0), unsqueeze(end, 0), vec({1}), vec({axis}));
}
inline Value slice_end() { return scalar(std::numeric_limits<int64_t>::max()); }
inline Value add(Value x, Value y) { return std::make_shared<ops::Add>(x, y); }
inline Value mul(Value x, Value y) { return std::make_shared<ops::Multiply>(x, y); }
inline Value swap_time_channels(Value x) { return std::make_shared<ops::Transpose>(x, vec({0, 2, 1})); }
inline Value linear(Value count) { return std::make_shared<ops::Range>(scalar(0), count, scalar(1), element::i64); }

struct RowSelection {
    Value sources, destinations, extra_src, extra_dst, resets, counts, keep;
    RowSelection(Value rows, Value control, Value reset, Value batch) {
        rows = reshape(rows, {-1, 2});
        auto src = gather(rows, scalar(1), 1);
        auto dst = gather(rows, scalar(0), 1);
        sources = slice(src, scalar(0), batch, 0);
        destinations = slice(dst, scalar(0), batch, 0);
        extra_src = slice(src, batch, slice_end(), 0);
        extra_dst = slice(dst, batch, slice_end(), 0);
        resets = reshape(reset, {-1});
        auto flags = reshape(control, {-1});
        counts = slice(flags, scalar(0), batch, 0);
        keep = std::make_shared<ops::NotEqual>(gather(flags, scalar(-1), 0), scalar(0));
    }
};

// A bank is logically [slots, features]. Gather original rows before any stores;
// active stores only replace their written features in the destination row.
struct Bank {
    Value cleared, sources, destinations, extra_values;
    const RowSelection& rows;
    Bank(Value input, const RowSelection& r) : rows(r) {
        auto flat = squeeze(squeeze(input, 0), 0);
        auto reset_shape = cat({unsqueeze(dim(r.resets, 0), 0), unsqueeze(dim(flat, 1), 0)}, 0);
        auto zeros = std::make_shared<ops::Broadcast>(fp(0), reset_shape);
        cleared = std::make_shared<ops::ScatterNDUpdate>(flat, unsqueeze(r.resets, 1), zeros);
        sources = gather(cleared, r.sources, 0);
        destinations = gather(cleared, r.destinations, 0);
        extra_values = gather(cleared, r.extra_src, 0);
    }
    Value write(Value active) const {
        // One destination per row-map entry. GGML generates these in slot order,
        // including the inactive rows that need a copy.
        auto indices = unsqueeze(cat({rows.destinations, rows.extra_dst}, 0), 1);
        auto values = cat({reshape(active, {0, -1}), extra_values}, 0);
        auto result = std::make_shared<ops::ScatterNDUpdate>(cleared, indices, values);
        return unsqueeze(unsqueeze(result, 0), 0);
    }
};
inline Value store_prefix(Value original, Value prefix) {
    return cat({prefix, slice(original, dim(prefix, 1), slice_end(), 1)}, 1);
}

inline OutputVector conv_body(const OutputVector& a) {
    auto x = squeeze(a[2], 0);
    const auto channels = a[3].get_partial_shape()[2].get_length();
    const auto taps = a[3].get_partial_shape()[3].get_length();
    RowSelection rows(a[1], a[4], a[5], dim(x, 0));
    Bank bank(a[0], rows);
    auto history = reshape(bank.sources, {0, -1, channels});
    auto indices = add(unsqueeze(rows.counts, 1), unsqueeze(linear(scalar(taps - 1)), 0));
    auto window = gather(history, indices, 1, 1);
    auto columns = cat({window, f32(x)}, 1);
    auto weights = reshape(f32(a[3]), {channels, 1, 1, taps});
    auto convolved = std::make_shared<ops::GroupConvolution>(swap_time_channels(columns), weights, Strides{1},
                                                             CoordinateDiff{0}, CoordinateDiff{0}, Strides{1});
    Value result = unsqueeze(swap_time_channels(convolved), 0);
    if (a[2].get_element_type() != element::f32)
        result = std::make_shared<ops::Convert>(result, a[2].get_element_type());
    // Keep the history and pending tokens, or commit only the last history window.
    auto begin = std::make_shared<ops::Select>(rows.keep, scalar(0), dim(x, 1));
    auto update = slice(columns, begin, slice_end(), 1);
    return {result, bank.write(store_prefix(reshape(bank.destinations, {0, -1, channels}), update))};
}

inline Value expand_heads(Value x, int64_t heads, int64_t expanded) {
    if (heads == expanded)
        return x;
    std::vector<int64_t> indices(expanded);
    for (int64_t h = 0; h < expanded; ++h)
        indices[h] = h % heads;
    return gather(x, vec(indices), 2);
}
inline Value dot(Value state, Value key) {
    return std::make_shared<ops::ReduceSum>(mul(state, unsqueeze(key, 2)), scalar(3), false);
}
inline Value step(Value state, Value key, Value value, Value gate, Value beta) {
    auto decayed = mul(state, unsqueeze(std::make_shared<ops::Exp>(gate), 2));
    auto delta = mul(beta, std::make_shared<ops::Subtract>(value, dot(decayed, key)));
    return add(decayed, mul(unsqueeze(delta, 3), unsqueeze(key, 2)));
}
inline OutputVector current_tokens(Value state, Value q, Value k, Value v, Value g, Value beta, float scale) {
    auto idx = std::make_shared<ops::Parameter>(element::i64, PartialShape{});
    auto carried = std::make_shared<ops::Parameter>(element::f32, state.get_partial_shape());
    ParameterVector params{idx, carried};
    OutputVector inputs{q, k, v, g, beta}, token;
    for (const auto& input : inputs) {
        auto p = std::make_shared<ops::Parameter>(input.get_element_type(), input.get_partial_shape());
        params.push_back(p);
        token.push_back(gather(p, idx, 1));
    }
    auto next = step(carried, token[1], token[2], token[3], token[4]);
    auto attention = unsqueeze(mul(dot(next, token[0]), fp(scale)), 1);
    auto yes = ops::Constant::create(element::boolean, {}, {true});
    auto body = std::make_shared<Model>(OutputVector{yes, next, attention}, params);
    auto loop = std::make_shared<ops::Loop>(dim(v, 1), yes);
    loop->set_function(body);
    loop->set_special_body_ports({0, 0});
    loop->set_merged_input(carried, state, next);
    for (size_t i = 0; i < inputs.size(); ++i)
        loop->set_invariant_input(params[i + 2], inputs[i]);
    return {loop->get_concatenated_slices(attention, 0, 1, 1, -1, 1), loop->get_iter_value(next, -1)};
}

inline OutputVector gdn_body(const OutputVector& a) {
    const auto qshape = a[0].get_partial_shape(), kshape = a[1].get_partial_shape(), vshape = a[2].get_partial_shape();
    const int64_t Hq = qshape[2].get_length(), Hk = kshape[2].get_length(), Hv = vshape[2].get_length();
    const int64_t K = kshape[3].get_length(), V = vshape[3].get_length(), G = a[4].get_partial_shape()[3].get_length();
    const int64_t P = a[6].get_partial_shape()[3].get_length() / (Hk * K);
    RowSelection rows(a[9], a[10], a[11], dim(a[2], 0));
    Bank states(a[3], rows), keys(a[6], rows), values(a[7], rows), gates(a[8], rows);
    Value state = reshape(states.sources, {0, Hv, V, K});
    auto pk = expand_heads(reshape(keys.sources, {0, P, Hk, K}), Hk, Hv);
    auto pv = reshape(values.sources, {0, P, Hv, V});
    auto pgb = reshape(gates.sources, {0, 2 * P, Hv, G});
    for (int64_t t = 0; t < P; ++t) {
        auto time = scalar(t);
        auto b = slice(gather(pgb, scalar(P + t), 1), scalar(0), scalar(1), 2);
        auto next = step(state, gather(pk, time, 1), gather(pv, time, 1), gather(pgb, time, 1), b);
        auto active = reshape(std::make_shared<ops::Greater>(rows.counts, time), {-1, 1, 1, 1});
        state = std::make_shared<ops::Select>(active, next, state);
    }
    auto q = expand_heads(f32(a[0]), Hq, Hv), k = expand_heads(f32(a[1]), Hk, Hv);
    auto computed = current_tokens(state, q, k, f32(a[2]), f32(a[4]), f32(a[5]), 1.0f / std::sqrt(float(K)));
    auto save = [&](Value x) { return slice(f32(x), scalar(0), scalar(P), 1); };
    auto saved_g = save(a[4]), saved_b = save(a[5]);
    auto gb = reshape(gates.destinations, {0, 2 * P, Hv, G});
    auto gb_gate = slice(gb, scalar(0), scalar(P), 1), gb_beta = slice(gb, scalar(P), slice_end(), 1);
    if (G > 1) {
        auto old_prefix = slice(gb_beta, scalar(0), dim(saved_b, 1), 1);
        saved_b = cat({saved_b, slice(old_prefix, scalar(1), slice_end(), 3)}, 3);
    }
    Value output = computed[0];
    if (a[0].get_element_type() != element::f32)
        output = std::make_shared<ops::Convert>(output, a[0].get_element_type());
    auto committed = std::make_shared<ops::Select>(rows.keep, state, computed[1]);
    return {output, states.write(committed),
            keys.write(store_prefix(reshape(keys.destinations, {0, P, Hk, K}), save(a[1]))),
            values.write(store_prefix(reshape(values.destinations, {0, P, Hv, V}), save(a[2]))),
            gates.write(cat({store_prefix(gb_gate, saved_g), store_prefix(gb_beta, saved_b)}, 1))};
}

inline OutputVector build(const OutputVector& args, bool conv) {
    return conv ? conv_body(args) : gdn_body(args);
}
} // namespace ov::recurrent_replay
