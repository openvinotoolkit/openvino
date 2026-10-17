// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

namespace {
// Put the reshape target's single -1 on the axis carrying the runtime token count, recovered by
// element-count conservation: product(tgt) / product(res_ps's static dims). No-op for a static
// res_ps, where the target is already exact and a -1 could land on the wrong axis.
void place_dynamic_token_axis(std::vector<int64_t>& tgt, const ov::PartialShape& res_ps) {
    if (std::find(tgt.begin(), tgt.end(), -1) != tgt.end()) {
        // The decoder can identify the runtime axis exactly from ggml strides/permutations. Keep
        // that stronger contract, especially for zero-length captures where element-count
        // inference below is intentionally impossible.
        return;
    }
    if (res_ps.rank().is_dynamic()) {
        return;
    }
    int64_t slice_static = 1;
    bool res_has_dyn = false;
    for (int64_t i = 0; i < res_ps.rank().get_length(); ++i) {
        if (res_ps[i].is_static()) {
            slice_static *= res_ps[i].get_length();
        } else {
            res_has_dyn = true;
        }
    }
    if (!res_has_dyn || slice_static == 0) {
        return;
    }
    int64_t out_prod = 1;
    for (auto d : tgt) {
        out_prod *= d;
    }
    if (out_prod % slice_static != 0) {
        return;
    }
    const int64_t token_len = out_prod / slice_static;
    for (auto& d : tgt) {
        if (d == token_len) {
            d = -1;
            break;
        }
    }
}

// A dense source viewed through non-dense strides: merged-projection slices ([hd, nh, T] out of
// [n_qkv, T]) and multi-sequence caches ([hd, nh, n_kv, n_seq] out of a [F, ctx * n_seq] cache).
// The flat source is regrouped into a mixed-radix grid whose radices are the ratios between
// consecutive view strides, so every view dimension becomes one grid axis that a plain Slice selects
// from. Returns an empty vector when the layout cannot be expressed that way.
OutputVector materialize_grid_view(const ov::Output<Node>& input,
                                   const ov::Shape& dst_shape,
                                   const std::vector<int64_t>& dst_byte_strides,
                                   int64_t element_size,
                                   int64_t offset_elements,
                                   int dynamic_axis) {
    const size_t rank = dst_shape.size();
    if (dst_byte_strides.size() != rank || element_size <= 0 || offset_elements < 0) {
        return {};
    }
    // Chain of view dimensions that actually select something, in ggml order (dim 0 innermost).
    std::vector<size_t> chain;  // ggml dimension indices, ordered by strictly descending stride
    std::vector<int64_t> stride(rank, 0);
    for (size_t g = 0; g < rank; ++g) {
        if (dst_byte_strides[g] <= 0 || dst_byte_strides[g] % element_size != 0) {
            return {};
        }
        stride[g] = dst_byte_strides[g] / element_size;
        // The dynamic axis selects along its stride even when it was captured with a single element.
        if (dst_shape[rank - 1 - g] > 1 || static_cast<int>(rank - 1 - g) == dynamic_axis) {
            chain.insert(chain.begin(), g);
        }
    }
    if (chain.empty()) {
        return {};
    }
    // Strides must grow with the ggml dimension index, otherwise the view permutes its source.
    for (size_t i = 1; i < chain.size(); ++i) {
        if (stride[chain[i - 1]] <= stride[chain[i]]) {
            return {};
        }
    }

    std::vector<int64_t> radix;     // grid axis sizes; the outermost stays -1 (inferred)
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    int64_t remainder = offset_elements;
    for (size_t i = 0; i < chain.size(); ++i) {
        const int64_t t = stride[chain[i]];
        const int64_t extent = static_cast<int64_t>(dst_shape[rank - 1 - chain[i]]);
        const bool dynamic = static_cast<int>(rank - 1 - chain[i]) == dynamic_axis;
        int64_t size = -1;
        if (i > 0) {
            const int64_t outer = stride[chain[i - 1]];
            if (outer % t != 0) {
                return {};
            }
            size = outer / t;
        }
        const int64_t coordinate = remainder / t;
        remainder %= t;
        if (dynamic) {
            // The extent is only known at run time. A leading offset cannot be expressed, and an inner
            // axis keeps its whole grid extent (the consumer trims it, e.g. FLASH_ATTN_EXT to the mask).
            if (coordinate != 0) {
                return {};
            }
            begin.push_back(0);
            end.push_back(std::numeric_limits<int64_t>::max());
        } else {
            if (size > 0 && coordinate + extent > size) {
                return {};
            }
            begin.push_back(coordinate);
            end.push_back(coordinate + extent);
        }
        radix.push_back(size);
    }
    // Innermost remainder: elements below the smallest selecting stride.
    const int64_t tail = stride[chain.back()];
    if (tail > 1) {
        radix.push_back(tail);
        begin.push_back(remainder);
        end.push_back(remainder + 1);
    } else if (remainder != 0) {
        return {};
    }

    std::vector<int64_t> axes(radix.size());
    std::iota(axes.begin(), axes.end(), 0);
    auto grid = std::make_shared<ov::op::v1::Reshape>(
        input, ov::op::v0::Constant::create(ov::element::i64, {radix.size()}, radix), false);
    ov::Output<Node> selected = std::make_shared<ov::op::v8::Slice>(
        grid,
        ov::op::v0::Constant::create(ov::element::i64, {begin.size()}, begin),
        ov::op::v0::Constant::create(ov::element::i64, {end.size()}, end),
        ov::op::v0::Constant::create(ov::element::i64, {radix.size()}, std::vector<int64_t>(radix.size(), 1)),
        ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes));

    std::vector<int64_t> target(dst_shape.begin(), dst_shape.end());
    if (dynamic_axis >= 0) {
        target[dynamic_axis] = -1;
    }
    return {std::make_shared<ov::op::v1::Reshape>(
        selected, ov::op::v0::Constant::create(ov::element::i64, {target.size()}, target), false)};
}

// Re-expresses a view of a (possibly non-dense) source in the element space of the source's dense
// logical tensor, which is what the translated operand holds. Shapes are OV order, strides ggml order
// (bytes). Returns {logical stride of every ggml dim..., logical offset, element size}, or an empty
// vector when some view element does not map to a source element through one affine map.
std::vector<int64_t> to_logical_view(const ov::Shape& src_shape,
                                     const std::vector<int64_t>& src_strides,
                                     const ov::Shape& dst_shape,
                                     const std::vector<int64_t>& dst_strides,
                                     int64_t byte_offset,
                                     int dynamic_axis) {
    const size_t src_rank = src_shape.size();
    const size_t dst_rank = dst_shape.size();
    std::vector<int64_t> extent(src_rank), logical(src_rank);
    int64_t element_size = 0;
    int64_t total = 1;
    for (size_t g = 0; g < src_rank; ++g) {
        extent[g] = static_cast<int64_t>(src_shape[src_rank - 1 - g]);
        logical[g] = total;
        total *= extent[g];
        if (extent[g] > 1) {
            if (src_strides[g] <= 0) {
                return {};
            }
            element_size = element_size == 0 ? src_strides[g] : std::min(element_size, src_strides[g]);
        }
    }
    if (total == 0) {
        return {};
    }
    if (element_size == 0) {
        element_size = src_strides.empty() ? 0 : src_strides[0];
    }
    if (element_size <= 0) {
        return {};
    }
    // Source dims that select something, by descending byte stride.
    std::vector<size_t> order;
    for (size_t g = 0; g < src_rank; ++g) {
        if (extent[g] > 1) {
            order.push_back(g);
        }
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return src_strides[a] > src_strides[b];
    });
    const auto to_logical = [&](int64_t bytes, int64_t& out) {
        out = 0;
        for (size_t k = 0; k < order.size(); ++k) {
            const size_t g = order[k];
            const int64_t c = bytes / src_strides[g];
            if (k > 0 && c >= extent[g]) {
                return false;
            }
            bytes -= c * src_strides[g];
            out += c * logical[g];
        }
        return bytes == 0;
    };
    const auto source_address = [&](int64_t index) {
        int64_t address = 0;
        for (size_t g = 0; g < src_rank; ++g) {
            address += (index / logical[g]) % extent[g] * src_strides[g];
        }
        return address;
    };

    std::vector<int64_t> result(dst_rank + 2, 0);
    int64_t count = 1;
    int64_t last = 0;
    for (size_t g = 0; g < dst_rank; ++g) {
        const auto n = static_cast<int64_t>(dst_shape[dst_rank - 1 - g]);
        count *= n;
        const bool moves = n > 1 || static_cast<int>(dst_rank - 1 - g) == dynamic_axis;
        if (moves && (dst_strides[g] <= 0 || !to_logical(dst_strides[g], result[g]))) {
            return {};
        }
        if (!moves) {
            // A unit axis never moves; give it a stride that keeps the grid ordering valid.
            result[g] = g == 0 ? 1 : result[g - 1] * std::max<int64_t>(1, dst_shape[dst_rank - g]);
        }
        last += (n - 1) * result[g];
    }
    int64_t offset = 0;
    if (!to_logical(byte_offset, offset) || count == 0 || offset + last >= total) {
        return {};
    }
    constexpr int64_t max_checked = int64_t{1} << 24;
    if (count > max_checked) {
        return {};
    }
    // Verify every element: the affine logical map must land on the same byte the view addresses.
    std::vector<int64_t> idx(dst_rank, 0);
    for (int64_t e = 0; e < count; ++e) {
        int64_t index = offset, address = byte_offset;
        for (size_t g = 0; g < dst_rank; ++g) {
            index += idx[g] * result[g];
            address += idx[g] * dst_strides[g];
        }
        if (source_address(index) != address) {
            return {};
        }
        for (size_t g = 0; g < dst_rank; ++g) {
            if (++idx[g] < static_cast<int64_t>(dst_shape[dst_rank - 1 - g])) {
                break;
            }
            idx[g] = 0;
        }
    }
    result[dst_rank] = offset;
    result[dst_rank + 1] = element_size;
    return result;
}

// A KV-cache read whose live length and first stream are only known at run time. The cache holds
// `streams` blocks of rows of `row_width` features (transposed: rows of cells per feature); the view
// takes `active_streams` blocks from a runtime start, `length` rows from `row_base`, splits the first
// heads * head_size features into heads, and arranges (stream, cell, head, dim) in `order`.
//   descriptor: {streams, row_width, heads, head_size, active_streams, row_base, transposed,
//                order[0..3], static_length}
//   inputs:     cache[, length[, first_stream]]
OutputVector translate_kv_view(const NodeContext& context, const std::vector<int64_t>& kv) {
    FRONT_END_OP_CONVERSION_CHECK(kv.size() == 12, "kv_view requires 12 values");
    const int64_t streams = kv[0], row_width = kv[1], heads = kv[2], head_size = kv[3];
    const int64_t active_streams = kv[4], row_base = kv[5];
    const bool transposed = kv[6] != 0;
    const std::vector<int64_t> order{kv[7], kv[8], kv[9], kv[10]};
    FRONT_END_OP_CONVERSION_CHECK(streams > 0 && row_width > 0 && heads > 0 && head_size > 0 && active_streams > 0 &&
                                      heads * head_size <= row_width && row_base >= 0,
                                  "Invalid kv_view descriptor");
    const auto constant = [](const std::vector<int64_t>& values) {
        return ov::op::v0::Constant::create(ov::element::i64, {values.size()}, values)->output(0);
    };
    const auto as_i64 = [](ov::Output<Node> value) {
        if (value.get_element_type() != ov::element::i64) {
            value = std::make_shared<ov::op::v0::Convert>(value, ov::element::i64);
        }
        return value;
    };
    const auto length = context.get_input_size() >= 2 ? as_i64(context.get_input(1)) : constant({kv[11]});
    const auto first = context.get_input_size() >= 3 ? as_i64(context.get_input(2)) : constant({0});

    // A cache in the [1, ctx, n_heads_kv, head_size] layout read from its first rows as whole heads: take the live
    // tokens with VariadicSplit, the read side of the GPU plugin's StatelessKVFusion (see LowerSetRowsStateless). The
    // fused StatelessKV hands attention a view of those tokens without copying the cache.
    if (is_kvcache_4d_shape(context.get_input(0).get_partial_shape()) && !transposed && streams == 1 &&
        active_streams == 1 && row_base == 0 && context.get_input_size() < 3) {
        const auto cache_shape = context.get_input(0).get_partial_shape();
        if (cache_shape[2].get_length() == heads && cache_shape[3].get_length() == head_size) {
            auto lengths = std::make_shared<ov::op::v0::Concat>(OutputVector{length, constant({-1})}, 0);
            auto tokens = std::make_shared<ov::op::v1::VariadicSplit>(
                context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {}, {1}), lengths);
            ov::Output<Node> y = tokens->output(0);
            if (order != std::vector<int64_t>{0, 1, 2, 3}) {
                y = std::make_shared<ov::op::v1::Transpose>(y, constant(order));
            }
            return rename_outputs_with_suffix({y}, context.get_name());
        }
    }

    // [streams, cells, features] or, transposed, [streams, features, cells]. Inferring the cell count
    // also accepts a cache bound with only its live prefix.
    const int64_t cell_axis = transposed ? 2 : 1;
    const int64_t feature_axis = transposed ? 1 : 2;
    ov::Output<Node> x = std::make_shared<ov::op::v1::Reshape>(
        context.get_input(0),
        constant(transposed ? std::vector<int64_t>{streams, row_width, -1} : std::vector<int64_t>{streams, -1, row_width}),
        false);
    auto begin = std::make_shared<ov::op::v0::Concat>(OutputVector{first, constant({row_base})}, 0);
    auto end = std::make_shared<ov::op::v0::Concat>(
        OutputVector{std::make_shared<ov::op::v1::Add>(first, constant({active_streams})),
                     std::make_shared<ov::op::v1::Add>(constant({row_base}), length)},
        0);
    x = std::make_shared<ov::op::v8::Slice>(x, begin, end, constant({1, 1}), constant({0, cell_axis}));
    if (heads * head_size < row_width) {
        x = std::make_shared<ov::op::v8::Slice>(x,
                                                constant({0}),
                                                constant({heads * head_size}),
                                                constant({1}),
                                                constant({feature_axis}));
    }
    // Canonical [stream, cell, head, dim].
    if (transposed) {
        x = std::make_shared<ov::op::v1::Reshape>(x, constant({0, heads, head_size, -1}), true);
        x = std::make_shared<ov::op::v1::Transpose>(x, constant({0, 3, 1, 2}));
    } else {
        x = std::make_shared<ov::op::v1::Reshape>(x, constant({0, -1, heads, head_size}), true);
    }
    if (order != std::vector<int64_t>{0, 1, 2, 3}) {
        x = std::make_shared<ov::op::v1::Transpose>(x, constant(order));
    }
    return rename_outputs_with_suffix({x}, context.get_name());
}

}  // namespace

// GGML_OP_VIEW. Both decoders describe a view with the same generic facts: a KV-cache descriptor
// ("kv_view"), a slice/reshape description ("view_slice", "view_reshape"), or
// the ggml byte strides and offset ("view_input_strides", "view_output_strides", "view_offset_bytes",
// "output_dynamic_axis"), which are materialized without knowing who produced them.
OutputVector translate_view(const NodeContext& context) {
    // A KV-cache read described by the decoder (runtime length and first stream as extra inputs).
    const auto kv_view = context.get_attribute<std::vector<int64_t>>("kv_view", {});
    if (!kv_view.empty()) {
        num_inputs_check(context, 1, 3);
        return translate_kv_view(context, kv_view);
    }
    num_inputs_check(context, 1, 1);

    const auto& output_shape = context.get_output_shape();
    const auto& input_shape = context.get_input_shape(0);
    if ((output_shape.is_static() && ov::shape_size(output_shape.to_shape()) == 0) ||
        (input_shape.is_static() && ov::shape_size(input_shape.to_shape()) == 0)) {
        // Empty recurrent-cache reorder views have no values and their CPY
        // consumer is routed directly to the cache base tensor.
        return {context.get_input(0)};
    }

    // A view described by the decoder rather than by raw strides:
    //   "view_slice"     {ov_axis, start, length}: a negative start is an end-anchored tail running to the
    //                    axis end (a conv window's last d_conv-1 columns); a negative length ends that many
    //                    elements before the axis end, which keeps a head slice of a dynamic axis dynamic
    //                    (attention rows stacked ahead of a recurrent state);
    //   "view_reshape"   the view's own layout in OpenVINO order, -1 on the runtime (token) axis.
    const auto view_slice = context.get_attribute<std::vector<int64_t>>("view_slice", {});
    const auto view_reshape = context.get_attribute<std::vector<int64_t>>("view_reshape", {});
    if (!view_slice.empty() || !view_reshape.empty()) {
        auto input = context.get_input(0);
        const auto input_ov_shape = input.get_partial_shape();
        const auto input_ggml_shape = context.get_attribute<ov::Shape>("input_ggml_shape", {});
        // An operand that lost leading axes is lifted back to the described base shape before slicing; an
        // operand with at least the base's rank is already the logical tensor and is used as is.
        if (!input_ggml_shape.empty() && input_ov_shape.rank().is_static() &&
            static_cast<size_t>(input_ov_shape.rank().get_length()) < input_ggml_shape.size()) {
            input = std::make_shared<ov::op::v1::Reshape>(
                input,
                ov::op::v0::Constant::create(ov::element::i64, {input_ggml_shape.size()}, input_ggml_shape),
                false);
        }

        ov::Output<ov::Node> result = input;
        bool whole_concat_input = false;
        if (!view_slice.empty()) {
            FRONT_END_OP_CONVERSION_CHECK(view_slice.size() == 3, "view_slice requires {axis, start, length}");
            const int64_t axis = view_slice[0], start = view_slice[1], len = view_slice[2];
            // Selecting a complete Concat input needs no packing or copy. This also keeps SSM token
            // outputs independent of the final state when converting to paged execution.
            const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(input.get_node_shared_ptr());
            if (concat && concat->get_input_size() == 2 && concat->get_axis() == axis && axis >= 0 &&
                concat->get_input_partial_shape(1).rank().is_static()) {
                const auto& tail = concat->get_input_partial_shape(1)[axis];
                if (tail.is_static() && tail.get_length() > 0) {
                    const auto count = tail.get_length();
                    const bool prefix = start == 0 && len == -count;
                    const bool suffix = start == -count && len == count;
                    if (prefix || suffix) {
                        result = concat->input_value(suffix ? 1 : 0);
                        whole_concat_input = true;
                    }
                }
            }
            if (!whole_concat_input) {
                if (axis >= 0 && static_cast<size_t>(axis) < input_ggml_shape.size() &&
                    start >= static_cast<int64_t>(input_ggml_shape[axis])) {
                    // A one-past-end recurrent cache view represents an empty slot-reorder destination.
                    // Its CPY consumer is routed to the cache base, so no standalone slice is needed.
                    return {std::move(input)};
                }
                const int64_t end = len < 0 ? len : start < 0 ? std::numeric_limits<int64_t>::max() : start + len;
                result = std::make_shared<ov::op::v8::Slice>(input,
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {start}),
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {end}),
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {axis}));
            }
        }

        // ggml views rearrange or drop singleton dims (a select-expert slice yields [1, tok, 1, 2048] but the
        // view's layout is [1, 1, tok, 2048]); reshape to the view's own layout so the token axis lands where
        // its consumer expects it.
        auto tgt = view_reshape;
        if (!tgt.empty()) {
            place_dynamic_token_axis(tgt, result.get_partial_shape());
            result = std::make_shared<ov::op::v1::Reshape>(result,
                                                           ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt),
                                                           false);
            result.get_node_shared_ptr()->set_friendly_name("view_reshape_" + context.get_name());
        }

        // Preserve producer names when the view reuses an existing value.
        if (result == context.get_input(0) || (whole_concat_input && tgt.empty())) {
            return {std::move(result)};
        }
        return rename_outputs_with_suffix({std::move(result)}, context.get_name());
    }

    // A view that preserves the source strides is a rectangular crop of
    // the logical source tensor; its byte offset identifies the first element. This covers the
    // ordinary multi-axis views used by backend validation without specializing their consumers.
    if (input_shape.is_static() && output_shape.is_static()) {
        const auto src_shape = input_shape.to_shape();
        const auto dst_shape = output_shape.to_shape();
        const auto src_strides = context.get_attribute<std::vector<int64_t>>("view_input_strides", {});
        const auto dst_strides = context.get_attribute<std::vector<int64_t>>("view_output_strides", {});
        const int64_t byte_offset = context.get_attribute<int64_t>("view_offset_bytes", 0);
        const int dynamic_axis = context.get_attribute<int>("output_dynamic_axis", -1);
        FRONT_END_OP_CONVERSION_CHECK(dynamic_axis >= -1 && dynamic_axis < static_cast<int>(dst_shape.size()),
                                      "Invalid VIEW output_dynamic_axis");

        // `dynamic` is an OV axis that must be laid out densely even if it was captured with one element.
        const auto is_dense = [](const ov::Shape& shape, const std::vector<int64_t>& strides, int dynamic = -1) {
            if (strides.size() != shape.size() || strides.empty() || strides[0] <= 0) {
                return false;
            }
            int64_t stride = strides[0];
            for (size_t i = 0; i < shape.size(); ++i) {
                const auto extent = static_cast<int64_t>(shape[shape.size() - 1 - i]);
                const bool dynamic_dim = static_cast<int>(shape.size() - 1 - i) == dynamic;
                if ((extent != 1 || dynamic_dim) && strides[i] != stride) {
                    return false;
                }
                stride *= extent;
            }
            return true;
        };

        // With a dynamic axis the dense path selects a flat [offset, end) range, which is only the view when
        // nothing but unit axes sits outside the operand's dynamic axis (a [1, 1, ?, F] cache). Otherwise the
        // selection lies behind the dynamic axis (gemma-4's per-layer input: layer l of [hd, T, n_layer]
        // starts at l * T * hd) and the strided paths below select it per axis.
        const auto flat_tail_selects_view = [&]() {
            const auto live = context.get_input(0).get_partial_shape();
            if (dynamic_axis < 0 || live.is_static()) {
                return true;
            }
            if (byte_offset != 0 || live.rank().is_dynamic()) {
                return false;
            }
            for (int64_t i = 0; i < live.rank().get_length(); ++i) {
                if (live[i].is_dynamic()) {
                    return true;
                }
                if (live[i] != 1) {
                    return false;
                }
            }
            return true;
        };
        if (is_dense(src_shape, src_strides) && is_dense(dst_shape, dst_strides, dynamic_axis) &&
            src_strides[0] == dst_strides[0] && byte_offset >= 0 && byte_offset % src_strides[0] == 0 &&
            flat_tail_selects_view()) {
            auto input = context.get_input(0);
            const auto live_shape = input.get_partial_shape();
            std::vector<int64_t> target(dst_shape.begin(), dst_shape.end());
            if (dynamic_axis >= 0) {
                target[dynamic_axis] = -1;
            }
            const int64_t offset = byte_offset / src_strides[0];
            const auto count = static_cast<int64_t>(ov::shape_size(dst_shape));
            FRONT_END_OP_CONVERSION_CHECK(offset + count <= static_cast<int64_t>(ov::shape_size(src_shape)),
                                          "VIEW exceeds its source buffer");
            if (byte_offset == 0 && src_shape == dst_shape && (live_shape.is_static() || dynamic_axis >= 0)) {
                return {input};
            }
            // A runtime-sized cache can already be expanded by its input binding or state update.
            if (byte_offset == 0 && dynamic_axis >= 0 && live_shape.compatible(output_shape) &&
                live_shape.rank().is_static() && live_shape[dynamic_axis].is_dynamic()) {
                return {input};
            }
            ov::Output<Node> selected = input;
            if (offset != 0 || ov::shape_size(src_shape) != ov::shape_size(dst_shape) ||
                (live_shape.is_dynamic() && dynamic_axis < 0)) {
                FRONT_END_OP_CONVERSION_CHECK(dynamic_axis < 0 || offset == 0,
                                              "An offset dynamic VIEW requires explicit runtime slice bounds");
                auto flat = std::make_shared<ov::op::v1::Reshape>(
                    input, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
                const int64_t end = dynamic_axis < 0 ? offset + count : std::numeric_limits<int64_t>::max();
                selected = std::make_shared<ov::op::v8::Slice>(
                    flat,
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {offset}),
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {end}),
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            }
            // Without a dynamic axis the target is fully static, so a runtime-sized operand (e.g. the
            // s_copy index list bound as [1, 1, 1, ?]) still reshapes unambiguously to the view extent.
            auto result = std::make_shared<ov::op::v1::Reshape>(
                selected, ov::op::v0::Constant::create(ov::element::i64, {target.size()}, target), false);
            return rename_outputs_with_suffix({std::move(result)}, context.get_name());
        }

        if (src_shape.size() == dst_shape.size() && src_shape.size() == src_strides.size() &&
            src_strides == dst_strides && byte_offset >= 0) {
            std::vector<int64_t> starts(src_shape.size(), 0);
            std::vector<size_t> dims(src_shape.size());
            std::iota(dims.begin(), dims.end(), 0);
            std::sort(dims.begin(), dims.end(), [&](size_t a, size_t b) {
                return src_strides[a] > src_strides[b];
            });

            int64_t remaining = byte_offset;
            bool valid = true;
            for (size_t d : dims) {
                if (src_shape[src_shape.size() - 1 - d] == 1 && dst_shape[dst_shape.size() - 1 - d] == 1) {
                    continue;
                }
                if (src_strides[d] <= 0) {
                    valid = false;
                    break;
                }
                starts[d] = remaining / src_strides[d];
                remaining %= src_strides[d];
                if (starts[d] < 0 || starts[d] + static_cast<int64_t>(dst_shape[src_shape.size() - 1 - d]) >
                                         static_cast<int64_t>(src_shape[src_shape.size() - 1 - d])) {
                    valid = false;
                    break;
                }
            }
            valid = valid && remaining == 0;

            if (valid) {
                std::vector<int64_t> begin(src_shape.size());
                std::vector<int64_t> end(src_shape.size());
                std::vector<int64_t> axes(src_shape.size());
                std::vector<int64_t> step(src_shape.size(), 1);
                for (size_t ov_axis = 0; ov_axis < src_shape.size(); ++ov_axis) {
                    const size_t ggml_dim = src_shape.size() - 1 - ov_axis;
                    begin[ov_axis] = starts[ggml_dim];
                    end[ov_axis] = begin[ov_axis] + static_cast<int64_t>(dst_shape[ov_axis]);
                    if (static_cast<int>(ov_axis) == dynamic_axis) {
                        FRONT_END_OP_CONVERSION_CHECK(begin[ov_axis] == 0 && dst_shape[ov_axis] == src_shape[ov_axis],
                                                      "Dynamic VIEW crop requires explicit runtime slice bounds");
                        end[ov_axis] = std::numeric_limits<int64_t>::max();
                    }
                    axes[ov_axis] = static_cast<int64_t>(ov_axis);
                }
                auto result = std::make_shared<ov::op::v8::Slice>(
                    context.get_input(0),
                    ov::op::v0::Constant::create(ov::element::i64, {begin.size()}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {end.size()}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {step.size()}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes));
                return rename_outputs_with_suffix({std::move(result)}, context.get_name());
            }

        }

        if (src_shape.size() == dst_shape.size() && src_shape.size() == src_strides.size() &&
            src_strides.size() == dst_strides.size() && byte_offset >= 0) {
            std::vector<int64_t> starts(src_shape.size(), 0);
            std::vector<int64_t> steps(src_shape.size(), 1);
            std::vector<size_t> dims(src_shape.size());
            std::iota(dims.begin(), dims.end(), 0);
            std::sort(dims.begin(), dims.end(), [&](size_t a, size_t b) {
                return src_strides[a] > src_strides[b];
            });

            int64_t remaining = byte_offset;
            bool valid = true;
            for (size_t d : dims) {
                if (src_shape[src_shape.size() - 1 - d] == 1 && dst_shape[dst_shape.size() - 1 - d] == 1) {
                    continue;
                }
                if (src_strides[d] <= 0 || dst_strides[d] <= 0 || dst_strides[d] % src_strides[d] != 0) {
                    valid = false;
                    break;
                }
                steps[d] = dst_strides[d] / src_strides[d];
                starts[d] = remaining / src_strides[d];
                remaining %= src_strides[d];
                const size_t ov_axis = src_shape.size() - 1 - d;
                const int64_t last = starts[d] + steps[d] * (static_cast<int64_t>(dst_shape[ov_axis]) - 1);
                if (starts[d] < 0 || last >= static_cast<int64_t>(src_shape[ov_axis])) {
                    valid = false;
                    break;
                }
            }

            if (valid && remaining == 0) {
                std::vector<int64_t> begin(src_shape.size());
                std::vector<int64_t> end(src_shape.size());
                std::vector<int64_t> step(src_shape.size());
                std::vector<int64_t> axes(src_shape.size());
                for (size_t ov_axis = 0; ov_axis < src_shape.size(); ++ov_axis) {
                    const size_t ggml_dim = src_shape.size() - 1 - ov_axis;
                    begin[ov_axis] = starts[ggml_dim];
                    step[ov_axis] = steps[ggml_dim];
                    end[ov_axis] = begin[ov_axis] + step[ov_axis] * (static_cast<int64_t>(dst_shape[ov_axis]) - 1) + 1;
                    if (static_cast<int>(ov_axis) == dynamic_axis) {
                        FRONT_END_OP_CONVERSION_CHECK(begin[ov_axis] == 0 && step[ov_axis] == 1 &&
                                                          dst_shape[ov_axis] == src_shape[ov_axis],
                                                      "Dynamic strided VIEW requires explicit runtime slice bounds");
                        end[ov_axis] = std::numeric_limits<int64_t>::max();
                    }
                    axes[ov_axis] = static_cast<int64_t>(ov_axis);
                }
                auto result = std::make_shared<ov::op::v8::Slice>(
                    context.get_input(0),
                    ov::op::v0::Constant::create(ov::element::i64, {begin.size()}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {end.size()}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {step.size()}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes));
                return rename_outputs_with_suffix({std::move(result)}, context.get_name());
            }
        }

        if (is_dense(src_shape, src_strides) && src_strides[0] > 0 && byte_offset >= 0 &&
            byte_offset % src_strides[0] == 0) {
            auto grid_view = materialize_grid_view(context.get_input(0),
                                                   dst_shape,
                                                   dst_strides,
                                                   src_strides[0],
                                                   byte_offset / src_strides[0],
                                                   dynamic_axis);
            if (!grid_view.empty()) {
                return rename_outputs_with_suffix(std::move(grid_view), context.get_name());
            }
        }

        // A view of a non-dense source (e.g. Vcur [hd, nh, T] carved out of a fused QKV row, then viewed as
        // [hd * nh, T]): the operand is the source's logical tensor, so re-express the view's byte strides and
        // offset as element strides over that dense logical tensor and select it as a grid view.
        if (src_shape.size() == src_strides.size() && dst_shape.size() == dst_strides.size() && byte_offset >= 0) {
            const auto logical_strides = to_logical_view(src_shape, src_strides, dst_shape, dst_strides, byte_offset, dynamic_axis);
            if (!logical_strides.empty()) {
                const int64_t element_size = logical_strides.back();
                std::vector<int64_t> strides(logical_strides.begin(), logical_strides.end() - 2);
                for (auto& s : strides) {
                    s *= element_size;
                }
                auto grid_view = materialize_grid_view(context.get_input(0),
                                                       dst_shape,
                                                       strides,
                                                       element_size,
                                                       logical_strides[logical_strides.size() - 2],
                                                       dynamic_axis);
                if (!grid_view.empty()) {
                    return rename_outputs_with_suffix(std::move(grid_view), context.get_name());
                }
            }
        }

        if (byte_offset == 0 && ov::shape_size(src_shape) == ov::shape_size(dst_shape)) {
            auto target_shape = context.get_attribute<std::vector<int64_t>>("view_reshape", {});
            if (target_shape.empty()) {
                target_shape.assign(dst_shape.begin(), dst_shape.end());
            }
            auto target = ov::op::v0::Constant::create(ov::element::i64, {target_shape.size()}, target_shape);
            auto result = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), target, false);
            return rename_outputs_with_suffix({std::move(result)}, context.get_name());
        }

        // Strides were supplied but describe no layout handled above: passing the source through would
        // silently read the wrong elements.
        FRONT_END_OP_CONVERSION_CHECK(src_strides.empty() && dst_strides.empty(),
                                      "Unsupported VIEW layout: cannot materialize the supplied strides and offset");
    }
    return {context.get_input(0)};
}

}  // namespace ov::frontend::gguf::op
