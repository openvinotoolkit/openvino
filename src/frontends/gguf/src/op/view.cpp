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

// Cases 2-5 are shared by both ingest paths: the llama.cpp cgraph decoder classifies a ggml view
// into them (see ggml-decoder.cpp::compute_op_case) and the native .gguf builder describes its own
// views the same way. The native builder can supply an optional second shape-reference input
// for case 3 to preserve the runtime token layout. Builder-only case 104 also uses that input.
// See its comment below and docs/frontend_design.md for the builder-only cases.
OutputVector translate_view(const NodeContext& context) {
    // A KV-cache read described by the decoder (runtime length and first stream as extra inputs).
    const auto kv_view = context.get_attribute<std::vector<int64_t>>("kv_view", {});
    if (!kv_view.empty()) {
        num_inputs_check(context, 1, 3);
        return translate_kv_view(context, kv_view);
    }
    num_inputs_check(context, 1, 2);

    const auto& output_shape = context.get_output_shape();
    const auto& input_shape = context.get_input_shape(0);
    if ((output_shape.is_static() && ov::shape_size(output_shape.to_shape()) == 0) ||
        (input_shape.is_static() && ov::shape_size(input_shape.to_shape()) == 0)) {
        // Empty recurrent-cache reorder views have no values and their CPY
        // consumer is routed directly to the cache base tensor.
        return {context.get_input(0)};
    }

    if (context.get_op_case() == 2) {
        auto dst_shape = context.get_output_shape().to_shape();
        return rename_outputs_with_suffix(
            {process_view_input(context, 0, static_cast<int>(dst_shape[2] * dst_shape[3]))},
            context.get_name());
    }
    if (context.get_attribute<int>("op_case", 0) == 5) {
        // qwen3.5 interleaved Q/gate split. The source is a contiguous [F, T] projection whose feature
        // axis packs G groups of `group_stride` elements, and this view selects a `inner_len`-wide
        // sub-block at `inner_off` within every group (e.g. Qcur_full packs [Q(256), gate(256)] per
        // head; the Q view takes inner [0,256), the gate view inner [256,512)). A plain contiguous
        // slice would scramble heads, so restore the source layout, reshape to expose the per-group
        // stride, slice the inner sub-window on the stride axis, then reshape to the view's own layout.
        auto input = context.get_input(0);
        auto iw = context.get_attribute<std::vector<int64_t>>("view_interleaved", {});
        FRONT_END_OP_CONVERSION_CHECK(iw.size() == 4,
                                      "interleaved view expects {group_count, group_stride, inner_off, inner_len}");
        const int64_t group_count = iw[0];
        const int64_t group_stride = iw[1];
        const int64_t inner_off = iw[2];
        const int64_t inner_len = iw[3];

        // Restore the original ggml (base) shape if the OV input was already reshaped, mirroring
        // op_case 3. The base is [F, T] (ggml) == OV [.., T, F]; we then split F into G x group_stride.
        auto input_ggml_shape = context.get_attribute<ov::Shape>("input_ggml_shape", {});
        auto input_ov_shape = input.get_partial_shape();
        if (!input_ggml_shape.empty() && input_ov_shape.rank().is_static() &&
            static_cast<size_t>(input_ov_shape.rank().get_length()) != input_ggml_shape.size()) {
            input = std::make_shared<ov::op::v1::Reshape>(
                input,
                ov::op::v0::Constant::create(ov::element::i64, {input_ggml_shape.size()}, input_ggml_shape),
                false);
        }

        // Split the feature axis to expose the per-group stride without disturbing the leading dims:
        // [1, 1, T, F] -> [1, 1, T, group_count, group_stride]. special_zero (0 copies the input dim at
        // that position) keeps the leading [1, 1, T] intact -- including the dynamic token axis -- so
        // only the last dim F is factored into the two static group dims.
        auto expose = ov::op::v0::Constant::create(ov::element::i64,
                                                   {5},
                                                   std::vector<int64_t>{0, 0, 0, group_count, group_stride});
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(input, expose, /*special_zero=*/true);

        // Slice the inner sub-window [inner_off, inner_off+inner_len) on the group_stride axis (4).
        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {inner_off});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {inner_off + inner_len});
        auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axis4 = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
        ov::Output<ov::Node> sliced = std::make_shared<ov::op::v8::Slice>(reshaped, begin, end, stride, axis4);

        // Reshape to the view's own OV layout; -1 lands on the (dynamic) token axis.
        auto tgt = context.get_attribute<std::vector<int64_t>>("view_reshape", {});
        if (!tgt.empty()) {
            place_dynamic_token_axis(tgt, sliced.get_partial_shape());
            auto res =
                std::make_shared<ov::op::v1::Reshape>(sliced,
                                                      ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt),
                                                      false);
            res->set_friendly_name("gdn_view_reshape_" + context.get_name());
            return rename_outputs_with_suffix({std::move(res)}, context.get_name());
        }
        return rename_outputs_with_suffix({std::move(sliced)}, context.get_name());
    }
    if (context.get_attribute<int>("op_case", 0) == 4) {
        // qwen3-next GDN packed output view. The GATED_DELTA_NET result is [1, 1, T + S_v, S_v*H_v]
        // with the T attention rows stacked on top of the S_v recurrent-state rows (see
        // translate_gated_delta_net). attn_output-0 reads the first T rows; new_state-0 reads the last
        // S_v rows. T (token count) is dynamic, S_v is static, so we slice OV axis 2 with bounds
        // anchored to the static S_v tail (negative indexing), then reshape to the view's own layout.
        auto input = context.get_input(0);
        auto gdn = context.get_attribute<std::vector<int64_t>>("gdn_view", {});
        FRONT_END_OP_CONVERSION_CHECK(gdn.size() == 2, "GDN view expects {part, s_v}");
        const int64_t part = gdn[0];  // 0 = attn (first T rows), 1 = state (last S_v rows)
        const int64_t s_v = gdn[1];

        auto axis2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ov::Output<ov::Node> sliced;
        if (part == 0) {
            // first T rows == everything except the last S_v rows: [0, -S_v)
            auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {-s_v});
            sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, step, axis2);
        } else {
            // last S_v rows: [-S_v, end)
            auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-s_v});
            auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {std::numeric_limits<int64_t>::max()});
            sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, step, axis2);
        }

        // Reshape the sliced [1,1,rows,S_v*H_v] block to the view's ggml layout (OV order). The attn
        // block keeps the dynamic token axis (-1); the state block is fully static.
        auto tgt = context.get_attribute<std::vector<int64_t>>("view_reshape", {});
        if (!tgt.empty()) {
            if (part == 0) {
                place_dynamic_token_axis(tgt, sliced.get_partial_shape());
            }
            auto res =
                std::make_shared<ov::op::v1::Reshape>(sliced,
                                                      ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt),
                                                      false);
            return rename_outputs_with_suffix({std::move(res)}, context.get_name());
        }
        return rename_outputs_with_suffix({std::move(sliced)}, context.get_name());
    }
    if (context.get_attribute<int>("op_case", 0) == 3) {
        auto input = context.get_input(0);
        auto input_ov_shape = input.get_partial_shape();
        auto input_ggml_shape = context.get_attribute<ov::Shape>("input_ggml_shape", {});

        // Input already reshaped: restore the original ggml (base) shape before slicing.
        if (!input_ggml_shape.empty() && input_ov_shape.rank().is_static() &&
            static_cast<size_t>(input_ov_shape.rank().get_length()) != input_ggml_shape.size()) {
            input = std::make_shared<ov::op::v1::Reshape>(
                input,
                ov::op::v0::Constant::create(ov::element::i64, {input_ggml_shape.size()}, input_ggml_shape),
                false);
        }

        // The decoder computes the slice as {ov_axis, start, len} from ggml's ne/nb/offset -- this
        // handles both a shrunk dim (topk 64->8, start 0) and an offset-selected sub-block
        // (per-expert ffn_moe_weighted, start selects the expert). The frontend never inspects
        // ggml strides.
        auto slice = context.get_attribute<std::vector<int64_t>>("view_slice", {});
        ov::Output<ov::Node> result = input;
        bool whole_concat_input = false;
        // Selecting a complete Concat input needs no packing or copy. This also keeps
        // SSM token outputs independent of the final state when converting to paged execution.
        const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(input.get_node_shared_ptr());
        if (slice.size() == 3 && concat && concat->get_input_size() == 2 && concat->get_axis() == slice[0] &&
            slice[0] >= 0 && concat->get_input_partial_shape(1).rank().is_static()) {
            const auto& tail = concat->get_input_partial_shape(1)[slice[0]];
            if (tail.is_static() && tail.get_length() > 0) {
                const auto count = tail.get_length();
                const bool prefix = slice[1] == 0 && slice[2] == -count;
                const bool suffix = slice[1] == -count && slice[2] == count;
                if (prefix || suffix) {
                    result = concat->input_value(suffix ? 1 : 0);
                    whole_concat_input = true;
                }
            }
        }
        if (!whole_concat_input && slice.size() == 3) {
            const int64_t axis = slice[0], start = slice[1], len = slice[2];
            if (axis >= 0 && static_cast<size_t>(axis) < input_ggml_shape.size() &&
                start >= static_cast<int64_t>(input_ggml_shape[axis])) {
                // A one-past-end recurrent cache view represents an empty
                // slot-reorder destination. Its CPY consumer is routed to the
                // cache base, so no standalone slice is needed.
                return {std::move(input)};
            }
            auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {start});
            // A negative start is an end-anchored TAIL slice (conv_state_last: last d_conv-1 columns
            // of a dynamic-length conv window); its end must run to the axis end, not start+len (which
            // would be <= 0). A non-negative start is an absolute sub-range, ending at start+len.
            const int64_t end_val = start < 0 ? std::numeric_limits<int64_t>::max() : start + len;
            auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val});
            auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
            result = std::make_shared<ov::op::v8::Slice>(input, begin, end, step, axes);
        }

        // Reshape the sliced result to the view's own (output) shape: ggml views rearrange/drop
        // singleton dims (e.g. select-expert slice yields [1,tok,1,2048] but the view's layout is
        // [1,1,tok,2048], token on axis 2). Skipping this leaves the dynamic token axis in the wrong
        // position, which later broadcasts incorrectly against the residual stream. The decoder
        // supplies the reshape target as "view_reshape" -- the output dims with -1 at the (dynamic)
        // token axis -- so the token dim stays dynamic and lands where the consumer expects it.
        auto tgt = context.get_attribute<std::vector<int64_t>>("view_reshape", {});
        if (!tgt.empty()) {
            place_dynamic_token_axis(tgt, result.get_partial_shape());
            result =
                std::make_shared<ov::op::v1::Reshape>(result,
                                                      ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt),
                                                      false);
            result.get_node_shared_ptr()->set_friendly_name("view_reshape_" + context.get_name());
        }
        // A second operand provides the runtime layout for a view of a packed recurrent
        // output. This preserves token placement when paged execution flattens the batch.
        if (context.get_input_size() == 2) {
            result = std::make_shared<ov::op::v1::Reshape>(result,
                                                           std::make_shared<ov::op::v3::ShapeOf>(context.get_input(1)),
                                                           false);
        }
        // Preserve producer names when the view reuses an existing value.
        if (result == context.get_input(0) || (whole_concat_input && tgt.empty() && context.get_input_size() == 1)) {
            return {std::move(result)};
        }
        return rename_outputs_with_suffix({std::move(result)}, context.get_name());
    }
    // op_case 104 (builder): layer-index slice for per-layer embedding.
    // Input [1, n_layer, T, D] -> slice the layer axis (1) -> [1, 1, T, D].
    if (context.get_op_case() == 104) {
        const int64_t layer_idx = context.get_attribute<int64_t>("layer_idx");
        auto input = context.get_input(0);
        auto start = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_idx});
        auto stop = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_idx + 1});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ov::Output<ov::Node> sliced = std::make_shared<ov::op::v8::Slice>(input, start, stop, step, axes);

        // The slice keeps the token count on the axis per_layer_embd stores it on (dim 1, since
        // it is layer-major). But every consumer is an elementwise op against the layer's own
        // activation, and OV broadcasts positionally, so both operands must agree which leading
        // axis holds the tokens: [1, T, ..] under plain SDPA, [T, 1, ..] once
        // ov::pass::SDPAToPagedAttention moves the token count into dim 0. Both hold the same T*D
        // values contiguously, so when the builder supplies the activation as a second
        // (shape-reference) input, reinterpret the slice into that operand's leading dims --
        // otherwise the multiply below broadcasts to a T x T outer product under PA.
        if (context.get_input_size() > 1) {
            const auto& ref = context.get_input(1);
            const auto ref_rank = ref.get_partial_shape().rank();
            FRONT_END_OP_CONVERSION_CHECK(ref_rank.is_static(),
                                          "VIEW case 104 shape reference must have a static rank");
            const auto d_ps = input.get_partial_shape();
            const int64_t rank = ref_rank.get_length();
            FRONT_END_OP_CONVERSION_CHECK(d_ps.rank().is_static() && d_ps[d_ps.rank().get_length() - 1].is_static(),
                                          "VIEW case 104 requires a static per-layer embedding width");
            const int64_t d = d_ps[d_ps.rank().get_length() - 1].get_length();

            std::vector<int64_t> lead(rank - 1);
            for (int64_t i = 0; i < rank - 1; ++i) {
                lead[i] = i;
            }
            auto lead_dims = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::ShapeOf>(ref, ov::element::i64),
                ov::op::v0::Constant::create(ov::element::i64, {lead.size()}, lead),
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            auto target = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{lead_dims, ov::op::v0::Constant::create(ov::element::i64, {1}, {d})},
                0);
            sliced = std::make_shared<ov::op::v1::Reshape>(sliced, target, false);
        }
        return rename_outputs_with_suffix({sliced.get_node_shared_ptr()}, context.get_name());
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
