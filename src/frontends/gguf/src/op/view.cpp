// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

namespace {
// Put the reshape target's single -1 on the axis carrying the runtime token count, recovered by
// element-count conservation: product(tgt) / product(res_ps's static dims). No-op for a static
// res_ps, where the target is already exact and a -1 could land on the wrong axis.
void place_dynamic_token_axis(std::vector<int64_t>& tgt, const ov::PartialShape& res_ps) {
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
}  // namespace

// Cases 2-5 are shared by both ingest paths: the llama.cpp cgraph decoder classifies a ggml view
// into them (see ggml-decoder.cpp::compute_op_case) and the native .gguf builder describes its own
// views the same way. Case 104 is builder-only: it takes a second (shape-reference) input the
// cgraph path does not supply, so it has a different arity than the shared cases. See its comment
// below, and docs/frontend_design.md for the other two builder-only cases in the frontend.
OutputVector translate_view(const NodeContext& context) {
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
        auto input_ggml_shape = context.get_attribute<ov::Shape>("input_ggml_shape");
        auto input_ov_shape = input.get_partial_shape();
        if (input_ov_shape.rank().is_static() &&
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
        auto input_ggml_shape = context.get_attribute<ov::Shape>("input_ggml_shape");

        // Input already reshaped: restore the original ggml (base) shape before slicing.
        if (input_ov_shape.rank().is_static() &&
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
        if (slice.size() == 3) {
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
        // A view that neither restores rank, slices nor reshapes is a pass-through: the value is
        // still the producer's output, so renaming it here would rename a node owned by another
        // ggml tensor (and the suffix compounds, since the helper appends).
        if (result == context.get_input(0)) {
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
            const auto d_ps = context.get_output_shape();
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

    // Generic llama.cpp view. A view that preserves the source strides is a rectangular crop of
    // the logical source tensor; its byte offset identifies the first element. This covers the
    // ordinary multi-axis views used by backend validation without specializing their consumers.
    if (input_shape.is_static() && output_shape.is_static()) {
        const auto src_shape = input_shape.to_shape();
        const auto dst_shape = output_shape.to_shape();
        const auto src_strides = context.get_attribute<std::vector<int64_t>>("view_input_strides", {});
        const auto dst_strides = context.get_attribute<std::vector<int64_t>>("view_output_strides", {});
        const int64_t byte_offset = context.get_attribute<int64_t>("view_offset_bytes", 0);

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

            // A ggml view may start inside a row and continue into the next row while preserving
            // the outer planes. Flatten the two innermost ggml dimensions for that case.
            if (src_shape.size() == 4 && src_shape[0] == dst_shape[0] && src_shape[1] == dst_shape[1] &&
                src_strides[0] > 0 && byte_offset % src_strides[0] == 0) {
                const int64_t begin_elem = byte_offset / src_strides[0];
                const int64_t src_plane = static_cast<int64_t>(src_shape[2] * src_shape[3]);
                const int64_t dst_plane = static_cast<int64_t>(dst_shape[2] * dst_shape[3]);
                if (begin_elem >= 0 && begin_elem + dst_plane <= src_plane) {
                    auto flat_shape =
                        ov::op::v0::Constant::create(ov::element::i64,
                                                     {3},
                                                     std::vector<int64_t>{static_cast<int64_t>(src_shape[0]),
                                                                          static_cast<int64_t>(src_shape[1]),
                                                                          src_plane});
                    auto flat = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), flat_shape, false);
                    auto selected = std::make_shared<ov::op::v8::Slice>(
                        flat,
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_elem}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_elem + dst_plane}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
                    auto target = ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape);
                    auto result = std::make_shared<ov::op::v1::Reshape>(selected, target, false);
                    return rename_outputs_with_suffix({std::move(result)}, context.get_name());
                }
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

        if (byte_offset == 0 && ov::shape_size(src_shape) == ov::shape_size(dst_shape)) {
            auto target = ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape);
            auto result = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), target, false);
            return rename_outputs_with_suffix({std::move(result)}, context.get_name());
        }
    }
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
