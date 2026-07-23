// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include <limits>
#include <vector>
namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    if (context.get_attribute<int>("op_case", 0) == 2) {
        auto dst_shape = context.get_output_shape().to_shape();
        return rename_outputs_with_suffix(
            {process_view_input(context, 0, static_cast<int>(dst_shape[2] * dst_shape[3]))},
            context.get_name());
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
        const int64_t part = gdn[0];   // 0 = attn (first T rows), 1 = state (last S_v rows)
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
            auto end = ov::op::v0::Constant::create(ov::element::i64, {1},
                                                    {std::numeric_limits<int64_t>::max()});
            sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, step, axis2);
        }

        // Reshape the sliced [1,1,rows,S_v*H_v] block to the view's ggml layout (OV order). The attn
        // block keeps the dynamic token axis (-1); the state block is fully static.
        auto tgt = context.get_attribute<std::vector<int64_t>>("view_reshape", {});
        if (!tgt.empty()) {
            if (part == 0) {
                // token axis is dynamic; place the single -1 on the axis holding the token count via
                // element-count conservation against the sliced result's static dims.
                const auto & res_ps = sliced.get_partial_shape();
                int64_t slice_static = 1;
                if (res_ps.rank().is_static()) {
                    for (int64_t i = 0; i < res_ps.rank().get_length(); ++i)
                        if (res_ps[i].is_static())
                            slice_static *= res_ps[i].get_length();
                }
                int64_t out_prod = 1;
                for (auto d : tgt)
                    out_prod *= d;
                if (slice_static != 0 && out_prod % slice_static == 0) {
                    int64_t token_len = out_prod / slice_static;
                    for (auto & d : tgt) {
                        if (d == token_len) {
                            d = -1;
                            break;
                        }
                    }
                }
            }
            auto res = std::make_shared<ov::op::v1::Reshape>(
                sliced, ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt), false);
            return rename_outputs_with_suffix({res}, context.get_name());
        }
        return rename_outputs_with_suffix({sliced}, context.get_name());
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
            auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {start});
            auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {start + len});
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
            // The decoder gives the (static) output layout. The sliced result carries the token
            // count on one dynamic axis; keep it dynamic by placing a single -1 on the output axis
            // that absorbs it. Determine that axis by conservation of element count: the product of
            // the sliced result's static dims must equal the product of the output's non--1 dims,
            // so the -1 axis is the one whose value equals (token count) = slice_static_product /
            // (output_static_product_without_that_axis). Concretely, if the sliced result has a
            // dynamic axis, exactly one output axis must be -1; pick the output axis whose value is
            // not needed to match the sliced result's static dims.
            const auto& res_ps = result.get_partial_shape();
            bool res_has_dyn = res_ps.rank().is_static() && [&] {
                for (int64_t i = 0; i < res_ps.rank().get_length(); ++i)
                    if (res_ps[i].is_dynamic())
                        return true;
                return false;
            }();
            if (res_has_dyn) {
                // product of static dims in the sliced result (excludes the single dynamic axis)
                int64_t slice_static = 1;
                for (int64_t i = 0; i < res_ps.rank().get_length(); ++i)
                    if (res_ps[i].is_static())
                        slice_static *= res_ps[i].get_length();
                // product of all output dims
                int64_t out_prod = 1;
                for (auto d : tgt)
                    out_prod *= d;
                // token count = out_prod / slice_static ; mark the axis holding it as -1
                if (slice_static != 0 && out_prod % slice_static == 0) {
                    int64_t token_len = out_prod / slice_static;
                    for (auto& d : tgt) {
                        if (d == token_len) {
                            d = -1;
                            break;
                        }
                    }
                }
            }
            result = std::make_shared<ov::op::v1::Reshape>(
                result, ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt), false);
        }
        return {result};
    }
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
