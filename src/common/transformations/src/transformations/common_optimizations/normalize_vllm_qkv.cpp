// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/normalize_vllm_qkv.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

// vLLM emits the QKV projection as
//   MatMul([B*S, H], W^T) -> VariadicSplit(axis=-1 i64, lengths=i64 [3]) -> q,k,v
// with rank-2 input. CPU QKVProjFusion was originally written for rank-3
// activations, axis=2 (literal positive), and i32 split-lengths. This pass
// canonicalizes both axes so the existing pattern can match without rank or
// dtype concessions.
//
// Three rewrites, all metadata-only:
//   1. Wrap the input in Reshape->[1, B*S, H] so the activation is rank-3.
//      The MatMul output gets re-flattened with Reshape->[B*S, sum(qkv)] for
//      downstream consumers.
//   2. Replace the VariadicSplit axis Constant with literal i32 axis=2.
//   3. Cast the split_lengths Constant to i32.
//
// This pass leaves the asymmetric Q/K/V split lengths (GQA: Hq != Hk) intact;
// the CPU QKVProjFusion pattern still has to accept asymmetric sizes, which
// is a one-line check relaxation in the plugin pattern.
NormalizeVLLMQKV::NormalizeVLLMQKV() {
    MATCHER_SCOPE(NormalizeVLLMQKV);
    using namespace pattern;

    auto input = any_input([](const ov::Output<ov::Node>& o) {
        auto r = o.get_partial_shape().rank();
        return r.is_static() && r.get_length() == 2;
    });
    auto weight = any_input();
    auto matmul = wrap_type<ov::op::v0::MatMul>({input, weight});
    auto axis = wrap_type<ov::op::v0::Constant>();
    auto lengths = wrap_type<ov::op::v0::Constant>();
    auto vsplit = wrap_type<ov::op::v1::VariadicSplit>({matmul, axis, lengths});

    auto callback = [=](Matcher& m) -> bool {
        const auto& pm = m.get_pattern_value_map();
        auto src = pm.at(input);
        auto mm = pm.at(matmul).get_node_shared_ptr();
        auto vs = pm.at(vsplit).get_node_shared_ptr();
        auto axis_c = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(axis).get_node_shared_ptr());
        auto lens_c = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(lengths).get_node_shared_ptr());
        if (!axis_c || !lens_c) return false;

        auto av = axis_c->cast_vector<int64_t>();
        if (av.size() != 1) return false;
        bool axis_is_last = false;
        if (av[0] == -1 || av[0] == 1) axis_is_last = true;
        if (!axis_is_last) return false;

        auto lv = lens_c->cast_vector<int64_t>();
        if (lv.size() != 3) return false;
        for (auto x : lv) if (x <= 0) return false;

        bool axis_already_canonical =
            (axis_c->get_element_type() == ov::element::i32) &&
            !axis_c->cast_vector<int64_t>().empty() &&
            axis_c->cast_vector<int64_t>()[0] == 2;
        bool lens_already_canonical =
            (lens_c->get_element_type() == ov::element::i32) &&
            lens_c->get_shape().size() == 1 && lens_c->get_shape()[0] == 3;
        bool input_already_rank3 =
            src.get_partial_shape().rank().is_static() && src.get_partial_shape().rank().get_length() == 3;
        if (axis_already_canonical && lens_already_canonical && input_already_rank3) return false;

        // 1. Reshape input [B*S, H] -> [1, B*S, H] (metadata only).
        auto src_shape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 0, 0});
        // reshape pattern: [1, 0, 0] with special_zero=true preserves dims.
        auto src_r3 = std::make_shared<ov::op::v1::Reshape>(src, src_shape_const, true);

        // Rebuild MatMul on the wrapped input, preserving original transpose flags.
        auto mm_orig = ov::as_type_ptr<ov::op::v0::MatMul>(mm);
        if (!mm_orig) return false;
        auto mm_new = std::make_shared<ov::op::v0::MatMul>(
            src_r3, mm_orig->input_value(1),
            mm_orig->get_transpose_a(), mm_orig->get_transpose_b());

        // 2. Replace axis Constant with i32 literal 2.
        auto axis_new = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {static_cast<int32_t>(2)});
        // 3. Cast split lengths to i32.
        auto lens_new = ov::op::v0::Constant::create(
            ov::element::i32, ov::Shape{3},
            {static_cast<int32_t>(lv[0]), static_cast<int32_t>(lv[1]), static_cast<int32_t>(lv[2])});

        auto vs_new = std::make_shared<ov::op::v1::VariadicSplit>(mm_new, axis_new, lens_new);

        // Each VariadicSplit output is rank-3 [1, B*S, len_i]; consumers
        // expect rank-2 [B*S, len_i]. Reflatten with Reshape->[-1, len_i].
        ov::OutputVector new_outs;
        for (size_t i = 0; i < 3; ++i) {
            auto flat_const = ov::op::v0::Constant::create(
                ov::element::i32, ov::Shape{2},
                {static_cast<int32_t>(-1), static_cast<int32_t>(lv[i])});
            auto r2 = std::make_shared<ov::op::v1::Reshape>(vs_new->output(i), flat_const, false);
            new_outs.push_back(r2->output(0));
        }

        // Rewire each existing VariadicSplit output to its rank-2 reshape.
        for (size_t i = 0; i < 3; ++i) {
            vs->output(i).replace(new_outs[i]);
        }
        ov::copy_runtime_info({mm, vs}, {src_r3, mm_new, vs_new});
        return true;
    };

    auto m = std::make_shared<Matcher>(vsplit, "NormalizeVLLMQKV");
    register_matcher(m, callback);
}

}  // namespace ov::pass
