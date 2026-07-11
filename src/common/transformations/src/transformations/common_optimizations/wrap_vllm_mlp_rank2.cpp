// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/wrap_vllm_mlp_rank2.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

// vLLM emits the MLP feature activation as flat [N, H] (rank 2), but the
// intel_cpu LLMMLPFusion pattern anchor and LLMMLPNode validator both
// require rank 3 ([B, S, H]). Rather than relax those in-plugin, this
// pass wraps a matched vLLM rank-2 MLP block with Unsqueeze(axis=0) at
// the shared input and Squeeze(axis=0) at the down_proj output. The
// interior sub-graph (gate_up matmul + VariadicSplit + Swish/Gelu +
// Multiply + down_proj matmul) then flows as rank 3 [1, N, H] and
// LLMMLPFusion matches unchanged.
//
// Match root: the down_proj MatMul whose "activation" input traces back
// (through the Multiply(Swish(gate), up) chain) to the shared source
// used by both gate and up projections. Only the combined-weight form
// (VariadicSplit) is handled — the pre-canonicalization forms are
// rewritten to VariadicSplit by NormalizeVLLMMLP, which runs first.
WrapVLLMMLPRank2::WrapVLLMMLPRank2() {
    MATCHER_SCOPE(WrapVLLMMLPRank2);
    using namespace pattern;

    auto callback = [=](Matcher& m) -> bool {
        auto down_proj = std::dynamic_pointer_cast<ov::op::v0::MatMul>(m.get_match_root());
        if (!down_proj) return false;

        // down_proj's first input is the gated multiply (optionally through Convert).
        auto in0 = down_proj->input_value(0);
        if (auto cvt = std::dynamic_pointer_cast<ov::op::v0::Convert>(in0.get_node_shared_ptr())) {
            in0 = cvt->input_value(0);
        }
        auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(in0.get_node_shared_ptr());
        if (!mul) return false;

        // Each multiply input is either Swish(x) / Gelu(x) or a plain VariadicSplit output.
        std::shared_ptr<ov::op::v1::VariadicSplit> vsplit;
        for (size_t i = 0; i < 2; ++i) {
            auto n = mul->get_input_node_shared_ptr(i);
            std::shared_ptr<ov::Node> inner = n;
            if (std::dynamic_pointer_cast<ov::op::v4::Swish>(n) ||
                std::dynamic_pointer_cast<ov::op::v7::Gelu>(n)) {
                inner = n->get_input_node_shared_ptr(0);
            }
            auto vs = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(inner);
            if (vs) {
                if (!vsplit) vsplit = vs;
                else if (vsplit.get() != vs.get()) return false;  // must be same VariadicSplit
            }
        }
        if (!vsplit) return false;

        // vsplit's data input is the gate_up MatMul (optionally with Convert).
        auto vs_in = vsplit->input_value(0);
        if (auto cvt = std::dynamic_pointer_cast<ov::op::v0::Convert>(vs_in.get_node_shared_ptr())) {
            vs_in = cvt->input_value(0);
        }
        auto gate_up_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(vs_in.get_node_shared_ptr());
        if (!gate_up_mm) return false;

        // shared source = MatMul's activation input
        auto shared_src = gate_up_mm->input_value(0);
        auto src_ps = shared_src.get_partial_shape();
        if (!src_ps.rank().is_static()) return false;
        if (src_ps.rank().get_length() != 2) return false;  // only rank-2 needs wrapping

        // down_proj output partial shape (also rank-2 [N, H_out])
        auto out_ps = down_proj->output(0).get_partial_shape();
        if (!out_ps.rank().is_static() || out_ps.rank().get_length() != 2) return false;

        // Insert Unsqueeze(axis=0) at gate_up_mm's activation input. Only
        // that one edge is rewired — other consumers of shared_src (e.g.
        // the residual add) still get the original rank-2 tensor.
        auto zero_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(shared_src, zero_axis);
        gate_up_mm->input(0).replace_source_output(unsqueezed->output(0));

        // Insert Squeeze(axis=0) after down_proj. All existing consumers
        // of down_proj must be routed through the Squeeze to preserve
        // the original rank-2 shape.
        auto zero_axis_sq = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto squeezed = std::make_shared<ov::op::v0::Squeeze>(down_proj->output(0), zero_axis_sq);
        for (auto& consumer : down_proj->output(0).get_target_inputs()) {
            if (consumer.get_node() == squeezed.get()) continue;
            consumer.replace_source_output(squeezed->output(0));
        }

        ov::copy_runtime_info(ov::NodeVector{gate_up_mm}, unsqueezed);
        ov::copy_runtime_info(ov::NodeVector{down_proj}, squeezed);
        return true;
    };

    auto down_proj_pat = wrap_type<ov::op::v0::MatMul>();
    auto m = std::make_shared<Matcher>(down_proj_pat, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
