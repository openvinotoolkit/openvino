// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moe_op_fusion.hpp"

#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "ov_ops/moe_compressed.hpp"

namespace ov::pass {

using ov::op::internal::GatherMatmul;
using ov::op::internal::GatherMatmulCompressed;
using ov::op::internal::MOECompressed;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;
namespace v7 = ov::op::v7;
namespace v8 = ov::op::v8;

// Logical K from a weight shape: rank-3 [E, ofm, K] or rank-4 [E, ofm, G, GS].
static size_t weight_logical_K(const ov::Shape& shape) {
    OPENVINO_ASSERT(shape.size() == 3 || shape.size() == 4, "MOE weight must be rank 3 or 4, got rank ", shape.size());
    return shape.size() == 4 ? shape[2] * shape[3] : shape[2];
}

Convert3GatherMatmulMoeBlockToMoeOp::Convert3GatherMatmulMoeBlockToMoeOp(bool has_batch_dim) {
    MATCHER_SCOPE(Convert3GatherMatmulMoeBlockToMoeOp);

    auto hidden_states_m = pattern::any_input();
    auto hidden_state_reshape = pattern::optional<v1::Reshape>({hidden_states_m, pattern::any_input()});
    auto unsqueeze_m = pattern::wrap_type<v0::Unsqueeze>({hidden_state_reshape, pattern::any_input()});

    auto gate_w_m = pattern::any_input();
    auto topk_indices_m = pattern::any_input();

    // Plain BGM (4 inputs: A, B, indices, bias)
    auto bgm_gate_4_m = pattern::wrap_type<GatherMatmul>({unsqueeze_m, gate_w_m, topk_indices_m, pattern::any_input()});
    // Compressed BGM (6 inputs: A, B, indices, bias, scale, zp)
    auto gate_scale_m = pattern::any_input();
    auto gate_zp_m = pattern::any_input();
    auto bgm_gate_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {unsqueeze_m, gate_w_m, topk_indices_m, pattern::any_input(), gate_scale_m, gate_zp_m});
    // Or-pattern
    auto bgm_gate_m = std::make_shared<pattern::op::Or>(OutputVector{bgm_gate_4_m, bgm_gate_6_m});

    // Gate activation: Swish (SwiGLU) or Gelu (GeGLU) with TANH or ERF approximation.
    auto swish_m = pattern::wrap_type<v4::Swish, v7::Gelu>({bgm_gate_m});

    auto up_w_m = pattern::any_input();
    auto bgm_up_4_m = pattern::wrap_type<GatherMatmul>({unsqueeze_m, up_w_m, topk_indices_m, pattern::any_input()});
    auto up_scale_m = pattern::any_input();
    auto up_zp_m = pattern::any_input();
    auto bgm_up_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {unsqueeze_m, up_w_m, topk_indices_m, pattern::any_input(), up_scale_m, up_zp_m});
    auto bgm_up_m = std::make_shared<pattern::op::Or>(OutputVector{bgm_up_4_m, bgm_up_6_m});

    auto swiglu_m = pattern::wrap_type<v1::Multiply>({swish_m, bgm_up_m});

    auto down_w_m = pattern::any_input();
    auto bgm_down_4_m = pattern::wrap_type<GatherMatmul>({swiglu_m, down_w_m, topk_indices_m, pattern::any_input()});
    auto down_scale_m = pattern::any_input();
    auto down_zp_m = pattern::any_input();
    auto bgm_down_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {swiglu_m, down_w_m, topk_indices_m, pattern::any_input(), down_scale_m, down_zp_m});
    auto bgm_down_m = std::make_shared<pattern::op::Or>(OutputVector{bgm_down_4_m, bgm_down_6_m});

    auto routing_m = pattern::any_input();
    auto routing_slice_m = pattern::optional<v8::Slice>(
        {routing_m, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto routing_transpose_m = pattern::wrap_type<v1::Transpose>({routing_slice_m, pattern::any_input()});
    auto routing_unsqueeze_m = pattern::wrap_type<v0::Unsqueeze>({routing_transpose_m, pattern::any_input()});

    auto final_mul_m = pattern::wrap_type<v1::Multiply>({bgm_down_m, routing_unsqueeze_m}, pattern::consumers_count(1));
    auto reduce_sum_m = pattern::wrap_type<v1::ReduceSum>({final_mul_m, pattern::any_input()}, {{"keep_dims", false}});
    auto end_reshape_shape_m = pattern::any_input();
    auto end_reshape_m = pattern::wrap_type<v1::Reshape>({reduce_sum_m, end_reshape_shape_m});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto hidden_states = pm.at(hidden_states_m);

        auto routing = pm.at(routing_m);
        auto topk_indices = pm.at(topk_indices_m);
        auto gate_w = pm.at(gate_w_m);
        auto up_w = pm.at(up_w_m);
        auto down_w = pm.at(down_w_m);

        // Extract expert_beta and detect activation type from the matched gate activation node.
        float expert_beta = 1.0f;
        ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
        auto activation_node = pm.at(swish_m).get_node_shared_ptr();

        if (auto swish_op = ov::as_type_ptr<v4::Swish>(activation_node)) {
            if (swish_op->get_input_size() > 1) {
                if (auto beta_const = ov::as_type_ptr<v0::Constant>(swish_op->get_input_node_shared_ptr(1))) {
                    expert_beta = beta_const->cast_vector<float>()[0];
                }
            }
        } else if (auto gelu_op = ov::as_type_ptr<v7::Gelu>(activation_node)) {
            if (gelu_op->get_approximation_mode() == ov::op::GeluApproximationMode::TANH) {
                activation_type = ov::op::internal::MOE::Activation_type::GEGLU_TANH;
            } else if (gelu_op->get_approximation_mode() == ov::op::GeluApproximationMode::ERF) {
                activation_type = ov::op::internal::MOE::Activation_type::GEGLU_ERF;
            } else {
                return false;
            }
        } else {
            OPENVINO_THROW("Unexpected node type matched for gate activation: ", activation_node);
        }

        std::shared_ptr<ov::Node> moe_node;
        const bool is_gate_compressed = pm.count(bgm_gate_6_m);
        const bool is_up_compressed = pm.count(bgm_up_6_m);
        const bool is_down_compressed = pm.count(bgm_down_6_m);

        // Bail out if BGMs are mixed (some compressed, some plain). The current
        // MOE/MOECompressed configs assume all-or-nothing compression.
        if ((is_gate_compressed != is_up_compressed) || (is_gate_compressed != is_down_compressed)) {
            return false;
        }

        const bool is_compressed = is_gate_compressed;

        if (is_compressed) {
            // MOE u2 mixed-precision bail: the fused moe_3gemm GPU kernel decodes gate/up/down
            // expert weights with a single weight dtype (gate's). NNCF mixed-precision
            // models can quantize the three GEMMs to different dtypes (e.g. gate=u2,
            // up=u8, down=u2), which the fused kernel mis-decodes. Bail out (keep the
            // per-GEMM-correct GatherMatmul form) unless all three share one dtype.
            {
                // A: only uniform-u2 layers fuse (moe_3gemm u2 x-read fixed for straight
                // packing); every other layer keeps the per-GEMM-correct GatherMatmul form.
                const auto _g = pm.at(gate_w_m).get_element_type();
                const auto _u = pm.at(up_w_m).get_element_type();
                const auto _d = pm.at(down_w_m).get_element_type();
                const bool _uniform = (_u == _g && _d == _g);
                // moe_3gemm fuses (a) uniform u2 (x-read fixed) or u4/i4 (the proven INT4
                // path), or (b) any per-GEMM combination of {u2, u8} -- the NNCF mixed-
                // precision case handled by per-GEMM dtype/zp dispatch in the kernel (A3).
                const auto _is_u2u8 = [](ov::element::Type t) {
                    return t == ov::element::u2 || t == ov::element::u8;
                };
                const bool _uniform_ok =
                    _uniform && (_g == ov::element::u2 || _g == ov::element::u4 || _g == ov::element::i4);
                const bool _mixed_u2u8 = _is_u2u8(_g) && _is_u2u8(_u) && _is_u2u8(_d);
                if (!(_uniform_ok || _mixed_u2u8)) {
                    return false;
                }
            }
            auto gate_zp = pm.at(gate_zp_m);
            auto up_zp = pm.at(up_zp_m);
            auto down_zp = pm.at(down_zp_m);

            // Non-zero scalar zp broadcast is only implemented for the u2 batched GEMV
            // kernels, so it is retained per-GEMM only when that GEMM's weights are u2.

            // A scalar (rank-0) zp constant is a folded per-tensor zero point.
            // An all-zero scalar zp is equivalent to symmetric quantization: normalize
            // it to the "absent zp" convention (dynamic-typed empty constant).
            // A non-zero scalar zp is passed through unchanged (u2 weights only):
            // MOE validation explicitly exempts rank-0 inputs from the num_experts
            // check, and the batched GEMV kernels broadcast the single element over
            // all experts/groups/channels.
            // Anything else (non-constant, packed dtype, or mixed sym/asym forms)
            // bails out and leaves the graph in the GatherMatmul form.
            auto normalize_scalar_zp = [&](ov::Output<ov::Node>& zp, ov::element::Type wt) -> bool {
                const auto& zp_shape = zp.get_partial_shape();
                if (!zp_shape.is_static() || zp_shape.rank().get_length() != 0) {
                    return true;  // not a scalar zp: nothing to do
                }
                const auto zp_const = ov::as_type_ptr<v0::Constant>(zp.get_node_shared_ptr());
                if (!zp_const) {
                    return false;
                }
                // Restrict to byte/wide integer types: cast_vector on packed (u2/u4/i4)
                // constants is not meaningful here and may throw.
                const auto zp_et = zp_const->get_element_type();
                if (zp_et != ov::element::i8 && zp_et != ov::element::u8 && zp_et != ov::element::i32 &&
                    zp_et != ov::element::u32 && zp_et != ov::element::i64) {
                    return false;
                }
                for (const auto v : zp_const->cast_vector<int64_t>()) {
                    if (v != 0) {
                        if (wt != ov::element::u2) {
                            return false;  // non-zero scalar zp only supported for u2 weights
                        }
                        // Keep the rank-0 scalar constant as-is: MOE validation exempts
                        // rank-0 inputs from the num_experts check, and the u2 batched
                        // GEMV kernels broadcast the single element (MOE_ZP_SCALAR).
                        return true;
                    }
                }
                zp = std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
                return true;
            };
            if (!normalize_scalar_zp(gate_zp, pm.at(gate_w_m).get_element_type()) ||
                !normalize_scalar_zp(up_zp, pm.at(up_w_m).get_element_type()) ||
                !normalize_scalar_zp(down_zp, pm.at(down_w_m).get_element_type())) {
                return false;
            }
            const size_t num_dynamic_zp = static_cast<size_t>(gate_zp.get_element_type() == ov::element::dynamic) +
                                          static_cast<size_t>(up_zp.get_element_type() == ov::element::dynamic) +
                                          static_cast<size_t>(down_zp.get_element_type() == ov::element::dynamic);
            if (num_dynamic_zp != 0 && num_dynamic_zp != 3) {
                return false;  // mixed symmetric/asymmetric GEMMs are not supported
            }

            // Populate compressed config from weight shapes
            auto wei_partial_shape = gate_w.get_partial_shape();
            OPENVINO_ASSERT(wei_partial_shape.is_static(), "MOE weight shape should be static.");
            auto weight_shape = wei_partial_shape.to_shape();

            auto topk_shape = topk_indices.get_partial_shape();
            OPENVINO_ASSERT(topk_shape[1].is_static(), "K dimension in moe topk input should be static.");

            auto gate_scale = pm.at(gate_scale_m);
            auto up_scale = pm.at(up_scale_m);
            auto down_scale = pm.at(down_scale_m);

            // group_size derived from scales; weight_logical_K handles rank-3/4.
            const auto gate_K = weight_logical_K(weight_shape);
            const auto up_K = weight_logical_K(pm.at(up_w_m).get_partial_shape().to_shape());
            const auto down_K = weight_logical_K(pm.at(down_w_m).get_partial_shape().to_shape());

            auto scale_num_groups = [](const ov::Output<ov::Node>& scale) -> size_t {
                const auto& ps = scale.get_partial_shape();
                if (!ps.is_static())
                    return SIZE_MAX;
                const auto s = ps.to_shape();
                return (s.size() >= 3) ? s[2] : 1;
            };
            const size_t gate_G = scale_num_groups(gate_scale);
            const size_t up_G = scale_num_groups(up_scale);
            const size_t down_G = scale_num_groups(down_scale);

            // MOECompressed requires one group_size shared by all three GEMMs.
            // A per-channel scale ([E, N, 1], group_size == K) is mathematically
            // identical to a group-wise scale with the same value repeated across
            // groups, so broadcast per-channel scale constants to the group-wise
            // num_groups used by the other GEMMs. Genuinely inconsistent group
            // sizes (two different group-wise gs) or non-constant scales bail out.
            size_t group_size = std::numeric_limits<size_t>::max();
            for (const auto& kg : {std::make_pair(gate_K, gate_G), {up_K, up_G}, {down_K, down_G}}) {
                const size_t K = kg.first;
                const size_t G = kg.second;
                if (G == SIZE_MAX)
                    return false;  // dynamic scale shape
                if (G > 1) {
                    if (K % G != 0)
                        return false;
                    const size_t gs = K / G;
                    if (group_size == std::numeric_limits<size_t>::max()) {
                        group_size = gs;
                    } else if (group_size != gs) {
                        return false;  // different group-wise group sizes
                    }
                }
            }

            // Broadcast a per-channel scale constant [E, N, 1] (or [E, N]) to
            // [E, N, K/group_size] by repeating each channel value across groups.
            auto broadcast_per_channel_scale = [&](ov::Output<ov::Node>& scale, size_t K) -> bool {
                const auto c = ov::as_type_ptr<v0::Constant>(scale.get_node_shared_ptr());
                if (!c)
                    return false;
                const auto s = c->get_shape();
                if (s.size() < 2 || K % group_size != 0)
                    return false;
                const size_t E = s[0];
                const size_t N = s[1];
                const size_t target_G = K / group_size;
                if (target_G == 1)
                    return true;  // nothing to expand
                const auto et = c->get_element_type();
                const size_t esz = et.size();
                const auto* src = static_cast<const char*>(c->get_data_ptr());
                std::vector<char> buf(E * N * target_G * esz);
                const size_t total = E * N * target_G;
                for (size_t i = 0; i < total; ++i) {
                    std::memcpy(buf.data() + i * esz, src + (i / target_G) * esz, esz);
                }
                scale = std::make_shared<v0::Constant>(et, ov::Shape{E, N, target_G}, buf.data());
                return true;
            };
            if (group_size != std::numeric_limits<size_t>::max()) {
                if ((gate_G == 1 && !broadcast_per_channel_scale(gate_scale, gate_K)) ||
                    (up_G == 1 && !broadcast_per_channel_scale(up_scale, up_K)) ||
                    (down_G == 1 && !broadcast_per_channel_scale(down_scale, down_K))) {
                    return false;
                }
                // A3: broadcast a per-channel (u8) zp [E, N, 1] to [E, N, K/group_size], the
                // same expansion as its scale, so zshape == sshape and the byfx reorder plus
                // in-kernel (group, channel) zp indexing line up. Scalar (u2) zps are rank-0
                // and were normalized above; they are left untouched here.
                if (num_dynamic_zp == 0) {
                    if ((gate_G == 1 && !broadcast_per_channel_scale(gate_zp, gate_K)) ||
                        (up_G == 1 && !broadcast_per_channel_scale(up_zp, up_K)) ||
                        (down_G == 1 && !broadcast_per_channel_scale(down_zp, down_K))) {
                        return false;
                    }
                }
            }

            // dynamic-typed zp = symmetric placeholder.
            const bool has_zp = num_dynamic_zp == 0;

            // Build MOECompressed with 12 inputs: hidden, routing, topk,
            // gate_w, gate_scale, gate_zp, up_w, up_scale, up_zp, down_w, down_scale, down_zp
            ov::OutputVector moe_inputs = {
                hidden_states,
                routing,
                topk_indices,
                gate_w,
                gate_scale,
                gate_zp,
                up_w,
                up_scale,
                up_zp,
                down_w,
                down_scale,
                down_zp,
            };

            MOECompressed::Config compressed_config{
                {ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU, 0.0f, expert_beta, 0, activation_type},
                gate_K,
                weight_shape[1],
                weight_shape[0],
                0,  // num_shared_expert
                static_cast<size_t>(topk_shape[1].get_length()),
                group_size,
                has_batch_dim,
                has_zp,
                ov::element::f16,
            };

            auto moe_compressed = std::make_shared<MOECompressed>(moe_inputs, compressed_config);

            // Insert Convert if output type was forced and differs from original
            if (moe_compressed->get_output_element_type(0) != hidden_states.get_element_type()) {
                moe_compressed->set_friendly_name(m.get_match_root()->get_friendly_name() + "/MOECompressed");
                auto convert = std::make_shared<v0::Convert>(moe_compressed, hidden_states.get_element_type());
                convert->set_friendly_name(m.get_match_root()->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), {moe_compressed, convert});
                moe_node = convert;
            } else {
                moe_node = moe_compressed;
            }
        } else {
            ov::op::internal::MOE::Config config{ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU,
                                                 0.0f,
                                                 expert_beta,
                                                 0,
                                                 activation_type};
            // Plain MOE with 6 inputs
            ov::OutputVector moe_inputs = {hidden_states, routing, topk_indices, gate_w, up_w, down_w};

            moe_node = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
        }

        moe_node->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), moe_node);
        ov::replace_node(m.get_match_root(), moe_node);

        register_new_node(moe_node);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(end_reshape_m, matcher_name);
    this->register_matcher(matcher, callback);
}

Convert2GatherMatmulMoeBlockToMoeOp::Convert2GatherMatmulMoeBlockToMoeOp(bool has_batch_dim) {
    MATCHER_SCOPE(Convert2GatherMatmulMoeBlockToMoeOp);

    auto hidden_states_m = pattern::any_input();
    auto hidden_state_reshape = pattern::optional<v1::Reshape>({hidden_states_m, pattern::any_input()});
    auto unsqueeze_m = pattern::wrap_type<v0::Unsqueeze>({hidden_state_reshape, pattern::any_input()});

    auto gate_up_w_m = pattern::any_input();
    auto topk_indices_m = pattern::any_input();
    auto gate_up_bias_m = pattern::any_input();

    // Plain BGM (4 inputs)
    auto bgm_gate_up_4_m = pattern::wrap_type<GatherMatmul>({unsqueeze_m, gate_up_w_m, topk_indices_m, gate_up_bias_m});
    // Compressed BGM (6 inputs)
    auto gate_up_scale_m = pattern::any_input();
    auto gate_up_zp_m = pattern::any_input();
    auto bgm_gate_up_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {unsqueeze_m, gate_up_w_m, topk_indices_m, gate_up_bias_m, gate_up_scale_m, gate_up_zp_m});
    auto bgm_gate_up_m = bgm_gate_up_4_m | bgm_gate_up_6_m;

    // Activation subgraph between gate_up and down BGMs
    auto slice1_m = pattern::wrap_type<v8::Slice>(
        {bgm_gate_up_m, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto clamp_m = pattern::wrap_type<v0::Clamp>({slice1_m});
    auto add1_m = pattern::wrap_type<v1::Add>({clamp_m, pattern::wrap_const()});

    auto slice2_m = pattern::wrap_type<v8::Slice>(
        {bgm_gate_up_m, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto minimum1_m = pattern::wrap_type<v1::Minimum>({slice2_m, pattern::wrap_const()});
    auto swish_beta_m = pattern::wrap_const();
    auto swish_m = pattern::wrap_type<v4::Swish>({minimum1_m, swish_beta_m});

    auto multiply2_m = pattern::wrap_type<v1::Multiply>({add1_m, swish_m});

    auto down_w_m = pattern::any_input();
    auto down_bias_m = pattern::any_input();
    // Plain BGM (4 inputs)
    auto bgm_down_4_m = pattern::wrap_type<GatherMatmul>({multiply2_m, down_w_m, topk_indices_m, down_bias_m});
    // Compressed BGM (6 inputs)
    auto down_scale_m = pattern::any_input();
    auto down_zp_m = pattern::any_input();
    auto bgm_down_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {multiply2_m, down_w_m, topk_indices_m, down_bias_m, down_scale_m, down_zp_m});
    auto bgm_down_m = bgm_down_4_m | bgm_down_6_m;

    // No-op Reshape between Transpose and Unsqueeze remains in the graph because
    // CommonOptimizations run after MoE passes in the GPU pipeline; match it as optional here.
    auto routing_transpose_order_m = pattern::wrap_type<v0::Constant>(pattern::value_matches("1, 0"));
    auto routing_transpose_m = pattern::wrap_type<v1::Transpose>({pattern::any_input(), routing_transpose_order_m});
    auto routing_reshape_m = pattern::optional<v1::Reshape>({routing_transpose_m, pattern::any_input()});
    auto routing_unsqueeze_m = pattern::wrap_type<v0::Unsqueeze>({routing_reshape_m, pattern::any_input()});

    auto final_mul_m = pattern::wrap_type<v1::Multiply>({bgm_down_m, routing_unsqueeze_m});
    auto reduce_sum_m = pattern::wrap_type<v1::ReduceSum>({final_mul_m, pattern::any_input()}, {{"keep_dims", false}});
    auto end_reshape_shape_m = pattern::any_input();
    auto end_reshape_m = pattern::wrap_type<v1::Reshape>({reduce_sum_m, end_reshape_shape_m});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto hidden_states = pm.at(hidden_states_m);

        // Bypass the [1,0] Transpose: moe_scatter_reduction expects tokens-major routing.
        // Order is enforced by the pattern (value_matches("1, 0")).
        auto routing_transpose_node = pm.at(routing_transpose_m).get_node_shared_ptr();
        ov::Output<ov::Node> routing = routing_transpose_node->input_value(0);
        auto topk_indices = pm.at(topk_indices_m);
        auto gate_up_w = pm.at(gate_up_w_m);
        auto gate_up_bias = pm.at(gate_up_bias_m);
        auto down_w = pm.at(down_w_m);
        auto down_bias = pm.at(down_bias_m);

        // Extract expert_beta from Swish beta
        auto swish_beta_const = ov::as_type_ptr<v0::Constant>(pm.at(swish_beta_m).get_node_shared_ptr());
        float expert_beta = swish_beta_const->cast_vector<float>()[0];

        // Extract expert_alpha from Clamp max
        auto clamp_node = pm.at(clamp_m).get_node_shared_ptr();
        auto clamp_op = ov::as_type_ptr<v0::Clamp>(clamp_node);
        OPENVINO_ASSERT(clamp_op, "Unexpected node type matched for clamp: ", *clamp_node);
        float expert_alpha = static_cast<float>(clamp_op->get_max());

        // gate_idx = start of the swish (slice2) lane on the step-2 axis.
        auto slice2_node = pm.at(slice2_m).get_node_shared_ptr();
        auto slice2_start_c = ov::as_type_ptr<v0::Constant>(slice2_node->input_value(1).get_node_shared_ptr());
        auto slice2_step_c = ov::as_type_ptr<v0::Constant>(slice2_node->input_value(3).get_node_shared_ptr());
        if (!slice2_start_c || !slice2_step_c) {
            return false;
        }
        const auto starts = slice2_start_c->cast_vector<int64_t>();
        const auto steps = slice2_step_c->cast_vector<int64_t>();
        if (starts.size() != steps.size()) {
            return false;
        }
        size_t gate_idx = 0;
        for (size_t i = 0; i < steps.size(); ++i) {
            if (steps[i] == 2) {
                gate_idx = static_cast<size_t>(starts[i]);
                break;
            }
        }

        std::shared_ptr<ov::Node> moe_node;

        const bool is_gate_up_compressed = pm.count(bgm_gate_up_6_m) > 0;
        const bool is_down_compressed = pm.count(bgm_down_6_m) > 0;

        // Bail out if BGMs are mixed (some compressed, some plain).
        if (is_gate_up_compressed != is_down_compressed) {
            return false;
        }

        const bool is_compressed = is_gate_up_compressed;
        if (is_compressed) {
            // The GEMM2 fused path lowers to cldnn::moe_gemm, whose only registered
            // impl (oneDNN) does not accept u2 weights. Keep u2 in GatherMatmul form
            // until moe_gemm gains u2 support, otherwise compilation fails.
            if (pm.at(gate_up_w_m).get_element_type() == ov::element::u2 ||
                pm.at(down_w_m).get_element_type() == ov::element::u2) {
                return false;
            }
            auto gate_up_zp = pm.at(gate_up_zp_m);
            auto down_zp = pm.at(down_zp_m);

            // Non-zero scalar zp broadcast is only implemented for the u2 fused
            // kernels, so it is retained per-GEMM only when that GEMM's weights are u2.

            // A scalar (rank-0) zp constant is a folded per-tensor zero point.
            // An all-zero scalar zp is equivalent to symmetric quantization: normalize
            // it to the "absent zp" convention (dynamic-typed empty constant).
            // A non-zero scalar zp is passed through unchanged (u2 weights only);
            // the fused backend broadcasts the single element over all
            // experts/groups/channels.
            // Anything else (non-constant, packed dtype, or mixed sym/asym forms)
            // bails out and leaves the graph in the GatherMatmul form.
            auto normalize_scalar_zp = [&](ov::Output<ov::Node>& zp, ov::element::Type wt) -> bool {
                const auto& zp_shape = zp.get_partial_shape();
                if (!zp_shape.is_static() || zp_shape.rank().get_length() != 0) {
                    return true;  // not a scalar zp: nothing to do
                }
                const auto zp_const = ov::as_type_ptr<v0::Constant>(zp.get_node_shared_ptr());
                if (!zp_const) {
                    return false;
                }
                // Restrict to byte/wide integer types: cast_vector on packed (u2/u4/i4)
                // constants is not meaningful here and may throw.
                const auto zp_et = zp_const->get_element_type();
                if (zp_et != ov::element::i8 && zp_et != ov::element::u8 && zp_et != ov::element::i32 &&
                    zp_et != ov::element::u32 && zp_et != ov::element::i64) {
                    return false;
                }
                for (const auto v : zp_const->cast_vector<int64_t>()) {
                    if (v != 0) {
                        if (wt != ov::element::u2) {
                            return false;  // non-zero scalar zp only supported for u2 weights
                        }
                        // Keep the rank-0 scalar constant as-is: the fused backend
                        // broadcasts the single element over all experts/groups/channels.
                        return true;
                    }
                }
                zp = std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
                return true;
            };
            if (!normalize_scalar_zp(gate_up_zp, pm.at(gate_up_w_m).get_element_type()) ||
                !normalize_scalar_zp(down_zp, pm.at(down_w_m).get_element_type())) {
                return false;
            }
            const size_t num_dynamic_zp = static_cast<size_t>(gate_up_zp.get_element_type() == ov::element::dynamic) +
                                          static_cast<size_t>(down_zp.get_element_type() == ov::element::dynamic);
            if (num_dynamic_zp != 0 && num_dynamic_zp != 2) {
                return false;  // mixed symmetric/asymmetric GEMMs are not supported
            }

            // Populate compressed config from weight shapes
            auto weight_shape = gate_up_w.get_shape();
            const size_t hidden = weight_logical_K(weight_shape);

            auto topk_indices_shape = topk_indices.get_partial_shape();
            auto topk_rank = topk_indices_shape.rank().get_length();
            OPENVINO_ASSERT(topk_indices_shape[topk_rank - 1].is_static(),
                            "K dimension in moe topk_indices input should be static.");

            auto gate_up_scale = pm.at(gate_up_scale_m);
            auto down_scale = pm.at(down_scale_m);

            // group_size derived from scales; weight_logical_K handles rank-3/4.
            const auto down_K = weight_logical_K(pm.at(down_w_m).get_partial_shape().to_shape());

            auto scale_num_groups = [](const ov::Output<ov::Node>& scale) -> size_t {
                const auto& ps = scale.get_partial_shape();
                if (!ps.is_static())
                    return SIZE_MAX;
                const auto s = ps.to_shape();
                return (s.size() >= 3) ? s[2] : 1;
            };
            const size_t gate_up_G = scale_num_groups(gate_up_scale);
            const size_t down_G = scale_num_groups(down_scale);

            // MOECompressed requires one group_size shared by both GEMMs.
            // A per-channel scale ([E, N, 1], group_size == K) is mathematically
            // identical to a group-wise scale with the same value repeated across
            // groups, so broadcast per-channel scale constants to the group-wise
            // num_groups used by the other GEMM. Genuinely inconsistent group
            // sizes (two different group-wise gs) or non-constant scales bail out.
            size_t group_size = std::numeric_limits<size_t>::max();
            for (const auto& kg : {std::make_pair(hidden, gate_up_G), {down_K, down_G}}) {
                const size_t K = kg.first;
                const size_t G = kg.second;
                if (G == SIZE_MAX)
                    return false;  // dynamic scale shape
                if (G > 1) {
                    if (K % G != 0)
                        return false;
                    const size_t gs = K / G;
                    if (group_size == std::numeric_limits<size_t>::max()) {
                        group_size = gs;
                    } else if (group_size != gs) {
                        return false;  // different group-wise group sizes
                    }
                }
            }

            // Broadcast a per-channel scale constant [E, N, 1] (or [E, N]) to
            // [E, N, K/group_size] by repeating each channel value across groups.
            auto broadcast_per_channel_scale = [&](ov::Output<ov::Node>& scale, size_t K) -> bool {
                const auto c = ov::as_type_ptr<v0::Constant>(scale.get_node_shared_ptr());
                if (!c)
                    return false;
                const auto s = c->get_shape();
                if (s.size() < 2 || K % group_size != 0)
                    return false;
                const size_t E = s[0];
                const size_t N = s[1];
                const size_t target_G = K / group_size;
                if (target_G == 1)
                    return true;  // nothing to expand
                const auto et = c->get_element_type();
                const size_t esz = et.size();
                const auto* src = static_cast<const char*>(c->get_data_ptr());
                std::vector<char> buf(E * N * target_G * esz);
                const size_t total = E * N * target_G;
                for (size_t i = 0; i < total; ++i) {
                    std::memcpy(buf.data() + i * esz, src + (i / target_G) * esz, esz);
                }
                scale = std::make_shared<v0::Constant>(et, ov::Shape{E, N, target_G}, buf.data());
                return true;
            };
            if (group_size != std::numeric_limits<size_t>::max()) {
                if ((gate_up_G == 1 && !broadcast_per_channel_scale(gate_up_scale, hidden)) ||
                    (down_G == 1 && !broadcast_per_channel_scale(down_scale, down_K))) {
                    return false;
                }
                // Broadcast a per-channel (u8) zp [E, N, 1] to [E, N, K/group_size], the
                // same expansion as its scale, so zshape == sshape and the (group, channel)
                // zp indexing line up. Scalar (u2) zps are rank-0 and were normalized
                // above; they are left untouched here.
                if (num_dynamic_zp == 0) {
                    if ((gate_up_G == 1 && !broadcast_per_channel_scale(gate_up_zp, hidden)) ||
                        (down_G == 1 && !broadcast_per_channel_scale(down_zp, down_K))) {
                        return false;
                    }
                }
            }

            // dynamic-typed zp = symmetric placeholder.
            const bool has_zp = num_dynamic_zp == 0;

            // Build MOECompressed inputs
            // GEMM2 compressed layout: hidden, routing, topk,
            // gate_up_w, gate_up_scale, [gate_up_zp,] gate_up_bias,
            // down_w, down_scale, [down_zp,] down_bias
            ov::OutputVector moe_inputs;
            moe_inputs.push_back(hidden_states);
            moe_inputs.push_back(routing);
            moe_inputs.push_back(topk_indices);

            // gate_up params
            moe_inputs.push_back(gate_up_w);
            moe_inputs.push_back(gate_up_scale);
            if (has_zp) {
                moe_inputs.push_back(gate_up_zp);
            }
            moe_inputs.push_back(gate_up_bias);

            // down params
            moe_inputs.push_back(down_w);
            moe_inputs.push_back(down_scale);
            if (has_zp) {
                moe_inputs.push_back(down_zp);
            }
            moe_inputs.push_back(down_bias);

            MOECompressed::Config compressed_config{
                {ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP, expert_alpha, expert_beta, gate_idx},
                hidden,
                weight_shape[1],
                weight_shape[0],
                0,  // num_shared_expert
                static_cast<size_t>(topk_indices_shape[topk_rank - 1].get_length()),
                group_size,
                has_batch_dim,
                has_zp,
                ov::element::dynamic,
            };

            moe_node = std::make_shared<MOECompressed>(moe_inputs, compressed_config);
        } else {
            const ov::op::internal::MOE::Config config{ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP,
                                                       expert_alpha,
                                                       expert_beta,
                                                       gate_idx};
            // Plain MOE with 7 inputs
            const ov::OutputVector moe_inputs =
                {hidden_states, routing, topk_indices, gate_up_w, gate_up_bias, down_w, down_bias};

            moe_node = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
        }

        moe_node->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), moe_node);
        ov::replace_node(m.get_match_root(), moe_node);

        register_new_node(moe_node);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(end_reshape_m, matcher_name);
    this->register_matcher(matcher, callback);
}

}  // namespace ov::pass
