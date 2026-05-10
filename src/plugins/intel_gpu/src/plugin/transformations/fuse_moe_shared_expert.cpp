// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_shared_expert.hpp"

#include <memory>
#include <optional>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

FuseMOESharedExpert::FuseMOESharedExpert() {
    using namespace ov::pass::pattern;

    // Match the MOE node (GEMM3_SWIGLU type, 6 inputs: hidden, routing, topk, gate, up, down)
    auto hidden_states_m = any_input();
    auto routing_weights_m = any_input();
    auto topk_m = any_input();
    auto gate_weight_m = any_input();
    auto up_weight_m = any_input();
    auto down_weight_m = any_input();

    auto is_gemm3_swiglu = [](const ov::Output<ov::Node>& output) {
        auto moe = ov::as_type_ptr<ov::op::internal::MOE>(output.get_node_shared_ptr());
        return moe && moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    };

    auto moe_base_m = wrap_type<ov::op::internal::MOE>({hidden_states_m, routing_weights_m, topk_m,
                                                         gate_weight_m, up_weight_m, down_weight_m},
                                                        is_gemm3_swiglu);

    // Match MOECompressed node (12 inputs: hidden, routing, topk, gate/scale/zp, up/scale/zp, down/scale/zp)
    auto gate_scale_m = any_input();
    auto gate_zp_m = any_input();
    auto up_scale_m = any_input();
    auto up_zp_m = any_input();
    auto down_scale_m = any_input();
    auto down_zp_m = any_input();

    auto moe_compressed_m = wrap_type<ov::op::internal::MOECompressed>(
        {hidden_states_m, routing_weights_m, topk_m,
         gate_weight_m, gate_scale_m, gate_zp_m,
         up_weight_m, up_scale_m, up_zp_m,
         down_weight_m, down_scale_m, down_zp_m},
        is_gemm3_swiglu);

    auto moe_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{moe_base_m, moe_compressed_m});

    // MoeOpFusion may insert Convert(f16→f32) between MOECompressed and Add
    // when the MOE output type (f16) differs from hidden_states type (f32).
    // Match this optional Convert so the pattern still fires.
    auto moe_convert_m = optional<ov::op::v0::Convert>({moe_m});

    // Shared expert subgraph:
    //   shared_gate = MatMul(shared_hidden, shared_gate_weight)
    //   shared_swish = Swish(shared_gate)
    //   shared_up   = MatMul(shared_hidden, shared_up_weight)
    //   shared_mul  = Mul(shared_swish, shared_up)
    //   shared_down = MatMul(shared_mul, shared_down_weight)
    //   Optional gating: sigmoid(MatMul(shared_hidden, gate_gate_weight)) * shared_down
    //   Optional reshape before Add
    auto shared_hidden_states_m = any_input();
    auto shared_gate_weight_m = any_input();
    auto shared_gate_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_states_m, shared_gate_weight_m});
    auto shared_swish_m = wrap_type<ov::op::v4::Swish>({shared_gate_m});
    auto shared_up_weight_m = any_input();
    auto shared_up_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_states_m, shared_up_weight_m});
    // Multiply is commutative: handle both input orders
    auto shared_mul_m_1 = wrap_type<ov::op::v1::Multiply>({shared_swish_m, shared_up_m});
    auto shared_mul_m_2 = wrap_type<ov::op::v1::Multiply>({shared_up_m, shared_swish_m});
    auto shared_mul_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_mul_m_1, shared_mul_m_2});
    auto shared_down_weight_m = any_input();
    auto shared_down_m = wrap_type<ov::op::v0::MatMul>({shared_mul_m, shared_down_weight_m});

    // Optional sigmoid gating: sigmoid(MatMul(hidden, gate_gate)) * down
    auto shared_gate_gate_wei_m = any_input();
    auto shared_gate_gate_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_states_m, shared_gate_gate_wei_m});
    auto shared_gate_sigmoid_m = wrap_type<ov::op::v0::Sigmoid>({shared_gate_gate_m});
    // Multiply is commutative: handle both input orders
    auto shared_expert_gated_m_1 = wrap_type<ov::op::v1::Multiply>({shared_gate_sigmoid_m, shared_down_m});
    auto shared_expert_gated_m_2 = wrap_type<ov::op::v1::Multiply>({shared_down_m, shared_gate_sigmoid_m});
    auto shared_expert_gated_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_expert_gated_m_1, shared_expert_gated_m_2});
    auto shared_expert_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_down_m, shared_expert_gated_m});
    auto shared_expert_reshaped_m = optional<ov::op::v1::Reshape>({shared_expert_m, any_input()});

    // Root: Add(MOE[→Convert], SharedExpert) or Add(SharedExpert, MOE[→Convert])
    auto add_1 = wrap_type<ov::op::v1::Add>({moe_convert_m, shared_expert_reshaped_m});
    auto add_2 = wrap_type<ov::op::v1::Add>({shared_expert_reshaped_m, moe_convert_m});
    auto root = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{add_1, add_2});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto root_node = pattern_map.at(root).get_node_shared_ptr();
        auto moe_node = pattern_map.at(moe_m).get_node_shared_ptr();
        auto moe = ov::as_type_ptr<ov::op::internal::MOE>(moe_node);
        if (!moe || transformation_callback(root_node)) {
            return false;
        }

        auto sh_gate_w = pattern_map.at(shared_gate_weight_m);
        auto sh_up_w   = pattern_map.at(shared_up_weight_m);
        auto sh_down_w = pattern_map.at(shared_down_weight_m);

        // any_input() may be spuriously bound on the non-gating branch — use sigmoid as ground truth.
        const bool has_gating = pattern_map.count(shared_gate_sigmoid_m) > 0;
        ov::Output<ov::Node> sh_gate_gate_w;
        if (has_gating) {
            sh_gate_gate_w = pattern_map.at(shared_gate_gate_wei_m);
        }

        auto moe_compressed = ov::as_type_ptr<ov::op::internal::MOECompressed>(moe_node);

        // Detect "F16 shared expert + compressed sparse experts" (mixed-precision) topology:
        //   * input MOE is MOECompressed (sparse experts are compressed)
        //   * all three shared weights are uncompressed f16/bf16/f32 Constants
        //     (possibly behind a decompression Convert, e.g. Const(bf16)→Convert(f32)→MatMul)
        // In this case, we normalize to the 22-input MOECompressed layout that
        // FuseMOE3GemmCompressed expects (with dummy scale/zp constants), and update
        // the Config to reflect the shared expert's precision.

        // Look through an optional decompression Convert to find the source Constant.
        auto get_source_constant = [](const ov::Output<ov::Node>& v)
                -> std::shared_ptr<ov::op::v0::Constant> {
            auto node = v.get_node_shared_ptr();
            if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
                node = convert->input_value(0).get_node_shared_ptr();
            }
            return ov::as_type_ptr<ov::op::v0::Constant>(node);
        };

        auto is_uncompressed_const = [&get_source_constant](const ov::Output<ov::Node>& v) {
            auto c = get_source_constant(v);
            if (!c) return false;
            const auto& et = c->get_element_type();
            return et == ov::element::f16 || et == ov::element::bf16 || et == ov::element::f32;
        };
        const bool is_mixed_precision_shared = moe_compressed &&
                                               is_uncompressed_const(sh_gate_w) &&
                                               is_uncompressed_const(sh_up_w) &&
                                               is_uncompressed_const(sh_down_w);

        // Attempt to decompose a weight decompression chain produced by NNCF:
        //   Const(u4/i4) → Convert(f16) → [Subtract(zp)] → Multiply(scale) → [Reshape] → [Convert(f32)]
        // Returns the raw weight, scale, optional zp, and group_size on success.
        struct DecompressedWeight {
            ov::Output<ov::Node> weight;
            ov::Output<ov::Node> scale;
            ov::Output<ov::Node> zp;       // empty if symmetric
            size_t group_size = 0;
            bool has_zp = false;
        };
        auto try_decompose_dequant = [](const ov::Output<ov::Node>& v)
                -> std::optional<DecompressedWeight> {
            auto node = v.get_node_shared_ptr();
            // 1. Optional trailing Convert (decompression, e.g. f16→f32)
            if (ov::as_type_ptr<ov::op::v0::Convert>(node))
                node = node->input_value(0).get_node_shared_ptr();
            // 2. Optional Reshape (e.g. [inter, groups, gs] → [inter, hidden])
            if (ov::as_type_ptr<ov::op::v1::Reshape>(node))
                node = node->input_value(0).get_node_shared_ptr();
            // 3. Multiply by scale
            auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(node);
            if (!mul) return std::nullopt;
            auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(
                mul->input_value(1).get_node_shared_ptr());
            if (!scale_const) return std::nullopt;
            auto next = mul->input_value(0).get_node_shared_ptr();
            // 4. Optional Subtract (asymmetric: weight_f16 - zp_f16)
            bool has_zp = false;
            std::shared_ptr<ov::op::v0::Constant> zp_const;
            if (auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(next)) {
                has_zp = true;
                auto zp_node = sub->input_value(1).get_node_shared_ptr();
                if (ov::as_type_ptr<ov::op::v0::Convert>(zp_node))
                    zp_node = zp_node->input_value(0).get_node_shared_ptr();
                zp_const = ov::as_type_ptr<ov::op::v0::Constant>(zp_node);
                if (!zp_const) return std::nullopt;
                next = sub->input_value(0).get_node_shared_ptr();
            }
            // 5. Convert from raw type to f16
            auto weight_cvt = ov::as_type_ptr<ov::op::v0::Convert>(next);
            if (!weight_cvt) return std::nullopt;
            // 6. Raw weight constant
            auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(
                weight_cvt->input_value(0).get_node_shared_ptr());
            if (!weight_const) return std::nullopt;
            auto wt = weight_const->get_element_type();
            if (wt != ov::element::u4 && wt != ov::element::i4 &&
                wt != ov::element::u8 && wt != ov::element::i8)
                return std::nullopt;
            // Derive group_size from weight shape
            auto shape = weight_const->get_shape();
            size_t group_size = (shape.size() == 3) ? shape[2] : shape.back();
            DecompressedWeight result;
            result.weight = weight_const;
            result.scale = scale_const;
            result.zp = has_zp ? ov::Output<ov::Node>(zp_const) : ov::Output<ov::Node>();
            result.group_size = group_size;
            result.has_zp = has_zp;
            return result;
        };

        // Detect compressed shared expert: all three shared weights have a valid
        // NNCF decompression chain (Const(u4/i4) → Convert → [Subtract] → Multiply → ...).
        auto gate_decomp = try_decompose_dequant(sh_gate_w);
        auto up_decomp   = try_decompose_dequant(sh_up_w);
        auto down_decomp = try_decompose_dequant(sh_down_w);
        const bool is_compressed_shared = moe_compressed &&
                                          !is_mixed_precision_shared &&
                                          gate_decomp && up_decomp && down_decomp;

        // When the weight is behind a decompression Convert (Const(bf16)→Convert(f32)→MatMul),
        // strip the Convert and pass the source Constant. If the source is bf16, insert an
        // explicit Convert(bf16→f16) because OpenCL kernels cannot consume bf16 directly.
        auto get_weight_input = [&get_source_constant](const ov::Output<ov::Node>& v) -> ov::Output<ov::Node> {
            if (ov::as_type_ptr<ov::op::v0::Convert>(v.get_node_shared_ptr())) {
                auto src = get_source_constant(v);
                if (src && src->get_element_type() == ov::element::bf16) {
                    return std::make_shared<ov::op::v0::Convert>(src, ov::element::f16);
                }
                if (src) return src;
            }
            return v;
        };

        OutputVector new_inputs;
        for (size_t i = 0; i < moe->get_input_size(); ++i) {
            new_inputs.push_back(moe->input_value(i));
        }

        std::shared_ptr<ov::Node> new_moe;
        if (is_mixed_precision_shared) {
            // Build 22-input MOECompressed: 12 base + 9 shared (3 wei + 6 dummy scale/zp) + gate_gate.
            // Dummy scale/zp are required as input placeholders so downstream patterns/plumbing
            // see a uniform input count regardless of shared-expert quantization.
            auto dummy_scalar = [](ov::element::Type et) {
                return ov::op::v0::Constant::create(et, ov::Shape{1}, {0.0f});
            };
            // Determine the effective element type for shared weights.
            // bf16 is not natively supported by OpenCL kernels, so map it to f16.
            const auto sh_wei_et_raw = get_source_constant(sh_gate_w)->get_element_type();
            const auto sh_wei_et = (sh_wei_et_raw == ov::element::bf16) ? ov::element::f16 : sh_wei_et_raw;

            auto sh_gate_w_in = get_weight_input(sh_gate_w);
            auto sh_up_w_in   = get_weight_input(sh_up_w);
            auto sh_down_w_in = get_weight_input(sh_down_w);

            // gate
            new_inputs.push_back(sh_gate_w_in);
            new_inputs.push_back(dummy_scalar(sh_wei_et));   // dummy scale
            new_inputs.push_back(dummy_scalar(sh_wei_et));   // dummy zp
            // up
            new_inputs.push_back(sh_up_w_in);
            new_inputs.push_back(dummy_scalar(sh_wei_et));
            new_inputs.push_back(dummy_scalar(sh_wei_et));
            // down
            new_inputs.push_back(sh_down_w_in);
            new_inputs.push_back(dummy_scalar(sh_wei_et));
            new_inputs.push_back(dummy_scalar(sh_wei_et));
            // gate_gate
            if (has_gating) {
                new_inputs.push_back(get_weight_input(sh_gate_gate_w));
            } else {
                size_t hidden_size = moe->get_output_partial_shape(0).rbegin()->get_length();
                new_inputs.push_back(ov::op::v0::Constant::create(
                    ov::element::f16, ov::Shape{hidden_size, 1}, std::vector<float>(hidden_size, 0.0f)));
            }

            // Update the config to advertise shared expert presence and precision.
            auto cfg = moe_compressed->get_config();
            cfg.num_shared_expert = 1;
            cfg.shared_weight_type = sh_wei_et;
            cfg.shared_group_size = 0;        // raw f16/f32 — no quantization
            cfg.shared_has_zp = false;
            // Derive shared expert intermediate size from the gate-weight constant shape
            // (logical layout is [hidden_size, inter_size] or [inter_size, hidden_size]).
            const auto& sh_shape = sh_gate_w_in.get_partial_shape();
            if (sh_shape.is_static() && cfg.hidden_size > 0) {
                const size_t total = ov::shape_size(sh_shape.to_shape());
                cfg.shared_inter_size = total / cfg.hidden_size;
            }

            new_moe = std::make_shared<ov::op::internal::MOECompressed>(new_inputs, cfg);
        } else if (is_compressed_shared) {
            // Build 22-input MOECompressed: 12 base + 9 shared (3 wei + 3 scale + 3 zp) + gate_gate.
            // Extract the raw weight/scale/zp from the dequant chain so FuseMOE3GemmCompressed
            // can match the 22-input pattern and the kernel receives compressed weights directly.
            auto& gd = gate_decomp.value();
            auto& ud = up_decomp.value();
            auto& dd = down_decomp.value();

            auto dummy_zp = [](ov::element::Type et) {
                return ov::op::v0::Constant::create(et, ov::Shape{1}, {0});
            };
            // gate
            new_inputs.push_back(gd.weight);
            new_inputs.push_back(gd.scale);
            new_inputs.push_back(gd.has_zp ? gd.zp : ov::Output<ov::Node>(dummy_zp(gd.weight.get_element_type())));
            // up
            new_inputs.push_back(ud.weight);
            new_inputs.push_back(ud.scale);
            new_inputs.push_back(ud.has_zp ? ud.zp : ov::Output<ov::Node>(dummy_zp(ud.weight.get_element_type())));
            // down
            new_inputs.push_back(dd.weight);
            new_inputs.push_back(dd.scale);
            new_inputs.push_back(dd.has_zp ? dd.zp : ov::Output<ov::Node>(dummy_zp(dd.weight.get_element_type())));
            // gate_gate (typically uncompressed — pass through get_weight_input)
            if (has_gating) {
                new_inputs.push_back(get_weight_input(sh_gate_gate_w));
            } else {
                size_t hidden_size = moe->get_output_partial_shape(0).rbegin()->get_length();
                new_inputs.push_back(ov::op::v0::Constant::create(
                    ov::element::f16, ov::Shape{hidden_size, 1}, std::vector<float>(hidden_size, 0.0f)));
            }

            auto cfg = moe_compressed->get_config();
            cfg.num_shared_expert = 1;
            cfg.shared_weight_type = gd.weight.get_element_type();
            cfg.shared_group_size = gd.group_size;
            cfg.shared_has_zp = gd.has_zp;
            const auto& sh_shape = gd.weight.get_partial_shape();
            if (sh_shape.is_static() && cfg.hidden_size > 0) {
                cfg.shared_inter_size = ov::shape_size(sh_shape.to_shape()) / cfg.hidden_size;
            }

            new_moe = std::make_shared<ov::op::internal::MOECompressed>(new_inputs, cfg);
        } else {
            // Backward-compatible behavior: append only weights (and gate_gate). Used by:
            //   * MOE (non-compressed sparse) path
            new_inputs.push_back(sh_gate_w);
            new_inputs.push_back(sh_up_w);
            new_inputs.push_back(sh_down_w);
            if (has_gating) {
                new_inputs.push_back(sh_gate_gate_w);
            } else {
                size_t hidden_size = moe->get_output_partial_shape(0).rbegin()->get_length();
                new_inputs.push_back(ov::op::v0::Constant::create(
                    ov::element::f16, ov::Shape{hidden_size, 1}, std::vector<float>(hidden_size, 0.0f)));
            }

            if (moe_compressed) {
                new_moe = std::make_shared<ov::op::internal::MOECompressed>(new_inputs, moe_compressed->get_config());
            } else {
                new_moe = std::make_shared<ov::op::internal::MOE>(new_inputs, moe->get_config());
            }
        }

        new_moe->set_friendly_name(root_node->get_friendly_name());
        ov::copy_runtime_info({moe_node, root_node}, new_moe);

        // If MoeOpFusion inserted a Convert between MOECompressed and Add,
        // wrap the replacement in a matching Convert to keep types consistent.
        std::shared_ptr<ov::Node> replacement = new_moe;
        if (pattern_map.count(moe_convert_m) > 0) {
            if (auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(
                    pattern_map.at(moe_convert_m).get_node_shared_ptr())) {
                replacement = std::make_shared<ov::op::v0::Convert>(new_moe, cvt->get_destination_type());
                replacement->set_friendly_name(root_node->get_friendly_name());
                new_moe->set_friendly_name(root_node->get_friendly_name() + "/fused_moe");
                ov::copy_runtime_info({moe_node, cvt, root_node}, replacement);
            }
        }
        ov::replace_node(root_node, replacement);

        // Disconnect the old MOECompressed from its inputs so the zombie node
        // doesn't inflate consumer counts of shared input nodes (routing, topk,
        // etc.). FuseMOE3GemmCompressed relies on consumers_count(1) predicates
        // on the routing subgraph, which would fail if the old node remains.
        auto dummy = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.0f});
        for (size_t i = 0; i < moe_node->get_input_size(); ++i) {
            moe_node->input(i).replace_source_output(dummy);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "FuseMOESharedExpert");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
