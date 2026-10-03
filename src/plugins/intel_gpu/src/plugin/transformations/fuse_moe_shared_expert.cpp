// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_shared_expert.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/constant_folding.hpp"
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

    // The fused MOE node may sit behind a Convert (e.g. f16 -> f32) left by the
    // preceding MOE fusions; the Add consumes the converted result.
    auto moe_converted_m = optional<ov::op::v0::Convert>({moe_m});
    auto moe_arm_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{moe_m, moe_converted_m});

    // Shared expert subgraph:
    //   shared_gate = MatMul(shared_hidden, shared_gate_weight)
    //   shared_swish = Swish(shared_gate)
    //   shared_up   = MatMul(shared_hidden, shared_up_weight)
    //   shared_mul  = Mul(shared_swish, shared_up)
    //   shared_down = MatMul(shared_mul, shared_down_weight)
    //   Optional gating: sigmoid(MatMul(shared_hidden, gate_gate_weight)) * shared_down
    //   Optional reshape before Add
    // Each shared-expert projection may consume the hidden states through its own
    // view adaptor (Reshape/Convert); bind them independently instead of forcing
    // one shared hidden node.
    auto make_hidden_arm = []() {
        return optional<ov::op::v0::Convert>({optional<ov::op::v1::Reshape>({any_input(), any_input()})});
    };
    auto shared_hidden_gate_m = make_hidden_arm();
    auto shared_hidden_up_m = make_hidden_arm();
    auto shared_hidden_gate_gate_m = make_hidden_arm();
    auto shared_gate_weight_m = any_input();
    auto shared_gate_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_gate_m, shared_gate_weight_m});
    auto shared_swish_m = wrap_type<ov::op::v4::Swish>({shared_gate_m});
    auto shared_up_weight_m = any_input();
    auto shared_up_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_up_m, shared_up_weight_m});
    // Multiply is commutative: handle both input orders
    auto shared_mul_m_1 = wrap_type<ov::op::v1::Multiply>({shared_swish_m, shared_up_m});
    auto shared_mul_m_2 = wrap_type<ov::op::v1::Multiply>({shared_up_m, shared_swish_m});
    auto shared_mul_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_mul_m_1, shared_mul_m_2});
    auto shared_down_weight_m = any_input();
    auto shared_down_m = wrap_type<ov::op::v0::MatMul>({shared_mul_m, shared_down_weight_m});

    // Optional sigmoid gating: sigmoid(MatMul(hidden, gate_gate)) * down
    auto shared_gate_gate_wei_m = any_input();
    auto shared_gate_gate_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_gate_gate_m, shared_gate_gate_wei_m});
    auto shared_gate_sigmoid_m = wrap_type<ov::op::v0::Sigmoid>({shared_gate_gate_m});
    // Multiply is commutative: handle both input orders
    auto shared_expert_gated_m_1 = wrap_type<ov::op::v1::Multiply>({shared_gate_sigmoid_m, shared_down_m});
    auto shared_expert_gated_m_2 = wrap_type<ov::op::v1::Multiply>({shared_down_m, shared_gate_sigmoid_m});
    auto shared_expert_gated_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_expert_gated_m_1, shared_expert_gated_m_2});
    auto shared_expert_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_down_m, shared_expert_gated_m});
    auto shared_expert_reshaped_m = optional<ov::op::v1::Reshape>({shared_expert_m, any_input()});

    // Root: Add(MOE, SharedExpert) or Add(SharedExpert, MOE)
    auto add_1 = wrap_type<ov::op::v1::Add>({moe_arm_m, shared_expert_reshaped_m});
    auto add_2 = wrap_type<ov::op::v1::Add>({shared_expert_reshaped_m, moe_arm_m});
    auto root = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{add_1, add_2});

    // Extract the plain constant feeding a (possibly converted) weight input.
    // Compressed weights arrive as Const(w) -> [Convert] -> Subtract(zp) -> Multiply(scale) -> [Reshape] -> [Convert]
    auto get_constant = [](const ov::Output<ov::Node>& value) -> std::shared_ptr<ov::op::v0::Constant> {
        auto node = value.get_node_shared_ptr();
        if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
            node = convert->input(0).get_source_output().get_node_shared_ptr();
        }
        return ov::as_type_ptr<ov::op::v0::Constant>(node);
    };

    // Decompose a compressed-weight dequant chain feeding a MatMul weight port into
    // {weight, zp, scale} plain constants. Returns false when the weight is not
    // a group-quantized dequant chain (caller then skips the fusion).
    // Captured by value: the callback outlives this constructor's scope.
    auto get_compressed_weight = [get_constant](const ov::Output<ov::Node>& matmul_weight_input,
                                                std::shared_ptr<ov::op::v0::Constant>& w,
                                                std::shared_ptr<ov::op::v0::Constant>& zp,
                                                std::shared_ptr<ov::op::v0::Constant>& scale) -> bool {
        // Peel layout adaptors (Reshape / Convert) above the dequant Multiply.
        auto node = matmul_weight_input.get_node_shared_ptr();
        for (int hops = 0; hops < 4; ++hops) {
            if (ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
                if (node->inputs().size() != 2) {
                    return false;
                }
                node = node->input(0).get_source_output().get_node_shared_ptr();
                continue;
            }
            if (ov::as_type_ptr<ov::op::v0::Convert>(node)) {
                if (node->inputs().size() != 1) {
                    return false;
                }
                node = node->input(0).get_source_output().get_node_shared_ptr();
                continue;
            }
            break;
        }
        auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(node);
        if (!mul) {
            return false;
        }
        // Multiply(dequant, scale) or Multiply(scale, dequant)
        auto lhs = mul->input(0).get_source_output();
        auto rhs = mul->input(1).get_source_output();
        auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(lhs.get_node_shared_ptr());
        auto scale_node = rhs.get_node_shared_ptr();
        if (!sub) {
            sub = ov::as_type_ptr<ov::op::v1::Subtract>(rhs.get_node_shared_ptr());
            scale_node = lhs.get_node_shared_ptr();
            if (!sub) {
                return false;
            }
        }
        // Subtract(Convert(w) | w, Convert(zp) | zp); port order is export-dependent,
        // disambiguate by element count: weights [N, G, 128] vs per-group zp [N, G, 1].
        auto c0 = get_constant(sub->input(0).get_source_output());
        auto c1 = get_constant(sub->input(1).get_source_output());
        if (!c0 || !c1) {
            return false;
        }
        // The weight ([N, G, 128]) has more elements than the per-group zero point ([N, G, 1]).
        if (ov::shape_size(c0->get_output_shape(0)) >= ov::shape_size(c1->get_output_shape(0))) {
            w = c0;
            zp = c1;
        } else {
            w = c1;
            zp = c0;
        }
        auto scale_const = get_constant(scale_node->output(0));
        if (!scale_const) {
            scale_const = ov::as_type_ptr<ov::op::v0::Constant>(scale_node);
        }
        if (!w || !zp || !scale_const) {
            return false;
        }
        scale = scale_const;
        return true;
    };

    // MatMul input order is export-dependent: resolve the weight port by trying both.
    // Captured by value: the callback outlives this constructor's scope.
    auto resolve_compressed_weight = [get_compressed_weight](const std::shared_ptr<ov::op::v0::MatMul>& mm,
                                                             std::shared_ptr<ov::op::v0::Constant>& w,
                                                             std::shared_ptr<ov::op::v0::Constant>& zp,
                                                             std::shared_ptr<ov::op::v0::Constant>& scale) -> bool {
        for (size_t port = 0; port < 2; ++port) {
            if (get_compressed_weight(mm->input(port).get_source_output(), w, zp, scale)) {
                return true;
            }
        }
        return false;
    };

    // The GPU shared-expert primitive consumes scale/zp through oneDNN's native
    // weights-decompression matmul, which requires group-major {groups, oc} byte
    // order. Optimum exports store them oc-major as [oc, groups, 1] (same as the
    // routed experts, whose custom kernels index them directly). Transpose the
    // shared scale/zp constants to [groups, oc] before fusing. Returns nullptr
    // when the constant's layout is not supported.
    auto transpose_group_constant = [](const std::shared_ptr<ov::op::v0::Constant>& c) -> std::shared_ptr<ov::op::v0::Constant> {
        const auto& shape = c->get_output_shape(0);
        // Expect oc-major [oc, groups] or [oc, groups, 1]; a leading expert dim or a
        // non-trivial trailing dim is not a shared-expert scale/zp layout.
        if (shape.size() < 2 || shape.size() > 3 || (shape.size() == 3 && shape[2] != 1)) {
            return nullptr;
        }
        const size_t N = shape[0], G = shape[1];
        if (N <= 1 || G <= 1) {
            return nullptr;
        }
        const auto et = c->get_output_element_type(0);
        const size_t bw = et.bitwidth();
        if (bw == 16 || bw == 32) {
            const size_t esz = et.size();
            const auto* src = static_cast<const uint8_t*>(c->get_data_ptr());
            std::vector<uint8_t> dst(N * G * esz, 0);
            for (size_t n = 0; n < N; ++n) {
                for (size_t g = 0; g < G; ++g) {
                    std::memcpy(dst.data() + (g * N + n) * esz, src + (n * G + g) * esz, esz);
                }
            }
            return std::make_shared<ov::op::v0::Constant>(et, ov::Shape{G, N}, dst.data());
        }
        if (bw == 8) {
            const auto* src = static_cast<const uint8_t*>(c->get_data_ptr());
            std::vector<uint8_t> dst(N * G, 0);
            for (size_t n = 0; n < N; ++n) {
                for (size_t g = 0; g < G; ++g) {
                    dst[g * N + n] = src[n * G + g];
                }
            }
            return std::make_shared<ov::op::v0::Constant>(et, ov::Shape{G, N}, dst.data());
        }
        if (bw == 4) {
            // u4/i4 pack two elements per byte, even index in the low nibble
            // (see Constant::set_unused_bits masking the last byte with 0x0F).
            const size_t total = N * G;
            const auto* src = static_cast<const uint8_t*>(c->get_data_ptr());
            auto read = [&](size_t i) -> uint8_t {
                const size_t byte = i >> 1;
                return (i & 1) ? static_cast<uint8_t>(src[byte] >> 4) : static_cast<uint8_t>(src[byte] & 0xF);
            };
            std::vector<uint8_t> dst((total + 1) / 2, 0);
            for (size_t n = 0; n < N; ++n) {
                for (size_t g = 0; g < G; ++g) {
                    const size_t i_new = g * N + n;
                    const uint8_t v = read(n * G + g);
                    if (i_new & 1) {
                        dst[i_new >> 1] |= static_cast<uint8_t>(v << 4);
                    } else {
                        dst[i_new >> 1] |= v;
                    }
                }
            }
            return std::make_shared<ov::op::v0::Constant>(et, ov::Shape{G, N}, dst.data());
        }
        return nullptr;
    };

    // OV_CAPTURE_CPY_AND_THIS starts with a default by-copy capture ('='), so the
    // helper lambdas used below are captured by value: the matcher callback runs
    // long after this constructor has returned, and by-reference captures of the
    // scope-local lambdas above would dangle.
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto root_node = pattern_map.at(root).get_node_shared_ptr();
        auto moe_node = pattern_map.at(moe_m).get_node_shared_ptr();
        auto moe = ov::as_type_ptr<ov::op::internal::MOE>(moe_node);
        if (!moe || transformation_callback(root_node)) {
            return false;
        }
        auto moe_compressed = ov::as_type_ptr<ov::op::internal::MOECompressed>(moe_node);

        if (moe_compressed) {
            // Compressed routed experts: the GPU primitive expects the shared expert
            // weights in compressed form at inputs 12..21:
            //   12-14 shared gate (w, scale, zp), 15-17 shared up (w, scale, zp),
            //   18-20 shared down (w, scale, zp), 21 shared gate_gate weight (plain).
            // The primitive always applies sigmoid(gate_gate @ hidden) to the shared
            // expert output, so only sigmoid-gated shared experts can be fused.
            if (pattern_map.count(shared_gate_sigmoid_m) == 0) {
                return false;
            }
            auto gate_mm = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(shared_gate_m).get_node_shared_ptr());
            auto up_mm = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(shared_up_m).get_node_shared_ptr());
            auto down_mm = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(shared_down_m).get_node_shared_ptr());
            if (!gate_mm || !up_mm || !down_mm) {
                return false;
            }
            std::array<std::shared_ptr<ov::op::v0::Constant>, 3> w_c, zp_c, scale_c;
            std::array<std::shared_ptr<ov::op::v0::MatMul>, 3> shared_mms = {gate_mm, up_mm, down_mm};
            for (size_t i = 0; i < 3; ++i) {
                if (!resolve_compressed_weight(shared_mms[i], w_c[i], zp_c[i], scale_c[i])) {
                    // Shared expert weights are not compressed - keep the original
                    // unfused subgraph rather than guessing an unsupported layout.
                    return false;
                }
                // The shared-expert GEMMs run through oneDNN weights-decompression with
                // group scales; per-channel (single-group) layouts are not handled here.
                // Expect oc-major [oc, groups{, 1}] with groups >= 2.
                const auto& sc = scale_c[i]->get_output_shape(0);
                if (sc.size() < 2 || sc.size() > 3 || (sc.size() == 3 && sc[2] != 1) || sc[1] < 2) {
                    return false;
                }
            }
            const auto& cfg = moe_compressed->get_config();
            // The kernel asserts the shared expert inter size equals the routed one and
            // derives the hidden size from the down weight; skip instead of crashing.
            const size_t shared_inter = w_c[0]->get_output_shape(0)[0];
            const size_t shared_hidden = w_c[2]->get_output_shape(0)[0];
            if ((cfg.inter_size != 0 && shared_inter != cfg.inter_size) ||
                (cfg.hidden_size != 0 && shared_hidden != cfg.hidden_size)) {
                return false;
            }

            auto gg_mm = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(shared_gate_gate_m).get_node_shared_ptr());
            if (!gg_mm) {
                return false;
            }
            std::shared_ptr<ov::op::v0::Constant> gate_gate_const = nullptr;
            // gate_gate runs through a plain GEMM inside the primitive; fold its
            // dequant chain (it is tiny: [1, hidden]) into a single constant.
            for (size_t port = 0; port < 2 && !gate_gate_const; ++port) {
                auto port_node = gg_mm->input(port).get_source_output().get_node_shared_ptr();
                gate_gate_const = get_constant(port_node->output(0));
                if (!gate_gate_const) {
                    // maybe the direct producer is a Convert/Reshape over the chain
                    auto p = port_node;
                    for (int hops = 0; hops < 3 && p; ++hops) {
                        if (ov::as_type_ptr<ov::op::v1::Reshape>(p) || ov::as_type_ptr<ov::op::v0::Convert>(p)) {
                            if (p->inputs().empty()) {
                                break;
                            }
                            p = p->input(0).get_source_output().get_node_shared_ptr();
                            gate_gate_const = ov::as_type_ptr<ov::op::v0::Constant>(p);
                            if (gate_gate_const) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
            if (!gate_gate_const) {
                // Locate the port that peels down to a dequant Multiply (pure-constant
                // subtree) and fold only that. The activation port can trace back
                // through the whole model, so it must never be cloned.
                auto peel_to_dequant = [](const ov::Output<ov::Node>& port_out) -> std::shared_ptr<ov::op::v1::Multiply> {
                    auto node = port_out.get_node_shared_ptr();
                    for (int hops = 0; hops < 4; ++hops) {
                        if (ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
                            if (node->inputs().size() != 2) {
                                return nullptr;
                            }
                            node = node->input(0).get_source_output().get_node_shared_ptr();
                        } else if (ov::as_type_ptr<ov::op::v0::Convert>(node)) {
                            if (node->inputs().size() != 1) {
                                return nullptr;
                            }
                            node = node->input(0).get_source_output().get_node_shared_ptr();
                        } else {
                            break;
                        }
                    }
                    auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(node);
                    if (!mul) {
                        return nullptr;
                    }
                    // both operands must resolve to Constants (dequant operands)
                    for (const auto& i : mul->inputs()) {
                        auto n = i.get_source_output().get_node_shared_ptr();
                        bool ok = ov::as_type_ptr<ov::op::v0::Constant>(n) != nullptr ||
                                  ov::as_type_ptr<ov::op::v1::Subtract>(n) != nullptr;
                        if (!ok) {
                            return nullptr;
                        }
                    }
                    return mul;
                };
                std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, int)> clone_chain =
                    [&](const ov::Output<ov::Node>& out, int depth) -> ov::Output<ov::Node> {
                    auto n = out.get_node_shared_ptr();
                    if (ov::as_type_ptr<ov::op::v0::Constant>(n) || depth > 16) {
                        return out;
                    }
                    OutputVector new_ins;
                    for (const auto& i : n->inputs()) {
                        new_ins.push_back(clone_chain(i.get_source_output(), depth + 1));
                    }
                    auto cloned = n->clone_with_new_inputs(new_ins);
                    return cloned->output(0);
                };
                for (size_t port = 0; port < 2 && !gate_gate_const; ++port) {
                    auto deq_mul = peel_to_dequant(gg_mm->input(port).get_source_output());
                    if (!deq_mul) {
                        continue;
                    }
                    try {
                        auto cloned_out = clone_chain(deq_mul->output(0), 0);
                        auto mini = std::make_shared<ov::Model>(ov::OutputVector{cloned_out},
                                                                ov::ParameterVector{},
                                                                "gate_gate_fold");
                        ov::pass::ConstantFolding().run_on_model(mini);
                        gate_gate_const = ov::as_type_ptr<ov::op::v0::Constant>(
                            mini->get_results()[0]->input_value(0).get_node_shared_ptr());
                    } catch (const std::exception&) {
                        gate_gate_const = nullptr;
                    }
                }
                if (!gate_gate_const) {
                    return false;
                }
            }

            // Transpose scale/zp to the group-major layout required by the
            // oneDNN weights-decompression GEMMs used for the shared expert.
            auto scale_t = std::array<std::shared_ptr<ov::op::v0::Constant>, 3>{transpose_group_constant(scale_c[0]),
                                                                                 transpose_group_constant(scale_c[1]),
                                                                                 transpose_group_constant(scale_c[2])};
            auto zp_t = std::array<std::shared_ptr<ov::op::v0::Constant>, 3>{transpose_group_constant(zp_c[0]),
                                                                              transpose_group_constant(zp_c[1]),
                                                                              transpose_group_constant(zp_c[2])};
            for (size_t i = 0; i < 3; ++i) {
                if (!scale_t[i] || !zp_t[i]) {
                    return false;
                }
            }

            OutputVector new_inputs;
            for (size_t i = 0; i < moe_compressed->get_input_size(); ++i) {
                new_inputs.push_back(moe_compressed->input_value(i));
            }
            new_inputs.push_back(w_c[0]->output(0));        // 12 shared gate weight
            new_inputs.push_back(scale_t[0]->output(0));    // 13 shared gate scale
            new_inputs.push_back(zp_t[0]->output(0));       // 14 shared gate zp
            new_inputs.push_back(w_c[1]->output(0));        // 15 shared up weight
            new_inputs.push_back(scale_t[1]->output(0));    // 16 shared up scale
            new_inputs.push_back(zp_t[1]->output(0));       // 17 shared up zp
            new_inputs.push_back(w_c[2]->output(0));        // 18 shared down weight
            new_inputs.push_back(scale_t[2]->output(0));    // 19 shared down scale
            new_inputs.push_back(zp_t[2]->output(0));       // 20 shared down zp
            new_inputs.push_back(gate_gate_const->output(0));  // 21 shared gate_gate weight

            auto config = moe_compressed->get_config();
            config.num_shared_expert = 1;
            auto new_moe = std::make_shared<ov::op::internal::MOECompressed>(new_inputs, config);
            auto out_et = root_node->get_output_element_type(0);
            std::shared_ptr<ov::Node> replacement = new_moe;
            if (new_moe->get_output_element_type(0) != out_et) {
                replacement = std::make_shared<ov::op::v0::Convert>(new_moe, out_et);
            }
            replacement->set_friendly_name(root_node->get_friendly_name());
            ov::copy_runtime_info({moe_node, root_node}, replacement);
            ov::replace_node(root_node, replacement);
            return true;
        }

        // Append shared expert weights to existing MOE inputs.
        OutputVector new_inputs;
        for (size_t i = 0; i < moe->get_input_size(); ++i) {
            new_inputs.push_back(moe->input_value(i));
        }
        new_inputs.push_back(pattern_map.at(shared_gate_weight_m));  // shared gate weight
        new_inputs.push_back(pattern_map.at(shared_up_weight_m));    // shared up weight
        new_inputs.push_back(pattern_map.at(shared_down_weight_m));  // shared down weight

        // any_input() may be spuriously bound on the non-gating branch — use sigmoid as ground truth.
        bool has_gating = pattern_map.count(shared_gate_sigmoid_m) > 0;
        if (has_gating) {
            new_inputs.push_back(pattern_map.at(shared_gate_gate_wei_m));
        } else {
            // No gate_gate: dummy keeps input count consistent.
            size_t hidden_size = moe->get_output_partial_shape(0).rbegin()->get_length();
            new_inputs.push_back(
                ov::op::v0::Constant::create(ov::element::f16, ov::Shape{hidden_size, 1}, std::vector<float>(hidden_size, 0.0f)));
        }

        auto new_moe = std::make_shared<ov::op::internal::MOE>(new_inputs, moe->get_config());
        new_moe->set_friendly_name(root_node->get_friendly_name());
        ov::copy_runtime_info({moe_node, root_node}, new_moe);
        ov::replace_node(root_node, new_moe);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "FuseMOESharedExpert");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
