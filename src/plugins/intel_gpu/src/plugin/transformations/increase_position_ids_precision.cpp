// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_position_ids_precision.hpp"

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/variable_extension.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

namespace ov::intel_gpu {

CollectSinCosConstantsForPrecisionPreservation::CollectSinCosConstantsForPrecisionPreservation(
    const std::shared_ptr<ConstantSet>& constants)
    : m_constants(constants) {
    OPENVINO_ASSERT(m_constants, "CollectSinCosConstantsForPrecisionPreservation requires a shared Constant set");
}

bool CollectSinCosConstantsForPrecisionPreservation::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_constants->clear();
    std::unordered_set<const ov::Node*> visited;
    ov::OutputVector pending;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type_any_of<ov::op::v0::Sin, ov::op::v0::Cos>(node)) {
            pending.push_back(node->input_value(0));
        }
    }

    while (!pending.empty()) {
        const auto output = pending.back();
        pending.pop_back();
        if (!output.get_element_type().is_real()) {
            continue;
        }
        const auto node = output.get_node_shared_ptr();
        if (!visited.insert(node.get()).second) {
            continue;
        }
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            if (output.get_element_type() == ov::element::f32) {
                m_constants->insert(node.get());
            }
            continue;
        }

        // Follow floating-point dependencies without an op allowlist, including through MatMul.
        // Integer/bool inputs (e.g. shapes, axes and indices) end the path even if derived from floats.
        // Only outer-graph input edges are followed; subgraph bodies are not traversed.
        for (const auto& input : node->input_values()) {
            if (input.get_element_type().is_real()) {
                pending.push_back(input);
            }
        }
    }

    return false;
}

// Shared by the GPU pipeline and accuracy regression tests to avoid duplicating the preservation policy.
bool skip_constant_precision_preservation(
    const std::shared_ptr<const ov::Node>& node,
    const CollectSinCosConstantsForPrecisionPreservation::ConstantSet& sincos_constants) {
    const auto targets = node->get_output_target_inputs(0);
    const ov::Node* next_node = node.get();
    if (!targets.empty()) {
        next_node = targets.begin()->get_node();
    }
    if (ov::is_type<ov::op::v0::Convert>(next_node)) {
        const auto convert_targets = next_node->get_output_target_inputs(0);
        if (!convert_targets.empty()) {
            next_node = convert_targets.begin()->get_node();
        }
    }
    if (ov::is_type_any_of<ov::op::v0::MatMul,
                           ov::op::internal::MOE,
                           ov::op::internal::GatherMatmulCompressed,
                           ov::op::internal::GroupedMatMulCompressed>(next_node)) {
        return false;
    }
    return sincos_constants.count(node.get()) == 0;
}

IncreasePositionIdsPrecisionForRoPE::IncreasePositionIdsPrecisionForRoPE() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    auto gemm_or_matmul = wrap_type<ov::intel_gpu::op::Gemm, ov::op::v0::MatMul>();
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({gemm_or_matmul, any_input()});
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({gemm_or_matmul, any_input()});
    auto concat_input = std::make_shared<Or>(OutputVector{gemm_or_matmul, transpose_m, reshape_m});
    auto concat = wrap_type<ov::op::v0::Concat>({concat_input, concat_input});
    auto sin = wrap_type<ov::op::v0::Sin>({concat});
    auto cos = wrap_type<ov::op::v0::Cos>({concat});

    auto sin_reshape = wrap_type<ov::op::v1::Reshape>({sin, wrap_type<ov::op::v0::Constant>()});
    auto sin_squeeze = wrap_type<ov::op::v0::Squeeze>({sin, wrap_type<ov::op::v0::Constant>()});
    auto sin_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({sin, wrap_type<ov::op::v0::Constant>()});
    // Adjust scale factor to positional embedding for LongRoPE
    auto sin_multiply = wrap_type<ov::op::v1::Multiply>({sin, wrap_type<ov::op::v0::Constant>()});
    auto sin_multiply_reshape = wrap_type<ov::op::v1::Reshape>({sin_multiply, wrap_type<ov::op::v0::Constant>()});

    auto cos_reshape = wrap_type<ov::op::v1::Reshape>({cos, wrap_type<ov::op::v0::Constant>()});
    auto cos_squeeze = wrap_type<ov::op::v0::Squeeze>({cos, wrap_type<ov::op::v0::Constant>()});
    auto cos_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({cos, wrap_type<ov::op::v0::Constant>()});
    // Adjust scale factor to positional embedding for LongRoPE
    auto cos_multiply = wrap_type<ov::op::v1::Multiply>({cos, wrap_type<ov::op::v0::Constant>()});
    auto cos_multiply_reshape = wrap_type<ov::op::v1::Reshape>({cos_multiply, wrap_type<ov::op::v0::Constant>()});

    auto sin_slice = wrap_type<ov::op::v1::StridedSlice>({sin, any_input(), any_input(), any_input()});
    auto sin_gather = wrap_type<ov::op::v8::Gather>({sin_slice, any_input(), any_input()});
    auto sin_unsqueeze2 = wrap_type<ov::op::v0::Unsqueeze>({sin_gather, wrap_type<ov::op::v0::Constant>()});

    auto cos_slice = wrap_type<ov::op::v1::StridedSlice>({cos, any_input(), any_input(), any_input()});
    auto cos_gather = wrap_type<ov::op::v8::Gather>({cos_slice, any_input(), any_input()});
    auto cos_unsqueeze2 = wrap_type<ov::op::v0::Unsqueeze>({cos_gather, wrap_type<ov::op::v0::Constant>()});

    auto rope_sin_input = std::make_shared<Or>(OutputVector{sin_reshape, sin_squeeze, sin_unsqueeze, sin_unsqueeze2, sin_multiply_reshape, sin});
    auto rope_cos_input = std::make_shared<Or>(OutputVector{cos_reshape, cos_squeeze, cos_unsqueeze, cos_unsqueeze2, cos_multiply_reshape, cos});

    auto rope = wrap_type<ov::op::internal::RoPE>({any_input(), rope_cos_input, rope_sin_input});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(gemm_or_matmul).get_node_shared_ptr());
        auto cos_node = ov::as_type_ptr<ov::op::v0::Cos>(pattern_map.at(cos).get_node_shared_ptr());
        auto sin_node = ov::as_type_ptr<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr());
        auto rope_node = pattern_map.at(rope).get_node_shared_ptr();

        if (!matmul_node || transformation_callback(matmul_node)) {
            return false;
        }

        const auto desired_et = ov::element::f32;
        const auto original_et = matmul_node->get_output_element_type(0);
        if (original_et == desired_et) {
            return false;
        }

        // Step 1: Ensure MatMul inputs are f32
        size_t input_idx = 0;
        bool is_changed = insert_converts_before_if_needed(matmul_node, desired_et, input_idx);

        // Step 2: Insert restore converts only if RoPE expects non-f32 precision.
        if (is_changed) {
            size_t output_idx = 0;
            auto rope_cos_et = rope_node->get_input_element_type(1);
            if (rope_cos_et != desired_et) {
                insert_converts_after_if_needed(cos_node, rope_cos_et, output_idx);
            }
            auto rope_sin_et = rope_node->get_input_element_type(2);
            if (rope_sin_et != desired_et) {
                insert_converts_after_if_needed(sin_node, rope_sin_et, output_idx);
            }
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope, "IncreasePositionIdsPrecisionForRoPE");
    this->register_matcher(m, callback);
}

IncreasePositionIdsPrecisionForQwen25VL::IncreasePositionIdsPrecisionForQwen25VL() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    // Qwen2.5-VL RoPE pattern:
    // position_ids -> Convert_to_i32 -> Unsqueeze -> Convert_to_f16 -> MatMul -> Transpose -> Concat -> Sin/Cos -> ... -> RoPE
    auto position_ids = any_input();
    auto convert_to_i32 = wrap_type<ov::op::v0::Convert>({position_ids});
    auto unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({convert_to_i32, any_input()});
    auto convert_to_f16 = wrap_type<ov::op::v0::Convert>({unsqueeze});

    auto broadcast_freq = wrap_type<ov::op::v3::Broadcast>({any_input(), any_input()});
    auto matmul = wrap_type<ov::op::v0::MatMul>({broadcast_freq, convert_to_f16});
    auto transpose = wrap_type<ov::op::v1::Transpose>({matmul, any_input()});
    auto concat = wrap_type<ov::op::v0::Concat>({transpose, transpose});

    auto sin = wrap_type<ov::op::v0::Sin>({concat});
    auto cos = wrap_type<ov::op::v0::Cos>({concat});
    auto sin_split = wrap_type<ov::op::v1::VariadicSplit>({sin, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()});
    auto cos_split = wrap_type<ov::op::v1::VariadicSplit>({cos, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()});
    auto sin_gather = wrap_type<ov::op::v8::Gather>({sin_split, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()});
    auto cos_gather = wrap_type<ov::op::v8::Gather>({cos_split, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()});
    auto sin_concat = wrap_type<ov::op::v0::Concat>({sin_gather, sin_gather, sin_gather, sin_gather, sin_gather, sin_gather});
    auto cos_concat = wrap_type<ov::op::v0::Concat>({cos_gather, cos_gather, cos_gather, cos_gather, cos_gather, cos_gather});
    auto sin_unsequeeze = wrap_type<ov::op::v0::Unsqueeze>({sin_concat, wrap_type<ov::op::v0::Constant>()});
    auto cos_unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({cos_concat, wrap_type<ov::op::v0::Constant>()});
    auto rope = wrap_type<ov::op::internal::RoPE>({any_input(), cos_unsqueeze, sin_unsequeeze});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto convert_node = ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(convert_to_f16).get_node_shared_ptr());
        auto broadcast_node = pattern_map.at(broadcast_freq).get_node_shared_ptr();
        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul).get_node_shared_ptr());
        auto sin_node = ov::as_type_ptr<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr());
        auto cos_node = ov::as_type_ptr<ov::op::v0::Cos>(pattern_map.at(cos).get_node_shared_ptr());

        if (!convert_node || !matmul_node || transformation_callback(convert_node)) {
            return false;
        }

        const auto desired_et = ov::element::f32;
        const auto original_et = convert_node->get_output_element_type(0);
        if (original_et == desired_et) {
            return false;
        }

        // Check if input is integer type (position_ids should be i32 or i64)
        auto input_et = convert_node->input_value(0).get_element_type();
        if (!input_et.is_integral()) {
            return false;
        }

        // 1. Change Convert output from f16 to f32 (position_ids path)
        auto new_convert = std::make_shared<ov::op::v0::Convert>(convert_node->input_value(0), desired_et);
        new_convert->set_friendly_name(convert_node->get_friendly_name() + "_increase_precision");
        copy_runtime_info(convert_node, new_convert);
        ov::replace_node(convert_node, new_convert);

        // 2. Insert Convert(f16->f32) after Broadcast (freq path) to match MatMul types
        //    MatMul needs both inputs to be the same type (f32)
        if (broadcast_node->get_output_element_type(0) != desired_et) {
            auto broadcast_to_f32 = std::make_shared<ov::op::v0::Convert>(broadcast_node->output(0), desired_et);
            broadcast_to_f32->set_friendly_name(broadcast_node->get_friendly_name() + "_to_f32");
            copy_runtime_info(broadcast_node, broadcast_to_f32);
            // Replace MatMul's input 0 (Broadcast) with the new Convert
            matmul_node->input(0).replace_source_output(broadcast_to_f32->output(0));
        }

        // 3. Insert Convert(f32->f16) after Sin/Cos to restore original precision
        size_t output_idx = 0;
        insert_converts_after_if_needed(sin_node, original_et, output_idx);
        insert_converts_after_if_needed(cos_node, original_et, output_idx);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope, "IncreasePositionIdsPrecisionForQwen25VL");
    this->register_matcher(m, callback);
}

IncreasePositionIdsPrecisionForQwen3VL::IncreasePositionIdsPrecisionForQwen3VL() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    // Qwen3-VL RoPE pattern:
    // position_ids -> Convert(i64->i32) -> Reshape(unsqueeze) -> Convert(i32->f16) -> MatMul(Broadcast, Convert)
    //   -> Reshape(transpose) -> Gather(select_channel) x3 -> ScatterNDUpdate chain -> Reshape -> Concat(self,self)
    //   -> Sin/Cos -> Reshape(unsqueeze) -> RoPE
    //
    // The intermediate path between MatMul and Sin/Cos is too complex to pattern-match,
    // so we match the beginning (up to MatMul->Reshape/Transpose) and use graph traversal to find downstream Sin/Cos.
    // Key difference from Qwen2.5-VL: Unsqueeze is decomposed to Reshape.
    auto position_ids = any_input();
    auto convert_to_i32 = wrap_type<ov::op::v0::Convert>({position_ids});
    auto reshape_0 = optional<ov::op::v1::Reshape>({convert_to_i32, wrap_const()});
    auto stridedslice_0 = optional<ov::op::v1::StridedSlice>({reshape_0, any_input(), any_input(), any_input()});
    auto reshape_unsqueeze = wrap_type<ov::op::v1::Reshape>({stridedslice_0, wrap_const()});
    auto unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({stridedslice_0, any_input()});
    auto reshape_or_unsqueeze = std::make_shared<Or>(OutputVector{reshape_unsqueeze, unsqueeze});
    auto convert_to_f16 = wrap_type<ov::op::v0::Convert>({reshape_or_unsqueeze});

    auto broadcast_freq = wrap_type<ov::op::v3::Broadcast>({any_input(), any_input()});
    auto matmul = wrap_type<ov::op::v0::MatMul>({broadcast_freq, convert_to_f16});

    auto reshape = wrap_type<ov::op::v1::Reshape>({matmul, any_input()});
    auto transpose = wrap_type<ov::op::v1::Transpose>({matmul, any_input()});
    auto reshape_or_transpose = std::make_shared<Or>(OutputVector{reshape, transpose});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto convert_node = ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(convert_to_f16).get_node_shared_ptr());
        auto broadcast_node = pattern_map.at(broadcast_freq).get_node_shared_ptr();
        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul).get_node_shared_ptr());

        if (!convert_node || !matmul_node || transformation_callback(convert_node)) {
            return false;
        }

        const auto desired_et = ov::element::f32;
        const auto original_et = convert_node->get_output_element_type(0);
        if (original_et == desired_et) {
            return false;
        }

        // Verify input is integer type (position_ids should be i32 or i64)
        auto input_et = convert_node->input_value(0).get_element_type();
        if (!input_et.is_integral()) {
            return false;
        }

        // Walk forward from MatMul to find Sin and Cos nodes through the
        // Reshape -> Gather -> ScatterNDUpdate -> Reshape -> Concat chain.
        // Only follow floating-point data outputs to stay on the data path.
        std::shared_ptr<ov::op::v0::Sin> sin_node;
        std::shared_ptr<ov::op::v0::Cos> cos_node;

        std::vector<ov::Node*> stack;
        std::set<ov::Node*> visited;
        stack.push_back(matmul_node.get());
        constexpr size_t max_nodes = 30;
        size_t nodes_visited = 0;

        while (!stack.empty() && nodes_visited < max_nodes && (!sin_node || !cos_node)) {
            auto* current = stack.back();
            stack.pop_back();

            for (auto& output : current->outputs()) {
                if (!output.get_element_type().is_real()) {
                    continue;
                }
                for (const auto& target_input : output.get_target_inputs()) {
                    auto consumer = target_input.get_node()->shared_from_this();
                    if (!visited.insert(consumer.get()).second) {
                        continue;
                    }
                    nodes_visited++;

                    if (auto sin_ptr = ov::as_type_ptr<ov::op::v0::Sin>(consumer)) {
                        sin_node = sin_ptr;
                    } else if (auto cos_ptr = ov::as_type_ptr<ov::op::v0::Cos>(consumer)) {
                        cos_node = cos_ptr;
                    } else {
                        stack.push_back(consumer.get());
                    }
                }
            }
        }

        if (!sin_node || !cos_node) {
            return false;
        }

        // 1. Change Convert output from f16 to f32 (position_ids path)
        auto new_convert = std::make_shared<ov::op::v0::Convert>(convert_node->input_value(0), desired_et);
        new_convert->set_friendly_name(convert_node->get_friendly_name() + "_increase_precision");
        copy_runtime_info(convert_node, new_convert);
        ov::replace_node(convert_node, new_convert);

        // 2. Insert Convert(f16->f32) after Broadcast (freq path) to match MatMul types
        if (broadcast_node->get_output_element_type(0) != desired_et) {
            auto broadcast_to_f32 = std::make_shared<ov::op::v0::Convert>(broadcast_node->output(0), desired_et);
            broadcast_to_f32->set_friendly_name(broadcast_node->get_friendly_name() + "_to_f32");
            copy_runtime_info(broadcast_node, broadcast_to_f32);
            matmul_node->input(0).replace_source_output(broadcast_to_f32->output(0));
        }

        // 3. Insert Convert(f32->f16) after Sin/Cos to restore original precision
        size_t output_idx = 0;
        insert_converts_after_if_needed(sin_node, original_et, output_idx);
        insert_converts_after_if_needed(cos_node, original_et, output_idx);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_or_transpose, "IncreasePositionIdsPrecisionForQwen3VL");
    this->register_matcher(m, callback);
}

namespace {

bool increase_sincos_angle_precision(const std::shared_ptr<ov::Node>& sin, const std::shared_ptr<ov::Node>& cos) {
    // Phase 1: inspect both upstream value paths before changing any live edge.
    // Traverse upstream from the values feeding Sin/Cos.
    const ov::OutputVector angle_inputs{sin->input_value(0), cos->input_value(0)};

    // Map each visited input value to its producer's floating-point input indices for FP32 reconstruction.
    // For example, Multiply records {0, 1}, while Reshape records {0}, leaving integer shape inputs unchanged.
    // Output keys distinguish Split output ports and deduplicate shared Sin/Cos dependencies.
    std::map<ov::Output<ov::Node>, std::vector<size_t>> floating_input_ports;
    bool has_required_arithmetic = false;
    bool has_fp16_in_angle_path = false;

    // Stack entry: (input_value, dependencies_processed); false schedules dependencies, true records the value.
    std::vector<std::pair<ov::Output<ov::Node>, bool>> traversal_stack;
    ov::OutputVector dependency_order;

    for (const auto& input_value : angle_inputs) {
        traversal_stack.emplace_back(input_value, false);
    }

    // Use an explicit DFS stack to avoid recursion depth limits on long upstream paths.
    // Unlike node-level traversal of all inputs, this walk follows only floating-point inputs and preserves output ports.
    // floating_input_ports also tracks visited values, so shared dependencies are inspected only once.
    //
    // Illustrative traversal: A -> B -> Sin, A -> C -> Cos; A is a leaf, all edges carry floating-point values.
    // Distinct B/C illustrate stack behavior only; the current matcher requires a shared Sin/Cos input.
    // Each entry is (value, dependencies_processed); the stack TOP is on the LEFT.
    //
    // Action                            | Stack: TOP -> bottom       | dependency_order
    // -----------------------------------------------------------------------------------
    // Push initial B, then C            | (C,false), (B,false)       | []
    // -----------------------------------------------------------------------------------
    // Pop C; schedule return and A      | (A,false), (C,true),       | []
    //                                   | (B,false)                  |
    // -----------------------------------------------------------------------------------
    // Pop leaf A; schedule recording    | (A,true), (C,true),        | []
    //                                   | (B,false)                  |
    // -----------------------------------------------------------------------------------
    // Pop A; record it                  | (C,true), (B,false)        | [A]
    // -----------------------------------------------------------------------------------
    // Pop C; record it                  | (B,false)                  | [A, C]
    // -----------------------------------------------------------------------------------
    // Pop B; schedule return and A      | (A,false), (B,true)        | [A, C]
    // -----------------------------------------------------------------------------------
    // Pop A; skip the visited value     | (B,true)                   | [A, C]
    // -----------------------------------------------------------------------------------
    // Pop B; record it                  | []                         | [A, C, B]
    // -----------------------------------------------------------------------------------
    //
    // Dependencies precede their consumers; leaves from separate branches need not all come first.
    // Process each value in two stages: false inspects it and schedules dependencies; true records it afterward.
    // This orders FP32 reconstruction so each input's FP32 counterpart is ready before rebuilding its consumer.
    while (!traversal_stack.empty()) {
        const auto [input_value, dependencies_processed] = traversal_stack.back();
        traversal_stack.pop_back();

        if (dependencies_processed) {
            dependency_order.push_back(input_value);
            continue;
        }
        if (floating_input_ports.count(input_value)) {
            continue;
        }

        const auto node = input_value.get_node_shared_ptr();

        // Require an upstream arithmetic operation; this does not prove that the path only computes angles.
        // These operations occur in the position-id precision patterns examined so far; this is not a traversal allowlist.
        // Extend this eligibility list only when a new model needs it (e.g. a Subtract/Divide-only angle path).
        has_required_arithmetic |= ov::is_type_any_of<ov::op::v1::Add,
                                                    ov::op::v1::Multiply,
                                                    ov::op::v0::MatMul,
                                                    ov::intel_gpu::op::Gemm>(node);

        const auto type = input_value.get_element_type();
        has_fp16_in_angle_path |= type == ov::element::f16;

        // Changing outer inputs alone cannot safely rebuild bodies or state/control dependencies.
        if ((type != ov::element::f16 && type != ov::element::f32) ||
            ov::is_type<ov::op::util::MultiSubGraphOp>(node) ||
            std::dynamic_pointer_cast<ov::op::util::VariableExtension>(node) ||
            !node->get_control_dependencies().empty()) {
            GPU_DEBUG_LOG << "IncreasePositionIdsPrecisionForLtxVideo: unsupported angle output " << input_value
                          << "; leaving Sin/Cos unchanged" << std::endl;
            return false;
        }

        std::vector<size_t> ports;
        // Input types select value dependencies without an op allowlist; integer/bool controls stay unchanged.
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            if (node->get_input_element_type(i).is_real()) {
                ports.push_back(i);
            }
        }

        const bool is_leaf = ov::is_type_any_of<ov::op::v0::Constant, ov::op::v0::Parameter>(node);
        const bool is_convert = ov::is_type<ov::op::v0::Convert>(node);

        // Only an existing integer-to-float Convert may enter from an integer value path.
        if ((!is_leaf && !is_convert && ports.empty()) ||
            (is_convert && ports.empty() && !node->get_input_element_type(0).is_integral_number())) {
            GPU_DEBUG_LOG << "IncreasePositionIdsPrecisionForLtxVideo: unsupported angle source " << input_value
                          << "; leaving Sin/Cos unchanged" << std::endl;
            return false;
        }

        floating_input_ports.emplace(input_value, ports);
        traversal_stack.emplace_back(input_value, true);
        for (const auto port : ports) {
            traversal_stack.emplace_back(input_value.get_node()->input_value(port), false);
        }
    }

    if (!has_required_arithmetic) {
        GPU_DEBUG_LOG << "IncreasePositionIdsPrecisionForLtxVideo: no angle arithmetic; leaving Sin/Cos unchanged"
                      << std::endl;
        return false;
    }

    if (!has_fp16_in_angle_path) {
        return false;
    }

    // Phase 2: prepare the FP32 path in dependency-first order without changing live graph edges.
    // - Reuse FP32 Constants/Parameters; upcast FP16 leaves and fold Constant upcasts (lost precision is not recovered).
    // - Bypass floating Converts, except reusable FP16 Parameter-to-FP32 input boundaries.
    // - Preserve integer sources and enter FP32 at existing integer-to-float Convert boundaries.
    // - Replace floating inputs only; retain integer/bool inputs and reuse unchanged FP32 nodes.
    // - Clone changed nodes, copy runtime info, and validate every output type before recording path mappings.
    // - Preserve output indices and share the FP32 path across Sin/Cos; return unchanged if both angle inputs are reused.
    const auto copy_info = [](const std::shared_ptr<ov::Node>& source, const std::shared_ptr<ov::Node>& replacement) {
        ov::copy_runtime_info(source, replacement);
        replacement->set_friendly_name(source->get_friendly_name() + "/fp32_" + replacement->get_name());
    };

    // Map each original value to its counterpart in the FP32 path, including reused values.
    // Example: A -> B -> Sin.
    // 1. Prepare A's FP32 counterpart and record A -> A_fp32.
    // 2. Look up A_fp32 when rebuilding B and use it as B's input.
    // 3. Record B -> B_fp32, then use B_fp32 as the new Sin input.
    // Shared values are prepared once; existing FP32 values can map to themselves.
    // This prepares the replacement path without rewiring the original graph.
    std::map<ov::Output<ov::Node>, ov::Output<ov::Node>> fp32_path_map;

    for (const auto& original_value : dependency_order) {
        if (fp32_path_map.count(original_value)) {
            continue;
        }

        const auto original_node = original_value.get_node_shared_ptr();
        const auto& original_floating_input_ports = floating_input_ports.at(original_value);

        // Choose the value to use in the FP32 path, then record original_value -> chosen value in fp32_path_map.
        // Constant/Parameter and Convert need explicit handling rather than generic cloning:
        //
        // Producer          | Condition                         | Value used in the FP32 path
        // ------------------|-----------------------------------|-----------------------------------------------
        // Constant/Parameter| Already FP32                      | Reuse the original value.
        // Constant          | FP16                              | Create an FP32 Constant by folding its upcast.
        // Parameter         | FP16                              | Create a Convert(FP32) from the Parameter.
        // Convert           | Parameter(FP16) -> FP32           | Reuse the existing Convert output.
        // Convert           | Other floating-point input        | Bypass it; use the input's FP32 counterpart.
        // Convert           | Integer input -> FP32             | Reuse the existing Convert output.
        // Convert           | Integer input -> FP16             | Create a Convert(FP32) from the integer input.
        //
        // Later consumers only need the recorded counterpart, regardless of whether it was reused, created or bypassed.
        if (ov::is_type_any_of<ov::op::v0::Constant, ov::op::v0::Parameter>(original_node)) {
            if (original_value.get_element_type() == ov::element::f32) {
                // Reuse Step01's preserved Constants and original FP32 Parameters directly.
                fp32_path_map.emplace(original_value, original_value);
            } else {
                // Convert remaining FP16 Constants/Parameters to FP32 for subsequent computation.
                // Phase 1 permits only FP16/FP32 values, so this non-FP32 value must be FP16.
                std::shared_ptr<ov::Node> promoted = std::make_shared<ov::op::v0::Convert>(original_value, ov::element::f32);
                if (ov::is_type<ov::op::v0::Constant>(original_node)) {
                    promoted = ov::util::get_constant_from_source(promoted);
                    OPENVINO_ASSERT(promoted, "Could not promote LTX angle Constant to FP32");
                }

                copy_info(original_node, promoted);
                fp32_path_map.emplace(original_value, promoted->output(0));
            }
        } else if (ov::is_type<ov::op::v0::Convert>(original_node)) {
            if (!original_floating_input_ports.empty()) {
                const auto input = original_node->input_value(0);
                // Reuse the existing Parameter(FP16) -> Convert(FP32) boundary.
                if (original_value.get_element_type() == ov::element::f32 && input.get_element_type() == ov::element::f16 &&
                    ov::is_type<ov::op::v0::Parameter>(input.get_node())) {
                    // This is an irreducible input boundary, not recoverable FP16 arithmetic.
                    fp32_path_map.emplace(original_value, original_value);
                } else {
                    fp32_path_map.emplace(original_value, fp32_path_map.at(input));
                }
            } else if (original_value.get_element_type() == ov::element::f32) {
                fp32_path_map.emplace(original_value, original_value);
            } else {
                // Keep coordinate generation integer and enter FP32 at the existing Convert boundary.
                auto converted = std::make_shared<ov::op::v0::Convert>(original_node->input_value(0), ov::element::f32);
                copy_info(original_node, converted);
                fp32_path_map.emplace(original_value, converted->output(0));
            }
        } else {
            auto inputs = original_node->input_values();
            // Replace only floating-point inputs; integer/bool inputs retain their original producers.
            for (const auto port : original_floating_input_ports) {
                inputs[port] = fp32_path_map.at(inputs[port]);
            }

            if (original_value.get_element_type() == ov::element::f32 && inputs == original_node->input_values()) {
                for (size_t i = 0; i < original_node->get_output_size(); ++i) {
                    fp32_path_map.emplace(original_node->output(i), original_node->output(i));
                }
            } else {
                // Example: original A(FP32) -> Convert(FP16) -> C; new path A(FP32) -> C_fp32.
                // original_node is C; original_value identifies one of C's output ports.
                // Floating-point entries in inputs were replaced via fp32_path_map (Convert output -> A here).
                // Cloning with these inputs bypasses the Convert in the new path.
                // rebuilt is the new C using those inputs; the original C and its connections stay unchanged.
                const auto rebuilt = original_node->clone_with_new_inputs(inputs);
                copy_info(original_node, rebuilt);

                // Verify every sibling before recording any output of this shared clone.
                for (size_t i = 0; i < original_node->get_output_size(); ++i) {
                    const auto expected = original_node->get_output_element_type(i).is_real()
                                              ? ov::element::f32
                                              : original_node->get_output_element_type(i);
                    if (rebuilt->get_output_element_type(i) != expected) {
                        GPU_DEBUG_LOG << "IncreasePositionIdsPrecisionForLtxVideo: incompatible rebuilt output "
                                      << rebuilt->output(i) << "; leaving Sin/Cos unchanged" << std::endl;
                        return false;
                    }
                }

                for (size_t i = 0; i < original_node->get_output_size(); ++i) {
                    fp32_path_map.emplace(original_node->output(i), rebuilt->output(i));
                }
            }
        }
    }

    if (fp32_path_map.at(angle_inputs[0]) == angle_inputs[0] &&
        fp32_path_map.at(angle_inputs[1]) == angle_inputs[1]) {
        return false;
    }

    // Phase 3: prepare both FP32 Sin/Cos branches before reconnecting either one.
    // Only their outputs are replaced; unrelated users of the old angle region keep their original edges.
    ov::NodeVector replacement_nodes;
    for (const auto& node : {sin, cos}) {
        auto rebuilt = node->clone_with_new_inputs({fp32_path_map.at(node->input_value(0))});
        copy_info(node, rebuilt);

        if (node->get_output_element_type(0) != ov::element::f32) {
            // Restore precision after Sin/Cos so downstream table and Concat inputs keep matching types.
            auto boundary = std::make_shared<ov::op::v0::Convert>(rebuilt, node->get_output_element_type(0));
            ov::copy_runtime_info(node, boundary);
            rebuilt = boundary;
        }

        rebuilt->set_friendly_name(node->get_friendly_name());
        replacement_nodes.push_back(rebuilt);
    }

    ov::replace_node(sin, replacement_nodes[0]);
    ov::replace_node(cos, replacement_nodes[1]);
    return true;
}

}  // namespace

IncreasePositionIdsPrecisionForLtxVideo::IncreasePositionIdsPrecisionForLtxVideo() {
    using namespace ov::pass::pattern;
    // Sin owns the rewrite; Cos is discovered from the exact same angle output.
    auto sin = wrap_type<ov::op::v0::Sin>({any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto sin_node = ov::as_type_ptr<ov::op::v0::Sin>(m.get_match_root());
        std::shared_ptr<ov::Node> cos_node;
        size_t sin_count = 0;
        size_t cos_count = 0;
        // TODO: Consider Sin/Cos branches with intervening nodes when generalizing precision handling.
        // In the models checked so far, Sin and Cos directly share the same input.
        // However, future models may have nodes between the branch point and Sin/Cos.
        // Consider these cases when generalizing precision handling.
        for (const auto& input : sin_node->input_value(0).get_target_inputs()) {
            const auto consumer = input.get_node()->shared_from_this();
            if (ov::is_type<ov::op::v0::Sin>(consumer)) {
                ++sin_count;
            } else if (ov::is_type<ov::op::v0::Cos>(consumer)) {
                ++cos_count;
                cos_node = consumer;
            }
        }
        if (sin_count != 1 || cos_count != 1) {
            GPU_DEBUG_LOG << "IncreasePositionIdsPrecisionForLtxVideo: missing or ambiguous Sin/Cos pair at "
                          << sin_node->input_value(0) << "; leaving graph unchanged" << std::endl;
            return false;
        }
        // With no RoPE requirement, opt-out is evaluated on both actual trigonometric roots.
        if (transformation_callback(sin_node) || transformation_callback(cos_node)) {
            return false;
        }
        return increase_sincos_angle_precision(sin_node, cos_node);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin, "IncreasePositionIdsPrecisionForLtxVideo");
    this->register_matcher(m, callback);
}

IncreasePositionIdsPrecisionForGPTOSS::IncreasePositionIdsPrecisionForGPTOSS() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    auto broadcast_freq = wrap_type<ov::op::v3::Broadcast>({any_input(), any_input()});
    auto convert_broadcast_freq = wrap_type<ov::op::v0::Convert>({broadcast_freq});

    auto convert_pos_id_to_i32 = wrap_type<ov::op::v0::Convert>({any_input()});
    auto unsqueeze_pos_id_1 = wrap_type<ov::op::v0::Unsqueeze>({convert_pos_id_to_i32, any_input()});
    auto unsqueeze_pos_id_2 = wrap_type<ov::op::v1::Reshape>({convert_pos_id_to_i32, any_input()});
    auto unsqueeze_pos_id = std::make_shared<Or>(OutputVector{unsqueeze_pos_id_1, unsqueeze_pos_id_2});
    auto convert_pos_id_to_f16 = wrap_type<ov::op::v0::Convert>({unsqueeze_pos_id});

    auto broadcast_freq_ = std::make_shared<Or>(OutputVector{broadcast_freq, convert_broadcast_freq});

    auto matmul_freq_pos_id = wrap_type<ov::op::v0::MatMul>({broadcast_freq_, convert_pos_id_to_f16});
    auto reshape_matmul = wrap_type<ov::op::v1::Reshape>({matmul_freq_pos_id, any_input()});
    auto transpose_matmul = wrap_type<ov::op::v1::Transpose>({matmul_freq_pos_id, any_input()});
    auto transpose_or_reshape = std::make_shared<Or>(OutputVector{transpose_matmul, reshape_matmul});

    auto sin_ = wrap_type<ov::op::v0::Sin>({transpose_or_reshape});
    auto sin_convert = wrap_type<ov::op::v0::Convert>({sin_});
    auto sin = std::make_shared<Or>(OutputVector{sin_, sin_convert});

    auto cos = wrap_type<ov::op::v0::Cos>({transpose_or_reshape});
    auto cos_convert = wrap_type<ov::op::v0::Convert>({cos});
    auto cos_ = std::make_shared<Or>(OutputVector{cos, cos_convert});

    auto scale_const_sin = wrap_type<ov::op::v0::Constant>();
    auto scale_const_sin_convert = wrap_type<ov::op::v0::Convert>({scale_const_sin});
    auto scale_const_sin_ = std::make_shared<Or>(OutputVector{scale_const_sin, scale_const_sin_convert});
    auto mul_sin_scale = wrap_type<ov::op::v1::Multiply>({sin, scale_const_sin_});

    auto scale_const_cos = wrap_type<ov::op::v0::Constant>();
    auto scale_const_cos_convert = wrap_type<ov::op::v0::Convert>({scale_const_cos});
    auto scale_const_cos_ = std::make_shared<Or>(OutputVector{scale_const_cos, scale_const_cos_convert});
    auto mul_cos_scale = wrap_type<ov::op::v1::Multiply>({cos_, scale_const_cos_});

    auto unsqueeze_mul_sin_scale_ = wrap_type<ov::op::v0::Unsqueeze>({mul_sin_scale, any_input()});
    auto reshape_mul_sin_scale_ = wrap_type<ov::op::v1::Reshape>({mul_sin_scale, any_input()});
    auto reshape_mul_sin_scale = std::make_shared<Or>(OutputVector{unsqueeze_mul_sin_scale_, reshape_mul_sin_scale_});

    auto unsqueeze_mul_cos_scale_ = wrap_type<ov::op::v0::Unsqueeze>({mul_cos_scale, any_input()});
    auto reshape_mul_cos_scale_ = wrap_type<ov::op::v1::Reshape>({mul_cos_scale, any_input()});
    auto reshape_mul_cos_scale = std::make_shared<Or>(OutputVector{unsqueeze_mul_cos_scale_, reshape_mul_cos_scale_});

    auto rope_qk = wrap_type<ov::op::internal::RoPE>({any_input(), reshape_mul_cos_scale, reshape_mul_sin_scale});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        if (transformation_callback(rope_qk)) {
            return false;
        }

        auto rope_node = ov::as_type_ptr<ov::op::internal::RoPE>(pattern_map.at(rope_qk).get_node_shared_ptr());

        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul_freq_pos_id).get_node_shared_ptr());
        auto mul_node1 = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_map.at(mul_sin_scale).get_node_shared_ptr());
        auto mul_node2 = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_map.at(mul_cos_scale).get_node_shared_ptr());
        if (!rope_node || !matmul_node || !mul_node1 || !mul_node2) {
            return false;
        }

        const auto desired_et = ov::element::f32;
        const auto original_et = rope_node->get_output_element_type(0);
        if (original_et == desired_et) {
            return false;
        }

        size_t idx = 0;
        insert_converts_before_if_needed(matmul_node, desired_et, idx);
        insert_converts_before_if_needed(mul_node1, desired_et, idx);
        insert_converts_before_if_needed(mul_node2, desired_et, idx);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope_qk, "IncreasePositionIdsPrecisionForGPTOSS");
    this->register_matcher(m, callback);
}

IncreasePositionIdsPrecisionForDirectMatMulSinCos::IncreasePositionIdsPrecisionForDirectMatMulSinCos() {
    using namespace ov::pass::pattern;

    auto matmul = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()});
    auto sin = wrap_type<ov::op::v0::Sin>({matmul});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul).get_node_shared_ptr());
        auto sin_node = ov::as_type_ptr<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr());
        std::vector<std::shared_ptr<ov::op::v0::Cos>> cos_nodes;

        if (!matmul_node || !sin_node || transformation_callback(matmul_node)) {
            return false;
        }

        auto reaches_position_ids = [](ov::Output<ov::Node> output) {
            std::set<ov::Node*> visited;
            while (visited.insert(output.get_node()).second) {
                const auto node = output.get_node_shared_ptr();
                if (const auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(node)) {
                    return parameter->get_friendly_name() == "position_ids" || parameter->output(0).get_names().count("position_ids") != 0;
                }

                if (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v0::Squeeze>(node) ||
                    ov::is_type<ov::op::v0::Unsqueeze>(node) || ov::is_type<ov::op::v8::Gather>(node)) {
                    output = node->input_value(0);
                    continue;
                }
                return false;
            }
            return false;
        };

        std::shared_ptr<ov::op::v0::Convert> position_ids_convert_node;
        for (auto position_ids_path : matmul_node->input_values()) {
            std::set<ov::Node*> visited;
            while (visited.insert(position_ids_path.get_node()).second) {
                const auto node = position_ids_path.get_node_shared_ptr();
                if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
                    if (convert->input_value(0).get_element_type().is_integral() && reaches_position_ids(convert->input_value(0))) {
                        position_ids_convert_node = convert;
                        break;
                    }
                }

                if (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v0::Squeeze>(node) ||
                    ov::is_type<ov::op::v0::Unsqueeze>(node) || ov::is_type<ov::op::v8::Gather>(node)) {
                    position_ids_path = node->input_value(0);
                    continue;
                }
                break;
            }

            if (position_ids_convert_node) {
                break;
            }
        }

        if (!position_ids_convert_node) {
            return false;
        }

        for (const auto& user : matmul_node->get_users()) {
            if (auto cos_node = ov::as_type_ptr<ov::op::v0::Cos>(user)) {
                cos_nodes.push_back(cos_node);
            } else if (user != sin_node && !ov::is_type<ov::op::util::ShapeOfBase>(user)) {
                return false;
            }
        }

        const auto desired_et = ov::element::f32;
        const auto original_et = matmul_node->get_output_element_type(0);
        if (cos_nodes.empty() || original_et == desired_et) {
            return false;
        }

        size_t input_idx = 0;
        if (!insert_converts_before_if_needed(matmul_node, desired_et, input_idx)) {
            return false;
        }

        auto promote_output = [&](const std::shared_ptr<ov::Node>& node) {
            if (auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node)) {
                type_relaxed->set_overridden_output_type(desired_et);
            }
            node->validate_and_infer_types();
        };
        promote_output(matmul_node);
        promote_output(sin_node);
        for (const auto& cos_node : cos_nodes) {
            promote_output(cos_node);
        }

        size_t output_idx = 0;
        for (const auto& cos_node : cos_nodes) {
            insert_converts_after_if_needed(cos_node, original_et, output_idx);
        }
        insert_converts_after_if_needed(sin_node, original_et, output_idx);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin, "IncreasePositionIdsPrecisionForDirectMatMulSinCos");
    this->register_matcher(m, callback);
}

IncreasePositionIdsPrecision::IncreasePositionIdsPrecision() = default;

bool IncreasePositionIdsPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForDirectMatMulSinCos>();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForRoPE>();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForQwen25VL>();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForQwen3VL>();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForGPTOSS>();
    symbolic_ctx_manager->register_pass<IncreasePositionIdsPrecisionForLtxVideo>();
    return symbolic_optimizations.run_on_model(model);
}

DisableFP16ComForGPTOSSROPEPattern::DisableFP16ComForGPTOSSROPEPattern() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    // for gpt-oss pattern
    auto freq_const = wrap_type<ov::op::v0::Constant>();
    auto broadcast_freq = wrap_type<ov::op::v3::Broadcast>({freq_const, any_input()});

    // position_id
    auto unsqueeze_pos_id = wrap_type<ov::op::v0::Unsqueeze>({any_input(), any_input()});
    auto convert_pos_id_to_f16 = wrap_type<ov::op::v0::Convert>({unsqueeze_pos_id});

    auto matmul_freq_pos_id = wrap_type<ov::op::v0::MatMul>({broadcast_freq, convert_pos_id_to_f16});
    auto transpose = wrap_type<ov::op::v1::Transpose>({matmul_freq_pos_id, any_input()});
    auto sin = wrap_type<ov::op::v0::Sin>({transpose});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sin_node = ov::as_type_ptr<ov::op::v0::Sin>(pattern_map.at(sin).get_node_shared_ptr());
        if (!sin_node || transformation_callback(sin_node)) {
            return false;
        }
        auto freq_const_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(freq_const).get_node_shared_ptr());
        ov::disable_conversion(freq_const_node, element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin, "DisableFP16ComForGPTOSSROPEPattern");
    this->register_matcher(m, callback);
}
}  // namespace ov::intel_gpu
