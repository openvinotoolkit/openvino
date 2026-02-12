// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/qdq_stripping.hpp"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <queue>
#include <unordered_set>

#include "itt.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/lpt_itt.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"

// Macro for conditional debug logging based on OV_QDQ_DEBUG_LOG environment variable
#define QDQ_DEBUG_LOG                                                               \
    if (static const bool debug = ov::util::getenv_bool("OV_QDQ_DEBUG_LOG"); debug) \
    std::cout

// Macro for model visualization dumping
#define VISUALIZE_MODEL(folder, filename, stage_desc)                                                                 \
    if (!folder.empty()) {                                                                                            \
        try {                                                                                                         \
            auto dump_path = (std::filesystem::path(folder) / filename).string();                                     \
            ov::pass::VisualizeTree(dump_path).run_on_model(f);                                                       \
            QDQ_DEBUG_LOG << "[ INFO ] Model visualized to: " << dump_path << std::endl;                              \
        } catch (const std::exception& e) {                                                                           \
            QDQ_DEBUG_LOG << "[ WARNING ] Failed to visualize " << stage_desc << " model: " << e.what() << std::endl; \
        }                                                                                                             \
    }

// Macro for model serialization
#define SERIALIZE_MODEL(folder, filename, stage_desc)                                                                 \
    if (!folder.empty()) {                                                                                            \
        try {                                                                                                         \
            auto xml_path = (std::filesystem::path(folder) / (std::string(filename) + ".xml")).string();              \
            auto bin_path = (std::filesystem::path(folder) / (std::string(filename) + ".bin")).string();              \
            ov::pass::Serialize(xml_path, bin_path).run_on_model(f);                                                  \
            QDQ_DEBUG_LOG << "[ INFO ] Model serialized to: " << xml_path << std::endl;                               \
        } catch (const std::exception& e) {                                                                           \
            QDQ_DEBUG_LOG << "[ WARNING ] Failed to serialize " << stage_desc << " model: " << e.what() << std::endl; \
        }                                                                                                             \
    }

namespace ov {
namespace pass {
namespace low_precision {

class WeightsDequantizationBlock : public ov::pass::pattern::op::Block {
public:
    WeightsDequantizationBlock() : Block({}, {}, "WeightsDequantizationBlock") {
        using namespace ov::pass::pattern;

        auto weights = wrap_type<ov::op::v0::Constant>();
        auto convert = wrap_type<ov::op::v0::Convert>({weights});

        auto sub_const = wrap_type<ov::op::v0::Constant>();
        auto sub_const_convert = optional<ov::op::v0::Convert>({sub_const});
        auto subtract = wrap_type<ov::op::v1::Subtract>({convert, sub_const_convert});

        auto mul_input = subtract | convert;
        auto mul_const = wrap_type<ov::op::v0::Constant>();
        auto multiply = wrap_type<ov::op::v1::Multiply>({mul_input, mul_const});

        m_inputs = ov::OutputVector{weights};
        m_outputs = ov::OutputVector{multiply};
        REGISTER_ANCHORS(this, weights, convert, sub_const, subtract, mul_input, mul_const, multiply);
    }
};

namespace {

// Helper to get scalar float value from a constant node
float get_const_float_value(const std::shared_ptr<Node>& node) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant || ov::shape_size(constant->get_shape()) != 1)
        return 0.0f;
    return constant->cast_vector<float>()[0];
}

// RAII helper to temporarily set environment variables and restore them on destruction
class EnvVarGuard {
public:
    EnvVarGuard(const std::vector<std::pair<std::string, std::string>>& vars_to_set) {
        for (const auto& [name, value] : vars_to_set) {
            // Save original value
            const char* original = std::getenv(name.c_str());
            m_original_values[name] = original ? std::optional<std::string>(original) : std::nullopt;

            // Set new value
#ifdef _WIN32
            _putenv_s(name.c_str(), value.c_str());
#else
            setenv(name.c_str(), value.c_str(), 1);
#endif
        }
    }

    ~EnvVarGuard() {
        // Restore original values
        for (const auto& [name, original_value] : m_original_values) {
            if (original_value.has_value()) {
#ifdef _WIN32
                _putenv_s(name.c_str(), original_value->c_str());
#else
                setenv(name.c_str(), original_value->c_str(), 1);
#endif
            } else {
#ifdef _WIN32
                _putenv_s(name.c_str(), "");
#else
                unsetenv(name.c_str());
#endif
            }
        }
    }

    EnvVarGuard(const EnvVarGuard&) = delete;
    EnvVarGuard& operator=(const EnvVarGuard&) = delete;

private:
    std::unordered_map<std::string, std::optional<std::string>> m_original_values;
};

}  // namespace

FQStrippingTransformation::FQStrippingTransformation(const std::set<size_t>& levels_to_strip)
    : levels_to_strip(levels_to_strip) {}

bool FQStrippingTransformation::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(FQStrippingTransformation);
    if (levels_to_strip.empty()) {
        return false;
    }

    // Check if model visualization is enabled
    auto visualize_folder = ov::util::getenv_string("OV_QDQ_VISUALIZE_MODEL_DIR");
    auto serialize_folder = ov::util::getenv_string("OV_QDQ_SERIALIZE_MODEL_DIR");

    // Set visualization env variables for the duration of model dumping
    std::unique_ptr<EnvVarGuard> vis_env_guard;
    if (!visualize_folder.empty()) {
        vis_env_guard = std::make_unique<EnvVarGuard>(
            std::vector<std::pair<std::string, std::string>>{{"OV_VISUALIZE_TREE_OUTPUT_SHAPES", "1"},
                                                             {"OV_VISUALIZE_TREE_OUTPUT_TYPES", "1"},
                                                             {"OV_VISUALIZE_TREE_IO", "1"}});
    }
    VISUALIZE_MODEL(visualize_folder, "01_initial", "initial");
    SERIALIZE_MODEL(serialize_folder, "01_initial", "initial");

    auto check_fq_constants = [&](const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) -> bool {
        auto is_scalar_const = [](const std::shared_ptr<Node>& node) -> bool {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (!constant) {
                return false;
            }
            return ov::shape_size(constant->get_shape()) == 1;
        };

        if (!is_scalar_const(fq->get_input_node_shared_ptr(1)) || !is_scalar_const(fq->get_input_node_shared_ptr(2)) ||
            !is_scalar_const(fq->get_input_node_shared_ptr(3)) || !is_scalar_const(fq->get_input_node_shared_ptr(4))) {
            return false;
        }

        // Check if ranges are valid (not degenerate)
        float input_low = get_const_float_value(fq->get_input_node_shared_ptr(1));
        float input_high = get_const_float_value(fq->get_input_node_shared_ptr(2));
        float output_low = get_const_float_value(fq->get_input_node_shared_ptr(3));
        float output_high = get_const_float_value(fq->get_input_node_shared_ptr(4));
        return std::abs(input_high - input_low) > 1e-6f && std::abs(output_high - output_low) > 1e-6f;
    };

    bool model_changed = false;
    const float ratio = 1.0f;
    const float threshold = 1.0f;

    QDQ_DEBUG_LOG << "\n[ INFO ] === QDQ Stripping Pass ===" << std::endl;
    QDQ_DEBUG_LOG << "[ INFO ] Total nodes in graph: " << f->get_ops().size() << std::endl;
    QDQ_DEBUG_LOG << "\n[ INFO ] === Nodes info dumping started ===" << std::endl;
    for (const auto& node : f->get_ordered_ops()) {
        QDQ_DEBUG_LOG << "  [ NODE ] Name: " << node->get_friendly_name() << ", type: " << node->get_type_name()
                      << ", ptr: " << node << std::endl;
    }
    QDQ_DEBUG_LOG << "[ INFO ] === Nodes info dumping completed ===\n" << std::endl;
    QDQ_DEBUG_LOG << "[ INFO ] Levels to strip: ";
    for (auto level : levels_to_strip) {
        QDQ_DEBUG_LOG << level << " ";
    }
    QDQ_DEBUG_LOG << std::endl;

    // Scale adjustment infrastructure
    std::unordered_set<ov::Node*> visited;
    float current_scale_divisor = 1.0f;
    auto backward_skip_predicate = [](ov::Node* n) {
        return ov::is_type<op::v0::ShapeOf>(n) || ov::is_type<op::v3::ShapeOf>(n);
    };

    auto apply_scale_to_weight = [&](const ov::pass::pattern::PatternValueMap& pattern_map,
                                     const std::shared_ptr<WeightsDequantizationBlock>& dq_block) {
        auto original_constant = dq_block->get_anchor("mul_const", pattern_map).value().get_node_shared_ptr();
        auto old_multiply = dq_block->get_anchor("multiply", pattern_map).value().get_node_shared_ptr();
        auto mul_input = dq_block->get_anchor("mul_input", pattern_map).value().get_node_shared_ptr();

        if (visited.find(old_multiply.get()) != visited.end()) {
            QDQ_DEBUG_LOG << "        [ DEBUG ]   Node " << old_multiply->get_friendly_name()
                          << " already visited, skipping scale adjustment" << std::endl;
            return;
        }

        QDQ_DEBUG_LOG << "        [ DEBUG ]     Dividing multiply " << old_multiply->get_friendly_name() << " by "
                      << current_scale_divisor << std::endl;

        // Create new scaled constant: divide dequantization scale by current_scale_divisor
        auto divisor_const =
            ov::op::v0::Constant::create(original_constant->get_output_element_type(0), {}, {current_scale_divisor});
        auto new_constant = ov::op::util::make_try_fold<ov::op::v1::Divide>(original_constant, divisor_const);

        // Create new multiply with the scaled constant
        auto new_multiply = old_multiply->clone_with_new_inputs({mul_input, new_constant});

        ov::replace_node(old_multiply, new_multiply);
        visited.insert(new_multiply.get());
    };

    // Helper to adjust FQ range constants (input_low, input_high, output_low, output_high)
    // by dividing them by current_scale_divisor. This is needed when scale propagation passes
    // through an un-stripped FQ — the quantization grid must shift to match the new value range.
    auto adjust_fq_ranges = [&](ov::Node* node) {
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node->shared_from_this());
        if (!fq)
            return;

        if (visited.count(node))
            return;

        QDQ_DEBUG_LOG << "        [ DEBUG ] Adjusting FQ ranges for: " << fq->get_friendly_name()
                      << " by dividing by " << current_scale_divisor << std::endl;

        auto divisor_const = ov::op::v0::Constant::create(ov::element::f32, {}, {current_scale_divisor});

        // Adjust all 4 range constants: input_low(1), input_high(2), output_low(3), output_high(4)
        for (size_t idx = 1; idx <= 4; ++idx) {
            auto original_const = fq->get_input_node_shared_ptr(idx);
            auto new_const = ov::op::util::make_try_fold<ov::op::v1::Divide>(original_const, divisor_const);
            fq->input(idx).replace_source_output(new_const);
        }

        visited.insert(node);
    };

    auto adjust_weights_scale = [&](ov::Node* node) {
        QDQ_DEBUG_LOG << "    [ INFO ] adjust_weights_scale called for node with type: " << node->get_type_name()
                      << ", with name: " << node->get_friendly_name() << ", node: " << node << std::endl;
        using namespace ov::pass::pattern;
        const auto node_shared = node->shared_from_this();
        // Case 1: Convolution + Add (bias) - scale both Add's constant and Conv weights
        // The bias pattern came from ONNX FE
        {
            // Conv with weights DQ
            auto conv_weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto conv_pattern = wrap_type<ov::op::v1::Convolution>({any_input(), conv_weights_dq_block});

            // Bias DQ and reshape with shape computation
            // Shape computation: ShapeOf(conv) -> ShapeOf -> Subtract -> Broadcast, ShapeOf(bias) -> Concat
            auto conv_shape = wrap_type<ov::op::v3::ShapeOf>({conv_pattern});
            auto conv_rank = wrap_type<ov::op::v3::ShapeOf>({conv_shape});
            auto rank_minus_2 = wrap_type<ov::op::v1::Subtract>({conv_rank, any_input()});
            auto tail = wrap_type<ov::op::v3::Broadcast>({any_input(), rank_minus_2});

            auto bias_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto c_dim = wrap_type<ov::op::v3::ShapeOf>({bias_dq_block});
            auto target_shape = wrap_type<ov::op::v0::Concat>({any_input(), c_dim, tail});
            auto reshape_pattern = wrap_type<ov::op::v1::Reshape>({bias_dq_block, target_shape});

            auto add_pattern = wrap_type<ov::op::v1::Add>({conv_pattern, reshape_pattern});
            auto matcher = std::make_shared<Matcher>(add_pattern, "ConvAddPattern");

            if (matcher->match(node_shared)) {
                QDQ_DEBUG_LOG << "        [ INFO ]   Matched Conv+Add(bias) pattern" << std::endl;
                auto pattern_map = matcher->get_pattern_value_map();
                apply_scale_to_weight(pattern_map, conv_weights_dq_block);
                apply_scale_to_weight(pattern_map, bias_dq_block);
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    visited.insert(in.get_node());
                }
                return;
            }
        }

        // Case 2: MatMul + Add (bias) - scale both MatMul weights and bias
        {
            auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto matmul_pattern = wrap_type<ov::op::v0::MatMul>({any_input(), weights_dq_block});

            auto bias_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto add_pattern = wrap_type<ov::op::v1::Add>({matmul_pattern, bias_dq_block});
            auto matcher = std::make_shared<Matcher>(add_pattern, "MatMulAddPattern");

            if (matcher->match(node_shared)) {
                QDQ_DEBUG_LOG << "        [ INFO ]   Matched MatMul+Add(bias) pattern" << std::endl;
                auto pattern_map = matcher->get_pattern_value_map();
                apply_scale_to_weight(pattern_map, weights_dq_block);
                apply_scale_to_weight(pattern_map, bias_dq_block);
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    visited.insert(in.get_node());
                }
                return;
            }
        }

        // Case 3: MatMul with weights (no bias)
        {
            auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto matmul_pattern = wrap_type<ov::op::v0::MatMul>({any_input(), weights_dq_block});
            auto matcher = std::make_shared<Matcher>(matmul_pattern, "MatMulPattern");

            if (matcher->match(node_shared)) {
                QDQ_DEBUG_LOG << "        [ INFO ]   Matched MatMul with weights pattern" << std::endl;
                auto pattern_map = matcher->get_pattern_value_map();
                apply_scale_to_weight(pattern_map, weights_dq_block);
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    visited.insert(in.get_node());
                }
                return;
            }
        }

        // Case 4: Multiply with weights
        {
            auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
            auto multiply_pattern = wrap_type<ov::op::v1::Multiply>({any_input(), weights_dq_block});
            auto matcher = std::make_shared<Matcher>(multiply_pattern, "MultiplyPattern");

            if (matcher->match(node_shared)) {
                QDQ_DEBUG_LOG << "        [ INFO ]   Matched Multiply with weights pattern" << std::endl;
                auto pattern_map = matcher->get_pattern_value_map();
                apply_scale_to_weight(pattern_map, weights_dq_block);
                for (const auto& in : matcher->get_match_root()->input_values()) {
                    visited.insert(in.get_node());
                }
                return;
            }
        }

        // Case 5: FakeQuantize (un-stripped) — adjust its range constants so scale propagates through
        if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
            QDQ_DEBUG_LOG << "        [ INFO ]   Matched un-stripped FQ in backward path" << std::endl;
            adjust_fq_ranges(node);
            return;
        }
    };

    // Forward propagation callback: handle Add nodes (backward-propagate into other branch)
    // and FakeQuantize nodes (adjust range constants so scale propagates through)
    auto forward_propagate_callback = [&](ov::Node* node) {
        // Handle un-stripped FakeQuantize: adjust its range constants
        if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
            QDQ_DEBUG_LOG << "    [ FORWARD ] Adjusting un-stripped FQ: " << node->get_friendly_name() << std::endl;
            adjust_fq_ranges(node);
            return;
        }

        if (!ov::is_type<ov::op::v1::Add>(node))
            return;

        QDQ_DEBUG_LOG << "    [ FORWARD ] Reached Add: " << node->get_friendly_name() << std::endl;

        // For each input of the Add, backward-propagate the scale to adjust weights
        // on branches that haven't been visited yet (the "other" branch of the residual)
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            auto input_node = node->get_input_node_ptr(i);
            if (visited.count(input_node))
                continue;

            QDQ_DEBUG_LOG << "    [ FORWARD ]   Backward propagating scale=" << current_scale_divisor
                          << " into Add input " << i << ": " << input_node->get_friendly_name() << std::endl;
            ov::op::util::visit_path(input_node, visited, adjust_weights_scale, backward_skip_predicate);
        }
    };

    // Forward propagation skip predicate: stop at scale-invariant nodes (MVN, Softmax)
    auto forward_skip_predicate = [](ov::Node* n) {
        return ov::is_type_any_of<ov::op::v0::MVN, ov::op::v6::MVN,
                                  ov::op::v1::Softmax, ov::op::v8::Softmax>(n->shared_from_this());
    };

    // Single-pass: strip each FQ and immediately propagate scale before processing the next FQ.
    // This ensures that forward propagation from FQ_A adjusts downstream FQ_B's ranges,
    // so when FQ_B is processed its y_scale is recomputed from the adjusted (smaller) ranges.
    // Without this, cascaded FQs would each independently compute y_scale from original ranges,
    // potentially double-scaling weights.
    for (const auto& node : f->get_ordered_ops()) {
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node);
        if (!fq || transformation_callback(node)) {
            continue;
        }

        if (!levels_to_strip.count(fq->get_levels())) {
            continue;
        }

        QDQ_DEBUG_LOG << "\n======== Processing FQ: " << fq->get_friendly_name() << " (levels=" << fq->get_levels()
                      << ") ========" << std::endl;

        // Check if FQ has valid constants
        if (!check_fq_constants(fq)) {
            QDQ_DEBUG_LOG << "  [ DEBUG ] Skipped: invalid or degenerate FQ constants" << std::endl;
            continue;
        }

        // Compute y_scale (dequantization scale) for this FQ:
        //   y_scale = (input_high - input_low) / (levels - 1)
        // Note: these ranges may have been adjusted by forward propagation from a preceding FQ,
        // so y_scale reflects the current (post-adjustment) scale rather than the original.
        const auto& input_low = fq->input_value(1);
        const auto& input_high = fq->input_value(2);

        auto levels_minus_one_node = ov::op::v0::Constant::create(
            input_high.get_element_type(), ov::Shape{}, {static_cast<float>(fq->get_levels() - 1)});
        auto input_range_node =
            ov::op::util::make_try_fold<ov::op::v1::Subtract>(input_high, input_low);
        auto y_scale_node =
            ov::op::util::make_try_fold<ov::op::v1::Divide>(input_range_node, levels_minus_one_node);

        // Fold the subgraph to a constant and extract the scalar value
        auto y_scale_const = ov::as_type_ptr<ov::op::v0::Constant>(y_scale_node);
        if (!y_scale_const) {
            QDQ_DEBUG_LOG << "  [ DEBUG ] Skipped: could not fold y_scale to a constant" << std::endl;
            continue;
        }
        float y_scale = y_scale_const->cast_vector<float>()[0];

        QDQ_DEBUG_LOG << "  [ DEBUG ] Input range: [" << get_const_float_value(input_low.get_node_shared_ptr()) << ", "
                      << get_const_float_value(input_high.get_node_shared_ptr()) << "] = "
                      << (get_const_float_value(input_high.get_node_shared_ptr()) -
                          get_const_float_value(input_low.get_node_shared_ptr()))
                      << std::endl;
        QDQ_DEBUG_LOG << "  [ DEBUG ] Y scale (dequant scale): " << y_scale << " (levels=" << fq->get_levels() << ")" << std::endl;

        // Remember the FQ's input node before stripping (this is the node that feeds into the FQ)
        auto propagation_root = fq->get_input_node_shared_ptr(0);

        QDQ_DEBUG_LOG << "  [ INFO ] Removing FQ: " << fq->get_friendly_name() << std::endl;
        OPENVINO_ASSERT(replace_output_update_name(fq->output(0), fq->input_value(0)), "FQ stripping failed");
        model_changed = true;

        if (y_scale > threshold) {
            current_scale_divisor = y_scale * ratio;

            QDQ_DEBUG_LOG << "  [ INFO ] y_scale=" << y_scale << " > threshold=" << threshold
                          << ", running scale propagation (scale_divisor=" << current_scale_divisor
                          << ") from: " << propagation_root->get_friendly_name() << std::endl;

            // Step 1: Backward propagation from FQ position to scale weights feeding into it
            QDQ_DEBUG_LOG << "  [ INFO ] --- Backward propagation ---" << std::endl;
            ov::op::util::visit_path(propagation_root.get(), visited, adjust_weights_scale, backward_skip_predicate);

            // Step 2: Forward propagation from FQ position to balance Add branches
            // and adjust downstream un-stripped FQ ranges (including other int16 FQs
            // that haven't been processed yet — their ranges will be adjusted so their
            // y_scale is recomputed correctly when we reach them in the topological walk).
            QDQ_DEBUG_LOG << "  [ INFO ] --- Forward propagation ---" << std::endl;
            std::unordered_set<ov::Node*> forward_visited;
            ov::op::util::visit_path_forward(propagation_root.get(), forward_visited,
                                             forward_propagate_callback, forward_skip_predicate);
        }
        QDQ_DEBUG_LOG << "========================================" << std::endl;
    }

    // Dump model after FQ removal and scale adjustment
    VISUALIZE_MODEL(visualize_folder, "02_after_fq_removal", "after FQ removal");
    SERIALIZE_MODEL(serialize_folder, "02_after_fq_removal", "after FQ removal");

    QDQ_DEBUG_LOG << "\n[ INFO ] === QDQ Stripping Pass Completed ===" << std::endl;

    // Dump final model
    VISUALIZE_MODEL(visualize_folder, "03_final", "final");
    SERIALIZE_MODEL(serialize_folder, "03_final", "final");

    return model_changed;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov