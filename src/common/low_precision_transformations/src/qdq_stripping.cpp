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
    float max_y_scale = 0.0f;

    QDQ_DEBUG_LOG << "\n[ INFO ] === QDQ Stripping Pass ===" << std::endl;
    QDQ_DEBUG_LOG << "[ INFO ] Total nodes in graph: " << f->get_ops().size() << std::endl;
    QDQ_DEBUG_LOG << "[ INFO ] Levels to strip: ";
    for (auto level : levels_to_strip) {
        QDQ_DEBUG_LOG << level << " ";
    }
    QDQ_DEBUG_LOG << std::endl;

    NodeVector scale_invariant_nodes;

    // Process each FQ node
    for (const auto& node : f->get_ordered_ops()) {
        if (ov::is_type_any_of<ov::op::v0::MVN, ov::op::v6::MVN, ov::op::v1::Softmax, ov::op::v8::Softmax>(node)) {
            scale_invariant_nodes.push_back(node);
            continue;
        }

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

        // Compute y_scale (dequantization scale) for this FQ using opset operations:
        //   From QuantizeLinear -> FakeQuantize conversion (ONNX frontend):
        //     input_low  = y_scale * (output_low  - zero_point)
        //     input_high = y_scale * (output_high - zero_point)
        //   Therefore:
        //     y_scale (dequant scale) = (input_high - input_low) / (levels - 1)
        //
        //   We use (levels - 1) instead of (output_high - output_low) because
        //   ConvertQuantizeDequantize may fold the Dequantize into the FQ,
        //   making the output range no longer the integer type range.
        //   The "levels" attribute always encodes the original integer range.
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

        if (y_scale > max_y_scale) {
            max_y_scale = y_scale;
        }

        QDQ_DEBUG_LOG << "  [ INFO ] Removing FQ: " << fq->get_friendly_name() << std::endl;
        OPENVINO_ASSERT(replace_output_update_name(fq->output(0), fq->input_value(0)), "FQ stripping failed");
        model_changed = true;
        QDQ_DEBUG_LOG << "========================================" << std::endl;
    }

    QDQ_DEBUG_LOG << "  [ INFO ] Max y_scale (dequant scale) across model: " << max_y_scale << std::endl;

    // Dump model after FQ removal
    VISUALIZE_MODEL(visualize_folder, "02_after_fq_removal", "after FQ removal");
    SERIALIZE_MODEL(serialize_folder, "02_after_fq_removal", "after FQ removal");

    const auto threshold = 1.f;
    if (max_y_scale <= threshold) {
        QDQ_DEBUG_LOG << "  [ INFO ] No scale adjustment needed, skipping" << std::endl;
    }

    if (max_y_scale > threshold && scale_invariant_nodes.empty()) {
        QDQ_DEBUG_LOG << "  [ INFO ] Scale adjustment is needed, but no scale-invariant nodes found, so this stage "
                         "is skipped"
                      << std::endl;
    }

    if (max_y_scale > threshold && !scale_invariant_nodes.empty()) {
        QDQ_DEBUG_LOG << "\n======== Applying backward scale adjustment ========" << std::endl;
        std::unordered_set<ov::Node*> visited;
        auto skip_node_predicate = [](ov::Node* n) {
            return false;
        };

        // Allow directly forcing scale_divisor via environment variable
        float scale_divisor;
        auto force_scale_str = ov::util::getenv_string("OV_QDQ_FORCE_SCALE");
        if (!force_scale_str.empty()) {
            try {
                scale_divisor = std::stof(force_scale_str);
                QDQ_DEBUG_LOG << "  [ INFO ] Using scale_divisor directly from env: " << scale_divisor
                              << " (original max_y_scale: " << max_y_scale << ")" << std::endl;
            } catch (const std::exception& e) {
                QDQ_DEBUG_LOG << "  [ WARNING ] Invalid OV_QDQ_FORCE_SCALE value: " << force_scale_str
                              << ", falling back to multiplier approach" << std::endl;
                force_scale_str.clear();
            }
        }

        // If not directly set, compute from max_y_scale with optional multiplier
        if (force_scale_str.empty()) {  
            float max_y_scale_multiplier = 1.0f;
            auto multiplier_str = ov::util::getenv_string("OV_QDQ_SCALE_MULTIPLIER");
            if (!multiplier_str.empty()) {
                try {
                    max_y_scale_multiplier = std::stof(multiplier_str);
                    QDQ_DEBUG_LOG << "  [ INFO ] Using max_y_scale multiplier from env: " << max_y_scale_multiplier
                                  << std::endl;
                } catch (const std::exception& e) {
                    QDQ_DEBUG_LOG << "  [ WARNING ] Invalid OV_QDQ_SCALE_MULTIPLIER value: " << multiplier_str
                                  << ", using default 1.0" << std::endl;
                }
            }
            scale_divisor = max_y_scale * max_y_scale_multiplier;
            QDQ_DEBUG_LOG << "  [ INFO ] Scale divisor: " << scale_divisor
                          << " (max_y_scale: " << max_y_scale << " * " << max_y_scale_multiplier << ")" << std::endl;
        }

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
                          << scale_divisor << std::endl;

            // Create new scaled constant: divide dequantization scale by scale_divisor
            auto divisor_const =
                ov::op::v0::Constant::create(original_constant->get_output_element_type(0), {}, {scale_divisor});
            auto new_constant = ov::op::util::make_try_fold<ov::op::v1::Divide>(original_constant, divisor_const);

            // Create new multiply with the scaled constant
            auto new_multiply = old_multiply->clone_with_new_inputs({mul_input, new_constant});

            ov::replace_node(old_multiply, new_multiply);
            visited.insert(new_multiply.get());
        };

        auto adjust_weights_scale = [&](ov::Node* node) {
            QDQ_DEBUG_LOG << "    [ INFO ] adjust_weights_scale called for node with type: " << node->get_type_name()
                          << ", with name: " << node->get_friendly_name() << std::endl;
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
                    return;
                }
            }

            // Case 2: MatMul with weights
            {
                auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
                auto matmul_pattern = wrap_type<ov::op::v0::MatMul>({any_input(), weights_dq_block});
                auto matcher = std::make_shared<Matcher>(matmul_pattern, "MatMulPattern");

                if (matcher->match(node_shared)) {
                    QDQ_DEBUG_LOG << "        [ INFO ]   Matched MatMul with weights pattern" << std::endl;
                    auto pattern_map = matcher->get_pattern_value_map();
                    apply_scale_to_weight(pattern_map, weights_dq_block);
                    return;
                }
            }

            // Case 3: Multiply with weights
            {
                auto weights_dq_block = std::make_shared<WeightsDequantizationBlock>();
                auto multiply_pattern = wrap_type<ov::op::v1::Multiply>({any_input(), weights_dq_block});
                auto matcher = std::make_shared<Matcher>(multiply_pattern, "MultiplyPattern");

                if (matcher->match(node_shared)) {
                    QDQ_DEBUG_LOG << "        [ INFO ]   Matched Multiply with weights pattern" << std::endl;
                    auto pattern_map = matcher->get_pattern_value_map();
                    apply_scale_to_weight(pattern_map, weights_dq_block);
                    return;
                }
            }
        };
        for (const auto& node : scale_invariant_nodes) {
            QDQ_DEBUG_LOG << "  [ INFO ] Processing scale-invariant node: " << node->get_friendly_name()
                          << " type=" << node->get_type_name() << std::endl;
            ov::op::util::visit_path(node.get(), visited, adjust_weights_scale, skip_node_predicate);
        }
        QDQ_DEBUG_LOG << "========================================" << std::endl;
    }

    QDQ_DEBUG_LOG << "\n[ INFO ] === QDQ Stripping Pass Completed ===" << std::endl;

    // Dump final model
    VISUALIZE_MODEL(visualize_folder, "03_final", "final");
    SERIALIZE_MODEL(serialize_folder, "03_final", "final");

    return model_changed;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov