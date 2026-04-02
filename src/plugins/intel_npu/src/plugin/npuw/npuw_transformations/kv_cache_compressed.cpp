// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_compressed.hpp"

#include <limits>

#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "../util.hpp"

namespace opp = ov::pass::pattern;
namespace {
constexpr auto g_key_cache_name = "key";
constexpr auto g_value_cache_name = "value";
constexpr auto g_past_name = "past_key_values";
constexpr auto g_present_name = "present";

bool isKey(const std::string& n) {
    return n.find(g_key_cache_name) != std::string::npos;
}
bool isValue(const std::string& n) {
    return n.find(g_value_cache_name) != std::string::npos;
}
bool isPast(const std::string& n) {
    return n.find(g_past_name) != std::string::npos;
}
bool isPresent(const std::string& n) {
    return n.find(g_present_name) != std::string::npos;
}
std::string cacheName(bool is_key) {
    return is_key ? g_key_cache_name : g_value_cache_name;
}

// ── V2 helper functions (ONNX DynamicQuantizeLinear style) ────────────────────

std::shared_ptr<ov::Node> find_min_value(const ov::Output<ov::Node>& input, size_t reduce_axes) {
    const auto& input_min = std::make_shared<ov::op::v1::ReduceMin>(
        input,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {reduce_axes}),
        true);
    const auto& zero_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0});
    return std::make_shared<ov::op::v1::Minimum>(zero_node, input_min);
}

std::shared_ptr<ov::Node> find_max_value(const ov::Output<ov::Node>& input, size_t reduce_axes) {
    const auto& input_max = std::make_shared<ov::op::v1::ReduceMax>(
        input,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {reduce_axes}),
        true);
    const auto& zero_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0});
    return std::make_shared<ov::op::v1::Maximum>(zero_node, input_max);
}

std::shared_ptr<ov::Node> quantize_linear(ov::Output<ov::Node> x,
                                          ov::Output<ov::Node> x_span,
                                          ov::Output<ov::Node> quant_range_span,
                                          ov::Output<ov::Node> y_zero_point) {
    const auto& x_scaled =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Multiply>(x, quant_range_span), x_span);
    const auto& x_rounded = std::make_shared<ov::op::v5::Round>(x_scaled, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    const auto& y_zero_point_f32 = std::make_shared<ov::op::v0::Convert>(y_zero_point, ov::element::f32);
    const auto& result_shifted = std::make_shared<ov::op::v1::Add>(x_rounded, y_zero_point_f32);
    const auto& result_clamped = std::make_shared<ov::op::v0::Clamp>(result_shifted, 0, 255);
    return std::make_shared<ov::op::v0::Convert>(result_clamped, ov::element::u8);
}

ov::OutputVector dynamic_quantize_linear(std::shared_ptr<ov::Node> input, size_t reduction_axe) {
    const auto& x = input;
    // quantization range for uint8 is [0, 255] per ONNX DynamicQuantizeLinear spec
    const auto& quant_range_min = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0});
    const auto& quant_range_max = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {255});
    const auto& quant_range_span = std::make_shared<ov::op::v1::Subtract>(quant_range_max, quant_range_min);

    const auto& x_max = find_max_value(x, reduction_axe);
    const auto& x_min = find_min_value(x, reduction_axe);
    const auto& x_span = std::make_shared<ov::op::v1::Subtract>(x_max, x_min);
    const auto& y_scale = std::make_shared<ov::op::v1::Divide>(x_span, quant_range_span);
    // ONNX spec: intermediate_zero_point = qmin - min(x) / y_scale
    // Note: this is qmin - (x_min / y_scale), NOT (qmin - x_min) / y_scale
    const auto& x_min_div_scale = std::make_shared<ov::op::v1::Divide>(x_min, y_scale);
    const auto& intermediate_zero_point =
        std::make_shared<ov::op::v5::Round>(std::make_shared<ov::op::v1::Subtract>(quant_range_min, x_min_div_scale),
                                            ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    const auto& y_zero_point =
        std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Clamp>(intermediate_zero_point, 0, 255),
                                              ov::element::u8);

    const auto& y = quantize_linear(x, x_span, quant_range_span, y_zero_point);
    return {y, y_scale, y_zero_point};
}

// ── V3 helper function (compiler pattern style) ──────────────────────────────

ov::OutputVector build_dynamic_quantize_linear(const ov::Output<ov::Node>& input,
                                               size_t reduction_axis,
                                               const std::string& name_prefix) {
    auto make_name = [&name_prefix](const std::string& suffix) {
        return name_prefix + "/" + suffix;
    };

    auto reduceAxis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {reduction_axis});
    auto cst_255 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {255.0f});
    auto cst_inv255 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f / 255.0f});
    auto cst_zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.0f});

    auto reduceMin = std::make_shared<ov::op::v1::ReduceMin>(input, reduceAxis, true);
    reduceMin->set_friendly_name(make_name("ReduceMin"));

    auto reduceMax = std::make_shared<ov::op::v1::ReduceMax>(input, reduceAxis, true);
    reduceMax->set_friendly_name(make_name("ReduceMax"));

    auto minClamped = std::make_shared<ov::op::v0::Clamp>(reduceMin, std::numeric_limits<float>::lowest(), 0.0);
    minClamped->set_friendly_name(make_name("Clamp_min"));

    auto maxClamped = std::make_shared<ov::op::v0::Clamp>(reduceMax, 0.0, std::numeric_limits<float>::max());
    maxClamped->set_friendly_name(make_name("Clamp_max"));

    auto subtractSpan = std::make_shared<ov::op::v1::Subtract>(maxClamped, minClamped);
    subtractSpan->set_friendly_name(make_name("Subtract_span"));

    auto multiplyScale = std::make_shared<ov::op::v1::Multiply>(subtractSpan, cst_inv255);
    multiplyScale->set_friendly_name(make_name("Multiply_scale"));

    auto multiplyInput = std::make_shared<ov::op::v1::Multiply>(input, cst_255);
    multiplyInput->set_friendly_name(make_name("Multiply_input_255"));

    // ONNX spec: zp = qmin - (minClamped / scale)
    auto cst_qmin = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {-128.0f});
    auto minDivScale = std::make_shared<ov::op::v1::Divide>(minClamped, multiplyScale);
    minDivScale->set_friendly_name(make_name("Divide_min_by_scale"));

    auto zpFloat = std::make_shared<ov::op::v1::Subtract>(cst_qmin, minDivScale);
    zpFloat->set_friendly_name(make_name("Subtract_zp"));

    auto divideSpan = std::make_shared<ov::op::v1::Divide>(multiplyInput, subtractSpan);
    divideSpan->set_friendly_name(make_name("Divide_span"));

    auto roundZp = std::make_shared<ov::op::v5::Round>(zpFloat, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    roundZp->set_friendly_name(make_name("Round_zp"));

    auto roundSpan = std::make_shared<ov::op::v5::Round>(divideSpan, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    roundSpan->set_friendly_name(make_name("Round_span"));

    auto clampZp = std::make_shared<ov::op::v0::Clamp>(roundZp, -128.0, 127.0);
    clampZp->set_friendly_name(make_name("Clamp_zp"));

    auto addQuant = std::make_shared<ov::op::v1::Add>(roundSpan, clampZp);
    addQuant->set_friendly_name(make_name("Add_quant"));

    auto clampOutput = std::make_shared<ov::op::v0::Clamp>(addQuant, -128.0, 127.0);
    clampOutput->set_friendly_name(make_name("Clamp_output"));

    auto convertZp = std::make_shared<ov::op::v0::Convert>(clampZp, ov::element::i8);
    convertZp->set_friendly_name(make_name("Convert_zp"));

    auto convertOutput = std::make_shared<ov::op::v0::Convert>(clampOutput, ov::element::i8);
    convertOutput->set_friendly_name(make_name("Convert_output"));

    return {convertOutput->output(0), multiplyScale->output(0), convertZp->output(0)};
}

}  // anonymous namespace

// ── V2: ONNX DynamicQuantizeLinear style, u8 [0, 255] ────────────────────────

ov::npuw::DecomposeDynamicQuantize2::DecomposeDynamicQuantize2() {
    auto dynamic_quantize = opp::wrap_type<ov::op::internal::DynamicQuantize>({opp::any_input()});
    auto callback = [=](opp::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto dq_ptr = node_to_output.at(dynamic_quantize).get_node_shared_ptr();

        // This pass only handles asymmetric u8 (ONNX DynamicQuantizeLinear style).
        // Symmetric or non-u8 nodes are handled by other passes.
        auto dq_node = ov::as_type_ptr<ov::op::internal::DynamicQuantize>(dq_ptr);
        const auto& attrs = dq_node->get_attrs();
        if (attrs.quantization_type != ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric ||
            attrs.quantization_dt != ov::element::u8)
            return false;

        LOG_DEBUG("Found DynamicQuantize : " << dq_ptr->get_friendly_name() << " decomposing");
        LOG_BLOCK();

        constexpr size_t k_embedding_index = 3;
        constexpr size_t v_embedding_index = 2;

        auto reduction_axis = isKey(dq_ptr->get_friendly_name()) ? k_embedding_index : v_embedding_index;

        auto dq_input = dq_ptr->input_value(0);
        auto has_zp = dq_ptr->outputs().size() == 3;

        auto dq_results = dynamic_quantize_linear(dq_input.get_node_shared_ptr(), reduction_axis);

        dq_ptr->output(0).replace(dq_results[0]);
        dq_ptr->output(1).replace(dq_results[1]);
        if (has_zp) {
            dq_ptr->output(2).replace(dq_results[2]);
        }

        return true;
    };

    register_matcher(std::make_shared<opp::Matcher>(dynamic_quantize, "DecomposeDynamicQuantize"), callback);
}

// ── V3: compiler pattern style, i8 [-128, 127] ───────────────────────────────

ov::npuw::DecomposeDynamicQuantize3::DecomposeDynamicQuantize3() {
    auto dynamic_quantize = opp::wrap_type<ov::op::internal::DynamicQuantize>({opp::any_input()});
    auto callback = [=](opp::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto dq_ptr = node_to_output.at(dynamic_quantize).get_node_shared_ptr();

        // This pass only handles symmetric i8 (signed integer, range [-128,127]).
        // Asymmetric or non-i8 nodes are handled by other passes.
        auto dq_node = ov::as_type_ptr<ov::op::internal::DynamicQuantize>(dq_ptr);

        LOG_DEBUG("Found DynamicQuantize : " << dq_ptr->get_friendly_name() << " decomposing");
        LOG_BLOCK();

        constexpr size_t k_embedding_index = 3;
        constexpr size_t v_embedding_index = 2;

        auto reduction_axis = isKey(dq_ptr->get_friendly_name()) ? k_embedding_index : v_embedding_index;

        auto dq_input = dq_ptr->input_value(0);
        auto has_zp = dq_ptr->outputs().size() == 3;

        auto dq_results = build_dynamic_quantize_linear(dq_input, reduction_axis, dq_ptr->get_friendly_name());

        dq_ptr->output(0).replace(dq_results[0]);
        dq_ptr->output(1).replace(dq_results[1]);
        if (has_zp) {
            dq_ptr->output(2).replace(dq_results[2]);
        }

        return true;
    };

    register_matcher(std::make_shared<opp::Matcher>(dynamic_quantize, "DecomposeDynamicQuantize"), callback);
}

// ── V1: handcrafted symmetric-style, i8 [-127, 127] ──────────────────────────

ov::npuw::DecomposeDynamicQuantize::DecomposeDynamicQuantize() {
    auto dynamic_quantize = opp::wrap_type<ov::op::internal::DynamicQuantize>({opp::any_input()});

    auto callback = [=](opp::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto dq_ptr = node_to_output.at(dynamic_quantize).get_node_shared_ptr();

        // DecomposeDynamicQuantize2 (asymmetric u8) runs first in the pass manager.
        // This pass handles whatever remains — dispatch on quantization_dt for the output range.
        auto dq_node = ov::as_type_ptr<ov::op::internal::DynamicQuantize>(dq_ptr);
        const auto& attrs = dq_node->get_attrs();

        const auto quant_dt = attrs.quantization_dt;
        float quant_max, quant_min;
        if (quant_dt == ov::element::i8) {
            quant_max = 127.0f;
            quant_min = -127.0f;
        } else if (quant_dt == ov::element::i4) {
            quant_max = 7.0f;
            quant_min = -7.0f;
        } else {
            LOG_WARN("DecomposeDynamicQuantize: unsupported dtype " << quant_dt);
            return false;
        }

        LOG_DEBUG("Found DynamicQuantize : " << dq_ptr->get_friendly_name() << " decomposing");
        LOG_BLOCK();

        // dequantize k and v caches in different layouts
        constexpr size_t k_embedding_index = 3;
        constexpr size_t v_embedding_index = 2;

        auto reduction_axis = isKey(dq_ptr->get_friendly_name()) ? k_embedding_index : v_embedding_index;

        //
        auto cst_0 =
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f / quant_max});  // for scale normalization
        auto cst_1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});     // optional offset

        auto dq_input = dq_ptr->input_value(0);
        auto has_zp = dq_ptr->outputs().size() == 3;

        auto set_friendly_name = [&dq_ptr](auto node, auto name) {
            node->set_friendly_name(dq_ptr->get_friendly_name() + "/" + name);
        };

        // --- Compute min and max per token using reduction dim ---
        auto reduce_min = std::make_shared<ov::op::v1::ReduceMin>(
            dq_input,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {reduction_axis}),  // axis = feature dim D
            true);                                                                           // keepdims

        set_friendly_name(reduce_min, "ReduceMin");

        auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(
            dq_input,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {reduction_axis}),  // axis = feature dim D
            true);

        set_friendly_name(reduce_max, "ReduceMax");

        // optional clamp
        auto clamped_min = std::make_shared<ov::op::v0::Clamp>(reduce_min, quant_min, quant_max);
        auto clamped_max = std::make_shared<ov::op::v0::Clamp>(reduce_max, quant_min, quant_max);

        set_friendly_name(clamped_min, "clamp_min");
        set_friendly_name(clamped_max, "clamp_max");

        // --- Compute range and scale ---
        auto range = std::make_shared<ov::op::v1::Subtract>(clamped_max, clamped_min);  // max - min
        auto scale = std::make_shared<ov::op::v1::Multiply>(range, cst_0);              // scale = range / quant_max

        set_friendly_name(range, "subtract_range");
        set_friendly_name(scale, "scale");

        auto scale_converted = std::make_shared<ov::op::v0::Convert>(scale, ov::element::f32);
        set_friendly_name(scale_converted, "scale_converted");

        std::shared_ptr<ov::Node> zp, zp_clamped;
        if (has_zp) {
            // --- Compute zero-point ---
            auto zp_float = std::make_shared<ov::op::v1::Divide>(cst_1, scale);  // optional: (0 - min)/scale
            set_friendly_name(zp_float, "Zp/Divide");

            auto zp_rounded = std::make_shared<ov::op::v5::Round>(zp_float, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
            set_friendly_name(zp_rounded, "Zp/Round");

            zp_clamped = std::make_shared<ov::op::v0::Clamp>(zp_rounded, quant_min, quant_max);
            set_friendly_name(zp_clamped, "Zp/Clamp");

            zp = std::make_shared<ov::op::v0::Convert>(zp_clamped, ov::element::i8);  // int8 zero-point
            set_friendly_name(zp, "Zp/Convert");
        }

        // --- Quantize input ---
        auto normalized = std::make_shared<ov::op::v1::Divide>(dq_input, scale);  // input / scale
        set_friendly_name(normalized, "Divide");

        auto rounded = std::make_shared<ov::op::v5::Round>(normalized, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        set_friendly_name(rounded, "Round");

        std::shared_ptr<ov::Node> with_zp = rounded;
        if (has_zp) {
            with_zp = std::make_shared<ov::op::v1::Add>(rounded, zp_clamped);  // add zero-point
            set_friendly_name(with_zp, "ZP/Add");
        }
        auto quantized_clamped = std::make_shared<ov::op::v0::Clamp>(with_zp, quant_min, quant_max);
        set_friendly_name(quantized_clamped, "Clamp2");

        // TODO: this convert might be optional???
        auto quantized_output = std::make_shared<ov::op::v0::Convert>(quantized_clamped, quant_dt);
        set_friendly_name(quantized_output, "Convert2");

        // --- Optional: expose outputs ---
        // --- Rewire outputs ---
        dq_ptr->output(0).replace(quantized_output->output(0));
        dq_ptr->output(1).replace(scale_converted->output(0));

        if (has_zp) {
            dq_ptr->output(2).replace(zp->output(0));
        }

        return true;
    };

    register_matcher(std::make_shared<opp::Matcher>(dynamic_quantize, "DecomposeDynamicQuantize"), callback);
}

// inject DynamicQuantize/Dynamic dequantize ops on kv-cache passes before matmuls
void ov::npuw::run_kv_cache_dynamic_quantization_passes(const std::shared_ptr<ov::Model>& model,
                                                        const KVCacheCompressionParams& params) {
    //  Validate SDPA patterns and extract key nodes
    auto pattern_nodes_list = ov::npuw::util::find_all_sdpa_pattern_nodes(model);

    if (pattern_nodes_list.empty()) {
        LOG_WARN("Failed to find SDPA pattern nodes in model "
                 << model->get_friendly_name()
                 << ". Will skip past-cache dequantization rewiring and keep present-cache quantization only.");
    }

    // in per token quantization lets use 1 scale/zp per token, effectively means shape should be clear from embeddings
    auto clear_embedding_index = [](auto shape_node, bool k_tensor) {
        constexpr size_t k_embedding_index = 3;
        constexpr size_t v_embedding_index = 2;

        auto kv_shape = shape_node->get_output_partial_shape(0);
        auto embedding_index = k_tensor ? k_embedding_index : v_embedding_index;

        std::vector<ov::Dimension> new_dims;
        for (int64_t i = 0; i < kv_shape.rank().get_length(); ++i) {
            if (i == embedding_index) {  // head_dim axis → set to 1
                new_dims.push_back(1);
            } else {
                new_dims.push_back(kv_shape[i]);
            }
        }
        return new_dims;
    };

    auto create_parameter_with_name = [&model](auto precision, auto dims, auto name) {
        auto new_param = std::make_shared<ov::op::v0::Parameter>(precision, dims);
        new_param->set_friendly_name(name);
        new_param->output(0).get_tensor().set_names({name});
        model->add_parameters({new_param});
        return new_param;
    };

    auto create_result_with_name = [&model](auto result_node, auto name) {
        auto new_res = std::make_shared<ov::op::v0::Result>(result_node);
        new_res->set_friendly_name(name);
        new_res->output(0).get_tensor().set_names({name});
        model->add_results({new_res});
    };

    size_t pattern_index = 0;
    auto make_name = [&pattern_index](auto base_name) {
        return base_name + std::string("/") + std::to_string(pattern_index);
    };

    // Shared lambda: create DynamicQuantize node on a given input, expose scale/zp as Results.
    // Returns the DynamicQuantize node so caller can wire its output(0) as needed.
    auto create_dynamic_quantize =
        [&make_name, &create_result_with_name](
            const ov::Output<ov::Node>& dq_input,
            bool is_key,
            const KVCacheCompressionConfig& cfg) -> std::shared_ptr<ov::op::internal::DynamicQuantize> {
        const std::string kv_name = cacheName(is_key);

        const bool is_asym = cfg.quantization_type == KVCacheCompressionConfig::QuantizationType::Asymmetric;

        auto rank = dq_input.get_partial_shape().size();
        std::vector<uint64_t> shape_group_size(rank, 1);

        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_dt = cfg.quantization_dt;
        dq_config.quantization_type = is_asym ? ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric
                                              : ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
        dq_config.scale_dt = ov::element::f32;
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
        dq_config.group_sizes = shape_group_size;

        if (is_asym) {
            dq_config.zp_dt = cfg.quantization_dt;  // zp same type as quantized data
        }

        auto kv_dyn_quant = std::make_shared<ov::op::internal::DynamicQuantize>(dq_input, dq_config);
        kv_dyn_quant->set_friendly_name(make_name("DynamicQuantize") + "/" + kv_name);

        create_result_with_name(kv_dyn_quant->output(1),
                                make_name("DynamicQuantize") + "/" + g_present_name + "/" + kv_name + "/scale");

        if (is_asym) {
            create_result_with_name(kv_dyn_quant->output(2),
                                    make_name("DynamicQuantize") + "/" + g_present_name + "/" + kv_name + "/zp");
        }

        return kv_dyn_quant;
    };

    // helper to recreate dequantisation nodes - TODO: probably better to insert Dequantize Node, than decompose it or
    // not.
    auto create_dequant_nodes = [&model, &create_parameter_with_name, &clear_embedding_index, &make_name](
                                    std::shared_ptr<ov::Node> start_node,
                                    bool isKey,
                                    const KVCacheCompressionConfig& cfg) {
        const std::string node_name = isKey ? g_key_cache_name : g_value_cache_name;

        // TODO: adding back slash here kills partitioning - fix that
        auto make_dq_name = [&make_name, &node_name](auto base_name) {
            return make_name("DynamicQuantize") + node_name + "/" + base_name;
        };
        auto make_dq_param_name = [&make_name, &node_name](auto base_name) {
            return make_name("DynamicQuantize") + "/" + g_past_name + "/" + node_name + "/" + base_name;
        };

        // Take snapshot of the ORIGINAL edge BEFORE building new nodes
        auto concat_consumers_set = start_node->output(0).get_target_inputs();

        LOG_INFO("Found Dequantize for " << node_name << " insertion point after: " << start_node->get_name()
                                         << ") shape: " << start_node->get_shape());

        // reconstruct k and v caches intgeres values  to matmul using one of Dequantize approach
        // use zp on quant/dequant only in case od assym
        std::shared_ptr<ov::Node> fp_subtracted_zp = start_node;
        if (cfg.quantization_type == KVCacheCompressionConfig::QuantizationType::Asymmetric) {
            //  Subtract zero-point - TODO: share this memory with DynamicQuantize/read/assign?
            auto zp = create_parameter_with_name(cfg.quantization_dt,
                                                 clear_embedding_index(start_node, isKey),
                                                 make_dq_param_name("zp"));

            // this probably to be optimized by compiler - but for now we need it to avoid types mismatch
            auto converted_zp = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f32);
            converted_zp->set_friendly_name(make_dq_name("zp_convert"));

            fp_subtracted_zp = std::make_shared<ov::op::v1::Subtract>(start_node, converted_zp);
            fp_subtracted_zp->set_friendly_name(make_dq_name("zp_sub"));
        }

        // Multiply by scale
        auto fp_scale = create_parameter_with_name(ov::element::f32,
                                                   clear_embedding_index(fp_subtracted_zp, isKey),
                                                   make_dq_param_name("scale"));

        auto dequantized = std::make_shared<ov::op::v1::Multiply>(fp_subtracted_zp, fp_scale);
        dequantized->set_friendly_name(make_dq_name("scale"));

        // substitute concat consumers to read from dq chain
        for (auto&& concat_consumer : concat_consumers_set) {
            concat_consumer.replace_source_output(dequantized);
        }
    };

    bool quantize_inserted_before_concat = false;

    for (auto&& pattern_nodes : pattern_nodes_list) {
        if (!pattern_nodes.past_key_concat_node) {
            LOG_INFO("NO k-cache concat node");
        }
        if (!pattern_nodes.past_value_concat_node) {
            LOG_INFO("NO v-cache concat node");
        }
        if (!pattern_nodes.past_value_concat_node || !pattern_nodes.past_key_concat_node) {
            LOG_INFO("skipping Dequantization for this pattern:");
            continue;
        }

        create_dequant_nodes(pattern_nodes.past_key_concat_node->input_value(0).get_node_shared_ptr(),
                             true,
                             params.key);
        create_dequant_nodes(pattern_nodes.past_value_concat_node->input_value(0).get_node_shared_ptr(),
                             false,
                             params.value);

        pattern_index++;
    }

    model->validate_nodes_and_infer_types();

    // For DynamicQuantize we need just to iterate over present.key/values results and insert DynQuante ops
    auto original_results = model->get_results();
    for (const auto& result : original_results) {
        auto result_name = result->get_friendly_name();
        LOG_BLOCK();
        LOG_DEBUG("Probing result  for DynamicQuantize: " << result_name);

        auto is_key = ov::npuw::util::isPresentKeyValuesKey(result_name);
        auto is_value = ov::npuw::util::isPresentKeyValuesValue(result_name);

        if (!is_key && !is_value) {
            LOG_DEBUG("Not a key and not a value - SKIP");
            continue;
        }

        if (is_key && is_value) {
            LOG_WARN("Invalid result name for DynamicQuantize: " << result_name);
            continue;
        }

        auto kv_result = result;
        pattern_index = is_key ? is_key.value() : is_value.value();
        auto kv_name = cacheName(is_key.has_value());

        LOG_DEBUG("is a " << kv_name << ": with index=" << pattern_index << ", shape=" << result->get_input_shape(0));

        auto dq_input_value = kv_result->input_value(0);
        auto dq_input_node = dq_input_value.get_node_shared_ptr();

        //  might be a convert layer - that substitute, if no just insert before it
        if (ov::is_type<ov::op::v0::Convert>(dq_input_node) && dq_input_node->get_users().size() == 1) {
            LOG_DEBUG("Bypassing PPP-inserted Convert: " << dq_input_node->get_friendly_name() << " ("
                                                         << dq_input_node->input_value(0).get_element_type() << " -> "
                                                         << dq_input_node->get_element_type() << ")");
            dq_input_value = dq_input_node->input_value(0);
        }

        const auto& cfg = is_key.has_value() ? params.key : params.value;
        auto kv_dyn_quant = create_dynamic_quantize(dq_input_value, is_key.has_value(), cfg);

        kv_result->input(0).replace_source_output(kv_dyn_quant->output(0));

        LOG_DEBUG("Done");
    }
    model->validate_nodes_and_infer_types();

    // TODO: for now internal op DynamicQuantize not easily gets converted into IE so run decompose here
    ov::pass::Manager manager("insert_dq_internal_ops");
    manager.register_pass<ov::npuw::DecomposeDynamicQuantize2>();  // asymmetric u8
    manager.register_pass<ov::npuw::DecomposeDynamicQuantize>();   // symmetric i8/i4
    manager.run_passes(model);

    model->validate_nodes_and_infer_types();

    LOG_INFO("DynamicQuantize passes complete");
}
