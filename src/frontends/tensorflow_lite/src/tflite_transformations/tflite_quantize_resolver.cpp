// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite_transformations/tflite_quantize_resolver.hpp"

#include <memory>

#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "tflite_ops/tflite_quantize.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::pass;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::frontend::tensorflow_lite;

pass::TFLQuantizeConvert::TFLQuantizeConvert() {
    auto tfl_quantize_label = wrap_type<tensorflow_lite::TFLQuantize>();
    auto convert_label = wrap_type<opset10::Convert>({tfl_quantize_label});

    matcher_pass_callback callback = [=](Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto tfl_quantize_node = pattern_map.at(tfl_quantize_label);
        auto convert_node = pattern_map.at(convert_label);
        auto convert = ov::as_type_ptr<opset10::Convert>(convert_node);
        if (!convert)
            return false;
        auto type = convert->get_destination_type();
        if (type != element::f32)
            return false;
        auto tfl_quantize = ov::as_type_ptr<TFLQuantize>(tfl_quantize_node);
        if (!tfl_quantize)
            return false;
        tfl_quantize->set_type(type);
        convert->output(0).replace(tfl_quantize->output(0));
        return true;
    };

    auto m =
        std::make_shared<pattern::Matcher>(convert_label, "ov::frontend::tensorflow_lite::pass::TFLQuantizeResolver");
    register_matcher(m, callback);
}

ov::Shape get_quant_shape(const ov::Output<ov::Node>& output,
                          const std::shared_ptr<ov::frontend::tensorflow_lite::QuantizationInfo>& quantization,
                          const size_t& size) {
    auto shape = ov::Shape{};
    if (size > 1) {
        FRONT_END_GENERAL_CHECK(output.get_partial_shape().rank().is_static(),
                                "Per-Channel Quantization of tensor with dynamic rank");
        auto rank = output.get_partial_shape().size();
        shape = ov::Shape(rank, 1);
        shape[quantization->get_axis()] = size;
    }
    return shape;
}

void fuse_zp_to_weights(ov::Output<ov::Node>& output, std::vector<int64_t>& zero_point, const ov::Shape& zp_shape) {
    if (std::all_of(zero_point.begin(), zero_point.end(), [](const int64_t& i) {
            return i == 0;
        }))
        return;
    auto type = output.get_element_type();
    if (type != ov::element::u8 && type != ov::element::i8)
        return;
    auto rank = output.get_partial_shape().size();
    vector<int64_t> axes_vec(rank);
    std::iota(axes_vec.begin(), axes_vec.end(), 0);

    auto axes = Constant::create(ov::element::i64, {axes_vec.size()}, axes_vec);
    auto max_value = make_shared<ReduceMax>(output, axes, false)->output(0);
    auto min_value = make_shared<ReduceMin>(output, axes, false)->output(0);

    auto check_in_bounds = [&](ov::Output<ov::Node>& value) -> bool {
        shared_ptr<ov::opset10::Constant> constant;
        if (rank == 0) {
            constant = ov::as_type_ptr<ov::opset10::Constant>(output.get_node_shared_ptr());
        } else {
            constant = ov::util::get_constant_from_source(value);
        }
        if (!constant)
            return false;
        auto weight = constant->cast_vector<int64_t>()[0];
        return std::all_of(zero_point.begin(), zero_point.end(), [&](const int64_t& zp) {
            auto new_value = weight - zp;
            return new_value >= -128 && new_value <= 127;
        });
    };
    if (!check_in_bounds(min_value) || !check_in_bounds(max_value))
        return;
    output = std::make_shared<Convert>(output, ov::element::i32);
    auto zp_node = ov::opset10::Constant::create(ov::element::i32, zp_shape, zero_point);
    output = std::make_shared<ov::opset10::Subtract>(output, zp_node);
    output = std::make_shared<ov::opset10::Convert>(output, ov::element::i8);
    output = ov::util::get_constant_from_source(output);  // TODO: Check Me
    zero_point = {0};
}

pass::TFLQuantizeReplacer::TFLQuantizeReplacer() {
    auto tfl_quantize_label = wrap_type<tensorflow_lite::TFLQuantize>();
    matcher_pass_callback callback = [=](Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto tfl_quantize_node = pattern_map.at(tfl_quantize_label);
        auto tfl_quantize = ov::as_type_ptr<TFLQuantize>(tfl_quantize_node);
        if (!tfl_quantize)
            return false;
        auto quantization = tfl_quantize->get_info();
        FRONT_END_GENERAL_CHECK(
            quantization != nullptr,
            "Internal operation TFLQuantized representing quantized tensor doesn't have quantization details");

        auto in_type = tfl_quantize->get_input_element_type(0);
        auto out_type = tfl_quantize->get_type();

        auto is_constant = ov::is_type<Constant>(tfl_quantize->get_input_node_shared_ptr(0));  // for Constant case

        FRONT_END_GENERAL_CHECK(in_type == out_type || in_type == element::f32 || out_type == element::f32,
                                "TFLQuantized types do not match: in_type = ",
                                in_type,
                                " out_type = ",
                                out_type);

        // non constant case -- FQ case
        Output<Node> output = tfl_quantize->get_input_source_output(0);

        auto zp = quantization->get_zero_point();
        auto scale = quantization->get_scale();

        auto zp_shape = get_quant_shape(output, quantization, zp.size());
        auto scale_shape = get_quant_shape(output, quantization, scale.size());

        auto zp_node = ov::opset10::Constant::create(element::f32, zp_shape, zp);
        auto scale_node = ov::opset10::Constant::create(element::f32, scale_shape, scale);

        if (is_constant) {
            fuse_zp_to_weights(output, zp, zp_shape);
            output = make_shared<Convert>(output, element::f32);
            disable_constant_folding(output.get_node_shared_ptr());
            if (std::any_of(zp.begin(), zp.end(), [](const int64_t& i) {
                    return i != 0;
                }))
                output = std::make_shared<ov::opset10::Subtract>(output, zp_node);
            output = std::make_shared<ov::opset10::Multiply>(output, scale_node);
            tfl_quantize->output(0).replace(output);
            return true;
        }
        if (in_type != element::f32) {
            output = make_shared<Convert>(output, element::f32);
        }

        auto levels = 1 << tfl_quantize->get_original_type().bitwidth();
        auto is_signed = tfl_quantize->get_original_type().is_signed();

        Output<Node> input_low, input_high, output_low, output_high;
        output_low = std::make_shared<opset10::Multiply>(std::make_shared<opset10::Negative>(scale_node), zp_node);
        output_high = std::make_shared<opset10::Multiply>(
            scale_node,
            std::make_shared<opset10::Subtract>(ov::opset10::Constant::create(element::f32, {}, {levels - 1}),
                                                zp_node));
        if (in_type != element::f32) {
            auto low = is_signed ? (-levels / 2) : 0;
            auto high = levels - low - 1;
            input_low = ov::opset10::Constant::create(element::f32, {}, {low});
            input_high = ov::opset10::Constant::create(element::f32, {}, {high});
        } else {
            input_low = output_low;
            input_high = output_high;
        }
        if (out_type != element::f32) {
            auto low = is_signed ? (-levels / 2) : 0;
            auto high = levels - low - 1;
            output_low = ov::opset10::Constant::create(element::f32, {}, {low});
            output_high = ov::opset10::Constant::create(element::f32, {}, {high});
        }
        input_low = ov::util::get_constant_from_source(input_low);
        input_high = ov::util::get_constant_from_source(input_high);
        output_low = ov::util::get_constant_from_source(output_low);
        output_high = ov::util::get_constant_from_source(output_high);
        output =
            std::make_shared<opset10::FakeQuantize>(output, input_low, input_high, output_low, output_high, levels);
        if (out_type != element::f32) {
            output = make_shared<Convert>(output, out_type);
        }
        tfl_quantize->output(0).replace(output);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(tfl_quantize_label,
                                                "ov::frontend::tensorflow_lite::pass::TFLQuantizeReplacer");
    register_matcher(m, callback);
}

bool pass::TFLQuantizeResolver::run_on_model(const std::shared_ptr<ov::Model>& m) {
    ov::pass::Manager manager("Frontend:TFLite:TFLQuantizeResolver");
    manager.register_pass<pass::TFLQuantizeConvert>();
    manager.register_pass<pass::TFLQuantizeReplacer>();
    manager.run_passes(m);
    return true;
}
