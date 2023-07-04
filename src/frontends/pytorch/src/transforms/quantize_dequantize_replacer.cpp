// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantize_dequantize_replacer.hpp"

#include <memory>
#include <utility>

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"


namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::opset10;

QuantizeDequantizeReplacer::QuantizeDequantizeReplacer() {
    auto dequantize_node = ov::pass::pattern::wrap_type<ov::op::util::QuantizedPtNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto dequantize_node = cast_quantized_fw_node(m.get_match_root(), ov::op::util::QuantizedPtNodeType::DEQUANTIZE);
        if (!dequantize_node)
            return false;

        ov::pass::NodeRegistry rg;
        std::shared_ptr<ov::Node> dequantize_input, scale, zero_point, dtype, axis;

        auto quantize_node = dequantize_node->input_value(0).get_node_shared_ptr();
        if (auto quantize_per_tensor = cast_quantized_fw_node(quantize_node, ov::op::util::QuantizedPtNodeType::QUANTIZE_PER_TENSOR)) {
            const auto quantize_input = quantize_per_tensor->input_value(0).get_node_shared_ptr();
            scale = quantize_per_tensor->input_value(1).get_node_shared_ptr();
            zero_point = quantize_per_tensor->input_value(2).get_node_shared_ptr();
            dtype = quantize_per_tensor->input_value(3).get_node_shared_ptr();

            // Quantize
            const auto scale_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(scale, quantize_input));
            const auto zero_point_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(zero_point, quantize_input));
            const auto scaled_input = context.mark_node(std::make_shared<opset10::Divide>(quantize_input, scale_convert));
            const auto scaled_input_with_zero_pt =
                context.mark_node(std::make_shared<opset10::Add>(scaled_input, zero_point_convert));
            const auto quantized_input = context.mark_node(
                std::make_shared<opset10::Round>(scaled_input_with_zero_pt, opset10::Round::RoundMode::HALF_TO_EVEN));

            if (dtype == element::u8) {
                const auto clamp =
                    context.mark_node(std::make_shared<opset10::Clamp>(quantized_input,
                                                                    std::numeric_limits<unsigned char>::lowest(),
                                                                    std::numeric_limits<unsigned char>::max()));
                dequantize_input = context.mark_node(std::make_shared<opset10::Convert>(clamp, element::u8));
            } else if (dtype == element::i8) {
                const auto clamp = context.mark_node(std::make_shared<opset10::Clamp>(quantized_input,
                                                                                    std::numeric_limits<char>::lowest(),
                                                                                    std::numeric_limits<char>::max()));
                dequantize_input = context.mark_node(std::make_shared<opset10::Convert>(clamp, element::i8));
            } else {
                dequantize_input = context.mark_node(std::make_shared<opset10::Convert>(quantized_input, element::i32));
            }

        } else if (auto quantize_per_channel = cast_quantized_fw_node(quantized_input_node, ov::op::util::QuantizedPtNodeType::QUANTIZE_PER_CHANNEL)) {
            return false; // TODO if needed
        } else {
            return false;
        }

        // Dequantize
        const auto input_convert_f32 = rg.make<opset10::Convert>(dequantize_input, element::f32);
        const auto scale_convert_f32 = rg.make<opset10::Convert>(scale, element::f32);
        const auto zero_point_convert_f32 = rg.make<opset10::Convert>(zero_point, element::f32);

        const auto input_sub_zero_pt =
            rg.make<opset10::Subtract>(input_convert_f32, zero_point_convert_f32));
        const auto dequantized_input = rg.make<opset10::Multiply>(input_sub_zero_pt, scale_convert_f32));

        copy_runtime_info_and_name(quantized_node, rg.get(), {dequantize_input});
        replace_node(quantized_node, dequantized_input);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(dequantized_node, "ov::frontend::pytorch::pass::QuantizeDequantizeReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
