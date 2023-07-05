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
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::opset10;

QuantizeDequantizeReplacer::QuantizeDequantizeReplacer() {
    auto dequantize_node = ov::pass::pattern::wrap_type<ov::frontend::pytorch::QuantizedPtNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto dequantize_node = cast_quantized_fw_node(m.get_match_root(), "aten::dequantize");
        if (!dequantize_node)
            return false;

        ov::pass::NodeRegistry rg;
        std::shared_ptr<ov::Node> quantize_input, dequantize_input, scale, zero_point, axis;

        auto quantize_node = dequantize_node->input_value(0).get_node_shared_ptr();
        if (auto quantize_per_tensor = cast_quantized_fw_node(quantize_node, "aten::quantize_per_tensor")) {
            quantize_input = quantize_per_tensor->input_value(0).get_node_shared_ptr();
            scale = quantize_per_tensor->input_value(1).get_node_shared_ptr();
            zero_point = quantize_per_tensor->input_value(2).get_node_shared_ptr();
            auto const dtype_node = quantize_per_tensor->input_value(3).get_node_shared_ptr();
            auto const const_dtype_node = std::dynamic_pointer_cast<opset10::Constant>(dtype_node);
            auto const dtype = convert_dtype(const_dtype_node->cast_vector<int64_t>().at(0));

            // Quantize
            const auto scale_convert = rg.make<opset10::ConvertLike>(scale, quantize_input);
            const auto zero_point_convert = rg.make<opset10::ConvertLike>(zero_point, quantize_input);
            const auto scaled_input = rg.make<opset10::Divide>(quantize_input, scale_convert);
            const auto scaled_input_with_zero_pt = rg.make<opset10::Add>(scaled_input, zero_point_convert);
            const auto quantized_input =
                rg.make<opset10::Round>(scaled_input_with_zero_pt, opset10::Round::RoundMode::HALF_TO_EVEN);

            if (dtype == element::u8) {
                const auto clamp = rg.make<opset10::Clamp>(quantized_input,
                                                           std::numeric_limits<unsigned char>::lowest(),
                                                           std::numeric_limits<unsigned char>::max());
                dequantize_input = rg.make<opset10::Convert>(clamp, element::u8);
            } else if (dtype == element::i8) {
                const auto clamp = rg.make<opset10::Clamp>(quantized_input,
                                                           std::numeric_limits<char>::lowest(),
                                                           std::numeric_limits<char>::max());
                dequantize_input = rg.make<opset10::Convert>(clamp, element::i8);
            } else {
                dequantize_input = rg.make<opset10::Convert>(quantized_input, element::i32);
            }

        } else if (auto quantize_per_channel = cast_quantized_fw_node(quantize_node, "aten::quantize_per_channel")) {
            return false;  // TODO if needed
        } else {
            return false;
        }

        // Dequantize
        const auto input_convert_f32 = rg.make<opset10::Convert>(dequantize_input, element::f32);
        const auto scale_convert_f32 = rg.make<opset10::Convert>(scale, element::f32);
        const auto zero_point_convert_f32 = rg.make<opset10::Convert>(zero_point, element::f32);

        const auto input_sub_zero_pt = rg.make<opset10::Subtract>(input_convert_f32, zero_point_convert_f32);
        const auto dequantized_input = rg.make<opset10::Multiply>(input_sub_zero_pt, scale_convert_f32);

        copy_runtime_info_and_name(dequantize_node, rg.get(), {dequantize_input});
        replace_node(dequantize_node, dequantized_input);

        copy_runtime_info_and_name(quantize_node, rg.get(), {quantize_input});
        replace_node(quantize_node, dequantize_input);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(dequantize_node,
                                                          "ov::frontend::pytorch::pass::QuantizeDequantizeReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
