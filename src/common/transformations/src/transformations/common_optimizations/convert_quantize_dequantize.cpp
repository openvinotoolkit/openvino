// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

// ConvertQuantizeDequantize converts Quantize/Dequantize pair to a single FakeQuantize.
// Since Quantize is decomposed to FakeQuantize and Dequantize is decomposed to Subtract->Multiply,
// the full pattern to match is presented on the left hand side of the graph below.
// On the right hand side is the graph after transformation.
// Currently transformation supports only i8, u8, i16, u16 quantized data type.
// That implies 'levels' attribute to be 256 or 65536, as well as (output_low, output_high)
// be (-128, 127) or (0, 255) or (-32768, 32767) or (0, 65535) (depends on type and depends
// on sign of the quantized data type). Another limitation is that 'zero_point' and 'scale' have to be broadcastable to
// the output of FakeQuantize.
//
//
//                                   |  |  |  |  |
//                                   |  |  |  |  |
//                                   v  v  v  v  v
//                                  +------------+
//                                  |FakeQuantize|
//                                  +------------+
//                                        |
//                                        v
//                              +---------------------+
//                              |      Convert        |
//                              |(e.g. from f32 to u8)|
//                              +---------+-----------+                            |  |  |  |  |
//                                        |                                        |  |  |  |  |
//                                        v                                        v  v  v  v  v
//                              +---------------------+                           +------------+
//                              |      Convert        |            ====>          |FakeQuantize|
//                              |  (from u8 to f32)   |                           +------------+
//                              +---------+-----------+                                 |
//                                        |                                             v
//                                        v
//                  +----------+    +------------+
//                  |zero point|--->|  Subtract  |
//                  +----------+    +-----+------+
//                                        |
//                                        v
//                   +---------+    +------------+
//                   |  scale  |--->|  Multiply  |
//                   +---------+    +-----+------+
//                                        |
//                                        v
//

ov::pass::ConvertQuantizeDequantize::ConvertQuantizeDequantize() {
    MATCHER_SCOPE(ConvertQuantizeDequantize);
    auto data_pattern = pass::pattern::any_input();
    auto input_low_pattern = pass::pattern::any_input();
    auto input_high_pattern = pass::pattern::any_input();
    auto output_low_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto output_high_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto fq_pattern = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>(
        {data_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});
    auto convert1_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Convert>(
        {fq_pattern},
        pattern::type_matches_any({element::i8, element::u8, element::i16, element::u16}));
    auto convert2_pattern =
        ov::pass::pattern::wrap_type<ov::op::v0::Convert>({convert1_pattern}, pattern::type_matches(element::f32));
    auto zero_point_pattern = pass::pattern::any_input();
    auto sub_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({convert2_pattern, zero_point_pattern},
                                                                          pattern::consumers_count(1));
    auto scale_pattern = pass::pattern::any_input();
    auto mul_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({sub_pattern, scale_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto data = pattern_map[data_pattern];
        auto input_low = pattern_map[input_low_pattern];
        auto input_high = pattern_map[input_high_pattern];
        auto output_low = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map[output_low_pattern].get_node_shared_ptr());
        if (!output_low)
            return false;
        auto output_high =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map[output_high_pattern].get_node_shared_ptr());
        if (!output_high)
            return false;
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(pattern_map[fq_pattern].get_node_shared_ptr());
        if (!fq)
            return false;
        auto zero_point = pattern_map[zero_point_pattern];
        auto scale = pattern_map[scale_pattern];
        auto convert1 = pattern_map[convert1_pattern];
        auto convert2 = pattern_map[convert2_pattern];
        auto mul = pattern_map[mul_pattern].get_node_shared_ptr();

        // convert1 and convert2 should have only one input
        if (convert1.get_target_inputs().size() != 1)
            return false;
        if (convert2.get_target_inputs().size() != 1)
            return false;

        // we support:
        // i8 or u8: 'levels' attribute must be 256
        // i16 or u16: 'levels' attribute must be 65536
        size_t levels = fq->get_levels();
        if (levels != 256 && levels != 65536)
            return false;

        // check if (out_low_val, out_high_val) is (-128, 127) or (0, 255) or (-32768, 32767) or (0, 65535)
        float out_low_val;
        if (!op::util::get_single_value(output_low, out_low_val))
            return false;
        float out_high_val;
        if (!op::util::get_single_value(output_high, out_high_val))
            return false;
        const auto& type = convert1.get_element_type();
        switch (type) {
        case element::Type_t::i8:
            if (out_low_val != -128 || out_high_val != 127)
                return false;
            break;
        case element::Type_t::u8:
            if (out_low_val != 0 || out_high_val != 255)
                return false;
            break;
        case element::Type_t::i16:
            if (out_low_val != -32768 || out_high_val != 32767)
                return false;
            break;
        case element::Type_t::u16:
            if (out_low_val != 0 || out_high_val != 65535)
                return false;
            break;
        default:
            return false;
        }

        std::shared_ptr<Node> new_out_low =
            std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v1::Subtract>(output_low, zero_point),
                                                   scale);
        std::shared_ptr<Node> new_out_high =
            std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v1::Subtract>(output_high, zero_point),
                                                   scale);

        // check if new_out_low/high shapes are broadcastable to FQ's input
        auto data_shape = data.get_partial_shape();
        if (data_shape.rank().is_dynamic())
            return false;
        auto out_low_shape = new_out_low->get_output_partial_shape(0);
        if (out_low_shape.rank().is_dynamic() || out_low_shape.rank().get_length() > data_shape.rank().get_length())
            return false;
        auto out_high_shape = new_out_high->get_output_partial_shape(0);
        if (out_high_shape.rank().is_dynamic() || out_high_shape.rank().get_length() > data_shape.rank().get_length())
            return false;

        std::shared_ptr<Node> const_out_low = ov::util::get_constant_from_source(new_out_low);
        if (const_out_low)
            new_out_low = const_out_low;
        std::shared_ptr<Node> const_out_high = ov::util::get_constant_from_source(new_out_high);
        if (const_out_high)
            new_out_high = const_out_high;

        auto new_fq =
            std::make_shared<ov::op::v0::FakeQuantize>(data, input_low, input_high, new_out_low, new_out_high, levels);
        new_fq->set_friendly_name(mul->get_friendly_name());

        copy_runtime_info({fq, convert1.get_node_shared_ptr(), convert2.get_node_shared_ptr()}, new_fq);
        replace_node(mul, new_fq);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}
