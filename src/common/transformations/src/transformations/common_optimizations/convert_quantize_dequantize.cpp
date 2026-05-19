// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;

namespace ov::pass {

// ConvertQuantizeDequantize converts Quantize/Dequantize pair to a single FakeQuantize.
// Since Quantize is decomposed to FakeQuantize and Dequantize is decomposed to Subtract->Multiply,
// the full pattern to match is presented on the left hand side of the graph below.
// On the right hand side is the graph after transformation.
// Currently transformation supports only i8, u8, i16, u16 quantized data type.
// That implies 'levels' attribute to be 256 or 65536, as well as (output_low, output_high)
// be (-128, 127) or (0, 255) or (-32768, 32767) or (0, 65535) (depends on type and depends
// on sign of the quantized data type). Another limitation is that 'zero_point' and 'scale' have to be broadcastable to
// the output of FakeQuantize.
// Mixed precision is supported: the quantizer (FakeQuantize input) and dequantizer (scale/zero_point)
// can use different floating-point precisions (e.g., FakeQuantize on fp32, dequantizer scale in fp16).
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

ConvertQuantizeDequantize::ConvertQuantizeDequantize(const ov::element::TypeVector& supported_low_precisions) {
    MATCHER_SCOPE(ConvertQuantizeDequantize);
    // Allow floating-point input types (fp32, fp16, bf16) for mixed precision support
    auto data_pattern = pattern::any_input([](const Output<Node>& output) {
        return output.get_element_type().is_real();
    });
    auto input_low_pattern = pattern::any_input();
    auto input_high_pattern = pattern::any_input();
    auto output_low_pattern = pattern::wrap_type<v0::Constant>();
    auto output_high_pattern = pattern::wrap_type<v0::Constant>();
    auto fq_pattern = pattern::wrap_type<v0::FakeQuantize>(
        {data_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});
    auto convert1_pattern = pattern::wrap_type<v0::Convert>(
        {fq_pattern},
        pattern::type_matches_any(supported_low_precisions) && pattern::consumers_count(1));
    // Allow mixed precision: dequantizer can use fp16 even if quantizer uses fp32
    auto convert2_pattern = pattern::wrap_type<v0::Convert>({convert1_pattern}, pattern::consumers_count(1));

    auto zero_point_pattern = pattern::any_input();
    auto sub_pattern =
        pattern::optional<v1::Subtract>({convert2_pattern, zero_point_pattern}, pattern::consumers_count(1));
    auto scale_pattern = pattern::any_input();
    auto mul_pattern = pattern::wrap_type<v1::Multiply>({sub_pattern, scale_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto data = pattern_map.at(data_pattern);
        auto input_low = pattern_map.at(input_low_pattern);
        auto input_high = pattern_map.at(input_high_pattern);
        auto output_low = ov::as_type_ptr<v0::Constant>(pattern_map.at(output_low_pattern).get_node_shared_ptr());
        if (!output_low)
            return false;
        auto output_high = ov::as_type_ptr<v0::Constant>(pattern_map.at(output_high_pattern).get_node_shared_ptr());
        if (!output_high)
            return false;
        auto fq = ov::as_type_ptr<v0::FakeQuantize>(pattern_map.at(fq_pattern).get_node_shared_ptr());
        if (!fq)
            return false;
        auto scale = pattern_map.at(scale_pattern);
        auto convert1 = pattern_map.at(convert1_pattern);
        auto convert2 = pattern_map.at(convert2_pattern);
        auto mul = pattern_map.at(mul_pattern).get_node_shared_ptr();

        // Validate convert2 outputs floating-point type (fp32, fp16, bf16)
        const auto& convert2_type = convert2.get_element_type();
        if (!convert2_type.is_real()) {
            return false;
        }

        static const std::unordered_set<size_t> supported_levels{256, 65536};
        const auto levels = fq->get_levels();
        if (!supported_levels.count(levels))
            return false;

        float out_low_val;
        if (!op_util::get_single_value(output_low, out_low_val))
            return false;
        float out_high_val;
        if (!op_util::get_single_value(output_high, out_high_val))
            return false;

#define PRECISION_LIMITS_FOR(type)                                                                           \
    {                                                                                                        \
        ov::element::type,                                                                                   \
            std::make_pair(                                                                                  \
                static_cast<float>(std::numeric_limits<ov::fundamental_type_for<ov::element::type>>::min()), \
                static_cast<float>(std::numeric_limits<ov::fundamental_type_for<ov::element::type>>::max())) \
    }

        static const std::unordered_map<ov::element::Type_t, std::pair<float, float>> supported_intervals{
            PRECISION_LIMITS_FOR(i8),
            PRECISION_LIMITS_FOR(u8),
            PRECISION_LIMITS_FOR(i16),
            PRECISION_LIMITS_FOR(u16)};
#undef PRECISION_LIMITS_FOR

        const auto& type = convert1.get_element_type();
        // check if (out_low_val, out_high_val) pair is mapped on the expected precision ranges
        auto interval_it = supported_intervals.find(type);
        if (interval_it == supported_intervals.end() ||
            interval_it->second != std::make_pair(out_low_val, out_high_val)) {
            return false;
        }

        // Get the element type of the FakeQuantize output bounds (typically f32)
        const auto& fq_output_type = output_low->get_element_type();
        const auto& scale_type = scale.get_element_type();

        // Perform arithmetic in FP32 to avoid overflow (e.g., 65535 overflows FP16 max ~65504),
        // then convert the final result to the scale's precision to preserve dequantizer precision info.
        const bool has_zero_point = pattern_map.count(zero_point_pattern);
        std::shared_ptr<Node> new_out_low = output_low, new_out_high = output_high;
        if (has_zero_point) {
            auto zero_point = pattern_map.at(zero_point_pattern);
            // Ensure zero_point is in FP32 for safe arithmetic
            if (zero_point.get_element_type() != fq_output_type) {
                zero_point = std::make_shared<v0::Convert>(zero_point, fq_output_type);
            }
            new_out_low = std::make_shared<v1::Subtract>(new_out_low, zero_point);
            new_out_high = std::make_shared<v1::Subtract>(new_out_high, zero_point);
        }
        // Ensure scale is in FP32 for safe multiplication
        auto scale_fp32 = scale;
        if (scale_type != fq_output_type) {
            scale_fp32 = std::make_shared<v0::Convert>(scale, fq_output_type);
        }
        new_out_low = std::make_shared<v1::Multiply>(new_out_low, scale_fp32);
        new_out_high = std::make_shared<v1::Multiply>(new_out_high, scale_fp32);
        // Convert the result to scale's original precision to preserve dequantizer precision info
        if (scale_type != fq_output_type) {
            new_out_low = std::make_shared<v0::Convert>(new_out_low, scale_type);
            new_out_high = std::make_shared<v0::Convert>(new_out_high, scale_type);
        }

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
            std::make_shared<v0::FakeQuantize>(data, input_low, input_high, new_out_low, new_out_high, levels);

        // Preserve the output precision of the dequantizer (e.g., fp16)
        std::shared_ptr<Node> final_output = new_fq;
        const auto& mul_output_type = mul->get_output_element_type(0);
        if (mul_output_type != new_fq->get_output_element_type(0)) {
            // Convert FakeQuantize output to match the original dequantizer output type
            auto convert_output = std::make_shared<v0::Convert>(new_fq, mul_output_type);
            convert_output->set_friendly_name(mul->get_friendly_name());
            copy_runtime_info({fq, convert1.get_node_shared_ptr(), convert2.get_node_shared_ptr()},
                              {new_fq, convert_output});
            final_output = convert_output;
        } else {
            new_fq->set_friendly_name(mul->get_friendly_name());
            copy_runtime_info({fq, convert1.get_node_shared_ptr(), convert2.get_node_shared_ptr()}, new_fq);
        }

        replace_node(mul, final_output);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
