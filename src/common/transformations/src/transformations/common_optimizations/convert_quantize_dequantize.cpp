// Copyright (C) 2018-2025 Intel Corporation
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

#define LOG_INFO(...)                           \
    do {                                        \
        if (std::getenv("QDQ_STRIPPING_LOG")) { \
            std::cout << __VA_ARGS__;           \
        }                                       \
    } while (0)

ov::pass::ConvertQuantizeDequantize::ConvertQuantizeDequantize(
    const ov::element::TypeVector& supported_low_precisions,
    const ov::element::TypeVector& supported_original_precisions,
    const bool ignore_consumers_count_check) {
    MATCHER_SCOPE(ConvertQuantizeDequantize);

    using namespace ov::pass::pattern;
    using namespace ov::op;

    auto data_pattern = any_input(type_matches_any(supported_original_precisions));
    auto input_low_pattern = any_input();
    auto input_high_pattern = any_input();
    auto output_low_pattern = wrap_type<v0::Constant>();
    auto output_high_pattern = wrap_type<v0::Constant>();
    auto fq_pattern = wrap_type<v0::FakeQuantize>(
        {data_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});
    op::Predicate convert1_predicate = ignore_consumers_count_check
                                           ? type_matches_any(supported_low_precisions)
                                           : type_matches_any(supported_low_precisions) && consumers_count(1);
    auto convert1_pattern = wrap_type<v0::Convert>({fq_pattern}, convert1_predicate);
    op::Predicate convert2_predicate = ignore_consumers_count_check
                                           ? type_matches_any(supported_original_precisions)
                                           : type_matches_any(supported_original_precisions) && consumers_count(1);
    auto convert2_pattern = wrap_type<v0::Convert>({convert1_pattern}, convert2_predicate);

    auto zero_point_pattern = any_input();
    op::Predicate sub_predicate = ignore_consumers_count_check ? op::Predicate() : consumers_count(1);
    auto sub_pattern = optional<v1::Subtract>({convert2_pattern, zero_point_pattern}, sub_predicate);
    auto scale_pattern = any_input();
    auto mul_pattern = wrap_type<v1::Multiply>({sub_pattern, scale_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        LOG_INFO("[ INFO ] ConvertQuantizeDequantize matched\n");

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto data = pattern_map.at(data_pattern);
        auto input_low = pattern_map.at(input_low_pattern);
        auto input_high = pattern_map.at(input_high_pattern);
        auto output_low =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(output_low_pattern).get_node_shared_ptr());
        if (!output_low)
            return false;
        auto output_high =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(output_high_pattern).get_node_shared_ptr());
        if (!output_high)
            return false;
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(pattern_map.at(fq_pattern).get_node_shared_ptr());
        if (!fq)
            return false;
        auto scale = pattern_map.at(scale_pattern);
        auto convert1 = pattern_map.at(convert1_pattern);
        auto convert2 = pattern_map.at(convert2_pattern);
        auto mul = pattern_map.at(mul_pattern).get_node_shared_ptr();

        static const std::unordered_set<size_t> supported_levels{256, 65536};
        const auto levels = fq->get_levels();
        if (!supported_levels.count(levels))
            return false;

        float out_low_val;
        if (!ov::op::util::get_single_value(output_low, out_low_val))
            return false;
        float out_high_val;
        if (!ov::op::util::get_single_value(output_high, out_high_val))
            return false;

        static const std::unordered_map<ov::element::Type_t, std::pair<float, float>> supported_intervals{
            {ov::element::i8, {-128.f, 127.f}},
            {ov::element::u8, {0.f, 255.f}},
            {ov::element::i16, {-32768.f, 32767.f}},
            {ov::element::u16, {0.f, 65535.f}}};
        const auto& type = convert1.get_element_type();
        LOG_INFO("[ INFO ] Matched nodes info\n");
        LOG_INFO("\t data = " << data << std::endl);
        LOG_INFO("\t fq = " << fq << std::endl);
#define LOG_CONSTANT_VALUE(name, value)                                                                             \
    do {                                                                                                            \
        auto node = value.get_node_shared_ptr();                                                                    \
        if (ov::is_type<ov::op::v0::Convert>(node)) {                                                               \
            node = node->get_input_node_shared_ptr(0);                                                              \
        }                                                                                                           \
        if (auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(node)) {                                        \
            if (ov::shape_size(const_node->get_shape()) == 1) {                                                     \
                LOG_INFO("\t " << name << " = " << value                                                            \
                               << " (constant value: " << const_node->cast_vector<float>()[0] << ")" << std::endl); \
            } else {                                                                                                \
                LOG_INFO("\t " << name << " = " << value << std::endl);                                             \
            }                                                                                                       \
        } else {                                                                                                    \
            LOG_INFO("\t " << name << " = " << value << std::endl);                                                 \
        }                                                                                                           \
    } while (0);

        LOG_CONSTANT_VALUE("input_low", input_low);
        LOG_CONSTANT_VALUE("input_high", input_high);
        LOG_CONSTANT_VALUE("output_low", pattern_map.at(output_low_pattern));
        LOG_CONSTANT_VALUE("output_high", pattern_map.at(output_high_pattern));
        LOG_INFO("\t convert1 = " << convert1 << std::endl);
        LOG_INFO("\t convert2 = " << convert2 << std::endl);
        if (pattern_map.count(zero_point_pattern)) {
            const auto& shift = pattern_map.at(zero_point_pattern);
            const auto& subtract = pattern_map.at(sub_pattern);
            LOG_CONSTANT_VALUE("shift", shift);
            LOG_INFO("\t subtract = " << subtract << std::endl);
        } else {
            LOG_INFO("\t zero_point not present\n");
        }
        LOG_CONSTANT_VALUE("scale", scale);
        LOG_INFO("\t mul = " << mul << std::endl);

        if (supported_intervals.count(type) == 0 ||
            supported_intervals.at(type) != std::make_pair(out_low_val, out_high_val)) {
            LOG_INFO("[ INFO ] ConvertQuantizeDequantize: unsupported intervals\n");
            return false;
        }

        const bool has_zero_point = pattern_map.count(zero_point_pattern);
        std::shared_ptr<Node> new_out_low = output_low, new_out_high = output_high;
        if (has_zero_point) {
            const auto& zero_point = pattern_map.at(zero_point_pattern);
            new_out_low = std::make_shared<ov::op::v1::Subtract>(new_out_low, zero_point);
            new_out_high = std::make_shared<ov::op::v1::Subtract>(new_out_high, zero_point);
        }
        new_out_low = std::make_shared<ov::op::v1::Multiply>(new_out_low, scale);
        new_out_high = std::make_shared<ov::op::v1::Multiply>(new_out_high, scale);

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
        LOG_INFO("[ INFO ] ConvertQuantizeDequantize: transformation finished\n");

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}
