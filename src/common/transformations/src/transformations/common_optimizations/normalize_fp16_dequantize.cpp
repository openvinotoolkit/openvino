// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/normalize_fp16_dequantize.hpp"

#include <memory>

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

namespace ov::pass {

// Handles the case where the Dequantize output is FP16 instead of FP32, which
// prevents ConvertQuantizeDequantize from detecting and fusing the sequence back
// into a single FakeQuantize. Rewrites the Dequantize output to FP32 by moving
// the FP16 cast to the end:
//
//   FQ -> Conv1(->int) -> Conv2(int->f16) -> [Sub(f16_zp)] -> Mul(f16_scale)
//                             |
//                             v
//   FQ -> Conv1(->int) -> Conv2(int->f32) -> [Sub(f32_zp)] -> Mul(f32_scale) -> Conv(f32->f16)

NormalizeDequantizeFP16::NormalizeDequantizeFP16() {
    MATCHER_SCOPE(NormalizeDequantizeFP16);

    auto fq_pattern = pattern::wrap_type<v0::FakeQuantize>(
        {pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});

    auto conv1_pattern = pattern::wrap_type<v0::Convert>(
        {fq_pattern},
        pattern::type_matches_any({ov::element::u8, ov::element::i8, ov::element::u16, ov::element::i16}) &&
            pattern::consumers_count(1));

    auto conv2_pattern =
        pattern::wrap_type<v0::Convert>({conv1_pattern},
                                        pattern::type_matches(ov::element::f16) && pattern::consumers_count(1));

    auto zp_pattern = pattern::any_input();
    auto sub_pattern = pattern::optional<v1::Subtract>({conv2_pattern, zp_pattern}, pattern::consumers_count(1));

    auto scale_pattern = pattern::any_input();
    auto mul_pattern = pattern::wrap_type<v1::Multiply>({sub_pattern, scale_pattern});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root()))
            return false;

        auto to_f32 = [](const Output<Node>& val) -> Output<Node> {
            if (val.get_element_type() == ov::element::f32)
                return val;
            Output<Node> cast = std::make_shared<v0::Convert>(val, ov::element::f32);
            if (auto cst = ov::util::get_constant_from_source(cast))
                return cst->output(0);
            return cast;
        };

        const bool has_zp = map.count(zp_pattern) > 0;
        auto scale = to_f32(map.at(scale_pattern));
        auto zp = has_zp ? to_f32(map.at(zp_pattern)) : Output<Node>{};

        auto conv2_node = map.at(conv2_pattern).get_node_shared_ptr();
        auto mul_node = map.at(mul_pattern).get_node_shared_ptr();

        // Normalize f16 DQ -> f32 DQ + f16 cast.
        auto new_conv2 = std::make_shared<v0::Convert>(map.at(conv1_pattern), ov::element::f32);

        NodeVector old_dq_nodes{conv2_node, mul_node};
        NodeVector new_dq_nodes{new_conv2};
        Output<Node> prev = new_conv2;

        if (has_zp) {
            old_dq_nodes.push_back(map.at(sub_pattern).get_node_shared_ptr());
            auto new_sub = std::make_shared<v1::Subtract>(prev, zp);
            new_dq_nodes.push_back(new_sub);
            prev = new_sub;
        }

        auto new_mul = std::make_shared<v1::Multiply>(prev, scale);
        new_dq_nodes.push_back(new_mul);

        auto cast_to_f16 = std::make_shared<v0::Convert>(new_mul, ov::element::f16);
        cast_to_f16->set_friendly_name(mul_node->get_friendly_name());
        new_dq_nodes.push_back(cast_to_f16);

        copy_runtime_info(old_dq_nodes, new_dq_nodes);
        replace_node(mul_node, cast_to_f16);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(mul_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
