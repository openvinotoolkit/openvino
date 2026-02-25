// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <type_traits>

#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu {

template <class T>
bool match_conv_mul_add_fq(const std::shared_ptr<const ov::Node>& node) {
    static_assert(std::is_same_v<T, ov::op::v1::Subtract> || std::is_same_v<T, ov::op::v1::Multiply>,
                  "match_conv_mul_add_fq supports only Subtract and Multiply");

    using namespace ov::pass::pattern;

    auto conv_m = wrap_type<ov::op::v1::Convolution>(
        {any_input(type_matches_any({ov::element::i8, ov::element::u8})), any_input()});
    auto mul0_m = wrap_type<ov::op::v1::Multiply>({conv_m, any_input()});
    auto add_m = wrap_type<ov::op::v1::Add>({mul0_m, any_input()});
    auto fq_m = wrap_type<ov::op::v0::FakeQuantize>({add_m, any_input(), any_input(), any_input(), any_input()},
                                                    type_matches_any({ov::element::i8, ov::element::u8}));
    auto final_m = wrap_type<T>({fq_m, any_input()});

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(final_m);
    if (!matcher->match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& pattern_map = matcher->get_pattern_value_map();
    const auto fq = pattern_map.at(fq_m).get_node_shared_ptr();
    const auto conv = pattern_map.at(conv_m).get_node_shared_ptr();

    return conv->get_input_element_type(0) == fq->get_output_element_type(0);
}

enum class FQMulAddPattern : std::uint8_t { ConvMulAdd, ConvAddMul };

bool match_fq_mul_conv_bias_same_types(const std::shared_ptr<const ov::Node>& node, FQMulAddPattern pattern);

bool match_conv_fq_same_types(const std::shared_ptr<const ov::Node>& node);

bool match_conv_stride_oc_ic_limit(const std::shared_ptr<const ov::Node>& node,
                                   const std::vector<int64_t>& strides,
                                   const ov::Shape& kernel_shape,
                                   size_t oc_ic_limit);

bool match_conv_mul_add(const std::shared_ptr<const ov::Node>& node);

}  // namespace ov::intel_cpu
