// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::pattern;

namespace ov::intel_cpu {

bool match_fq_mul_conv_bias_same_types(const std::shared_ptr<const ov::Node>& node, FQMulAddPattern pattern) {
    auto convMulAdd_conv = wrap_type<ov::op::v1::Convolution>();
    auto convMulAdd_mul = wrap_type<ov::op::v1::Multiply>({convMulAdd_conv, any_input()});
    auto convMulAdd_add = wrap_type<ov::op::v1::Add>({convMulAdd_mul, any_input()});
    auto convMulAdd_fq =
        wrap_type<ov::op::v0::FakeQuantize>({convMulAdd_add, any_input(), any_input(), any_input(), any_input()});
    Matcher convMulAdd_matcher(convMulAdd_fq);
    auto convAddMul_conv = wrap_type<ov::op::v1::Convolution>();
    auto convAddMul_add = wrap_type<ov::op::v1::Add>({convAddMul_conv, any_input()});
    auto convAddMul_mul = wrap_type<ov::op::v1::Multiply>({convAddMul_add, any_input()});
    auto convAddMul_fq =
        wrap_type<ov::op::v0::FakeQuantize>({convAddMul_mul, any_input(), any_input(), any_input(), any_input()});
    Matcher convAddMul_matcher(convAddMul_fq);
    auto matcher = (pattern == FQMulAddPattern::ConvMulAdd) ? convMulAdd_matcher : convAddMul_matcher;
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }
    const auto& pattern_map = matcher.get_pattern_value_map();
    auto conv = pattern_map.at((pattern == FQMulAddPattern::ConvMulAdd) ? convMulAdd_conv : convAddMul_conv);

    return conv.get_node_shared_ptr()->get_input_element_type(0) == node->get_output_element_type(0);
}

bool match_conv_fq_same_types(const std::shared_ptr<const ov::Node>& node) {
    auto conv = wrap_type<ov::op::v1::Convolution>();
    auto fq = wrap_type<ov::op::v0::FakeQuantize>({conv, any_input(), any_input(), any_input(), any_input()});
    Matcher matcher(fq);
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& pattern_map = matcher.get_pattern_value_map();
    const auto conv_node = pattern_map.at(conv).get_node_shared_ptr();

    return conv_node->get_input_element_type(0) == node->get_output_element_type(0);
}

bool match_conv_stride_oc_ic_limit(const std::shared_ptr<const ov::Node>& node,
                                   const std::vector<int64_t>& strides,
                                   const ov::Shape& kernel_shape,
                                   size_t oc_ic_limit) {
    const auto weights_shape = "OC, IC, " + std::to_string(kernel_shape[0]) + ", " + std::to_string(kernel_shape[1]);
    const auto weights_m = any_input(has_static_shape() && shape_matches(weights_shape));
    const auto conv_m = wrap_type<ov::op::v1::Convolution>({any_input(), weights_m}, {{"strides", strides}});
    Matcher matcher(conv_m);
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& symbols = matcher.get_symbols();
    const auto oc = symbols.at("OC").i();
    const auto ic = symbols.at("IC").i();
    return (oc >= 0 && static_cast<size_t>(oc) < oc_ic_limit) || (ic >= 0 && static_cast<size_t>(ic) < oc_ic_limit);
}

bool match_conv_mul_add(const std::shared_ptr<const ov::Node>& node) {
    auto conv_m = wrap_type<ov::op::v1::Convolution>();
    auto mul_m = wrap_type<ov::op::v1::Multiply>({conv_m, any_input()});
    auto add_m = wrap_type<ov::op::v1::Add>({mul_m, any_input()});

    auto matcher = std::make_shared<Matcher>(add_m);
    return matcher->match(std::const_pointer_cast<ov::Node>(node));
}

}  // namespace ov::intel_cpu
