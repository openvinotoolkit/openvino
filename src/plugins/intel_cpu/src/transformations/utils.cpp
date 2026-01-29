// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu {

template <class T>
bool match_conv_add_mul_fq(const std::shared_ptr<const ov::Node>& node) {
    auto conv_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>(
        {ov::pass::pattern::any_input(ov::pass::pattern::type_matches_any({ov::element::i8, ov::element::u8})),
         ov::pass::pattern::any_input()});
    auto add_m = ov::pass::pattern::wrap_type<ov::op::v1::Add>({conv_m, ov::pass::pattern::any_input()});
    auto mul0_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_m, ov::pass::pattern::any_input()});
    auto fq_m = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>(
        {mul0_m,
         ov::pass::pattern::any_input(),
         ov::pass::pattern::any_input(),
         ov::pass::pattern::any_input(),
         ov::pass::pattern::any_input()},
        ov::pass::pattern::type_matches_any({ov::element::i8, ov::element::u8}));
    auto final_m = ov::pass::pattern::wrap_type<T>({fq_m, ov::pass::pattern::any_input()});

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(final_m);
    if (!matcher->match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& pattern_map = matcher->get_pattern_value_map();
    const auto fq = pattern_map.at(fq_m).get_node_shared_ptr();
    const auto conv = pattern_map.at(conv_m).get_node_shared_ptr();

    return conv->get_input_element_type(0) == fq->get_output_element_type(0);
}

template bool match_conv_add_mul_fq<ov::op::v1::Subtract>(const std::shared_ptr<const ov::Node>& node);
template bool match_conv_add_mul_fq<ov::op::v1::Multiply>(const std::shared_ptr<const ov::Node>& node);

bool match_fq_mul_conv_bias_same_types(const std::shared_ptr<const ov::Node>& node) {
    auto conv_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>();
    auto conv_or_bias_m = ov::pass::pattern::optional<ov::op::v1::Add>({conv_m, ov::pass::pattern::any_input()});
    auto mul_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({conv_or_bias_m, ov::pass::pattern::any_input()});
    auto fq_m = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>({mul_m,
                                                                        ov::pass::pattern::any_input(),
                                                                        ov::pass::pattern::any_input(),
                                                                        ov::pass::pattern::any_input(),
                                                                        ov::pass::pattern::any_input()});
    ov::pass::pattern::Matcher matcher(fq_m);
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& pattern_map = matcher.get_pattern_value_map();
    const auto conv = pattern_map.at(conv_m).get_node_shared_ptr();

    return conv->get_input_element_type(0) == node->get_output_element_type(0);
}

bool match_conv_stride_oc_ic_limit(const std::shared_ptr<const ov::Node>& node,
                                   const ov::Strides& strides,
                                   const ov::Shape& kernel_shape,
                                   size_t oc_ic_limit) {
    const auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
    if (!conv) {
        return false;
    }

    if (strides.size() < 2 || kernel_shape.size() < 2) {
        return false;
    }

    const auto& conv_strides = conv->get_strides();
    if (conv_strides.size() < 2 || conv_strides[0] != strides[0] || conv_strides[1] != strides[1]) {
        return false;
    }

    const auto weights_shape = "OC, IC, " + std::to_string(kernel_shape[0]) + ", " + std::to_string(kernel_shape[1]);
    const auto weights_m = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape() &&
                                                        ov::pass::pattern::shape_matches(weights_shape));
    const auto conv_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Convolution>({ov::pass::pattern::any_input(), weights_m},
                                                              {{"strides", strides}});
    ov::pass::pattern::Matcher matcher(conv_m);
    if (!matcher.match(std::const_pointer_cast<ov::Node>(node))) {
        return false;
    }

    const auto& symbols = matcher.get_symbols();
    const auto oc = symbols.at("OC").i();
    const auto ic = symbols.at("IC").i();
    return (oc >= 0 && static_cast<size_t>(oc) < oc_ic_limit) || (ic >= 0 && static_cast<size_t>(ic) < oc_ic_limit);
}

bool match_conv_mul_add(const std::shared_ptr<const ov::Node>& node) {
    auto conv_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>();
    auto mul_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({conv_m, ov::pass::pattern::any_input()});
    auto add_m = ov::pass::pattern::wrap_type<ov::op::v1::Add>({mul_m, ov::pass::pattern::any_input()});

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(add_m);
    return matcher->match(std::const_pointer_cast<ov::Node>(node));
}

}  // namespace ov::intel_cpu
