// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/eye_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/gather_nd_base.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace pass {

std::shared_ptr<Node> EyeDecomposition::make_eye_model(const Output<Node>& height,
                                                       const Output<Node>& width,
                                                       const Output<Node>& k,
                                                       element::Type dtype) {
    const auto zero_int = register_new_node(op::v0::Constant::create(element::i64, Shape{1}, {0}));
    const auto zero = register_new_node(op::v0::Constant::create(dtype, Shape{1}, {0}));
    const auto one = register_new_node(op::v0::Constant::create(dtype, ov::Shape{1}, {1}));

    const auto k_neg = register_new_node<op::v0::Negative>(k);
    const auto k_axis = register_new_node<op::v0::Concat>(OutputVector{k_neg, k}, 0);

    const auto eye_shape = register_new_node<op::v0::Concat>(OutputVector{height, width}, 0);

    // Calculate eye zero padding and internal square eye size.
    const auto pad_start =
        register_new_node<op::v1::Minimum>(eye_shape, register_new_node<op::v1::Maximum>(zero_int, k_axis));
    const auto shape_pad_diff = register_new_node<op::v1::Subtract>(eye_shape, pad_start);
    const auto eye_size = register_new_node<op::v1::ReduceMin>(shape_pad_diff, zero_int, true);
    const auto pad_end = register_new_node<op::v1::Subtract>(shape_pad_diff, eye_size);

    // Make 1d-ey as eye_size times of (1, zeros(eye_size)), trimmed at end by eye_size elements at end.
    const auto zeros = register_new_node<op::v0::Tile>(zero, eye_size);
    const auto one_followed_by_zeros = register_new_node<op::v0::Concat>(OutputVector{one, zeros}, 0);
    const auto eye_1d = register_new_node<op::v1::Pad>(register_new_node<op::v0::Tile>(one_followed_by_zeros, eye_size),
                                                       zero_int,
                                                       register_new_node<op::v0::Negative>(eye_size),
                                                       ov::op::PadMode::CONSTANT);
    // Reshape 1d-eye to 2d-eye
    const auto square_eye =
        register_new_node<op::v1::Reshape>(eye_1d,
                                           register_new_node<op::v0::Concat>(OutputVector{eye_size, eye_size}, 0),
                                           false);

    // Pad Eye to get final shape
    return register_new_node<op::v1::Pad>(square_eye, pad_start, pad_end, ov::op::PadMode::CONSTANT);
}

ov::pass::EyeDecomposition::EyeDecomposition() {
    auto p_height = pattern::any_input();
    auto p_width = pattern::any_input();
    auto p_k = pattern::wrap_type<op::v0::Constant>(pattern::type_matches_any({element::i32, element::i64}));
    auto p_eye = pattern::wrap_type<op::v9::Eye>({p_height, p_width, p_k});

    ngraph::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto m_eye = std::dynamic_pointer_cast<ov::op::v9::Eye>(m.get_match_root());

        if ((!m_eye) || transformation_callback(m_eye)) {
            return false;
        }

        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto dtype = m_eye->get_out_type();
        const auto width = pattern_to_output.at(p_width);
        const auto height = pattern_to_output.at(p_height);
        const auto k = pattern_to_output.at(p_k);

        const auto eye = make_eye_model(height, width, k, dtype);

        eye->set_friendly_name(m_eye->get_friendly_name());
        ov::copy_runtime_info(m_eye, get_new_nodes());
        ov::replace_node(m_eye, eye);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(p_eye, "EyeDecomposition");
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
