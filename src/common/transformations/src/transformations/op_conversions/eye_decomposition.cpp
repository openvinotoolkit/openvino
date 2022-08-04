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
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace pass {

/** \brief Check if output is rank one with static dimension and data type can be i32 or i64. */
const auto is_static_rank_one_int_shape = [](const Output<Node>& output) -> bool {
    return pattern::type_matches_any({element::i32, element::i64})(output) && pattern::rank_equals(1)(output) &&
           pattern::has_static_dim(0);
};

/** \brief Predicate to check eye k node is valid. */
const auto k_predicate = [](const Output<Node>& output) -> bool {
    return is_static_rank_one_int_shape(output) && (output.get_partial_shape()[0].get_length() == 1);
};

/** \brief Predicate to check eye batch node is valid. */
const auto batch_predicate = [](const Output<Node>& output) -> bool {
    return is_static_rank_one_int_shape(output) && output.get_partial_shape()[0].get_length();
};

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
    const auto eye_2d =
        register_new_node<op::v1::Reshape>(eye_1d,
                                           register_new_node<op::v0::Concat>(OutputVector{eye_size, eye_size}, 0),
                                           false);

    // Pad Eye to get final shape
    return register_new_node<op::v1::Pad>(eye_2d, pad_start, pad_end, ov::op::PadMode::CONSTANT);
}

std::shared_ptr<Node> EyeDecomposition::make_eye_batches(const Output<Node>& eye, const Output<Node>& batch) {
    const auto zero_int = register_new_node(op::v0::Constant::create(element::i64, Shape{1}, {0}));
    const auto eye_tile = register_new_node<op::v0::Constant>(element::i64, Shape{2}, 1);

    // `batch_repeats` repeat eye matrix as tile only in higher dimensions than 1 by number(s) in batch parameter.
    const auto batch_repeats = register_new_node<op::v0::Concat>(OutputVector{batch, eye_tile}, 0);

    return register_new_node<op::v0::Tile>(eye, batch_repeats);
}

EyeDecomposition::EyeDecomposition() {
    auto p_height = pattern::any_input();
    auto p_width = pattern::any_input();
    auto p_k = pattern::wrap_type<op::v0::Constant>(k_predicate);
    auto p_batch = pattern::wrap_type<op::v0::Constant>(batch_predicate);

    auto p_eye_no_batch = pattern::wrap_type<op::v9::Eye>({p_height, p_width, p_k});
    auto p_eye_batch = pattern::wrap_type<op::v9::Eye>({p_height, p_width, p_k, p_batch});

    auto p_eye = std::make_shared<pattern::op::Or>(OutputVector{p_eye_batch, p_eye_no_batch});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto m_eye = std::dynamic_pointer_cast<op::v9::Eye>(m.get_match_root());

        if ((!m_eye) || transformation_callback(m_eye)) {
            return false;
        }

        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto dtype = m_eye->get_out_type();
        const auto width = pattern_to_output.at(p_width);
        const auto height = pattern_to_output.at(p_height);
        const auto k = pattern_to_output.at(p_k);

        auto eye = make_eye_model(height, width, k, dtype);

        if (m_eye->get_input_size() == p_eye_batch->get_input_size()) {
            eye = make_eye_batches(eye, pattern_to_output.at(p_batch));
        }

        eye->set_friendly_name(m_eye->get_friendly_name());
        ov::copy_runtime_info(m_eye, get_new_nodes());
        ov::replace_node(m_eye, eye);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(p_eye, "EyeDecomposition");
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
