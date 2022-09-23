// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/eye_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/gather_nd_base.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

using namespace ov::opset9;

/** \brief Check if output is rank one and data type can be i32 or i64. */
constexpr auto is_rank_one_int_shape = [](const auto& output) {
    return pattern::type_matches_any({element::i32, element::i64})(output) && pattern::has_static_shape()(output) &&
           pattern::rank_equals(1)(output);
};

/** \brief Predicate to check eye k node is valid. */
constexpr auto k_predicate = [](const auto& output) {
    return is_rank_one_int_shape(output) && (output.get_partial_shape()[0].get_length() == 1);
};

/** \brief Predicate to check eye batch node is valid. */
constexpr auto batch_predicate = [](const auto& output) {
    return is_rank_one_int_shape(output) && output.get_partial_shape()[0].get_length();
};

/**
 * \brief Make eye model which generate eye matrix.
 *
 * If 'k' is outside the eye dimension then result matrix will be filled with zeros.
 *
 * \param reg     Node registry used store created nodes.
 * \param height  Height of eye
 * \param width   Width of eye
 * \param k       Eye diagonal shift.
 * \param dtype   Data type of eye.
 *
 * \return Pointer to decomposed eye model.
 */
std::shared_ptr<Node> make_eye_model(NodeRegistry& reg,
                                     const Output<Node>& height,
                                     const Output<Node>& width,
                                     const Output<Node>& k,
                                     element::Type dtype) {
    const auto zero_int = reg.add(Constant::create(element::i64, Shape{1}, {0}));
    const auto zero = reg.add(Constant::create(dtype, Shape{1}, {0}));
    const auto one = reg.add(Constant::create(dtype, Shape{1}, {1}));

    const auto k_neg = reg.make<Negative>(k);
    const auto k_axis = reg.make<Concat>(OutputVector{k_neg, k}, 0);

    const auto eye_shape = reg.make<Concat>(OutputVector{height, width}, 0);

    // Calculate eye zero padding and internal square eye size.
    const auto pad_start = reg.make<Minimum>(eye_shape, reg.make<Maximum>(zero_int, k_axis));
    const auto shape_pad_diff = reg.make<Subtract>(eye_shape, pad_start);
    const auto eye_size = reg.make<ReduceMin>(shape_pad_diff, zero_int, true);
    const auto pad_end = reg.make<Subtract>(shape_pad_diff, eye_size);

    // Make 1d-eye as eye_size times of (1, zeros(eye_size)), trimmed at end by eye_size elements.
    const auto zeros = reg.make<Tile>(zero, eye_size);
    const auto one_followed_by_zeros = reg.make<Concat>(OutputVector{one, zeros}, 0);
    const auto eye_1d = reg.make<Pad>(reg.make<Tile>(one_followed_by_zeros, eye_size),
                                      zero_int,
                                      reg.make<Negative>(eye_size),
                                      op::PadMode::CONSTANT);
    // Reshape 1d-eye to 2d-eye
    const auto eye_2d = reg.make<Reshape>(eye_1d, reg.make<Concat>(OutputVector{eye_size, eye_size}, 0), false);

    // Pad Eye to get final shape
    return reg.make<Pad>(eye_2d, pad_start, pad_end, op::PadMode::CONSTANT);
}

/**
 * \brief Make eye model as basic 2D eye replicated as specified in batch size.
 *
 * \param reg    Node registry used store created nodes.
 * \param eye    Eye model.
 * \param batch  1-D tensor which defines leading batch dimensions of output eye shape.
 *
 * \return Pointer to decomposed eye model.
 */
std::shared_ptr<Node> make_eye_batches(NodeRegistry& reg, const Output<Node>& eye, const Output<Node>& batch) {
    const auto eye_tile = reg.make<Constant>(element::i64, Shape{2}, 1);

    // `batch_repeats` repeat eye matrix as tile only in higher dimensions than 1 by number(s) in batch parameter.
    const auto batch_repeats = reg.make<Concat>(OutputVector{batch, eye_tile}, 0);

    return reg.make<Tile>(eye, batch_repeats);
}

EyeDecomposition::EyeDecomposition() {
    MATCHER_SCOPE(EyeDecomposition);

    auto p_height = pattern::any_input();
    auto p_width = pattern::any_input();
    auto p_k = pattern::wrap_type<Constant>(k_predicate);
    auto p_batch = pattern::wrap_type<Constant>(batch_predicate);

    auto p_eye_no_batch = pattern::wrap_type<Eye>({p_height, p_width, p_k});
    auto p_eye_batch = pattern::wrap_type<Eye>({p_height, p_width, p_k, p_batch});

    auto p_eye = std::make_shared<pattern::op::Or>(OutputVector{p_eye_batch, p_eye_no_batch});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto m_eye = std::dynamic_pointer_cast<Eye>(m.get_match_root());

        if ((!m_eye) || transformation_callback(m_eye)) {
            return false;
        }

        NodeRegistry copy_reg;
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto dtype = m_eye->get_out_type();
        const auto width = pattern_to_output.at(p_width);
        const auto height = pattern_to_output.at(p_height);
        const auto k = pattern_to_output.at(p_k);

        auto eye = make_eye_model(copy_reg, height, width, k, dtype);

        if (pattern_to_output.find(p_batch) != pattern_to_output.end()) {
            eye = make_eye_batches(copy_reg, eye, pattern_to_output.at(p_batch));
        }

        eye->set_friendly_name(m_eye->get_friendly_name());
        ov::copy_runtime_info(m_eye, copy_reg.get());
        ov::replace_node(m_eye, eye);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(p_eye, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
