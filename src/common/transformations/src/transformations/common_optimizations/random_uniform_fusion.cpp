// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/random_uniform_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::RandomUniformFusion::RandomUniformFusion() {
    MATCHER_SCOPE(RandomUniformFusion);
    const auto data_pattern = pass::pattern::any_input();
    const auto ru_min_input_pattern = pass::pattern::any_input();
    const auto ru_max_input_pattern = pass::pattern::any_input();
    const auto random_uniform_pattern = ov::pass::pattern::wrap_type<ov::op::v8::RandomUniform>(
        {data_pattern, ru_min_input_pattern, ru_max_input_pattern},
        pattern::consumers_count(1));
    const auto const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    const auto optional_convert = ov::pass::pattern::optional<ov::op::v0::Convert>(random_uniform_pattern);

    const auto mul_add_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::Multiply, ov::op::v1::Add>({optional_convert, const_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto data = pattern_map.at(data_pattern);
        const auto random_uniform = pattern_map.at(random_uniform_pattern);
        const auto constant = pattern_map.at(const_pattern);
        const auto ru = ov::as_type_ptr<ov::op::v8::RandomUniform>(random_uniform.get_node_shared_ptr());
        if (!ru)
            return false;
        if (!ru->get_out_type().is_real())
            return false;

        const auto old_const = ov::as_type_ptr<ov::op::v0::Constant>(constant.get_node_shared_ptr());
        if (!old_const)
            return false;
        if (!old_const->get_element_type().is_real())
            return false;

        auto const_shape = old_const->get_shape();
        if (shape_size(const_shape) != 1)
            return false;

        const auto& value = old_const->cast_vector<double>();
        auto new_const = ov::op::v0::Constant::create(ru->get_out_type(), Shape{}, value);

        const auto& mul_add = pattern_map.at(mul_add_pattern);
        const auto mul_add_ptr = std::dynamic_pointer_cast<ov::Node>(mul_add.get_node_shared_ptr());
        const auto new_mul_add1 = mul_add_ptr->clone_with_new_inputs({ru->input_value(1), new_const});
        const auto new_mul_add2 = mul_add_ptr->clone_with_new_inputs({ru->input_value(2), new_const});

        const auto& folded_const1 = ov::util::get_constant_from_source(new_mul_add1);
        const auto& folded_const2 = ov::util::get_constant_from_source(new_mul_add2);

        const auto new_ru = ru->clone_with_new_inputs(
            {data, folded_const1 ? folded_const1 : new_mul_add1, folded_const2 ? folded_const2 : new_mul_add2});

        if (pattern_map.count(optional_convert)) {
            auto cvt = pattern_map.at(optional_convert).get_node_shared_ptr();
            if (!cvt->get_element_type().is_real())
                return false;
            const auto new_ru_conv = cvt->clone_with_new_inputs({new_ru});
            copy_runtime_info({ru, cvt, mul_add.get_node_shared_ptr()},
                              {new_mul_add1, new_mul_add2, new_ru, new_ru_conv});
            new_ru_conv->set_friendly_name(m.get_match_root()->get_friendly_name());
            ov::replace_node(m.get_match_root(), new_ru_conv);
        } else {
            copy_runtime_info({ru, mul_add.get_node_shared_ptr()}, {new_mul_add1, new_mul_add2, new_ru});
            new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
            ov::replace_node(m.get_match_root(), new_ru);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_add_pattern, matcher_name);
    this->register_matcher(m, callback);
}
