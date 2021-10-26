// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/convert_nonzero_to_range.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNonZeroToRange, "ConvertNonZeroToRange", 0);

ngraph::pass::ConvertNonZeroToRange::ConvertNonZeroToRange() {
    MATCHER_SCOPE(ConvertNonZeroToRange);

    auto one_const_m = ngraph::pattern::wrap_type<opset8::Constant>(ngraph::pattern::rank_equals(1));
    auto target_shape_m = ngraph::pattern::any_input(ngraph::pattern::rank_equals(1));

    auto bcast_with_2_inputs = ngraph::pattern::wrap_type<ngraph::op::util::BroadcastBase>({ one_const_m, target_shape_m },
                                                                                           ngraph::pattern::consumers_count(1));
    auto axis_mapping = ngraph::pattern::wrap_type<opset8::Constant>();
    auto bcast_with_3_inputs = ngraph::pattern::wrap_type<ngraph::op::util::BroadcastBase>({ one_const_m, target_shape_m, axis_mapping },
                                                                                           ngraph::pattern::consumers_count(1));

    auto bcast_m = std::make_shared<ngraph::pattern::op::Or>(OutputVector{ bcast_with_2_inputs, bcast_with_3_inputs });
    auto nonzero_m = ngraph::pattern::wrap_type<opset8::NonZero>({ bcast_m });

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto target_shape = pattern_map.at(target_shape_m);
        const auto one_const = as_type_ptr<opset8::Constant>(pattern_map.at(one_const_m).get_node_shared_ptr());
        const auto nonzero = as_type_ptr<opset8::NonZero>(pattern_map.at(nonzero_m).get_node_shared_ptr());
        const auto bcast = nonzero->get_input_node_shared_ptr(0);

        if (!one_const || !nonzero || target_shape.get_shape()[0] != 1ul) {
            return false;
        }

        const auto constant_values = one_const->cast_vector<std::int64_t>();
        if (std::any_of(constant_values.begin(), constant_values.end(), [](const std::int64_t x) { return x == 0; })) {
            return false;
        }

        const auto constant_et = target_shape.get_element_type();
        const auto range_start = ngraph::opset8::Constant::create(constant_et, {}, { 0 });
        const auto range_stop = ngraph::op::util::make_try_fold<opset8::Squeeze>(target_shape, opset8::Constant::create(constant_et, { 1 }, { 0 }));
        const auto range_step = ngraph::opset8::Constant::create(constant_et, {}, { 1 });
        const auto range = ngraph::op::util::make_try_fold<opset8::Range>(range_start, range_stop, range_step, nonzero->get_output_element_type(0));
        const auto unsqueeze_after = ngraph::op::util::make_try_fold<opset8::Unsqueeze>(range, opset8::Constant::create(constant_et, { 1 }, { 0 }));

        ngraph::copy_runtime_info({ one_const, bcast, nonzero}, {range, unsqueeze_after, range_stop});
        unsqueeze_after->set_friendly_name(nonzero->get_friendly_name());
        ngraph::replace_node(nonzero, unsqueeze_after);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nonzero_m, matcher_name);
    this->register_matcher(m, callback);
}
