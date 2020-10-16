// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <transformations/smart_reshape/strided_slice_squeeze.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::StridedSliceSqueeze, "StridedSliceSqueeze", 0);

ngraph::pass::StridedSliceSqueeze::StridedSliceSqueeze() {
    auto ss_label = ngraph::pattern::wrap_type<opset5::StridedSlice>(pattern::consumers_count(1));
    auto squeeze_label = ngraph::pattern::wrap_type<opset5::Squeeze>({ss_label, ngraph::pattern::wrap_type<opset5::Constant>()});

    matcher_pass_callback callback = [](pattern::Matcher &m) -> bool {
        const auto & squeeze = m.get_match_root();
        const auto & const_axes = std::dynamic_pointer_cast<ngraph::opset5::Constant>(squeeze->get_input_node_shared_ptr(1));

        auto slice = std::dynamic_pointer_cast<ngraph::opset5::StridedSlice>(squeeze->get_input_node_shared_ptr(0));
        if (!const_axes || !slice)
            return false;

        const auto & slice_plan = get_slice_plan(slice);
        if (slice_plan.begins.empty() || slice_plan.reshape_in_shape != slice_plan.reshape_out_shape || !slice_plan.reverse_axes.empty())
            return false;

        const auto & axes = const_axes->cast_vector<int64_t>();

        auto begin = std::dynamic_pointer_cast<ngraph::opset5::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto end = std::dynamic_pointer_cast<ngraph::opset5::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto strides = std::dynamic_pointer_cast<ngraph::opset5::Constant>(slice->input_value(3).get_node_shared_ptr());
        if (!begin || !end || !strides)
            return false;

        auto begin_vec = begin->cast_vector<int64_t>();
        auto end_vec = end->cast_vector<int64_t>();
        auto strides_vec = strides->cast_vector<int64_t>();
        auto begin_mask = slice->get_begin_mask();
        auto end_mask = slice->get_end_mask();
        auto new_axis_mask = slice->get_new_axis_mask();
        auto shrink_axis_mask = slice->get_shrink_axis_mask();
        auto ellipsis_mask = slice->get_ellipsis_mask();

        for (const auto & axis : axes) {
            if ((slice_plan.ends[axis] - slice_plan.begins[axis]) != 1 && slice_plan.strides[axis] == 1)
                return false;
            begin_vec[axis] = slice_plan.begins[axis];
            end_vec[axis] = slice_plan.ends[axis];
            strides_vec[axis] = 1;
            begin_mask[axis] = 0;
            end_mask[axis] = 0;
            new_axis_mask[axis] = 0;
            shrink_axis_mask[axis] = 0;
            ellipsis_mask[axis] = 0;
        }

        auto new_slice = std::make_shared<opset5::StridedSlice>(
                slice->input_value(0),
                opset5::Constant::create(element::i64, {begin_vec.size()}, begin_vec),
                opset5::Constant::create(element::i64, {end_vec.size()}, end_vec),
                opset5::Constant::create(element::i64, {strides_vec.size()}, strides_vec),
                begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask);
        return replace_node_update_name(slice, new_slice);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(squeeze_label, "StridedSliceSqueeze");
    register_matcher(m, callback);
}
