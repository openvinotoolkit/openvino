// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reduce_merge.hpp"

#include <memory>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"

using namespace ngraph;

template <typename T>
std::shared_ptr<Node> create_pattern() {
    auto input = pattern::any_input();
    auto first_axis = pattern::any_input();
    auto reduce = pattern::wrap_type<T>({input, first_axis});
    auto second_axis = pattern::any_input();
    return pattern::wrap_type<T>({reduce, second_axis});
}

template <typename T>
bool fuse_reduce_operations(const std::shared_ptr<Node>& node) {
    const auto bottom_reduce = as_type_ptr<T>(node);
    if (!bottom_reduce) {
        return false;
    }

    const auto top_reduce = as_type_ptr<T>(bottom_reduce->get_input_node_shared_ptr(0));
    if (!top_reduce) {
        return false;
    }

    if (top_reduce->get_keep_dims() != bottom_reduce->get_keep_dims()) {
        return false;
    }

    if (!top_reduce->get_keep_dims()) {
        const auto first_axes = top_reduce->get_reduction_axes();
        const auto second_axes = bottom_reduce->get_reduction_axes();

        // check if each axis in the second set is smaller than every axis in the first set
        if (!std::all_of(first_axes.begin(), first_axes.end(), [&second_axes](const size_t first_axis) {
                return std::all_of(second_axes.begin(), second_axes.end(), [first_axis](const size_t second_axis) {
                    return first_axis > second_axis;
                });
            })) {
            return false;
        }
    }

    std::shared_ptr<Node> axes =std::make_shared<opset9::Concat>(OutputVector{top_reduce->input_value(1),
                                                      bottom_reduce->input_value(1)},
                                         int64_t(0));
    if (auto constant = ov::get_constant_from_source(axes)) {
        axes = constant;
    }
    axes->set_friendly_name(bottom_reduce->get_friendly_name() + "/Axes");
    auto new_reduce =
        bottom_reduce->copy_with_new_inputs({top_reduce->input_value(0), axes->get_default_output()});
    new_reduce->set_friendly_name(bottom_reduce->get_friendly_name());

    copy_runtime_info({top_reduce, bottom_reduce}, {axes, new_reduce});
    ngraph::replace_node(bottom_reduce, new_reduce);
    return true;
}

pass::ReduceMerge::ReduceMerge() {
    MATCHER_SCOPE(ReduceMerge);

    auto reducel1_pattern = create_pattern<opset9::ReduceL1>();
    auto reducel2_pattern = create_pattern<opset9::ReduceL2>();
    auto reduce_log_and_pattern = create_pattern<opset9::ReduceLogicalAnd>();
    auto reduce_log_or_pattern = create_pattern<opset9::ReduceLogicalOr>();
    auto reduce_max_pattern = create_pattern<opset9::ReduceMax>();
    auto reduce_mean_pattern = create_pattern<opset9::ReduceMean>();
    auto reduce_min_pattern = create_pattern<opset9::ReduceMin>();
    auto reduce_prod_pattern = create_pattern<opset9::ReduceProd>();
    auto reduce_sum_pattern = create_pattern<opset9::ReduceSum>();

    auto pattern = std::make_shared<pattern::op::Or>(OutputVector{reducel1_pattern,
                                                                  reducel2_pattern,
                                                                  reduce_log_and_pattern,
                                                                  reduce_log_or_pattern,
                                                                  reduce_max_pattern,
                                                                  reduce_mean_pattern,
                                                                  reduce_min_pattern,
                                                                  reduce_prod_pattern,
                                                                  reduce_sum_pattern});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto node = m.get_match_root();
        if (ov::is_type<op::util::ArithmeticReductionKeepDims>(node)) {
            return fuse_reduce_operations<op::util::ArithmeticReductionKeepDims>(node);
        } else if (ov::is_type<op::util::LogicalReductionKeepDims>(node)) {
            return fuse_reduce_operations<op::util::LogicalReductionKeepDims>(node);
        } else {
            return false;
        }
    };
    auto m = std::make_shared<pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, callback);
}
