// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reduce_merge.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;
namespace op_util = ov::op::util;

namespace ov::pass {

namespace {
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

    auto in_rank = top_reduce->get_input_partial_shape(0).rank();
    if (in_rank.is_dynamic()) {
        return false;
    }

    auto top_axis_const = ov::util::get_constant_from_source(top_reduce->input_value(1));
    auto bottom_axis_const = ov::util::get_constant_from_source(bottom_reduce->input_value(1));
    if (!top_axis_const || !bottom_axis_const) {
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

    // Align reduce axes constants by shape and type
    const bool dtype_match =
        top_reduce->input_value(1).get_element_type() == bottom_reduce->input_value(1).get_element_type();
    for (auto& reduce : {top_reduce, bottom_reduce}) {
        const auto reduce_axes_output = reduce->input_value(1);
        const auto reduce_axes_node = reduce_axes_output.get_node_shared_ptr();
        const auto reduce_axes_rank = reduce_axes_output.get_partial_shape().rank();
        if (reduce_axes_rank == Dimension(0)) {
            const auto unsqueeze_const = v0::Constant::create(reduce_axes_node->get_element_type(), {}, {0});
            const auto unsqueeze = std::make_shared<v0::Unsqueeze>(reduce_axes_output, unsqueeze_const);
            reduce->inputs()[1].replace_source_output(unsqueeze);
            copy_runtime_info(reduce_axes_node, {unsqueeze_const, unsqueeze});
        }
        if (!dtype_match) {
            const auto cast = std::make_shared<v0::Convert>(reduce->input_value(1), ov::element::i64);
            reduce->inputs()[1].replace_source_output(cast);
            copy_runtime_info(reduce_axes_node, cast);
        }
    }

    const auto first_axes = top_reduce->get_reduction_axes();
    const auto second_axes = bottom_reduce->get_reduction_axes();
    std::vector<int64_t> axes_values;
    axes_values.reserve(first_axes.size() + second_axes.size());
    axes_values.insert(axes_values.end(), first_axes.begin(), first_axes.end());
    axes_values.insert(axes_values.end(), second_axes.begin(), second_axes.end());
    std::shared_ptr<Node> axes = v0::Constant::create(element::i64, Shape{axes_values.size()}, axes_values);

    axes->set_friendly_name(bottom_reduce->get_friendly_name() + "/Axes");
    auto new_reduce = bottom_reduce->copy_with_new_inputs({top_reduce->input_value(0), axes->get_default_output()});
    new_reduce->set_friendly_name(bottom_reduce->get_friendly_name());

    copy_runtime_info({top_reduce, bottom_reduce}, {axes, new_reduce});
    ov::replace_node(bottom_reduce, new_reduce);
    return true;
}
}  // namespace

ReduceMerge::ReduceMerge() {
    MATCHER_SCOPE(ReduceMerge);

    auto reducel1_pattern = create_pattern<v4::ReduceL1>();
    auto reducel2_pattern = create_pattern<v4::ReduceL2>();
    auto reduce_log_and_pattern = create_pattern<v1::ReduceLogicalAnd>();
    auto reduce_log_or_pattern = create_pattern<v1::ReduceLogicalOr>();
    auto reduce_max_pattern = create_pattern<v1::ReduceMax>();
    auto reduce_mean_pattern = create_pattern<v1::ReduceMean>();
    auto reduce_min_pattern = create_pattern<v1::ReduceMin>();
    auto reduce_prod_pattern = create_pattern<v1::ReduceProd>();
    auto reduce_sum_pattern = create_pattern<v1::ReduceSum>();

    auto pattern_node = std::make_shared<pattern::op::Or>(OutputVector{reducel1_pattern,
                                                                       reducel2_pattern,
                                                                       reduce_log_and_pattern,
                                                                       reduce_log_or_pattern,
                                                                       reduce_max_pattern,
                                                                       reduce_mean_pattern,
                                                                       reduce_min_pattern,
                                                                       reduce_prod_pattern,
                                                                       reduce_sum_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto node = m.get_match_root();
        if (ov::is_type<op_util::ArithmeticReductionKeepDims>(node)) {
            return fuse_reduce_operations<op_util::ArithmeticReductionKeepDims>(node);
        } else if (ov::is_type<op_util::LogicalReductionKeepDims>(node)) {
            return fuse_reduce_operations<op_util::LogicalReductionKeepDims>(node);
        } else {
            return false;
        }
    };
    auto m = std::make_shared<pattern::Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
