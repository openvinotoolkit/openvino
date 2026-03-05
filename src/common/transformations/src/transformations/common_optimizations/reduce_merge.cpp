// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reduce_merge.hpp"

#include <memory>
#include <set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape_util.hpp"
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

/**
 * @brief Remaps reduction axes of a "bottom" Reduce operation through the axes removed by a preceding "top" Reduce.
 *
 * Given:
 *  - @p axes_top   : axes reduced (removed) by the first/top Reduce
 *  - @p axes_bottom: axes reduced by the second/bottom Reduce in the resulting tensor rank (after @p axes_top removal)
 *
 * This function converts each axis in @p axes_bottom back to the corresponding axis index in the original tensor
 * (i.e., the tensor rank before @p axes_top axes were removed), producing a set of axes that can be used to express
 * the bottom reduction directly on the original input.
 *
 * @param axes_bottom Axes indices in the tensor after the top reduction has removed @p axes_top dimensions.
 * @param axes_top    Axes indices removed by the top reduction in the original tensor.
 * @return A set of axes indices in the original tensor rank corresponding to @p axes_bottom.
 *
 * @note The result is returned as a sorted unique set.
 */
std::set<size_t> remap_axes(const ov::AxisSet& axes_bottom, const ov::AxisSet& axes_top) {
    // Compute the highest index that needs to be mapped to set the required rank
    // Create a vector of original axis indices: [0, 1, ..., required_rank-1]
    const size_t max_bottom_axis = *axes_bottom.rbegin();
    const size_t required_rank = max_bottom_axis + axes_top.size() + 1;
    std::vector<size_t> original_axes(required_rank);
    std::iota(original_axes.begin(), original_axes.end(), 0);

    // Remove axes_top from original_axes
    auto reduced_axes = ov::util::reduce(original_axes, axes_top);

    // Map each axes_bottom index in the reduced axes back to the original indices
    std::set<size_t> remapped_axes;
    for (const size_t bottom_axis : axes_bottom) {
        remapped_axes.insert(reduced_axes[bottom_axis]);
    }
    return remapped_axes;
}

ov::Output<ov::Node> cast(const ov::Output<ov::Node>& in, const element::Type& target_type) {
    if (in.get_element_type() != target_type) {
        const auto convert_op = std::make_shared<ov::op::v0::Convert>(in, target_type);
        copy_runtime_info(in.get_node_shared_ptr(), convert_op);
        return convert_op;
    }
    return in;
}

ov::Output<ov::Node> make_1d(const ov::Output<ov::Node>& in) {
    const auto& ps = in.get_partial_shape();
    if (ps.rank() == ov::Dimension(0)) {
        const auto unsq_axis = ov::op::v0::Constant::create(in.get_element_type(), {}, {0});
        const auto unsq = std::make_shared<ov::op::v0::Unsqueeze>(in, unsq_axis);
        copy_runtime_info(in.get_node_shared_ptr(), {unsq_axis, unsq});
        return unsq;
    }
    return in;
}

template <typename T>
std::shared_ptr<Node> create_pattern() {
    auto input = pattern::any_input(pattern::has_static_rank());
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

    const ov::AxisSet axes_top = top_reduce->get_reduction_axes();
    const ov::AxisSet axes_bottom = bottom_reduce->get_reduction_axes();
    std::shared_ptr<Node> axes = nullptr;

    if (!axes_top.empty() && !axes_bottom.empty()) {
        // if both axes are constants, we can merge them into a single constant
        std::set<size_t> fused_axes{axes_top.begin(), axes_top.end()};

        if (top_reduce->get_keep_dims()) {
            fused_axes.insert(axes_bottom.begin(), axes_bottom.end());
        } else {
            std::set<size_t> axes_remapped = remap_axes(axes_bottom, axes_top);
            fused_axes.insert(axes_remapped.begin(), axes_remapped.end());
        }

        axes = op::v0::Constant::create(element::i64,
                                        Shape{fused_axes.size()},
                                        std::vector<int64_t>(fused_axes.begin(), fused_axes.end()));
    } else if (top_reduce->get_keep_dims()) {
        // if the axes input of any of reduce is not constant, but keep_dims is true,
        // we can merge them by concatenating axes inputs
        auto axes1_input = top_reduce->input_value(1);
        auto axes2_input = bottom_reduce->input_value(1);

        ov::Output<ov::Node> axes1 = axes1_input;
        ov::Output<ov::Node> axes2 = axes2_input;

        // Align reduce axes constants by type
        const bool dtype_match = axes1_input.get_element_type() == axes2_input.get_element_type();
        if (!dtype_match) {
            axes1 = cast(axes1_input, element::i64);
            axes2 = cast(axes2_input, element::i64);
        }

        // Align reduce axes constants by shape
        axes1 = make_1d(axes1);
        axes2 = make_1d(axes2);

        axes = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{axes1, axes2}, int64_t(0));
    } else {
        return false;
    }

    auto reduce_fused = bottom_reduce->copy_with_new_inputs({top_reduce->input_value(0), axes});
    reduce_fused->set_friendly_name(bottom_reduce->get_friendly_name());
    copy_runtime_info({top_reduce, bottom_reduce}, {axes, reduce_fused});
    ov::replace_node(bottom_reduce, reduce_fused);

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
