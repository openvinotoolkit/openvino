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
#include "openvino/op/unique.hpp"
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

    const ov::AxisSet axes_top = top_reduce->get_reduction_axes();
    const ov::AxisSet axes_bottom = bottom_reduce->get_reduction_axes();
    std::shared_ptr<Node> axes = nullptr;
    if (!axes_top.empty() && !axes_bottom.empty()) {
        // case when both axes are constants
        std::set<size_t> fused_axes;
        fused_axes.insert(axes_top.begin(), axes_top.end());

        if (top_reduce->get_keep_dims()) {
            fused_axes.insert(axes_bottom.begin(), axes_bottom.end());
        } else {
            for (size_t bottom_axis : axes_bottom) {
                size_t remaining = bottom_axis;
                size_t original_axis = 0;

                auto top_axes_iter = axes_top.begin();
                while (original_axis <= bottom_axis + axes_top.size()) {
                    bool removed = false;
                    if (top_axes_iter != axes_top.end() && *top_axes_iter == original_axis) {
                        removed = true;
                        ++top_axes_iter;
                    }

                    if (!removed) {
                        if (remaining == 0) {
                            fused_axes.insert(original_axis);
                            break;
                        }
                        --remaining;
                    }

                    ++original_axis;
                }
            }
        }

        axes = op::v0::Constant::create(element::i64,
                                        Shape{fused_axes.size()},
                                        std::vector<int64_t>(fused_axes.begin(), fused_axes.end()));
    } else if (top_reduce->get_keep_dims()) {
        auto axes1_input = top_reduce->input_value(1);
        auto axes2_input = bottom_reduce->input_value(1);

        auto cast = [](const ov::Output<ov::Node>& in, const element::Type& target_type) -> ov::Output<ov::Node> {
            if (in.get_element_type() != target_type) {
                const auto convert_op = std::make_shared<ov::op::v0::Convert>(in, target_type);
                copy_runtime_info(in.get_node_shared_ptr(), convert_op);
                return convert_op;
            }
            return in;
        };

        auto make_1d = [](const ov::Output<ov::Node>& in) -> ov::Output<ov::Node> {
            const auto& ps = in.get_partial_shape();
            if (ps.rank().is_static() && ps.rank().get_length() == 0) {
                const auto unsq_axis = ov::op::v0::Constant::create(in.get_element_type(), {}, {0});
                const auto unsq = std::make_shared<ov::op::v0::Unsqueeze>(in, unsq_axis);
                copy_runtime_info(in.get_node_shared_ptr(), {unsq_axis, unsq});
                return unsq;
            }
            return in;
        };

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
