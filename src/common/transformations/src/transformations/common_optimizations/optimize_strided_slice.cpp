// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/optimize_strided_slice.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/util/slice_plan.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

bool ov::pass::UselessSliceEraser::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(UselessSliceEraser);
    bool rewritten = false;
    for (auto& node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        rewritten = ov::op::util::process_subgraph(*this, node) || rewritten;

        bool is_slice = ov::is_type<ov::op::v1::StridedSlice>(node) || ov::is_type<ov::op::v8::Slice>(node);
        if (!is_slice || node->get_output_partial_shape(0).is_dynamic() ||
            node->get_input_partial_shape(0).is_dynamic())
            continue;
        if (node->get_input_shape(0) != node->get_output_shape(0))
            continue;

        auto stridesNode = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(3));
        if (stridesNode) {
            auto strides = stridesNode->cast_vector<int64_t>();
            if (!std::any_of(strides.begin(), strides.end(), [](int64_t strd) {
                    return strd < 0;
                })) {
                rewritten = replace_output_update_name(node->output(0), node->input_value(0)) || rewritten;
            }
        }
    }
    return rewritten;
}

namespace {

op::util::SlicePlan get_slice_plan(std::shared_ptr<ov::op::v1::StridedSlice> slice) {
    auto convert_mask_to_axis_set = [](const std::vector<int64_t>& mask) {
        ov::AxisSet axis_set{};
        for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
            if (mask[i] == 1)
                axis_set.emplace(i);
        }
        return axis_set;
    };

    auto data = slice->input_value(0).get_node_shared_ptr();
    auto begin = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
    auto end = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
    auto strides = ov::as_type_ptr<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());
    if (!begin || !end || !strides || slice->input(0).get_partial_shape().is_dynamic())
        return op::util::SlicePlan();

    auto begin_vec = begin->cast_vector<int64_t>();
    auto end_vec = end->cast_vector<int64_t>();
    auto strides_vec = strides->cast_vector<int64_t>();
    const auto begin_mask = convert_mask_to_axis_set(slice->get_begin_mask());
    const auto end_mask = convert_mask_to_axis_set(slice->get_end_mask());

    const auto plan = op::util::make_slice_plan(slice->input(0).get_shape(),
                                                begin_vec,
                                                end_vec,
                                                strides_vec,
                                                begin_mask,
                                                end_mask,
                                                convert_mask_to_axis_set(slice->get_new_axis_mask()),
                                                convert_mask_to_axis_set(slice->get_shrink_axis_mask()),
                                                convert_mask_to_axis_set(slice->get_ellipsis_mask()));
    return plan;
}

}  // namespace

bool ov::pass::GroupedStridedSliceOptimizer::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(GroupedStridedSliceOptimizer);
    bool graph_rewritten = false;
    struct planned_slice {
        std::shared_ptr<ov::op::v1::StridedSlice> ptr;
        op::util::SlicePlan plan;
    };

    std::map<ov::Output<Node>, std::vector<planned_slice>> source_to_ss_with_plan;
    for (const auto& node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        graph_rewritten = ov::op::util::process_subgraph(*this, node) || graph_rewritten;

        if (auto ss = ov::as_type_ptr<ov::op::v1::StridedSlice>(node)) {
            auto slice_plan = get_slice_plan(ss);
            if (slice_plan == op::util::SlicePlan())
                continue;
            source_to_ss_with_plan[ss->input_value(0)].push_back({ss, slice_plan});
        }
    }

    for (auto& pair : source_to_ss_with_plan) {
        if (pair.second.size() < 2)
            continue;

        bool valid_for_replacement = true;

        auto root_plan = pair.second[0].plan;
        for (const auto& ss_plan : pair.second) {
            valid_for_replacement &= (ss_plan.plan.begins.size() == root_plan.begins.size());
            valid_for_replacement &=
                (ss_plan.ptr->get_ellipsis_mask().empty() && ss_plan.ptr->get_new_axis_mask().empty() &&
                 ss_plan.ptr->get_shrink_axis_mask().empty());
        }

        if (!valid_for_replacement)
            continue;

        auto input_shape = pair.first.get_shape();
        auto axis = -1;

        struct OutputToPatrition {
            Output<Node> output;
            int64_t begin;
            int64_t end;
        };

        std::vector<OutputToPatrition> output_to_partition;
        for (size_t i = 0; i < input_shape.size(); ++i) {
            for (const auto& ss_plan : pair.second) {
                if (ss_plan.plan.begins[i] != 0 || ss_plan.plan.ends[i] != static_cast<int64_t>(input_shape[i])) {
                    if (axis == -1 || axis == static_cast<int>(i))
                        axis = static_cast<int>(i);
                    else
                        valid_for_replacement = false;
                    if (ss_plan.plan.strides[i] != 1)
                        valid_for_replacement = false;

                    for (auto& target_input : ss_plan.ptr->output(0).get_target_inputs()) {
                        if (is_type<ov::op::v0::Result>(target_input.get_node())) {
                            valid_for_replacement = false;
                            break;
                        }
                    }
                    output_to_partition.push_back(
                        {ss_plan.ptr->output(0), ss_plan.plan.begins[i], ss_plan.plan.ends[i]});
                }
                if (!valid_for_replacement)
                    break;
            }
            if (!valid_for_replacement)
                break;
        }

        if (!valid_for_replacement || output_to_partition.size() < 2 || axis == -1)
            continue;

        std::sort(output_to_partition.begin(),
                  output_to_partition.end(),
                  [](OutputToPatrition lhs, OutputToPatrition rhs) {
                      return lhs.begin < rhs.begin;
                  });

        std::vector<std::pair<Output<Node>, uint64_t>> output_to_size;
        int64_t prev_r = 0;
        for (auto& record : output_to_partition) {
            valid_for_replacement &= (record.begin >= prev_r);
            prev_r = record.end;
        }
        valid_for_replacement &= (static_cast<size_t>(prev_r) <= input_shape[axis]);
        if (!valid_for_replacement)
            continue;

        prev_r = 0;
        Output<Node> fake_output;
        for (auto& record : output_to_partition) {
            if (record.begin > prev_r)
                output_to_size.emplace_back(fake_output, record.begin - prev_r);
            prev_r = record.end;
            output_to_size.emplace_back(record.output, record.end - record.begin);
        }
        if (static_cast<size_t>(prev_r) < input_shape[axis]) {
            output_to_size.emplace_back(fake_output, input_shape[axis] - prev_r);
        }

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});

        std::vector<int64_t> size_splits;
        for (const auto& item : output_to_size)
            size_splits.push_back(item.second);
        auto size_splits_const =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{size_splits.size()}, size_splits);
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(pair.first, axis_const, size_splits_const);

        auto i = 0;
        NodeVector ops_to_replace;
        for (auto& record : output_to_size) {
            if (record.first != fake_output) {
                record.first.replace(variadic_split->output(i));
                ops_to_replace.push_back(record.first.get_node_shared_ptr());
            }
            ++i;
        }
        copy_runtime_info(ops_to_replace, variadic_split);
    }
    return graph_rewritten;
}

namespace {

struct SliceAttrs {
    int64_t start, stop, axis;
};

struct SliceWithAttrs {
    std::shared_ptr<op::v8::Slice> slice;
    SliceAttrs attrs;
};

bool slice_is_suitable_for_optimization(const std::shared_ptr<ov::op::v8::Slice>& op, SliceAttrs& attrs) {
    const auto& input_shape = op->get_input_partial_shape(0);
    const auto& data_rank = input_shape.rank();
    if (op->get_input_size() != 5 || data_rank.is_dynamic())
        return false;
    const auto rank = data_rank.get_length();

    auto get_scalar = [](const std::shared_ptr<ov::Node>& node, int64_t& value) -> bool {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (!constant)
            return false;
        if (shape_size(constant->get_shape()) != 1)
            return false;
        value = constant->cast_vector<int64_t>()[0];
        return true;
    };

    enum { START = 1, STOP, STRIDE, AXIS };

    int64_t stride;
    if (!get_scalar(op->get_input_node_shared_ptr(STRIDE), stride) || stride != 1)
        return false;
    if (!get_scalar(op->get_input_node_shared_ptr(AXIS), attrs.axis))
        return false;
    attrs.axis = attrs.axis >= 0 ? attrs.axis : attrs.axis + rank;

    if (input_shape[attrs.axis].is_dynamic())
        return false;
    const auto dimension = input_shape[attrs.axis].get_length();

    for (int i = START; i <= STOP; i++) {
        int64_t value;
        if (!get_scalar(op->get_input_node_shared_ptr(i), value))
            return false;
        value = value >= 0 ? value : value + dimension;
        value = std::max<int64_t>(std::min(value, dimension), 0);
        if (i == START)
            attrs.start = value;
        else if (i == STOP)
            attrs.stop = value;
    }

    return true;
}

}  // namespace

bool ov::pass::GroupedSliceToVSplitOptimization::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(GroupedSliceToVSplitOptimization);
    bool graph_rewritten = false;

    using OutputWithAxis = std::pair<ov::Output<ov::Node>, int64_t>;

    std::map<OutputWithAxis, std::vector<SliceWithAttrs>> source_to_op_with_attrs;

    std::vector<OutputWithAxis> ordered_outputs;
    for (const auto& node : model->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        graph_rewritten = ov::op::util::process_subgraph(*this, node) || graph_rewritten;

        if (auto op = ov::as_type_ptr<op::v8::Slice>(node)) {
            SliceAttrs attributes{};
            if (slice_is_suitable_for_optimization(op, attributes)) {
                OutputWithAxis current_output = {op->input_value(0), attributes.axis};
                source_to_op_with_attrs[current_output].push_back({op, attributes});
                if (std::find(ordered_outputs.begin(), ordered_outputs.end(), current_output) == ordered_outputs.end())
                    ordered_outputs.push_back(current_output);
            }
        }
    }
    // optimizing in reverse topological order for case if such VSplit-like Slices are chained
    std::reverse(ordered_outputs.begin(), ordered_outputs.end());
    for (const auto& output_with_axis : ordered_outputs) {
        const auto& output = output_with_axis.first;
        const auto& axis = output_with_axis.second;
        auto attributes = source_to_op_with_attrs[output_with_axis];

        if (attributes.size() < 2)
            continue;

        std::sort(attributes.begin(), attributes.end(), [](const SliceWithAttrs& lhs, const SliceWithAttrs& rhs) {
            if (lhs.attrs.start == rhs.attrs.start)
                return lhs.attrs.stop < rhs.attrs.stop;
            return lhs.attrs.start < rhs.attrs.start;
        });

        const int64_t& dimension = output.get_partial_shape()[axis].get_length();
        int64_t dimension_length_left = dimension;
        std::vector<int64_t> split_lengths;

        int64_t prev_stop = 0;
        bool valid_for_replacement = true;

        // they shouldn't overlap and no holes while slicing
        for (auto& slice_with_attrs : attributes) {
            const auto &start = slice_with_attrs.attrs.start, &stop = slice_with_attrs.attrs.stop;
            if (prev_stop != start) {
                valid_for_replacement = false;
                break;
            }
            int64_t sliced = stop - start;
            split_lengths.push_back((sliced > dimension_length_left ? -1 : sliced));
            dimension_length_left -= sliced;
            prev_stop = stop;
        }
        if (!valid_for_replacement)
            continue;
        if (std::count(split_lengths.begin(), split_lengths.end(), -1) > 1)
            continue;

        int64_t current_sum = 0;
        for (const auto& i : split_lengths)
            if (i != -1)
                current_sum += i;
        for (auto& i : split_lengths)
            if (i == -1) {
                i = dimension - current_sum;
                current_sum = dimension;  // we resolve -1 into actual value since we can use shape data
            }
        if (current_sum != dimension)
            continue;
        auto split_lengths_const =
            op::v0::Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths);
        auto axis_const = op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        auto variadic_split = std::make_shared<op::v1::VariadicSplit>(output, axis_const, split_lengths_const);

        auto i = 0;
        for (auto& slice_with_attrs : attributes) {
            graph_rewritten =
                ov::replace_output_update_name(slice_with_attrs.slice->output(0), variadic_split->output(i)) ||
                graph_rewritten;
            ov::copy_runtime_info(slice_with_attrs.slice, variadic_split);
            ++i;
        }
    }
    return graph_rewritten;
}

ov::pass::SliceSequenceToSingleSlice::SliceSequenceToSingleSlice() {
    MATCHER_SCOPE(SliceSequenceToSingleSlice);
    using namespace ov::op;
    using namespace ov::op::util;
    using namespace ov::pass::pattern;

    auto const_axes_1_pattern = wrap_type<v0::Constant>();
    auto const_axes_2_pattern = wrap_type<v0::Constant>();
    auto slice_1_pattern =
        wrap_type<v8::Slice>({any_input(), any_input(), any_input(), any_input(), const_axes_1_pattern},
                             consumers_count(1));
    auto slice_2_pattern =
        wrap_type<v8::Slice>({slice_1_pattern, any_input(), any_input(), any_input(), const_axes_2_pattern});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();
        auto slice_1 = pattern_to_output.at(slice_1_pattern);
        auto slice_2 = pattern_to_output.at(slice_2_pattern);

        auto const_axes_1 = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(const_axes_1_pattern));
        auto const_axes_2 = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(const_axes_2_pattern));

        auto axes_1_values = const_axes_1->cast_vector<int64_t>();
        auto axes_2_values = const_axes_2->cast_vector<int64_t>();

        // supported a simple scenario when the axes_1 values and axes_2 values don't intersect.
        for (const auto& axis : axes_1_values) {
            if (std::find(axes_2_values.begin(), axes_2_values.end(), axis) != axes_2_values.end()) {
                return false;
            }
        }

        auto begin = std::make_shared<v0::Concat>(OutputVector{slice_1->input_value(1), slice_2->input_value(1)}, 0);
        auto end = std::make_shared<v0::Concat>(OutputVector{slice_1->input_value(2), slice_2->input_value(2)}, 0);
        auto step = std::make_shared<v0::Concat>(OutputVector{slice_1->input_value(3), slice_2->input_value(3)}, 0);
        auto axes = std::make_shared<v0::Concat>(OutputVector{slice_1->input_value(4), slice_2->input_value(4)}, 0);
        auto one_slice = std::make_shared<ov::op::v8::Slice>(slice_1->input_value(0),
                                                             try_fold_unary_output(begin),
                                                             try_fold_unary_output(end),
                                                             try_fold_unary_output(step),
                                                             try_fold_unary_output(axes));

        ov::copy_runtime_info({slice_1, slice_2}, {one_slice, begin, end, step, axes});
        one_slice->set_friendly_name(slice_2->get_friendly_name());
        ov::replace_node(slice_2, one_slice);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(slice_2_pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::StridedSliceOptimization::StridedSliceOptimization(bool use_shapes) {
    m_use_shapes = use_shapes;
}

bool ov::pass::StridedSliceOptimization::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(StridedSliceOptimization);
    ov::pass::Manager manager("StridedSliceOptimization");
    manager.set_per_pass_validation(false);
    if (m_use_shapes) {
        manager.register_pass<UselessSliceEraser>();
        manager.register_pass<SharedOpOptimization>();
        manager.register_pass<GroupedStridedSliceOptimizer>();
        manager.register_pass<GroupedSliceToVSplitOptimization>();
    }

    manager.register_pass<SliceSequenceToSingleSlice>();
    return manager.run_passes(f);
}
