// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "itt.hpp"
#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <ngraph/pass/manager.hpp>
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::StridedSliceOptimization, "StridedSliceOptimization", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::UselessStridedSliceEraser, "UselessStridedSliceEraser", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::SliceToStridedSlice, "SliceToStridedSlice", 0);

using namespace ngraph;

namespace {
    Output<ngraph::Node> adjust_indices_if_needed(const Output<ngraph::Node>& indices,
                                              const std::vector<int64_t>& axes,
                                              uint64_t slice_indices_length,
                                              int64_t fill_in_value) {
    const bool are_axes_sorted = std::is_sorted(axes.begin(), axes.end());

    const auto indices_shape = indices.get_partial_shape();
    // if length of slice indices vector is known
    if (indices_shape.rank().is_static() && indices_shape.rank().get_length() == 1 && indices_shape[0].is_static()) {
        if (static_cast<uint64_t>(indices_shape[0].get_length()) >= slice_indices_length && are_axes_sorted) {
            // adjusting indices is not needed
            return indices;
        }
    }
    // Handle a case when starts/ends/steps lengths are less than provided axes
    // in order to ensure compatibility with `StridedSlice:v1` interface
    // Example:
    // data_shape: {3, 3, 3, 3}
    // starts: [1, 1] - after extending --> [0, 0, 1, 1]
    // ends: [2, 2] - after extending --> [0, 0, 2, 2]
    // steps : [1, 1] - after extending --> [1, 1, 1, 1] (`1` is neutral as a
    // strides value)
    // axes: [2, 3] - apply slice values to 2 and 3 dimension of input data
    // expected_output_shape: {3, 3, 1, 1}
    OutputVector adjusted_indices(slice_indices_length);
    std::vector<int64_t> target_axes(axes);
    const auto gather_axis = opset8::Constant::create(indices.get_element_type(), {}, {0});

    int added_indices_number = 0;
    for (uint64_t i = 0; i < slice_indices_length; ++i) {
        if (std::find(std::begin(axes), std::end(axes), i) == axes.end()) {
            adjusted_indices[i] = opset8::Constant::create(indices.get_element_type(), {1}, {fill_in_value});
            target_axes.insert(std::next(target_axes.begin(), i), i);
            ++added_indices_number;
        } else {
            adjusted_indices[i] = std::make_shared<opset8::Gather>(
                indices,
                opset8::Constant::create(indices.get_element_type(), {1}, {i - added_indices_number}),
                gather_axis);
        }
    }

    if (!are_axes_sorted) {
        OutputVector indices_tmp(adjusted_indices);
        for (size_t i = 0; i < target_axes.size(); ++i) {
            adjusted_indices[target_axes[i]] = indices_tmp[i];
        }
    }

    return std::make_shared<opset8::Concat>(adjusted_indices, 0);
}
std::vector<int64_t> axes_to_mask(const std::vector<int64_t>& axes, uint64_t slice_indices_length) {
    std::vector<int64_t> mask(slice_indices_length, 1);
    for (auto axis : axes) {
        mask[axis] = 0;
    }
    return mask;
}

}  // namespace

ngraph::pass::SliceToStridedSlice::SliceToStridedSlice() {
    MATCHER_SCOPE(SliceToStridedSlice);
    auto slice = pattern::wrap_type<ngraph::opset8::Slice>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto slice_node = std::dynamic_pointer_cast<ngraph::opset8::Slice>(m.get_match_root());
        if (!slice_node)
            return false;

        if (slice_node->get_input_size() < 4)
            return false;

        auto arg = slice_node->input_value(0);

        const auto start_const = get_constant_from_source(slice_node->input_value(1));
        const auto stop_const = get_constant_from_source(slice_node->input_value(2));
        const auto step_const = get_constant_from_source(slice_node->input_value(3));

        const auto& start_input = start_const ? start_const : slice_node->input_value(1);
        const auto& stop_input = stop_const ? stop_const : slice_node->input_value(2);
        const auto& step_input = step_const ? step_const : slice_node->input_value(3);

        std::shared_ptr<opset8::Constant> axes_const;
        if (slice_node->get_input_size() > 4) {
            axes_const = get_constant_from_source(slice_node->input_value(4));
        } else {
            axes_const = slice_node->get_default_const_axes(start_input);
        }

        if (!axes_const)
            return false;

        // auto raw_axes_vec = axes_const->cast_vector<int64_t>();
        // std::vector<uint64_t> axes_vec = get_normalized_axes_vector(node, data_rank, raw_axes_vec);
        auto axes_vec = axes_const->cast_vector<int64_t>();

        const size_t slice_indices_length = *std::max_element(std::begin(axes_vec), std::end(axes_vec)) + 1;
        const auto begin_end_mask = axes_to_mask(axes_vec, slice_indices_length);

        const auto& starts = adjust_indices_if_needed(start_input, axes_vec, slice_indices_length, 0);
        const auto& ends = adjust_indices_if_needed(stop_input, axes_vec, slice_indices_length, 0);
        const auto& steps = adjust_indices_if_needed(step_input, axes_vec, slice_indices_length, 1);

        const auto strided_slice = std::make_shared<opset8::StridedSlice>(arg, starts, ends, steps, begin_end_mask, begin_end_mask);


        strided_slice->set_friendly_name(slice_node->get_friendly_name());
        ngraph::copy_runtime_info(strided_slice, slice_node);
        ngraph::replace_node(slice_node, strided_slice);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(slice, matcher_name);
    register_matcher(m, callback);
}

bool ngraph::pass::UselessStridedSliceEraser::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(UselessStridedSliceEraser);
    bool rewritten = false;
    for (auto & node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                rewritten |= run_on_function(sub_graph);
            }
        }
        auto ss = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice>(node);
        if (!ss || ss->get_output_partial_shape(0).is_dynamic() || ss->get_input_partial_shape(0).is_dynamic())
            continue;
        if (ss->input(0).get_shape() != ss->output(0).get_shape())
            continue;

        auto stridesNode = std::dynamic_pointer_cast<ngraph::opset3::Constant>(ss->input_value(3).get_node_shared_ptr());
        if (stridesNode) {
            auto strides = stridesNode->cast_vector<int64_t>();
            if (!std::any_of(strides.begin(), strides.end(), [](int64_t strd) { return strd < 0;}))
                rewritten |= replace_output_update_name(ss->output(0), ss->input_value(0));
        }
    }
    return rewritten;
}

ngraph::SlicePlan get_slice_plan(std::shared_ptr<ngraph::opset1::StridedSlice> slice) {
    auto convert_mask_to_axis_set = [](const std::vector<int64_t>& mask) {
        ngraph::AxisSet axis_set{};
        for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
            if (mask[i] == 1)
                axis_set.emplace(i);
        }
        return axis_set;
    };

    auto data = slice->input_value(0).get_node_shared_ptr();
    auto begin = std::dynamic_pointer_cast<ngraph::opset1::Constant>(slice->input_value(1).get_node_shared_ptr());
    auto end = std::dynamic_pointer_cast<ngraph::opset1::Constant>(slice->input_value(2).get_node_shared_ptr());
    auto strides = std::dynamic_pointer_cast<ngraph::opset1::Constant>(slice->input_value(3).get_node_shared_ptr());
    if (!begin || !end || !strides || slice->input(0).get_partial_shape().is_dynamic())
        return ngraph::SlicePlan();

    auto begin_vec = begin->cast_vector<int64_t>();
    auto end_vec = end->cast_vector<int64_t>();
    auto strides_vec = strides->cast_vector<int64_t>();
    const auto begin_mask = convert_mask_to_axis_set(slice->get_begin_mask());
    const auto end_mask = convert_mask_to_axis_set(slice->get_end_mask());

    ngraph::SlicePlan plan = ngraph::make_slice_plan(slice->input(0).get_shape(),
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


bool strided_slices_perform_the_same(std::shared_ptr<ngraph::opset1::StridedSlice> lhs,
                                     std::shared_ptr<ngraph::opset1::StridedSlice> rhs) {
    auto lhs_plan = get_slice_plan(lhs);
    auto rhs_plan = get_slice_plan(rhs);

    auto empty_plan = ngraph::SlicePlan();
    if (lhs_plan == empty_plan || rhs_plan == empty_plan)
        return false;
    return lhs_plan == rhs_plan;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::SharedStridedSliceEraser, "SharedStridedSliceEraser", 0);

bool ngraph::pass::SharedStridedSliceEraser::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(SharedStridedSliceEraser);
    bool graph_rewritten = false;

    std::map<ngraph::Output<Node>, std::vector<std::shared_ptr<ngraph::opset1::StridedSlice>>> source_to_ss;
    for (const auto & node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_function(sub_graph);
            }
        }
        if (auto ss = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice>(node)) {
            source_to_ss[ss->input_value(0)].push_back(ss);
        }
    }

    for (auto& pair : source_to_ss) {
        if (pair.second.size() < 2)
            continue;
        auto root_ss = pair.second[0];
        for (auto& child_ss : pair.second) {
            if (root_ss->get_instance_id() != child_ss->get_instance_id() && strided_slices_perform_the_same(root_ss, child_ss)) {
                graph_rewritten |= replace_output_update_name(child_ss->output(0), root_ss->output(0));
            }
        }
    }
    return graph_rewritten;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupedStridedSliceOptimizer, "GroupedStridedSliceOptimizer", 0);

bool ngraph::pass::GroupedStridedSliceOptimizer::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(GroupedStridedSliceOptimizer);
    bool graph_rewritten = false;
    using planned_slice = std::pair<std::shared_ptr<ngraph::opset1::StridedSlice>, ngraph::SlicePlan>;

    std::map<ngraph::Output<Node>, std::vector<planned_slice>> source_to_ss_with_plan;
    for (const auto & node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_function(sub_graph);
            }
        }
        if (auto ss = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice>(node)) {
            auto slice_plan = get_slice_plan(ss);
            if (slice_plan == ngraph::SlicePlan())
                continue;
            source_to_ss_with_plan[ss->input_value(0)].push_back({ss, slice_plan});
        }
    }

    for (auto& pair : source_to_ss_with_plan) {
        if (pair.second.size() < 2)
            continue;

        bool valid_for_replacement = true;

        auto root_plan = pair.second[0].second;
        for (const auto & ss_plan : pair.second) {
            valid_for_replacement &= (ss_plan.second.begins.size() == root_plan.begins.size());
            valid_for_replacement &= (ss_plan.first->get_ellipsis_mask().empty() &&
                                      ss_plan.first->get_new_axis_mask().empty() &&
                                      ss_plan.first->get_shrink_axis_mask().empty());
        }

        if (!valid_for_replacement) continue;

        auto input_shape = pair.first.get_shape();
        auto axis = -1;

        using OutputToPatrition = struct {
            Output<Node> output;
            int64_t begin;
            int64_t end;
        };

        std::vector<OutputToPatrition> output_to_partition;
        for (size_t i = 0; i < input_shape.size(); ++i) {
            for (const auto & ss_plan : pair.second) {
                if (ss_plan.second.begins[i] != 0 || ss_plan.second.ends[i] != static_cast<int64_t>(input_shape[i])) {
                    if (axis == -1 || axis == static_cast<int>(i))
                        axis = static_cast<int>(i);
                    else
                        valid_for_replacement = false;
                    if (ss_plan.second.strides[i] != 1)
                        valid_for_replacement = false;

                    for (auto& target_input : ss_plan.first->output(0).get_target_inputs()) {
                        if (is_type<op::Result>(target_input.get_node())) {
                            valid_for_replacement = false;
                            break;
                        }
                    }
                    output_to_partition.push_back({ss_plan.first->output(0), ss_plan.second.begins[i], ss_plan.second.ends[i]});
                }
                if (!valid_for_replacement) break;
            }
            if (!valid_for_replacement) break;
        }

        if (!valid_for_replacement) continue;
        if (output_to_partition.size() < 2) continue;

        std::sort(output_to_partition.begin(), output_to_partition.end(),
                [](OutputToPatrition lhs, OutputToPatrition rhs)
            {return lhs.begin < rhs.begin;});

        std::vector<std::pair<Output<Node>, uint64_t>> output_to_size;
        int64_t prev_r = 0;
        for (auto & record : output_to_partition) {
            valid_for_replacement &= (record.begin >= prev_r);
            prev_r = record.end;
        }
        valid_for_replacement &= (static_cast<size_t>(prev_r) <= input_shape[axis]);
        if (!valid_for_replacement) continue;

        prev_r = 0;
        Output<Node> fake_output;
        for (auto & record : output_to_partition) {
            if (record.begin > prev_r)
                output_to_size.emplace_back(fake_output, record.begin - prev_r);
            prev_r = record.end;
            output_to_size.emplace_back(record.output, record.end - record.begin);
        }
        if (static_cast<size_t>(prev_r) < input_shape[axis]) {
            output_to_size.emplace_back(fake_output, input_shape[axis] - prev_r);
        }

        auto axis_const = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {axis});

        std::vector<int64_t> size_splits;
        for (const auto & item : output_to_size)
            size_splits.push_back(item.second);
        auto size_splits_const = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{size_splits.size()}, size_splits);
        auto variadic_split = std::make_shared<ngraph::opset1::VariadicSplit>(pair.first, axis_const, size_splits_const);

        auto i = 0;
        NodeVector ops_to_replace;
        for (auto & record : output_to_size) {
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

bool ngraph::pass::StridedSliceOptimization::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(StridedSliceOptimization);
    ngraph::pass::Manager manager(get_pass_config());
    manager.register_pass<ngraph::pass::SliceToStridedSlice>();
    manager.run_passes(f);

    bool rewritten = UselessStridedSliceEraser().run_on_function(f);
    rewritten |= SharedStridedSliceEraser().run_on_function(f);
    rewritten |= GroupedStridedSliceOptimizer().run_on_function(f);
    return rewritten;
}
