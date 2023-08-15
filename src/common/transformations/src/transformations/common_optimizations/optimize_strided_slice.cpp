// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/pass/manager.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/slice_plan.hpp>
#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <vector>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"

using namespace ov;

bool ov::pass::UselessStridedSliceEraser::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(UselessStridedSliceEraser);
    bool rewritten = false;
    for (auto& node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                rewritten |= run_on_model(sub_graph);
            }
        }
        auto ss = std::dynamic_pointer_cast<ov::op::v1::StridedSlice>(node);
        if (!ss || ss->get_output_partial_shape(0).is_dynamic() || ss->get_input_partial_shape(0).is_dynamic())
            continue;
        if (ss->input(0).get_shape() != ss->output(0).get_shape())
            continue;

        auto stridesNode = std::dynamic_pointer_cast<ov::op::v0::Constant>(ss->input_value(3).get_node_shared_ptr());
        if (stridesNode) {
            auto strides = stridesNode->cast_vector<int64_t>();
            if (!std::any_of(strides.begin(), strides.end(), [](int64_t strd) {
                    return strd < 0;
                }))
                rewritten |= replace_output_update_name(ss->output(0), ss->input_value(0));
        }
    }
    return rewritten;
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace {

ngraph::SlicePlan get_slice_plan(std::shared_ptr<ov::op::v1::StridedSlice> slice) {
    auto convert_mask_to_axis_set = [](const std::vector<int64_t>& mask) {
        ngraph::AxisSet axis_set{};
        for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
            if (mask[i] == 1)
                axis_set.emplace(i);
        }
        return axis_set;
    };

    auto data = slice->input_value(0).get_node_shared_ptr();
    auto begin = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(1).get_node_shared_ptr());
    auto end = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(2).get_node_shared_ptr());
    auto strides = std::dynamic_pointer_cast<ov::op::v0::Constant>(slice->input_value(3).get_node_shared_ptr());
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

bool strided_slices_perform_the_same(std::shared_ptr<ov::op::v1::StridedSlice> lhs,
                                     std::shared_ptr<ov::op::v1::StridedSlice> rhs) {
    auto lhs_plan = get_slice_plan(lhs);
    auto rhs_plan = get_slice_plan(rhs);

    auto empty_plan = ngraph::SlicePlan();
    if (lhs_plan == empty_plan || rhs_plan == empty_plan)
        return false;
    return lhs_plan == rhs_plan;
}

}  // namespace

bool ov::pass::SharedStridedSliceEraser::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(SharedStridedSliceEraser);
    bool graph_rewritten = false;

    std::map<ngraph::Output<Node>, std::vector<std::shared_ptr<ov::op::v1::StridedSlice>>> source_to_ss;
    for (const auto& node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_model(sub_graph);
            }
        }
        if (auto ss = std::dynamic_pointer_cast<ov::op::v1::StridedSlice>(node)) {
            source_to_ss[ss->input_value(0)].push_back(ss);
        }
    }

    for (auto& pair : source_to_ss) {
        if (pair.second.size() < 2)
            continue;
        auto root_ss = pair.second[0];
        for (auto& child_ss : pair.second) {
            if (root_ss->get_instance_id() != child_ss->get_instance_id() &&
                strided_slices_perform_the_same(root_ss, child_ss)) {
                graph_rewritten |= replace_output_update_name(child_ss->output(0), root_ss->output(0));
            }
        }
    }
    return graph_rewritten;
}

bool ov::pass::GroupedStridedSliceOptimizer::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(GroupedStridedSliceOptimizer);
    bool graph_rewritten = false;
    struct planned_slice {
        std::shared_ptr<ov::op::v1::StridedSlice> ptr;
        ngraph::SlicePlan plan;
    };

    std::map<ngraph::Output<Node>, std::vector<planned_slice>> source_to_ss_with_plan;
    for (const auto& node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_model(sub_graph);
            }
        }
        if (auto ss = std::dynamic_pointer_cast<ov::op::v1::StridedSlice>(node)) {
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

        auto axis_const = ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{}, {axis});

        std::vector<int64_t> size_splits;
        for (const auto& item : output_to_size)
            size_splits.push_back(item.second);
        auto size_splits_const =
            ov::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{size_splits.size()}, size_splits);
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

struct SliceAttrs {
    int64_t start, stop, axis;
};

struct SliceWithAttrs {
    std::shared_ptr<op::v8::Slice> slice;
    SliceAttrs attrs;
};

bool slice_is_suitable_for_optimization(const std::shared_ptr<ov::op::v8::Slice>& op, SliceAttrs& attrs) {
    const auto& data_rank = op->get_input_partial_shape(0).rank();
    if (op->get_input_size() != 5 || data_rank.is_dynamic())
        return false;

    for (size_t i = 1; i < 5; ++i) {
        auto input_as_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(i));
        if (!input_as_constant)
            return false;
        if (shape_size(input_as_constant->get_shape()) != 1)
            return false;

        int64_t value = input_as_constant->cast_vector<int64_t>()[0];

        if (((i == 1 || i == 2) && value < 0) || (i == 3 && value != 1))
            return false;
        else if (i == 1)
            attrs.start = value;
        else if (i == 2)
            attrs.stop = value;
        else if (i == 4)
            attrs.axis = value >= 0 ? value : value + data_rank.get_length();
    }
    if (attrs.axis < 0 || op->get_input_partial_shape(0)[attrs.axis].is_dynamic())
        return false;
    return true;
}

bool ov::pass::GroupedSliceToVSplitOptimization::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(GroupedSliceToVSplitOptimization);
    bool graph_rewritten = false;

    using OutputWithAxis = std::pair<ov::Output<ov::Node>, int64_t>;

    std::map<OutputWithAxis, std::vector<SliceWithAttrs>> source_to_op_with_attrs;

    std::vector<OutputWithAxis> ordered_outputs;
    for (const auto& node : model->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto multi_subgraph_op = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(node)) {
            for (const auto& sub_graph : multi_subgraph_op->get_functions()) {
                if (sub_graph)
                    graph_rewritten |= run_on_model(sub_graph);
            }
        }
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
            op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{split_lengths.size()}, split_lengths);
        auto axis_const = op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{}, {axis});
        auto variadic_split = std::make_shared<op::v1::VariadicSplit>(output, axis_const, split_lengths_const);

        auto i = 0;
        for (auto& slice_with_attrs : attributes) {
            graph_rewritten |=
                ov::replace_output_update_name(slice_with_attrs.slice->output(0), variadic_split->output(i));
            ov::copy_runtime_info(slice_with_attrs.slice, variadic_split);
            ++i;
        }
    }
    return graph_rewritten;
}

ov::pass::StridedSliceOptimization::StridedSliceOptimization(bool use_shapes) {
    m_use_shapes = use_shapes;
}

bool ov::pass::StridedSliceOptimization::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(StridedSliceOptimization);

    ov::pass::Manager manager(get_pass_config());
    using namespace ngraph::pass;
    REGISTER_PASS(manager, SliceToStridedSlice, m_use_shapes)
    manager.run_passes(f);

    bool rewritten = false;
    if (m_use_shapes) {
        rewritten = UselessStridedSliceEraser().run_on_model(f);
        // Execution of other passes is also needed even if 'rewritten' is already 'true'
        rewritten = SharedStridedSliceEraser().run_on_model(f) || rewritten;
        rewritten = GroupedStridedSliceOptimizer().run_on_model(f) || rewritten;
        rewritten = GroupedSliceToVSplitOptimization().run_on_model(f) || rewritten;
    }
    return rewritten;
}
