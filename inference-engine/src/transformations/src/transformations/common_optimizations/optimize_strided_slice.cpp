// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::StridedSliceOptimization, "StridedSliceOptimization", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::UselessStridedSliceEraser, "UselessStridedSliceEraser", 0);

bool ngraph::pass::UselessStridedSliceEraser::run_on_function(std::shared_ptr<ngraph::Function> f) {
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
                if (ss_plan.second.begins[i] != 0 || ss_plan.second.ends[i] != input_shape[i]) {
                    if (axis == -1 || axis == i)
                        axis = i;
                    else
                        valid_for_replacement = false;
                    if (ss_plan.second.strides[i] != 1)
                        valid_for_replacement = false;
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
        uint64_t prev_r = 0;
        for (auto & record : output_to_partition) {
            valid_for_replacement &= (record.begin >= prev_r);
            prev_r = record.end;
        }
        valid_for_replacement &= (prev_r <= input_shape[axis]);
        if (!valid_for_replacement) continue;

        prev_r = 0;
        Output<Node> fake_output;
        for (auto & record : output_to_partition) {
            if (record.begin > prev_r)
                output_to_size.emplace_back(fake_output, record.begin - prev_r);
            prev_r = record.end;
            output_to_size.emplace_back(record.output, record.end - record.begin);
        }
        if (prev_r < input_shape[axis]) {
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

