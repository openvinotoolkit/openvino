// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dimension_tracking.hpp"

#include <memory>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "dimension_tracker.hpp"
#include "itt.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

void ov::batch_util::mark_with_unique_dimension_labels(const std::shared_ptr<ov::Model>& f,
                                                       const ov::DimensionTracker& dt) {
    ov::label_t i = 1;
    for (auto& parameter : f->get_parameters()) {
        ov::PartialShape new_shape = ov::PartialShape::dynamic(parameter->get_partial_shape().rank());
        for (auto& dim : new_shape)
            dt.set_up_for_tracking(dim, i++);
        parameter->set_partial_shape(new_shape);
    }
}

void ov::batch_util::mark_batch(const std::shared_ptr<ov::opset1::Parameter>& parameter,
                                P2Btype& map,
                                const std::unordered_set<label_t>& batches) {
    auto& shape = parameter->get_partial_shape();
    if (map.count(parameter)) {  // we already marked this parameter as having a batch
        std::unordered_set<ov::label_t> intersection_in_all_three_sources_of_batch;
        auto mapped_batches = map[parameter];
        for (auto& dim : shape) {
            const auto& dim_label = ov::DimensionTracker::get_label(dim);
            if (batches.count(dim_label) && mapped_batches.count(dim_label)) {
                intersection_in_all_three_sources_of_batch.insert(dim_label);
            } else {
                ov::DimensionTracker::reset_tracking_info(dim);
            }
        }
    } else {
        // two cases possible:
        //     1) It is our first time marking batch for this node
        //     2) This node was marked as 'no_batch' previously. 'no_batch' has higher priority, batch won't be set
        for (auto& dim : shape) {
            const auto& dim_label = ov::DimensionTracker::get_label(dim);
            if (batches.count(dim_label)) {  // this is one of the batches
                map[parameter].insert(dim_label);
            } else {
                ov::DimensionTracker::reset_tracking_info(dim);
            }
        }
    }
    parameter->set_partial_shape(shape);
    parameter->validate_and_infer_types();
}

void ov::batch_util::mark_layout_independent_batch(const std::shared_ptr<ov::opset1::Parameter>& parameter,
                                                   const std::shared_ptr<ov::Node>& result,
                                                   P2Btype& map) {
    TensorLabel p_labels, r_labels;

    for (const auto& dim : result->get_output_partial_shape(0))
        if (const auto& label = ov::DimensionTracker::get_label(dim))
            r_labels.push_back(label);
    for (const auto& dim : parameter->get_partial_shape()) {
        if (const auto& label = ov::DimensionTracker::get_label(dim)) {
            if (std::find(r_labels.begin(), r_labels.end(), label) != r_labels.end()) {
                mark_batch(parameter, map, std::unordered_set<label_t>{label});
                return;
            }
        }
    }
    // we should have mark the intersecting batch already, otherwise no intersection == no batch
    mark_no_batch(parameter, map);
}

void ov::batch_util::mark_no_batch(const std::shared_ptr<ov::opset1::Parameter>& parameter, P2Btype& map) {
    if (map.count(parameter))
        map.erase(parameter);
    auto& shape = parameter->get_partial_shape();
    for (auto& dim : shape)
        ov::DimensionTracker::reset_tracking_info(dim);
    parameter->set_partial_shape(shape);
    parameter->validate_and_infer_types();
}

P2Btype ov::batch_util::find_batch(const std::shared_ptr<ov::Model>& f) {
    std::unordered_map<ov::Node::type_info_t, std::pair<size_t, size_t>> type_input_port_batch_index = {
        {ov::opset1::Convolution::get_type_info_static(), {0, 0}},
        {ov::opset1::GroupConvolution::get_type_info_static(), {0, 0}},
        {ov::opset1::ConvolutionBackpropData::get_type_info_static(), {0, 0}},
        {ov::opset1::GroupConvolutionBackpropData::get_type_info_static(), {0, 0}},
        {ov::opset1::DeformableConvolution::get_type_info_static(), {0, 0}},
        {ov::opset1::MatMul::get_type_info_static(), {0, 0}},  // transpose_a situation
    };

    P2Btype parameter_to_batch_labels;

    for (const auto& parameter : f->get_parameters()) {
        auto raw_parameter = parameter.get();
        std::vector<ngraph::Node*> layout_independent_results;

        std::deque<ngraph::Node*> nodes{raw_parameter};
        std::unordered_set<ngraph::Node*> visited;

        while (!nodes.empty()) {
            auto curr_node = nodes.front();
            nodes.pop_front();
            if (visited.count(curr_node))
                continue;
            visited.insert(curr_node);
            if (type_input_port_batch_index.count(curr_node->get_type_info())) {
                auto batch_placement = type_input_port_batch_index[curr_node->get_type_info()];
                const auto& shape = curr_node->input_value(batch_placement.first).get_partial_shape();
                const auto& batch_dim_label = ov::DimensionTracker::get_label(shape[batch_placement.second]);
                if (batch_dim_label == 0)
                    mark_no_batch(parameter, parameter_to_batch_labels);
                else
                    mark_batch(parameter, parameter_to_batch_labels, {batch_dim_label});
                continue;  // batch was or was not found at this point -- there is no point in searching further }
            }
            // node is not layout obvious -- checking if dims were propagated through
            bool all_outputs_labeled = true;
            for (const auto& output : curr_node->outputs()) {
                const auto& output_shape = output.get_partial_shape();
                bool name_stays = std::any_of(output_shape.cbegin(), output_shape.cend(), [](const Dimension& d) {
                    return ov::DimensionTracker::get_label(d) != 0;
                });
                all_outputs_labeled &= name_stays;
            }

            if (!all_outputs_labeled) {
                mark_no_batch(parameter, parameter_to_batch_labels);
                continue;  // label propagation stopped
            }

            if (ov::is_type<ov::opset1::Result>(curr_node))
                layout_independent_results.push_back(curr_node);

            for (const auto& output : curr_node->outputs()) {
                // we do not need to walk through shape-of sub-graphs
                for (const auto& t_input : output.get_target_inputs()) {
                    if (ov::is_type<ov::opset1::ConvertLike>(t_input.get_node()) ||
                        ov::is_type<ov::opset1::ShapeOf>(t_input.get_node()) ||
                        ov::is_type<ov::opset3::ShapeOf>(t_input.get_node()))
                        continue;
                    nodes.push_back(t_input.get_node());
                }
            }
        }

        for (auto& result : layout_independent_results)
            // there are no layout obvious operations on the Parameter-Result path
            // considering the outer-most matching dimension is batch
            mark_layout_independent_batch(parameter, result->shared_from_this(), parameter_to_batch_labels);
    }
    return parameter_to_batch_labels;
}

void ov::batch_util::restore_original_dimensions(
    const std::map<std::shared_ptr<ov::opset1::Parameter>, ov::PartialShape>& parameter_to_shape,
    bool leave_batch_dynamic) {
    for (const auto& item : parameter_to_shape) {
        const auto& batch_marked_shape = item.first->get_partial_shape();
        auto original_shape = item.second;
        OPENVINO_ASSERT(batch_marked_shape.rank().is_static() && original_shape.rank().is_static());
        OPENVINO_ASSERT(batch_marked_shape.size() == original_shape.size());

        for (size_t n = 0; n < batch_marked_shape.size(); ++n) {
            if (const auto& label = ov::DimensionTracker::get_label(batch_marked_shape[n])) {
                if (leave_batch_dynamic)
                    original_shape[n] = Dimension::dynamic();
                ov::DimensionTracker::set_label(original_shape[n], label);
            }
        }
        item.first->set_partial_shape(original_shape);
    }
}

bool ov::batch_util::check_batch_tracks_through_all_the_nodes(const std::shared_ptr<ov::Model>& f) {
    bool failed_to_propagate_batch = false;
    for (const auto& node : f->get_ordered_ops()) {
        bool any_input_has_batch = false;
        for (const auto& input : node->input_values()) {
            const auto& input_shape = input.get_partial_shape();
            bool name_stays = false;
            bool others_are_static = true;
            for (const auto& dim : input_shape)
                if (ov::DimensionTracker::get_label(dim) == 0)
                    others_are_static = others_are_static && dim.is_static();
                else
                    name_stays = true;
            any_input_has_batch |= name_stays && others_are_static;
        }
        bool all_outputs_has_batch = true;
        for (const auto& output : node->outputs()) {
            const auto& output_shape = output.get_partial_shape();
            bool name_stays = false;
            bool others_are_static = true;
            for (const auto& dim : output_shape)
                if (ov::DimensionTracker::get_label(dim) == 0)
                    others_are_static = others_are_static && dim.is_static();
                else
                    name_stays = true;
            all_outputs_has_batch &= name_stays;  // && others_are_static;
        }
        if (any_input_has_batch && !all_outputs_has_batch && !ov::is_type<ov::opset3::ShapeOf>(node) &&
            !ov::is_type<ov::opset1::ShapeOf>(node) && !ov::is_type<ov::opset1::ConvertLike>(node)) {
            failed_to_propagate_batch = true;
            node->validate_and_infer_types();
        }
    }
    const auto& results = f->get_results();
    for (const auto& result : results) {
        const auto& input_shape = result->get_input_partial_shape(0);
        bool name_stays = std::any_of(input_shape.cbegin(), input_shape.cend(), [](const ov::Dimension& d) {
            return ov::DimensionTracker::get_label(d);
        });
        failed_to_propagate_batch |= !name_stays;
    }
    return failed_to_propagate_batch;
}

bool ov::batch_util::detach_detection_output(const std::shared_ptr<ov::Model>& f) {
    ResultVector new_outputs, outputs_to_delete;
    for (auto& result_node : f->get_results()) {
        auto do_node = result_node->input_value(0).get_node_shared_ptr();
        if (ov::is_type<opset1::Convert>(do_node))  // cases with do->convert->result
            do_node = do_node->get_input_node_shared_ptr(0);
        if (ov::is_type<opset1::DetectionOutput>(do_node) || ov::is_type<opset8::DetectionOutput>(do_node)) {
            for (auto& new_result_src : do_node->input_values()) {
                auto new_result = std::make_shared<opset1::Result>(new_result_src);
                ngraph::copy_runtime_info(result_node, new_result);
                new_outputs.push_back(new_result);
            }
            outputs_to_delete.push_back(result_node);
        }
    }
    for (auto& result : outputs_to_delete)
        f->remove_result(result);
    f->add_results(new_outputs);
    return !new_outputs.empty() || !outputs_to_delete.empty();
}

bool ov::pass::FindBatch::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(FindBatch);
    auto te = std::make_shared<ov::TableOfEquivalence>();
    ov::DimensionTracker dt(te);

    bool model_has_changed = false;
    if (detach_do)
        model_has_changed |= batch_util::detach_detection_output(m);

    const auto& parameters = m->get_parameters();
    std::map<std::shared_ptr<ov::opset1::Parameter>, PartialShape> parameter_to_shape;
    for (const auto& parameter : parameters) {
        auto shape = parameter->get_partial_shape();
        if (shape.rank().is_dynamic())
            return model_has_changed;
        parameter_to_shape[parameter] = shape;
    }

    ov::batch_util::mark_with_unique_dimension_labels(m, dt);
    m->validate_nodes_and_infer_types();

    ov::batch_util::find_batch(m);

    if (!track) {
        ov::batch_util::restore_original_dimensions(parameter_to_shape, false);
        m->validate_nodes_and_infer_types();
        return true;
    }

    ov::batch_util::restore_original_dimensions(parameter_to_shape);

    m->validate_nodes_and_infer_types();

    bool failed_to_propagate_batch = ov::batch_util::check_batch_tracks_through_all_the_nodes(m);

    if (failed_to_propagate_batch) {  // restore original input shape with labels
        for (const auto& item : parameter_to_shape)
            item.first->set_partial_shape(item.second);
    } else {  // restore original input shape with batch labels
        ov::batch_util::restore_original_dimensions(parameter_to_shape, false);
    }
    m->validate_nodes_and_infer_types();
    return true;
}

// count all the non-constant dimensions (what we need to validate and infer)
// + count how many nodes we need to validate and infer to collect all the necessary shape information the old way
//
// collect all constant dimensions from Parameter (if there are any) and Constant shapes (label them)
// perform validate_and_infer runs for all the nodes including body-graphs
// count all the unknown dimension that will be needed for calculation (no duplicates allowed)
// count the number of Operation that would be needed to be shape-inference-ed to get all the dimensions
//

void shape_work(ov::DimensionTracker& dt,
                ov::PartialShape& shape) {
    if (shape.rank().is_dynamic())
        return;
    for (auto& d : shape) {
        if (d.is_static() || ov::DimensionTracker::has_label(d))
            continue;
        dt.set_up_for_tracking(d);
    }
}

bool ov::pass::SymbolicPOC::run_on_model(const std::shared_ptr<ov::Model> &m) {
//    size_t curr_label = 1; // must be a field in this class to pass to the body
    auto te = std::make_shared<ov::TableOfEquivalence>();
    ov::DimensionTracker dt(te);

    size_t num_ops_to_shape_infer = 0;

    for (const auto& op : m->get_ordered_ops()) {
        bool shape_infer_op = false;
        op->invalidate_values();
        for (auto &output: op->outputs()) {
            output.get_rt_info()["SKIP_INVALIDATION"] = true;
            output.get_rt_info()["TABLE_OF_EQUIVALENCE"] = te;
        }
        op->validate_and_infer_types();
        for (auto& output : op->outputs()) {
            auto shape = output.get_partial_shape();
            shape_work(dt, shape);
            OPENVINO_SUPPRESS_DEPRECATED_START
            output.get_tensor().set_tensor_type(output.get_element_type(), shape);
            OPENVINO_SUPPRESS_DEPRECATED_END
            if (shape.is_dynamic())
                shape_infer_op = true;
        }
        num_ops_to_shape_infer += size_t(shape_infer_op);
    }
    std::cout << "Overall num ops: " << m->get_ordered_ops().size() << std::endl;
    std::cout << "num_ops_to_shape_infer = " << num_ops_to_shape_infer << std::endl;

    std::unordered_set<size_t> known_labels;
    size_t new_num_ops_to_shape_infer = 0;
    for (const auto& op : m->get_ordered_ops()) {
        bool shape_infer_op = false;
        for (auto& output : op->outputs()) {
//            if (output.get_rt_info().count("SKIP_INVALIDATION"))
//                output.get_rt_info().erase("SKIP_INVALIDATION");
            for (const auto& dim : output.get_partial_shape()) {
                if (dim.is_static())
                    continue;
                auto label = ov::DimensionTracker::get_label(dim);
                if (!known_labels.count(label)) {
                    // incorporate table of equivalence knowledge? maybe not
                    known_labels.insert(label);
                    if (!ov::is_type<ov::op::v0::Parameter>(op) && !ov::is_type<ov::op::v0::Constant>(op))
                        shape_infer_op = true;
                }
            }
        }
        new_num_ops_to_shape_infer += size_t(shape_infer_op);
    }
    std::cout << "new_num_ops_to_shape_infer = " << new_num_ops_to_shape_infer << std::endl;
    std::cout << "Percent: " << size_t(float(new_num_ops_to_shape_infer) / float(num_ops_to_shape_infer) * 100) << "%" << std::endl;
    return false;
}


ov::pass::ChainedMaximumOptimization::ChainedMaximumOptimization() {
    MATCHER_SCOPE(ChainedMaximumOptimization);
    auto first_input = pattern::any_input();
    auto second_input = pattern::any_input();
    auto third_input = pattern::any_input();
    auto upper_max_label = pattern::wrap_type<opset1::Maximum>({first_input, second_input});
    auto lower_max_label = pattern::wrap_type<opset1::Maximum>({upper_max_label, third_input});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto first_labels = pattern_to_output.at(first_input).get_tensor().get_value_label();
        auto second_labels = pattern_to_output.at(second_input).get_tensor().get_value_label();
        auto third_labels = pattern_to_output.at(third_input).get_tensor().get_value_label();

        auto valid_labels = [](const ov::TensorLabel& labels) {
            return !labels.empty() && std::all_of(labels.begin(), labels.end(), [](const label_t& l){ return l != 0;});
        };
        bool replaced = false;
        auto intermidiate = pattern_to_output.at(upper_max_label);
        if (valid_labels(first_labels) && valid_labels(third_labels) && first_labels == third_labels) {
            // Maximum(second_input, third_input)
            intermidiate.replace(pattern_to_output.at(second_input));
            replaced = true;
        } else if (valid_labels(second_labels) && valid_labels(third_labels) && second_labels == third_labels) {
            // Maximum(first_input, third_input)
            intermidiate.replace(pattern_to_output.at(first_input));
            replaced = true;
        }
        return replaced;
    };

    auto m = std::make_shared<pattern::Matcher>(lower_max_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::NopBroadcast::NopBroadcast() {
    MATCHER_SCOPE(NopBroadcast);
    auto input_label = pattern::any_input(pattern::has_static_rank());
    auto shape_of = pattern::wrap_type<opset1::ShapeOf, opset3::ShapeOf>();
    auto ones = pattern::wrap_type<opset1::Constant>();
    auto maximum = pattern::wrap_type<opset1::Maximum>({shape_of, ones});
    auto broadcast = pattern::wrap_type<opset1::Broadcast, opset3::Broadcast>({input_label, maximum});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto constant = ov::as_type_ptr<opset1::Constant>(pattern_to_output.at(ones).get_node_shared_ptr());
        if (!constant)
            return false;
        auto valid_labels = [](const ov::TensorLabel& labels) {
            return !labels.empty() && std::all_of(labels.begin(), labels.end(), [](const label_t& l){ return l != 0;});
        };
        auto shape_of_labels = pattern_to_output.at(shape_of).get_tensor().get_value_label();
        if (!valid_labels(shape_of_labels))
            return false;
        auto input = pattern_to_output.at(input_label);
        ov::TensorLabel input_labels;
        for (const auto& dim : input.get_partial_shape()) {
            if (dim.get_max_length() == 0)
                return false;
            input_labels.push_back(ov::DimensionTracker::get_label(dim));
        }
        if (!valid_labels(input_labels))
            return false;
        auto constant_content = constant->cast_vector<int64_t>();
        bool all_ones = std::all_of(constant_content.begin(), constant_content.end(), [](const int64_t& i){ return i == 1; });
        if (constant_content.size() > input.get_partial_shape().size() || !all_ones)
            return false;
        auto output = pattern_to_output.at(broadcast);
        output.replace(input);
        std::cout << "BC replaces" << std::endl;
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(broadcast, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

bool labels_eq_or_eq_static_dims(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    auto lhs_label = ov::DimensionTracker::get_label(lhs);
    auto rhs_label = ov::DimensionTracker::get_label(rhs);
    bool labels_exist_and_equal = lhs_label != 0 && lhs_label == rhs_label;
    bool dims_are_static_and_equal = lhs.is_static() && lhs == rhs;
    return labels_exist_and_equal || dims_are_static_and_equal;
}

bool reshape_keeps_last_two_dims(const std::shared_ptr<ov::Node>& op) {
    const auto& before = op->get_input_partial_shape(0);
    const auto& after = op->get_output_partial_shape(0);
    if (before.rank().is_dynamic() || before.size() < 2)
        return false;
    if (after.rank().is_dynamic() || after.size() < 2)
        return false;
    for (size_t i = 2; i > 0 ; --i)
        if (!labels_eq_or_eq_static_dims(before[before.size() - i], after[after.size() - i]))
            return false;
    return true;
}

bool batches_are_equal(const std::shared_ptr<ov::Node>& op_0, const std::shared_ptr<ov::Node>& op_1) {
    auto input_0 = op_0->get_input_partial_shape(0);
    auto input_1 = op_1->get_input_partial_shape(0);
    for (size_t i = 0; i < input_0.size() - 2; ++i) // we are sure of its rank
        if (!labels_eq_or_eq_static_dims(input_0[i], input_1[i]))
            return false;
    auto output_0 = op_0->get_output_partial_shape(0);
    auto output_1 = op_1->get_output_partial_shape(0);
    for (size_t i = 0; i < output_0.size() - 2; ++i) // we are sure of its rank
        if (!labels_eq_or_eq_static_dims(output_0[i], output_1[i]))
            return false;
    return true;
}

bool pass_labels_through(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& output) {
    const auto &in_shape = input.get_partial_shape(), &out_shape = output.get_partial_shape();
    if (in_shape.rank().is_dynamic() || out_shape.rank().is_dynamic())
        return false;
    if (in_shape.size() != out_shape.size())
        return false;
    for (size_t i = 0; i < in_shape.size(); ++i)
        if (!labels_eq_or_eq_static_dims(in_shape[i], out_shape[i]))
            return false;
    return true;
}

bool output_has_single_target_input_and_its_bea_node_has_only_one_output_on_zero_port(const ov::Output<ov::Node>& output) {
    auto target_inputs = output.get_target_inputs();
    if (target_inputs.size() != 1)
        return false;
    auto node = target_inputs.begin()->get_node();
    if (node->get_output_size() != 1)
        return false;
    if (node->outputs()[0].get_index() != 0)
        return false;
    if (!ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node))
        return false;
    return true;
}


ov::pass::DeReshapeMatMul::DeReshapeMatMul() {
    MATCHER_SCOPE(DeReshapeMatMul);
    auto reshape_0 = pattern::wrap_type<opset1::Reshape>(pattern::has_static_rank());
    auto reshape_1 = pattern::wrap_type<opset1::Reshape>(pattern::has_static_rank());
    auto matmul = pattern::wrap_type<opset1::MatMul>({reshape_0, reshape_1});
5
    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        // find bottom Reshape check that output "batch" dims are equal to pre-first Reshape ones
        // and check that two last dims are same

        auto reshape_0_node = pattern_to_output.at(reshape_0).get_node_shared_ptr();
        auto reshape_1_node = pattern_to_output.at(reshape_1).get_node_shared_ptr();
        auto matmul_node = pattern_to_output.at(matmul).get_node_shared_ptr();
        std::cout << "MM matched" << std::endl;
        if (reshape_0_node->get_input_partial_shape(0).rank().is_dynamic() || reshape_0_node->get_output_partial_shape(0).rank().is_dynamic())
            return false;
        if (reshape_1_node->get_input_partial_shape(0).rank().is_dynamic() || reshape_1_node->get_output_partial_shape(0).rank().is_dynamic())
            return false;
        if (reshape_0_node->get_input_partial_shape(0).size() != reshape_1_node->get_input_partial_shape(0).size())
            return false;
        if (reshape_0_node->get_output_partial_shape(0).size() != reshape_1_node->get_output_partial_shape(0).size())
            return false;
        if (!reshape_keeps_last_two_dims(reshape_0_node) || !reshape_keeps_last_two_dims(reshape_1_node))
            return false;
        if (!batches_are_equal(reshape_0_node, reshape_1_node))
            return false;
        // proved MatMul could have been executed on the non-Reshaped input tensors

        std::vector<Node*> nodes_for_revalidation{matmul_node.get()};
        Output<Node> output = matmul_node->output(0);
        while (output_has_single_target_input_and_its_bea_node_has_only_one_output_on_zero_port(output)) {
            auto next_output = output.get_target_inputs().begin()->get_node()->output(0);
            if (!pass_labels_through(output, next_output))
                break;
            output = next_output;
            nodes_for_revalidation.push_back(output.get_node());
        }
        // to reduce number of Reshapes -- searching for Reshape on the output of the MatMul skipping nodes which don't influence output
        std::cout << output.get_node_shared_ptr() << std::endl;
        if (output.get_target_inputs().size() == 1)
            return false;
        auto reshape_output = ov::as_type<opset1::Reshape>(output.get_target_inputs().begin()->get_node());
        if (!reshape_output)
            return false; // we didn't find Reshape back on the output of the MatMul

        matmul_node->input(0).replace_source_output(reshape_0_node->input_value(0));
        matmul_node->input(1).replace_source_output(reshape_1_node->input_value(0));
        reshape_output->output(0).replace(reshape_output->input_value(0));
        for (auto& node : nodes_for_revalidation)
            node->validate_and_infer_types();
        std::cout << "MM replaced" << std::endl;
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(matmul, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

bool ov::pass::SymbolicOptimizations::run_on_model(const std::shared_ptr<ov::Model> &m) {
    RUN_ON_FUNCTION_SCOPE(SymbolicOptimizations);
    ov::pass::Manager manager(get_pass_config());
    manager.set_per_pass_validation(false);
    // label everything
    REGISTER_PASS(manager, SymbolicPOC)

    auto optimizations_0 = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(optimizations_0, ChainedMaximumOptimization)
    ADD_MATCHER(optimizations_0, NopBroadcast)
    optimizations_0->set_name("ov::pass::GraphRewrite::SymbolicOptimizations::0");
    auto optimizations_1 = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(optimizations_1, DeReshapeMatMul)
    optimizations_1->set_name("ov::pass::GraphRewrite::SymbolicOptimizations::1");
    // cleanup labels, erase SKIP_INVALIDATION
    REGISTER_PASS(manager, VisualizeTree, "model.svg")
    return manager.run_passes(m);
}
