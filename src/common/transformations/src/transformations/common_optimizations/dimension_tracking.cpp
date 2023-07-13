// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dimension_tracking.hpp"

#include <memory>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "ov_ops/fused_MHA_with_RPE.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/transpose_sinking/ts_slice.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"

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

void ov::batch_util::mark_batch(const std::shared_ptr<ov::op::v0::Parameter>& parameter,
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

void ov::batch_util::mark_layout_independent_batch(const std::shared_ptr<ov::op::v0::Parameter>& parameter,
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

void ov::batch_util::mark_no_batch(const std::shared_ptr<ov::op::v0::Parameter>& parameter, P2Btype& map) {
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
        {ov::op::v1::Convolution::get_type_info_static(), {0, 0}},
        {ov::op::v1::GroupConvolution::get_type_info_static(), {0, 0}},
        {ov::op::v1::ConvolutionBackpropData::get_type_info_static(), {0, 0}},
        {ov::op::v1::GroupConvolutionBackpropData::get_type_info_static(), {0, 0}},
        {ov::op::v1::DeformableConvolution::get_type_info_static(), {0, 0}},
        {ov::op::v0::MatMul::get_type_info_static(), {0, 0}},  // transpose_a situation
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

            if (ov::is_type<ov::op::v0::Result>(curr_node))
                layout_independent_results.push_back(curr_node);

            for (const auto& output : curr_node->outputs()) {
                // we do not need to walk through shape-of sub-graphs
                for (const auto& t_input : output.get_target_inputs()) {
                    if (ov::is_type<ov::op::v1::ConvertLike>(t_input.get_node()) ||
                        ov::is_type<ov::op::v0::ShapeOf>(t_input.get_node()) ||
                        ov::is_type<ov::op::v3::ShapeOf>(t_input.get_node()))
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
    const std::map<std::shared_ptr<ov::op::v0::Parameter>, ov::PartialShape>& parameter_to_shape,
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
        if (any_input_has_batch && !all_outputs_has_batch && !ov::is_type<ov::op::v3::ShapeOf>(node) &&
            !ov::is_type<ov::op::v0::ShapeOf>(node) && !ov::is_type<ov::op::v1::ConvertLike>(node)) {
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
        if (ov::is_type<ov::op::v0::Convert>(do_node))  // cases with do->convert->result
            do_node = do_node->get_input_node_shared_ptr(0);
        if (ov::is_type<ov::op::v0::DetectionOutput>(do_node) || ov::is_type<ov::op::v8::DetectionOutput>(do_node)) {
            for (auto& new_result_src : do_node->input_values()) {
                auto new_result = std::make_shared<ov::op::v0::Result>(new_result_src);
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
    std::map<std::shared_ptr<ov::op::v0::Parameter>, PartialShape> parameter_to_shape;
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

void shape_work(ov::DimensionTracker& dt, ov::PartialShape& shape) {
    if (shape.rank().is_dynamic())
        return;
    for (auto& d : shape) {
        bool is_static = d.is_static(), has_label = ov::DimensionTracker::has_label(d);
        if (is_static && has_label)
            dt.reset_tracking_info(d);  // remove labels from static dims on shapes to reduce visual clutter
        if (is_static || has_label)
            continue;
        dt.set_up_for_tracking(d);
    }
}

void special_case_range_label_propagation(const std::shared_ptr<ov::Node>& node) {
    /* Label propagation through specific Range operation
          start    shift
            |  \   /
            |   Add   step == 1
            \    /    /
               Range
    */
    if (!ov::is_type<ov::op::v0::Range>(node) && !ov::is_type<ov::op::v4::Range>(node))
        return;

    auto output_shape = node->get_output_partial_shape(0);
    if (output_shape.rank().is_dynamic() || output_shape.size() != 1)
        return;

    OPENVINO_SUPPRESS_DEPRECATED_START
    auto step_value = ov::get_constant_from_source(node->input_value(2));
    OPENVINO_SUPPRESS_DEPRECATED_END
    if (!step_value || step_value->cast_vector<int64_t>()[0] != 1)
        return;

    auto start_labels = node->get_input_tensor(0).get_value_label();
    if (start_labels.size() != 1 || start_labels[0] == ov::no_label)
        return;
    auto start_label = start_labels[0];

    auto stop_node = node->input_value(1).get_node_shared_ptr();
    if (!ov::is_type<ov::op::v1::Add>(stop_node))
        return;
    auto add_in0_labels = stop_node->get_input_tensor(0).get_value_label();
    if (add_in0_labels.size() != 1 || add_in0_labels[0] == ov::no_label)
        return;
    auto add_in0_label = add_in0_labels[0];

    auto add_in1_labels = stop_node->get_input_tensor(1).get_value_label();
    if (add_in1_labels.size() != 1 || add_in1_labels[0] == ov::no_label)
        return;
    auto add_in1_label = add_in1_labels[0];

    if (add_in0_label == start_label)
        ov::DimensionTracker::set_label(output_shape[0], add_in1_label);
    else if (add_in1_label == start_label)
        ov::DimensionTracker::set_label(output_shape[0], add_in0_label);
    node->set_output_type(0, node->get_output_element_type(0), output_shape);
}

bool ov::pass::SymbolicPOC::run_on_model(const std::shared_ptr<ov::Model>& m) {
    //    size_t curr_label = 1; // must be a field in this class to pass to the body
    auto te = std::make_shared<ov::TableOfEquivalence>();
    ov::DimensionTracker dt(te);

    size_t num_ops_to_shape_infer = 0;

    m->get_rt_info()["TABLE_OF_EQUIVALENCE"] = te;

    for (const auto& op : m->get_ordered_ops()) {
        bool shape_infer_op = false;
        op->invalidate_values();
        for (auto& output : op->outputs()) {
            output.get_rt_info()["SKIP_INVALIDATION"] = true;
            output.get_rt_info()["TABLE_OF_EQUIVALENCE"] = te;
        }
        op->validate_and_infer_types();
        // additional rules must be triggered here
        special_case_range_label_propagation(op);
        //
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
    //    std::cout << "Overall num ops: " << m->get_ordered_ops().size() << std::endl;
    //    std::cout << "num_ops_to_shape_infer = " << num_ops_to_shape_infer << std::endl;

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
    //    std::cout << "new_num_ops_to_shape_infer = " << new_num_ops_to_shape_infer << std::endl;
    //    std::cout << "Percent: " << size_t(float(new_num_ops_to_shape_infer) / float(num_ops_to_shape_infer) * 100) <<
    //    "%"
    //              << std::endl;
    return false;
}

ov::pass::ChainedMaximumOptimization::ChainedMaximumOptimization() {
    MATCHER_SCOPE(ChainedMaximumOptimization);
    auto first_input = pattern::any_input();
    auto second_input = pattern::any_input();
    auto third_input = pattern::any_input();
    auto upper_max_label = pattern::wrap_type<op::v1::Maximum>({first_input, second_input});
    auto lower_max_label = pattern::wrap_type<op::v1::Maximum>({upper_max_label, third_input});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto first_labels = pattern_to_output.at(first_input).get_tensor().get_value_label();
        auto second_labels = pattern_to_output.at(second_input).get_tensor().get_value_label();
        auto third_labels = pattern_to_output.at(third_input).get_tensor().get_value_label();

        auto valid_labels = [](const ov::TensorLabel& labels) {
            return !labels.empty() && std::all_of(labels.begin(), labels.end(), [](const label_t& l) {
                return l != 0;
            });
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
    auto shape_of = pattern::wrap_type<op::v0::ShapeOf, op::v3::ShapeOf>();
    auto ones = pattern::wrap_type<op::v0::Constant>();
    auto maximum = pattern::wrap_type<op::v1::Maximum>({shape_of, ones});
    auto broadcast = pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>({input_label, maximum});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto constant = ov::as_type_ptr<op::v0::Constant>(pattern_to_output.at(ones).get_node_shared_ptr());
        if (!constant)
            return false;
        auto valid_labels = [](const ov::TensorLabel& labels) {
            return !labels.empty() && std::all_of(labels.begin(), labels.end(), [](const label_t& l) {
                return l != 0;
            });
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
        bool all_ones = std::all_of(constant_content.begin(), constant_content.end(), [](const int64_t& i) {
            return i == 1;
        });
        if (constant_content.size() > input.get_partial_shape().size() || !all_ones)
            return false;
        auto output = pattern_to_output.at(broadcast);
        output.replace(input);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(broadcast, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::BroadcastOnes::BroadcastOnes() {
    MATCHER_SCOPE(BroadcastOnes);
    auto input = pattern::any_input(pattern::has_static_rank());
    auto ones = pattern::any_input();
    auto broadcast = pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast, op::v0::Tile>({input, ones},
                                                                                            pattern::has_static_rank());

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& broadcast = m.get_match_root();
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto constant_output_shape = ov::get_constant_from_source(broadcast->input_value(1));
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!constant_output_shape)
            return false;
        auto data = constant_output_shape->cast_vector<int64_t>();
        if (!std::all_of(begin(data), end(data), [](const int64_t& i) {
                return i == 1;
            }))
            return false;
        const auto& input_rank = broadcast->get_input_partial_shape(0).size();
        const auto& output_rank = broadcast->get_output_partial_shape(0).size();
        if (input_rank != output_rank)
            return false;
        broadcast->output(0).replace(broadcast->input_value(0));
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
    for (size_t i = 2; i > 0; --i)
        if (!labels_eq_or_eq_static_dims(before[before.size() - i], after[after.size() - i]))
            return false;
    return true;
}

bool batches_are_equal(const std::shared_ptr<ov::Node>& op_0, const std::shared_ptr<ov::Node>& op_1) {
    auto input_0 = op_0->get_input_partial_shape(0);
    auto input_1 = op_1->get_input_partial_shape(0);
    for (size_t i = 0; i < input_0.size() - 2; ++i)  // we are sure of its rank
        if (!labels_eq_or_eq_static_dims(input_0[i], input_1[i]))
            return false;
    auto output_0 = op_0->get_output_partial_shape(0);
    auto output_1 = op_1->get_output_partial_shape(0);
    for (size_t i = 0; i < output_0.size() - 2; ++i)  // we are sure of its rank
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

bool output_has_single_target_input_and_its_bea_node_has_only_one_output_on_zero_port(
    const ov::Output<ov::Node>& output) {
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

bool are_equal_int_constants(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
    if (lhs == rhs)
        return true;
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto lhs_constant = ov::get_constant_from_source(lhs);
    auto rhs_constant = ov::get_constant_from_source(rhs);
    OPENVINO_SUPPRESS_DEPRECATED_END
    if (!lhs_constant || !rhs_constant)
        return false;
    return lhs_constant->template cast_vector<int64_t>() == rhs_constant->template cast_vector<int64_t>();
}

ov::pass::DeReshapeMatMul::DeReshapeMatMul() {
    MATCHER_SCOPE(DeReshapeMatMul);

    auto reshape_0 = pattern::wrap_type<op::v1::Reshape>(pattern::has_static_rank());
    auto bea_0 = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({reshape_0, pattern::any_input()});
    auto or_0 = std::make_shared<pattern::op::Or>(OutputVector{reshape_0, bea_0});
    // FIXME: put all checks in the pattern of reshape and bea
    auto reshape_1 = pattern::wrap_type<op::v1::Reshape>(pattern::has_static_rank());
    auto bea_1 = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({reshape_1, pattern::any_input()});
    auto or_1 = std::make_shared<pattern::op::Or>(OutputVector{reshape_1, bea_1});
    // FIXME: put all checks in the pattern of reshape and bea
    auto matmul = pattern::wrap_type<op::v0::MatMul>({or_0, or_1});

    auto reshape_2 = pattern::wrap_type<op::v1::Reshape>({matmul, pattern::any_input()}, pattern::has_static_rank());

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        // find bottom Reshape check that output "batch" dims are equal to pre-first Reshape ones
        // and check that two last dims are same

        auto reshape_0_node = pattern_to_output.at(reshape_0).get_node_shared_ptr();
        auto reshape_1_node = pattern_to_output.at(reshape_1).get_node_shared_ptr();
        auto matmul_node = pattern_to_output.at(matmul).get_node_shared_ptr();
        if (reshape_0_node->get_input_partial_shape(0).rank().is_dynamic() ||
            reshape_0_node->get_output_partial_shape(0).rank().is_dynamic())
            return false;
        if (reshape_1_node->get_input_partial_shape(0).rank().is_dynamic() ||
            reshape_1_node->get_output_partial_shape(0).rank().is_dynamic())
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
        // to reduce number of Reshapes -- searching for Reshape on the output of the MatMul skipping nodes which don't
        // influence output
        if (output.get_target_inputs().size() != 1)
            return false;
        auto reshape_output = ov::as_type<op::v1::Reshape>(output.get_target_inputs().begin()->get_node());
        if (!reshape_output)
            return false;  // we didn't find Reshape back on the output of the MatMul

        reshape_0_node->output(0).replace(reshape_0_node->input_value(0));
        reshape_1_node->output(0).replace(reshape_1_node->input_value(0));
        reshape_output->output(0).replace(reshape_output->input_value(0));
        for (auto& node : nodes_for_revalidation)
            node->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_2, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::RemoveSliceBeforeGatherElements::RemoveSliceBeforeGatherElements() {
    MATCHER_SCOPE(RemoveSliceBeforeGatherElements);

    auto slice = pattern::wrap_type<op::v8::Slice>();
    auto gather = pattern::wrap_type<op::v6::GatherElements>({slice, pattern::any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_node = m.get_pattern_map();

        auto is_constant_and_all_values_equal = [](const Output<Node>& output, const int64_t& v) -> bool {
            OPENVINO_SUPPRESS_DEPRECATED_START
            const auto& constant = ov::get_constant_from_source(output);
            OPENVINO_SUPPRESS_DEPRECATED_END
            if (!constant)
                return false;
            const auto& values = constant->cast_vector<int64_t>();
            return std::all_of(values.begin(), values.end(), [&](const int64_t& i) {
                return i == v;
            });
        };

        const auto& slice_node = pattern_to_node.at(slice);
        if (!is_constant_and_all_values_equal(slice_node->input_value(1), 0) ||
            !is_constant_and_all_values_equal(slice_node->input_value(3), 1))
            return false;
        // we slice from 0 to
        const auto& gather_node = pattern_to_node.at(gather);
        gather_node->input(0).replace_source_output(slice_node->input_value(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(gather, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

bool check_constant_is_non_negative_get_int_value(const ov::Output<ov::Node>& output, std::vector<int64_t>& data) {
    const auto& node = output.get_node_shared_ptr();
    const auto& constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant)
        return false;
    data = constant->cast_vector<int64_t>();
    return std::all_of(data.begin(), data.end(), [](const int64_t& i) {
        return i >= 0;
    });
}
//
// ov::pass::ChainedVariadicSplitOptimization::ChainedVariadicSplitOptimization() {
//    MATCHER_SCOPE(ChainedVariadicSplitOptimization);
//
//    auto first_vsplit = pattern::any_input();
//    auto axis = pattern::wrap_type<op::v0::Constant>();
//    auto length = pattern::wrap_type<op::v0::Constant>();
//    auto second_vsplit = pattern::wrap_type<op::v1::VariadicSplit>({first_vsplit, axis, length});
//
//    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
//        const auto& pattern_to_node = m.get_pattern_map();
//        auto first = pattern_to_node.at(first_vsplit);
//        if (!ov::as_type_ptr<op::v1::VariadicSplit>(first))
//            return false;
//        auto second = pattern_to_node.at(second_vsplit);
//        if (!are_equal_int_constants(first->input_value(1), second->input_value(1)))
//            return false;
//        std::vector<int64_t> split_length_1, split_length_2;
//        if (!check_constant_is_non_negative_get_int_value(first->input_value(2), split_length_1))
//            return false;
//        if (!check_constant_is_non_negative_get_int_value(second->input_value(2), split_length_2))
//            return false;
//
//        auto output_index = second->input_value(0).get_index();
//        if (first->output(output_index).get_target_inputs().size() != 1)
//            return false;
//        std::cout << "Output index: " << output_index << std::endl;
//        std::cout << "First : " << first << std::endl;
//        std::cout << PartialShape{split_length_1} << std::endl;
//        std::cout << "Second: " << second << std::endl;
//        std::cout << PartialShape{split_length_2} << std::endl;
//
//        int64_t summ_first = 0;
//        for (const auto& i : split_length_1)
//            summ_first += i;
//        std::cout << "Summ of dim before: " << summ_first << std::endl;
//
//        // first split info collection
//        split_length_1.erase(split_length_1.begin() + output_index);
//        split_length_1.insert(split_length_1.begin() + output_index, split_length_2.begin(), split_length_2.end());
//        summ_first = 0;
//        for (const auto& i : split_length_1)
//            summ_first += i;
//        std::cout << "Summ of dim after: " << summ_first << std::endl;
//
//        std::cout << "Resulting length: " << PartialShape{split_length_1} << std::endl;
//        auto outputs_1 = first->outputs();
//        auto outputs_2 = second->outputs();
//        outputs_1.erase(outputs_1.begin() + output_index);
//        outputs_1.insert(outputs_1.begin() + output_index, outputs_2.begin(), outputs_2.end());
//
//        auto new_split_length =
//            std::make_shared<op::v0::Constant>(element::i64, Shape{split_length_1.size()}, split_length_1);
//        first->input(2).replace_source_output(new_split_length->output(0));
//        first->validate_and_infer_types();
//        for (size_t i = 0; i < outputs_1.size(); ++i)
//            for (auto& input : outputs_1[i].get_target_inputs())
//                input.replace_source_output(first->output(i));
//        std::cout << "Resulting VSplit" << std::endl;
//        std::cout << first << std::endl;
//        return true;
//    };
//
//    auto m = std::make_shared<pattern::Matcher>(second_vsplit, matcher_name);
//    register_matcher(m, matcher_pass_callback);
//}

ov::pass::RPE_Optimization::RPE_Optimization() {
    MATCHER_SCOPE(RPE_Optimization);

    auto sin = pattern::wrap_type<op::v6::GatherElements, op::v0::Unsqueeze>();  // any_input doesn't work here
    auto cos = pattern::wrap_type<op::v6::GatherElements, op::v0::Unsqueeze>();  // any_input doesn't work here

    auto source = pattern::any_input(pattern::has_static_rank());

    // rotate_half begin
    auto split_length =
        pattern::wrap_type<op::v0::Constant>(pattern::op::as_value_predicate([](std::shared_ptr<Node> n) -> bool {
            const auto& constant = ov::as_type_ptr<op::v0::Constant>(n);
            if (!constant)
                return false;
            const auto& value = constant->cast_vector<int64_t>();
            return value.size() == 2 && value[0] == value[1];
        }));  // make sure constant contains 2 elements with same content; it may be -1, but we fix it earlier
    auto axis = pattern::any_input();
    auto vsplit = pattern::wrap_type<op::v1::VariadicSplit>({source, axis, split_length});
    vsplit->set_output_size(2);
    auto minus_1 =
        pattern::wrap_type<op::v0::Constant>(pattern::op::as_value_predicate([](std::shared_ptr<Node> n) -> bool {
            const auto& constant = ov::as_type_ptr<op::v0::Constant>(n);
            if (!constant)
                return false;
            const auto& value = constant->cast_vector<int64_t>();
            return value.size() == 1 && value[0] == -1;
        }));  // make sure it is == -1
    auto neg = pattern::wrap_type<op::v1::Multiply>({vsplit->output(1), minus_1});
    auto concat = pattern::wrap_type<op::v0::Concat>({neg, vsplit->output(0)});  // make sure axis eq to vsplit eq -1
    // rotate half end
    auto mul_sin = pattern::wrap_type<op::v1::Multiply>({concat, sin});
    auto mul_cos = pattern::wrap_type<op::v1::Multiply>({source, cos});
    auto add = pattern::wrap_type<op::v1::Add>({mul_cos, mul_sin});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        auto value_map = m.get_pattern_value_map();

        auto input = value_map.at(source);
        auto concat_node = ov::as_type_ptr<op::v0::Concat>(value_map.at(concat).get_node_shared_ptr());
        if (!concat_node)
            return false;
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto split_axis_node = ov::get_constant_from_source(value_map.at(axis));
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!split_axis_node)
            return false;
        auto value = split_axis_node->cast_vector<int64_t>();
        if (value.size() != 1)
            return false;
        auto concat_axis = concat_node->get_concatenation_axis();
        concat_axis = concat_axis < 0 ? concat_axis + input.get_partial_shape().size() : concat_axis;
        auto split_axis = value[0];
        split_axis = split_axis < 0 ? split_axis + input.get_partial_shape().size() : split_axis;
        if (concat_axis != split_axis)
            return false;
        auto rope = std::make_shared<ov::op::internal::RPE>(input,
                                                            value_map.at(sin),
                                                            value_map.at(cos),
                                                            concat_node->get_axis());
        value_map.at(add).replace(rope->output(0));  // TODO: update fused names
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(add, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::Fused_RPE_MHA_Replacer::Fused_RPE_MHA_Replacer() {
    MATCHER_SCOPE(Fused_RPE_MHA_Replacer);
    auto source = pattern::any_input(pattern::has_static_rank());

    auto parameter_values = pattern::wrap_type<op::v0::Parameter>();
    auto parameter_keys = pattern::wrap_type<op::v0::Parameter>();

    auto sin = pattern::wrap_type<op::v6::GatherElements>();  // any_input doesn't work here
    auto cos = pattern::wrap_type<op::v6::GatherElements>();  // any_input doesn't work here

    auto bool_constant = pattern::wrap_type<op::v0::Constant>([](const ov::Output<Node>& out) -> bool {
        auto is_boolean = out.get_element_type() == element::boolean;
        auto shape = out.get_shape();
        auto is_2D_matrix = shape.size() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] == shape[3];
        return is_boolean && is_2D_matrix;
    });
    auto slice_1 = pattern::wrap_type<op::v8::Slice>(
        {bool_constant, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto bool_mask = pattern::wrap_type<op::v8::Slice>(
        {slice_1, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});

    auto attention_mask = pattern::any_input();

    auto vsplit_1 = pattern::wrap_type<op::v1::VariadicSplit>({source, pattern::any_input(), pattern::any_input()});
    vsplit_1->set_output_size(3);

    auto vsplit_2 =
        pattern::wrap_type<op::v1::VariadicSplit>({vsplit_1->output(0), pattern::any_input(), pattern::any_input()});
    vsplit_2->set_output_size(2);
    auto rope_1 = pattern::wrap_type<op::internal::RPE>({vsplit_2->output(0), sin, cos});
    auto concat_1 = pattern::wrap_type<op::v0::Concat>({rope_1, vsplit_2->output(1)});

    auto vsplit_3 =
        pattern::wrap_type<op::v1::VariadicSplit>({vsplit_1->output(1), pattern::any_input(), pattern::any_input()});
    vsplit_3->set_output_size(2);
    auto rope_2 = pattern::wrap_type<op::internal::RPE>({vsplit_3->output(0), sin, cos});
    auto concat_2 = pattern::wrap_type<op::v0::Concat>({rope_2, vsplit_3->output(1)});

    auto concat_with_prev_k = pattern::wrap_type<op::v0::Concat>({parameter_keys, concat_2});
    auto some_constant = pattern::wrap_type<op::v0::Constant>();
    auto multiply = pattern::wrap_type<op::v1::Multiply>({concat_with_prev_k, some_constant});

    auto matmul_1 = pattern::wrap_type<op::v0::MatMul>({concat_1, multiply});

    auto inf_constant = pattern::wrap_type<op::v0::Constant>();
    auto select = pattern::wrap_type<op::v1::Select>({bool_mask, matmul_1, inf_constant});

    auto add = pattern::wrap_type<op::v1::Add>({select, attention_mask});
    auto softmax = pattern::wrap_type<op::v1::Softmax>({add});

    auto concat_with_prev_v = pattern::wrap_type<op::v0::Concat>({parameter_values, vsplit_1->output(2)});
    auto matmul_2 = pattern::wrap_type<op::v0::MatMul>({softmax, concat_with_prev_v});

    auto transpose = pattern::wrap_type<op::v1::Transpose>({matmul_2, pattern::any_input()});
    auto reshape = pattern::wrap_type<op::v1::Reshape>({transpose, pattern::any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        auto value_map = m.get_pattern_value_map();
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto split_d_head = ov::get_constant_from_source(value_map.at(vsplit_1).get_node_shared_ptr()->input_value(2));
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!split_d_head)
            return false;
        auto d_head_value = split_d_head->cast_vector<int64_t>()[1];
        if (d_head_value < 0)
            return false;  // being extra cautious -- it may be -1, but we converted it earlier
        auto node = std::make_shared<ov::op::internal::FusedMHA_RPE>(value_map.at(source),
                                                                     value_map.at(sin),
                                                                     value_map.at(cos),
                                                                     value_map.at(parameter_keys),
                                                                     value_map.at(parameter_values),
                                                                     value_map.at(bool_mask),
                                                                     value_map.at(attention_mask),
                                                                     value_map.at(some_constant),
                                                                     d_head_value);

        value_map.at(reshape).replace(node->output(0));             // TODO: update fused names
        value_map.at(concat_with_prev_k).replace(node->output(1));  // Make sure we aren't making a loop in the graph
        value_map.at(concat_with_prev_v).replace(node->output(2));
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(reshape, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

template <class T>
bool all_secondary_inputs_from_same_source_or_equal_int_constants(const std::shared_ptr<T>& lhs,
                                                                  const std::shared_ptr<T>& rhs) {
    if (lhs->get_input_size() != rhs->get_input_size())
        return false;
    size_t input_size = lhs->get_input_size();
    for (size_t i = 1; i < input_size; ++i) {
        if (lhs->input_value(i) == rhs->input_value(i))
            continue;
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto lhs_constant = ov::get_constant_from_source(lhs->input_value(i));
        auto rhs_constant = ov::get_constant_from_source(rhs->input_value(i));
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!lhs_constant || !rhs_constant)
            return false;
        if (lhs_constant->template cast_vector<int64_t>() != rhs_constant->template cast_vector<int64_t>())
            return false;
    }
    return true;
}

bool gather_elements_perform_same(const std::shared_ptr<ov::op::v6::GatherElements>& lhs,
                                  const std::shared_ptr<ov::op::v6::GatherElements>& rhs) {
    // 0 input value is already checked -- it is the same, we only need to check input with idx 1 and axis value
    return lhs->get_axis() == rhs->get_axis() && lhs->input_value(1) == rhs->input_value(1);
}

template <class T>
bool shared_node_optimization_helper(const std::shared_ptr<ov::Model>& model,
                                     bool (*are_equal)(const std::shared_ptr<T>&, const std::shared_ptr<T>&)) {
    // only works for 0 index inputs
    bool graph_rewritten = false;

    std::map<ov::Output<ov::Node>, std::vector<std::shared_ptr<T>>> source_to_typed_op;
    for (const auto& node : model->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<ov::op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= shared_node_optimization_helper<T>(sub_graph, are_equal);
            }
        }
        if (auto op = ov::as_type_ptr<T>(node)) {
            source_to_typed_op[op->input_value(0)].push_back(op);
        }
    }
    for (auto& pair : source_to_typed_op) {
        if (pair.second.size() < 2)
            continue;
        auto root_op = pair.second[0];
        for (auto& child_op : pair.second) {
            if (root_op->get_instance_id() != child_op->get_instance_id() && are_equal(root_op, child_op)) {
                graph_rewritten |= replace_output_update_name(child_op->output(0), root_op->output(0));
            }
        }
    }
    return graph_rewritten;
}

bool ov::pass::SharedTileOptimization::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(SharedTileOptimization);
    return shared_node_optimization_helper<ov::op::v0::Tile>(
        model,
        all_secondary_inputs_from_same_source_or_equal_int_constants);
}

bool ov::pass::SharedGatherElementsOptimization::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(SharedGatherElementsOptimization);
    return shared_node_optimization_helper<ov::op::v6::GatherElements>(model, gather_elements_perform_same);
}

bool ov::pass::SharedTransposeOptimization::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(SharedTransposeOptimization);
    return shared_node_optimization_helper<ov::op::v1::Transpose>(
        model,
        all_secondary_inputs_from_same_source_or_equal_int_constants);
}

bool ov::pass::SharedSliceOptimization::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(SharedSliceOptimization);
    return shared_node_optimization_helper<ov::op::v8::Slice>(
        model,
        all_secondary_inputs_from_same_source_or_equal_int_constants);
}

struct SliceAttrs {
    int64_t start, stop, axis;
};

bool slice_is_suitable_for_optimization(const std::shared_ptr<ov::op::v8::Slice>& op, SliceAttrs& attrs) {
    if (op->get_input_size() != 5 || op->get_input_partial_shape(0).rank().is_dynamic())
        return false;

    const auto& data_rank = op->get_input_partial_shape(0).rank().get_length();
    for (size_t i = 1; i < 5; ++i) {
        auto input_as_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(i));
        if (!input_as_constant)
            return false;
        if (shape_size(input_as_constant->get_shape()) != 1)
            return false;

        int64_t value = input_as_constant->cast_vector<int64_t>()[0];

        if ((i == 1 || i == 2) && value < 0)
            return false;
        if (i == 1)
            attrs.start = value;
        if (i == 2)
            attrs.stop = value;
        if (i == 3 && input_as_constant->cast_vector<int64_t>()[0] != 1)
            return false;  // step should be equal 1 for this optimization
        if (i == 4)
            attrs.axis = value >= 0 ? value : value + data_rank;
    }
    if (op->get_input_partial_shape(0)[attrs.axis].is_dynamic())
        return false;
    return true;
}

bool ov::pass::GroupedSliceToVSplitOptimization::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(GroupedSliceToVSplitOptimization);
    bool graph_rewritten = false;

    struct SliceWithAttrs {
        std::shared_ptr<op::v8::Slice> slice;
        SliceAttrs attrs;
    };

    using OutputWithAxis = std::pair<ov::Output<ov::Node>, int64_t>;

    std::map<OutputWithAxis, std::vector<SliceWithAttrs>> source_to_typed_op;
    std::vector<OutputWithAxis> ordered_outputs;
    for (const auto& node : model->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<ov::op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_model(sub_graph);
            }
        }
        if (auto op = ov::as_type_ptr<op::v8::Slice>(node)) {
            SliceAttrs attributes{};
            if (slice_is_suitable_for_optimization(op, attributes)) {
                OutputWithAxis current_output = {op->input_value(0), attributes.axis};
                source_to_typed_op[current_output].push_back({op, attributes});
                if (std::find(ordered_outputs.begin(), ordered_outputs.end(), current_output) == ordered_outputs.end())
                    ordered_outputs.push_back(current_output);
            }
        }
    }
    std::reverse(ordered_outputs.begin(), ordered_outputs.end());
    for (const auto& output_with_axis : ordered_outputs) {
        const auto& axis = output_with_axis.second;
        const auto& output = output_with_axis.first;
        auto attributes = source_to_typed_op[output_with_axis];

        std::sort(attributes.begin(), attributes.end(), [](const SliceWithAttrs& lhs, const SliceWithAttrs& rhs) {
            return lhs.attrs.start < rhs.attrs.start;
        });
        int64_t prev_stop = 0;
        bool valid_for_replacement = true;
        for (auto& slice_with_attrs : attributes) {
            // they shouldn't overlap and no holes while slicing
            if (prev_stop != slice_with_attrs.attrs.start)
                valid_for_replacement = false;
            prev_stop = slice_with_attrs.attrs.stop;
        }
        if (!valid_for_replacement)
            continue;

        std::vector<int64_t> split_lengths;
        // we made sure that dimension is static before
        const int64_t& dimension = output.get_partial_shape()[axis].get_length();
        int64_t dimension_length_left = dimension;
        for (auto& slice_with_attrs : attributes) {
            int64_t sliced = slice_with_attrs.attrs.stop - slice_with_attrs.attrs.start;
            if (sliced > dimension_length_left)
                split_lengths.push_back(-1);
            else
                split_lengths.push_back(sliced);
            dimension_length_left -= sliced;
        }
        if (std::count(split_lengths.begin(), split_lengths.end(), -1) > 1)
            continue;

        int64_t current_sum = 0;
        for (const auto& i : split_lengths)
            if (i != -1)
                current_sum += i;
        for (auto& i : split_lengths)
            if (i == -1) {
                i = dimension - current_sum;
                current_sum = dimension;
            }
        if (current_sum != dimension)
            continue;  // there are some l
        auto split_lengths_const =
            op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{split_lengths.size()}, split_lengths);
        auto axis_const = op::v0::Constant::create(ngraph::element::i64, ngraph::Shape{}, {axis});
        auto variadic_split = std::make_shared<op::v1::VariadicSplit>(output, axis_const, split_lengths_const);

        auto i = 0;
        NodeVector ops_to_replace;
        for (auto& slice_with_attrs : attributes) {
            slice_with_attrs.slice->output(0).replace(variadic_split->output(i));
            ops_to_replace.push_back(slice_with_attrs.slice);
            ++i;
        }
        copy_runtime_info(ops_to_replace, variadic_split);
        graph_rewritten = true;
    }
    return graph_rewritten;
}

bool ov::pass::ApplyTableOfEquivalence::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(ApplyTableOfEquivalence);

    // get table of equivalence
    auto rt_info = m->get_rt_info();
    if (!rt_info.count("TABLE_OF_EQUIVALENCE"))
        return false;
    auto te = rt_info["TABLE_OF_EQUIVALENCE"].as<std::shared_ptr<ov::TableOfEquivalence>>();
    if (te == nullptr)
        return false;
    auto equivalence_table = te->get_equivalence_table();

    for (const auto& op : m->get_ordered_ops()) {
        for (auto& output : op->outputs()) {
            auto shape = output.get_partial_shape();
            for (auto& d : shape) {
                if (d.is_static())
                    continue;
                auto label = ov::DimensionTracker::get_label(d);
                if (label != ov::no_label && equivalence_table.count(label) &&
                    *equivalence_table[label]->begin() != ov::no_label) {
                    label = std::min(label, *equivalence_table[label]->begin());
                    ov::DimensionTracker::set_label(d, label);
                }
            }
            op->set_output_type(output.get_index(), output.get_element_type(), shape);
            auto value_labels = output.get_tensor().get_value_label();
            if (!value_labels.empty() &&
                !std::all_of(value_labels.begin(), value_labels.end(), [](const ov::label_t& i) {
                    return i == ov::no_label;
                })) {
                for (auto& label : value_labels) {
                    if (equivalence_table.count(label) && *equivalence_table[label]->begin() != ov::no_label)
                        label = std::min(label, *equivalence_table[label]->begin());
                }
                output.get_tensor().set_value_label(value_labels);
            }
        }
    }
    return false;
}

int64_t get_idx_of_label_in_source(const ov::Output<ov::Node> source, const ov::label_t& label) {
    int64_t idx = -1;
    if (label == ov::no_label)
        return idx;
    auto pshape = source.get_partial_shape();
    auto rank = pshape.rank();
    if (rank.is_dynamic())
        return idx;
    for (int64_t i = 0; i < rank.get_length(); ++i) {
        auto l = ov::DimensionTracker::get_label(pshape[i]);
        if (l == label) {
            idx = i;
            break;
        }
    }
    return idx;
}

std::shared_ptr<ov::Node> get_node_representing_label_from_source_by_idx(const ov::Output<ov::Node>& source,
                                                                         const ov::element::Type& et,
                                                                         const ov::Shape& shape,
                                                                         const int64_t& idx) {
    if (idx == -1)
        return nullptr;  // -1 index doesn't mean counting from the back -- it means we didn't find where the label
                         // comes from
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(source, et);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto indices = ov::op::v0::Constant::create(ov::element::i64, shape, {idx});
    auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, indices, axis);
    evaluate_both_bounds(gather->output(0));
    return gather;
}

std::shared_ptr<ov::Node> get_node_representing_label_from_source_by_label(const ov::Output<ov::Node>& source,
                                                                           const ov::element::Type& et,
                                                                           const ov::Shape& shape,
                                                                           const ov::label_t& label) {
    return get_node_representing_label_from_source_by_idx(source, et, shape, get_idx_of_label_in_source(source, label));
}

// label to source map
using LTS_map = std::unordered_map<ov::label_t, ov::Output<ov::Node>>;

void optimize_value_usage(ov::Output<ov::Node>& output, LTS_map& label_shape_source, LTS_map& label_value_source) {
    auto value_labels = output.get_tensor().get_value_label();
    if (value_labels.size() != 1)
        return;
    auto label = value_labels[0];
    if (label == ov::no_label)
        return;
    auto pshape = output.get_partial_shape();
    if (pshape.is_dynamic() || ov::shape_size(pshape.to_shape()) != 1)
        return;
    auto shape = pshape.to_shape();  // scalar of some form of tensor with one element
    auto et = output.get_element_type();

    auto default_et = ov::element::i64;
    auto default_shape = ov::Shape{1};

    std::shared_ptr<ov::Node> alternative_source = nullptr;

    if (label_shape_source.count(label)) {
        auto source = label_shape_source[label];
        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(source.get_node_shared_ptr());
        int64_t idx = get_idx_of_label_in_source(source, label);
        if (concat && idx != -1 && idx == concat->get_concatenation_axis() && concat->get_input_size() == 2) {
            // optimize using the knowledge of the Concat SI and what happens on the axis
            auto lhs_source = concat->input_value(0);
            const auto& lhs_pshape = lhs_source.get_partial_shape();
            auto rhs_source = concat->input_value(1);
            const auto& rhs_pshape = rhs_source.get_partial_shape();
            if (lhs_pshape.rank().is_static() && rhs_pshape.rank().is_static()) {
                auto lhs_label = ov::DimensionTracker::get_label(lhs_pshape[idx]);
                auto rhs_label = ov::DimensionTracker::get_label(rhs_pshape[idx]);
                std::shared_ptr<ov::Node> lhs_alternative = nullptr, rhs_alternative = nullptr;
                // get lhs_label from value or shape source
                if (label_value_source.count(lhs_label))
                    lhs_alternative = label_value_source[lhs_label].get_node_shared_ptr();
                if (!lhs_alternative && label_shape_source.count(lhs_label)) {
                    lhs_alternative = get_node_representing_label_from_source_by_label(label_shape_source[lhs_label],
                                                                                       default_et,
                                                                                       default_shape,
                                                                                       lhs_label);
                    if (lhs_alternative)
                        label_value_source[lhs_label] = lhs_alternative->output(0);
                }
                // get rhs_label from value or shape source
                if (label_value_source.count(rhs_label))
                    rhs_alternative = label_value_source[lhs_label].get_node_shared_ptr();
                if (!rhs_alternative && label_shape_source.count(rhs_label)) {
                    rhs_alternative = get_node_representing_label_from_source_by_label(label_shape_source[rhs_label],
                                                                                       default_et,
                                                                                       default_shape,
                                                                                       rhs_label);
                    if (rhs_alternative)
                        label_value_source[rhs_label] = rhs_alternative->output(0);
                }
                if (lhs_alternative && rhs_alternative) {
                    alternative_source = std::make_shared<ov::op::v1::Add>(lhs_alternative, rhs_alternative);
                    label_value_source[label] = alternative_source->output(0);
                }
            }
        }
    }
    if (!alternative_source && label_value_source.count(label)) {
        auto value_source = label_value_source[label];
        alternative_source = value_source.get_node_shared_ptr();
    }
    if (!alternative_source && label_shape_source.count(label)) {
        // replacement via constructing the label source and saving it for the future
        alternative_source = get_node_representing_label_from_source_by_label(label_shape_source[label],
                                                                              default_et,
                                                                              default_shape,
                                                                              label);
        if (alternative_source)
            label_value_source[label] = alternative_source->output(0);
    }
    if (alternative_source != nullptr) {
        auto value_source = alternative_source->output(0);
        if (value_source.get_shape() != shape && (shape.empty() || shape == ov::Shape{0}))
            value_source = std::make_shared<ov::op::v0::Squeeze>(value_source);
        else if (value_source.get_shape() != shape)
            value_source = std::make_shared<ov::op::v1::Reshape>(
                value_source,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape),
                false);
        if (value_source.get_element_type() != et)
            value_source = std::make_shared<ov::op::v0::Convert>(value_source, et);
        output.replace(value_source);
    } else {
        // in case we can not optimize it -- it is label which appeared just now on the value path
        label_value_source[label] = output;
    }
}

void save_shape_sources(const ov::Output<ov::Node>& output, LTS_map& label_shape_source) {
    auto shape = output.get_partial_shape();
    for (const auto& d : shape) {
        if (d.is_static())
            continue;
        auto label = ov::DimensionTracker::get_label(d);
        if (label == ov::no_label || label_shape_source.count(label))
            continue;
        label_shape_source[label] = output;
    }
}

bool ov::pass::OptimizeLabelsUsedAsValues::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(OptimizeLabelsUsedAsValues);
    LTS_map label_shape_source;
    LTS_map label_value_source;
    for (const auto& op : m->get_ordered_ops()) {
        for (auto& output : op->outputs()) {
            optimize_value_usage(output, label_shape_source, label_value_source);
            save_shape_sources(output, label_shape_source);
        }
    }
    return true;
}

bool ov::pass::SymbolicOptimizations::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(SymbolicOptimizations);

#define VISUALIZE(man, file) std::cout;  // << "Skipped visualization to " << (file) << std::endl;
                                         //#define VISUALIZE(man, file) REGISTER_PASS(man, VisualizeTree, file)

    ov::pass::Manager pre_symbolic_manager(get_pass_config());
    pre_symbolic_manager.set_per_pass_validation(false);
    VISUALIZE(pre_symbolic_manager, "00_before_BroadcastOnes.svg")
    REGISTER_PASS(pre_symbolic_manager, BroadcastOnes)
    VISUALIZE(pre_symbolic_manager, "0_after_BroadcastOnes_before_SharedTileOptimization.svg")
    REGISTER_PASS(pre_symbolic_manager, SharedTileOptimization)
    VISUALIZE(pre_symbolic_manager, "1_after_SharedTileOptimization_before_TSSliceBackward.svg")
    auto pre_optimizations_0 = pre_symbolic_manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(pre_optimizations_0, transpose_sinking::TSSliceBackward)
    pre_optimizations_0->set_name("ov::pass::GraphRewrite::PreSymbolicOptimizations::0");
    VISUALIZE(pre_symbolic_manager, "2_after_TSSliceBackward_before_SharedTransposeOptimization.svg")
    REGISTER_PASS(pre_symbolic_manager, SharedTransposeOptimization)
    VISUALIZE(pre_symbolic_manager, "3_after_SharedTransposeOptimization_before_SharedSliceOptimization.svg")
    REGISTER_PASS(pre_symbolic_manager, SharedSliceOptimization)
    VISUALIZE(pre_symbolic_manager, "4_after_SharedSliceOptimization_before_GroupedSliceToVSplitOptimization.svg")
    REGISTER_PASS(pre_symbolic_manager, GroupedSliceToVSplitOptimization)
    pre_symbolic_manager.run_passes(m);

    ov::pass::Manager manager(get_pass_config());
    manager.set_per_pass_validation(false);
    VISUALIZE(manager, "5_after_GroupedSliceToVSplitOptimization_before_SymbolicPOC.svg")
    REGISTER_PASS(manager, SymbolicPOC)
    VISUALIZE(manager, "6_after_SymbolicPOC_before_optimizations_0_ChainedMaximumOptimization_NopBroadcast.svg")

    auto optimizations_0 = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(optimizations_0, ChainedMaximumOptimization)
    ADD_MATCHER(optimizations_0, NopBroadcast)
    VISUALIZE(manager, "7_after_optimizations_0_before_BroadcastOnes.svg")
    REGISTER_PASS(manager, BroadcastOnes)
    optimizations_0->set_name("ov::pass::GraphRewrite::SymbolicOptimizations::0");
    manager.run_passes(m);

    ov::pass::Manager manager_1(get_pass_config());
    manager_1.set_per_pass_validation(false);
    VISUALIZE(manager_1, "8_after_BroadcastOnes_before_DeReshapeMatMul_RemoveSliceBeforeGatherElements.svg")
    auto optimizations_1 = manager_1.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(optimizations_1, DeReshapeMatMul)
    ADD_MATCHER(optimizations_1, RemoveSliceBeforeGatherElements)
    VISUALIZE(manager_1,
              "9_after_DeReshapeMatMul_RemoveSliceBeforeGatherElements_before_SharedGatherElementsOptimization.svg")
    REGISTER_PASS(manager_1, SharedGatherElementsOptimization)
    optimizations_1->set_name("ov::pass::GraphRewrite::SymbolicOptimizations::1");
    VISUALIZE(manager_1, "10_after_SharedGatherElementsOptimization_before_RPE_Optimization.svg")
    //    REGISTER_PASS(manager_1, RPE_Optimization)
    VISUALIZE(manager_1, "11_after_RPE_Optimization_before_OptimizeLabelsUsedAsValues.svg")
    REGISTER_PASS(manager_1, OptimizeLabelsUsedAsValues)
    VISUALIZE(manager_1, "12_after_OptimizeLabelsUsedAsValues_before_ApplyTableOfEquivalence.svg")
    REGISTER_PASS(manager_1, ApplyTableOfEquivalence)
    VISUALIZE(manager_1, "13_after_ApplyTableOfEquivalence_before_Fused_RPE_MHA_Replacer.svg")
    REGISTER_PASS(manager_1, Fused_RPE_MHA_Replacer)
    // cleanup labels, erase SKIP_INVALIDATION
    VISUALIZE(manager_1, "final_model.svg")
    manager_1.run_passes(m);
    return true;  // cleans up all the label information
}
