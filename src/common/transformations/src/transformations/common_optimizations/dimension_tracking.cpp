// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dimension_tracking.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/rt_info.hpp"
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

void ov::batch_util::mark_with_unique_dimension_labels(const std::shared_ptr<ov::Model>& m,
                                                       const ov::DimensionTracker& dt) {
    ov::label_t i = 1;
    for (auto& parameter : m->get_parameters()) {
        ov::PartialShape new_shape = ov::PartialShape::dynamic(parameter->get_partial_shape().rank());
        for (auto& dim : new_shape)
            dt.set_up_for_tracking(dim, i++);
        parameter->set_partial_shape(new_shape);
    }
    m->validate_nodes_and_infer_types();
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
        std::vector<ov::Node*> layout_independent_results;

        std::deque<ov::Node*> nodes{raw_parameter};
        std::unordered_set<ov::Node*> visited;

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
            // considering the outermost matching dimension is batch
            mark_layout_independent_batch(parameter, result->shared_from_this(), parameter_to_batch_labels);
    }
    return parameter_to_batch_labels;
}

void ov::batch_util::restore_original_dimensions(
    const std::shared_ptr<ov::Model>& model,
    const std::map<std::shared_ptr<ov::op::v0::Parameter>, ov::PartialShape>& parameter_to_shape,
    bool leave_batch_dynamic,
    bool clear_labels) {
    for (const auto& item : parameter_to_shape) {
        const auto& batch_marked_shape = item.first->get_partial_shape();
        auto original_shape = item.second;
        OPENVINO_ASSERT(batch_marked_shape.rank().is_static() && original_shape.rank().is_static());
        OPENVINO_ASSERT(batch_marked_shape.size() == original_shape.size());

        for (size_t n = 0; n < batch_marked_shape.size(); ++n) {
            if (const auto& label = ov::DimensionTracker::get_label(batch_marked_shape[n])) {
                if (leave_batch_dynamic)
                    original_shape[n] = Dimension::dynamic();
                if (!clear_labels)
                    ov::DimensionTracker::set_label(original_shape[n], label);
            }
        }
        item.first->set_partial_shape(original_shape);
    }
    std::unordered_map<std::shared_ptr<ov::op::v0::Result>, ov::PartialShape> output_to_shape;
    if (!clear_labels) {
        for (const auto& result : model->get_results())
            output_to_shape[result] = result->get_output_partial_shape(0);
    }

    model->validate_nodes_and_infer_types();

    if (!clear_labels) {
        for (const auto& item : output_to_shape) {
            auto labeled_shape = item.second, current_shape = item.first->get_output_partial_shape(0);
            auto labeled_rank = labeled_shape.rank(), current_rank = current_shape.rank();
            if (labeled_rank.is_static() && current_rank.is_static() && labeled_rank == current_rank) {
                for (size_t i = 0; i < labeled_shape.size(); ++i) {
                    auto label = ov::DimensionTracker::get_label(labeled_shape[i]);
                    if (label != ov::no_label)
                        ov::DimensionTracker::set_label(current_shape[i], label);
                }
                item.first->set_output_type(0, item.first->get_element_type(), current_shape);
            }
        }
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
                ov::copy_runtime_info(result_node, new_result);
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

namespace {

std::map<std::shared_ptr<ov::op::v0::Parameter>, ov::PartialShape> collect_original_input_shapes(
    const std::shared_ptr<ov::Model>& m) {
    const auto& parameters = m->get_parameters();
    std::map<std::shared_ptr<ov::op::v0::Parameter>, ov::PartialShape> parameter_to_shape;
    for (const auto& parameter : parameters) {
        auto shape = parameter->get_partial_shape();
        if (shape.rank().is_dynamic())
            return {};
        parameter_to_shape[parameter] = shape;
    }
    return parameter_to_shape;
}

}  // namespace

bool ov::pass::FindBatch::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(FindBatch);
    auto te = std::make_shared<ov::TableOfEquivalence>();
    ov::DimensionTracker dt(te);

    bool model_has_changed = false;
    if (detach_do)
        model_has_changed |= batch_util::detach_detection_output(m);

    auto parameter_to_shape = collect_original_input_shapes(m);
    if (parameter_to_shape.empty())
        return model_has_changed;

    ov::batch_util::mark_with_unique_dimension_labels(m, dt);

    ov::batch_util::find_batch(m);

    if (!track) {
        ov::batch_util::restore_original_dimensions(m, parameter_to_shape, false, false);
        return false;  // we have called validation on this model already
    }
    ov::batch_util::restore_original_dimensions(m, parameter_to_shape);
    bool failed_to_propagate_batch = ov::batch_util::check_batch_tracks_through_all_the_nodes(m);
    ov::batch_util::restore_original_dimensions(m, parameter_to_shape, false, failed_to_propagate_batch);
    return false;  // we have called validation on this model already
}
