// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/dimensions_tracking.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/ops.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::FindBatch, "FindBatch", 0);

void mark_batch(ngraph::opset1::Parameter* parameter, const std::set<std::string>& batches) {
    auto& shape = parameter->get_partial_shape();
    for (auto& dim : shape) {
        auto dim_name = dim.get_name();
        if (batches.count(dim_name)) {
            dim_name = "BATCH_" + dim_name;
        } else if (dim_name.rfind("BATCH_", 0) == 0 && !batches.count(dim_name.substr(6))) { // skip dims already marked as batch
            dim_name = "";
        }
        dim = ngraph::Dimension(dim.get_min_length(), dim.get_max_length(), dim_name);
    }
    parameter->set_partial_shape(shape);
}

void mark_layout_independent_batch(ngraph::opset1::Parameter* parameter, ngraph::Node* result) {
    std::vector<std::string> r_dim_names, p_dim_names;
    for (const auto& dim : parameter->get_partial_shape()) {
        if (!dim.get_name().empty())
            p_dim_names.push_back(dim.get_name());
        if (dim.get_name().rfind("BATCH_", 0) == 0)
            return; // batch was already set for the parameter
    }
    for (const auto& dim : result->get_output_partial_shape(0)) {
        const auto& name = dim.get_name();
        if (!name.empty() && find(p_dim_names.begin(), p_dim_names.end(), name) != p_dim_names.end()) {
            mark_batch(parameter, {name});
            return; // we marked outer-most matching dimension between parameter and result shapes
        }
    }
}

void mark_no_batch(ngraph::op::Parameter* parameter, ngraph::Node* node) {
    auto &shape = parameter->get_partial_shape();
    for (auto &dim : shape) {
        dim = ngraph::Dimension(dim.get_min_length(), dim.get_max_length(), "");
    }
    parameter->set_partial_shape(shape);
    NGRAPH_UNREACHABLE(node);
}

void mark_with_unique_dimension_names(std::shared_ptr<ngraph::Function> f) {
    size_t i = 0;
    for (auto & parameter : f->get_parameters()) {
        ngraph::PartialShape new_shape = ngraph::PartialShape::dynamic(parameter->get_partial_shape().rank());
        for (auto& dim : new_shape)
            dim = ngraph::Dimension{-1, "DIM_" + std::to_string(i++)};
        parameter->set_partial_shape(new_shape);
    }
}

void find_batch(std::shared_ptr<ngraph::Function> f) {
    std::map<ngraph::Node::type_info_t, std::pair<size_t, size_t>> type_input_port_batch_index = {
            {ngraph::opset1::Convolution::type_info, {0, 0}},
            {ngraph::opset1::GroupConvolution::type_info, {0, 0}},
            {ngraph::opset1::ConvolutionBackpropData::type_info, {0, 0}},
            {ngraph::opset1::GroupConvolutionBackpropData::type_info, {0, 0}},
            {ngraph::opset1::DeformableConvolution::type_info, {0, 0}},
            {ngraph::opset1::MatMul::type_info, {0, 0}},
    };


    for (auto& parameter : f->get_parameters()) {
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
                const auto &batch_dim_name = curr_node->input_value(batch_placement.first).get_partial_shape()[batch_placement.second].get_name();
                if (!batch_dim_name.empty())
                    mark_batch(raw_parameter, {batch_dim_name});
                else
                    mark_no_batch(raw_parameter, curr_node);
                continue; // batch was or was not found at this point -- there is no point in searching further            }
            }
            // node is not layout obvious -- checking if dims were propagated through
            bool all_outputs_marked_with_name = true;
            for (auto& output : curr_node->outputs()) {
                bool name_stays = false;
                const auto& output_shape = output.get_partial_shape();
                for (const auto &dim : output_shape)
                    name_stays |= !dim.get_name().empty();
                all_outputs_marked_with_name &= name_stays;
            }

            if (!all_outputs_marked_with_name) {
                mark_no_batch(raw_parameter, curr_node);
                continue; // dimensions name propagation stopped
            }

            if (ngraph::is_type<ngraph::opset1::Result>(curr_node))
                layout_independent_results.push_back(curr_node);

            for (const auto& output : curr_node->outputs()) {
                // we do not need to walk through shape-of sub-graphs
                for (const auto& t_input : output.get_target_inputs()) {
                    if (ngraph::is_type<ngraph::opset1::ShapeOf>(t_input.get_node()) ||
                            ngraph::is_type<ngraph::opset3::ShapeOf>(t_input.get_node()))
                        continue;
                    nodes.push_back(t_input.get_node());
                }
            }
        }

        for (const auto& result : layout_independent_results)
            // there are no layout obvious operations on the Parameter-Result path
            // considering the outer-most matching dimension is batch
            mark_layout_independent_batch(raw_parameter, result);
    }
}

void restore_original_dimensions_except_batch(const std::map<ngraph::opset1::Parameter*, ngraph::PartialShape>& parameter_to_shape) {
    for (const auto& item : parameter_to_shape) {
        const auto& batch_marked_shape = item.first->get_partial_shape();
        const auto& original_shape = item.second;

        auto original_shape_with_dynamic_batch = ngraph::PartialShape::dynamic(batch_marked_shape.rank());
        for (ngraph::Dimension::value_type n = 0; n < batch_marked_shape.rank().get_length(); ++n) {
            if (batch_marked_shape[n].get_name().rfind("BATCH_", 0) == 0)
                original_shape_with_dynamic_batch[n] = ngraph::Dimension(-1, batch_marked_shape[n].get_name());
            else
                original_shape_with_dynamic_batch[n] = original_shape[n];
        }
        item.first->set_partial_shape(original_shape_with_dynamic_batch);
    }
}

bool check_batch_tracks_through_all_the_nodes(std::shared_ptr<ngraph::Function> f) {
    bool failed_to_propagate_batch = false;
    for (auto &node : f->get_ordered_ops()) {
        bool any_input_has_batch = false;
        for (const auto &input : node->input_values()) {
            const auto &input_shape = input.get_partial_shape();
            bool name_stays = false;
            bool others_are_static = true;
            for (const auto &dim : input_shape)
                if (dim.get_name().empty())
                    others_are_static &= dim.is_static();
                else
                    name_stays = true;
            any_input_has_batch |= name_stays && others_are_static;
        }
        bool all_outputs_has_batch = true;
        for (const auto &output : node->outputs()) {
            const auto &output_shape = output.get_partial_shape();
            bool name_stays = false;
            bool others_are_static = true;
            for (const auto &dim : output_shape)
                if (dim.get_name().empty())
                    others_are_static &= dim.is_static();
                else
                    name_stays = true;
            all_outputs_has_batch &= name_stays; // && others_are_static;
        }
        if (any_input_has_batch && !all_outputs_has_batch &&
            !ngraph::is_type<ngraph::opset3::ShapeOf>(node) && !ngraph::is_type<ngraph::opset1::ShapeOf>(node)) {
            failed_to_propagate_batch = true;
            std::cout << "Lost batch: " << node << std::endl;
        }
    }
    const auto &results = f->get_results();
    for (const auto &result : results) {
        const auto &input_shape = result->get_input_partial_shape(0);
        bool name_stays = std::any_of(
                input_shape.cbegin(), input_shape.cend(), [](const ngraph::Dimension &d) { return !d.get_name().empty(); });
        failed_to_propagate_batch |= !name_stays;
    }
    return failed_to_propagate_batch;
}

bool ngraph::pass::FindBatch::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(FindBatch);

    const auto& parameters = f->get_parameters();
    std::map<opset1::Parameter*, PartialShape> parameter_to_shape;
    for (auto & parameter : parameters) {
        auto shape = parameter->get_partial_shape();
        if (shape.rank().is_dynamic())
            return false;
        parameter_to_shape[parameter.get()] = shape;
    }

    mark_with_unique_dimension_names(f);

    f->validate_nodes_and_infer_types();

    find_batch(f);

    restore_original_dimensions_except_batch(parameter_to_shape);

    f->validate_nodes_and_infer_types();

    bool failed_to_propagate_batch = check_batch_tracks_through_all_the_nodes(f);
    std::cout << (failed_to_propagate_batch ? "fail" : "success") << std::endl;
    for (const auto& p : f->get_parameters())
        std::cout << p << std::endl;
    NGRAPH_CHECK(!failed_to_propagate_batch);
    if (failed_to_propagate_batch) {
        // return function to the initial state
        for (const auto& item : parameter_to_shape) {
            item.first->set_partial_shape(item.second);
        }
        f->validate_nodes_and_infer_types();
    }
    return false;
}
