// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/pass/insert_loops.hpp"
#include "snippets/pass/loop_helpers.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"

#include <ngraph/rt_info.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

InsertLoops::InsertLoops(ov::PartialShape master_shape, size_t loop_depth, size_t vector_size, bool single_loop_body)
    : m_master_shape(std::move(master_shape)), m_loop_depth(loop_depth), m_vector_size(vector_size), m_single_loop_body(single_loop_body) {
    if (m_master_shape.size() < m_loop_depth)
        throw ngraph_error("InsertLoops can't insert loops: master shape rank is too small");
}

std::vector<bool> InsertLoops::calculate_inner_apply_increments(const ov::PartialShape& master,
                                                                const std::vector<ov::PartialShape>& shapes) {
    // Inner Loop applies increments if a dimension is not broadcasted
    std::vector<bool> apply_increments;
    apply_increments.reserve(shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(apply_increments),
                   [=](const ov::PartialShape& ps) { return utils::get_inner_dim(ps) != 1 && utils::get_inner_dim(master) != 1; });
    return apply_increments;
}
std::vector<bool> InsertLoops::calculate_outer_apply_increments(const std::vector<ov::PartialShape>& shapes) {
    // Outer Loop applies increments only if a corresponding lower dim was broadcasted (or all lower dims == 1)
    std::vector<bool> apply_increments;
    apply_increments.reserve(shapes.size());
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(apply_increments),
                   [=](const ov::PartialShape& ps) { return utils::get_outer_dim(ps) != 1 && utils::get_inner_dim(ps) == 1; });
    return apply_increments;
}
std::vector<int64_t> InsertLoops::calculate_finalization_offsets(const ov::PartialShape& master,
                                                                 const std::vector<ov::PartialShape>& shapes) {
    const auto inner_work_amount = utils::get_inner_dim(master).get_length();
    std::vector<int64_t> inner_finalization_offsets(shapes.size(), 0);
    std::transform(shapes.begin(), shapes.end(), inner_finalization_offsets.begin(),
                   [=](const ov::PartialShape& ps) {
                       return utils::get_outer_dim(ps) == 1 && utils::get_inner_dim(ps) != 1 ? -inner_work_amount : 0;
                   });
    return inner_finalization_offsets;
}

void insert_loops_explicitly(const ov::NodeVector& ops, const size_t vector_size) {
    ov::NodeVector body;
    ov::NodeVector body_remainder;
    ov::OutputVector body_parameters;
    std::vector<ov::Input<ov::Node>> body_results;

    // check for potential parameters for new Loop
    auto add_body_parameters = [](const std::shared_ptr<ov::Node>& op, ov::OutputVector& body_parameters) {
        for (const auto& input : op->inputs()) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            if (ov::is_type<op::LoopEnd>(parent) ||
                ov::is_type<op::Buffer>(parent) ||
                ov::is_type<ov::op::v0::Parameter>(parent) ||
                ov::is_type<op::Brgemm>(parent)) {
                body_parameters.push_back(input.get_source_output());
            }
        }
    };

    // check for potential results for new Loop
    auto add_body_results = [](const std::shared_ptr<ov::Node>& op, std::vector<ov::Input<ov::Node>>& body_results) {
        for (const auto& output : op->outputs()) {
            for (const auto& target_input : output.get_target_inputs()) {
                auto child = target_input.get_node();
                if (ov::is_type<op::LoopBegin>(child) ||
                    ov::is_type<op::Buffer>(child) ||
                    ov::is_type<ov::op::v0::Result>(child) ||
                    ov::is_type<op::Brgemm>(child)) {
                    body_results.push_back(target_input);
                }
            }
        }
    };

    // check for potential missing body ops for new loop
    std::function<void(const std::shared_ptr<ov::Node>& op, ov::NodeVector& body)> add_missing_body_ops;
    add_missing_body_ops = [&](const std::shared_ptr<ov::Node>& op, ov::NodeVector& body) {
        if (body_remainder.size()) {
            for (const auto& input : op->inputs()) {
                auto parent = input.get_source_output().get_node_shared_ptr();
                auto iter = std::find(body_remainder.begin(), body_remainder.end(), parent);
                if (iter != body_remainder.end()) {
                    *std::back_inserter(body) = std::move(*iter);
                    add_missing_body_ops(parent, body);
                    add_body_parameters(parent, body_parameters);
                    add_body_results(op, body_results);
                }
            }
        }
    };

    auto wrap_body_by_loop = [&](const ov::NodeVector& body, const ov::OutputVector& body_parameters, const std::vector<ov::Input<ov::Node>>& body_results) {
        NGRAPH_CHECK(!body_parameters.empty(), "The count of parameters for loop should be more than zero to create loop");
        NGRAPH_CHECK(!body_results.empty(), "The count of results for loop should be more than zero to create loop");
        std::vector<ov::PartialShape> body_shapes;
        const auto count_io = body_parameters.size() + body_results.size();
        body_shapes.reserve(count_io);
        std::transform(body_parameters.begin(), body_parameters.end(), std::back_inserter(body_shapes),
                       [](const ov::Output<ov::Node>& out) { return out.get_partial_shape(); });
        std::transform(body_results.begin(), body_results.end(), std::back_inserter(body_shapes),
                       [](const ov::Input<ov::Node>& in) { return in.get_partial_shape(); });

        auto body_master_shape = body_shapes.front();
        for (const auto& shape : body_shapes) {
            NGRAPH_CHECK(PartialShape::broadcast_merge_into(body_master_shape, shape, ::ngraph::op::AutoBroadcastType::NUMPY),
                         "Loop input and output must be numpy broadcastable");
        }
        const auto inner_work_amount = utils::get_inner_dim(body_master_shape).get_length();
        const auto outer_work_amount = utils::get_outer_dim(body_master_shape).get_length();

        auto apply_increments = InsertLoops::calculate_inner_apply_increments(body_master_shape, body_shapes);
        std::vector<int64_t> inner_finalization_offsets(body_shapes.size(), 0);
        if (outer_work_amount > 1) {
            inner_finalization_offsets = InsertLoops::calculate_finalization_offsets(body_master_shape, body_shapes);
        }

        const auto& inner_loop_begin = op::insertLoopBeginAfterOutputs(body_parameters);
        const auto& inner_loop_end = op::insertLoopEndBeforeInputs(
            body_results, inner_loop_begin, inner_work_amount, vector_size,
            apply_increments, inner_finalization_offsets);
        // set internal flag to enable scalar vs vector loop optimizations
        inner_loop_end->has_outer_loop = outer_work_amount > 1;
        // Due to features of topological sort, some Constants (Scalars) may appear right after Parameters in
        // sorted ops (so it's between Parameters and LoopBegin). Consequently, ScalarEmitters would be called
        // outside the Loop, and only the first Loop iteration would yield correct data (assuming the vector reg
        // assigned to scalar will get corrupted inside the loop body). To avoid such cases, we add control dependency
        // on LoopBegin to guarantee that the constants are executed inside the Loop.
        for (const auto& n : body) {
            if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(n)) {
                c->add_control_dependency(inner_loop_begin);
            }
        }

        if (outer_work_amount > 1) {
            std::vector<bool> apply_increments = InsertLoops::calculate_outer_apply_increments(body_shapes);
            std::vector<int64_t> outer_finalization_offsets(body_shapes.size(), 0);
            const auto& outer_loop_begin = op::insertLoopBegin(body_parameters);
            op::insertLoopEnd(body_results, outer_loop_begin, outer_work_amount, 1lu,
                apply_increments, outer_finalization_offsets);
        }
    };

    auto op_is_outside_loop = [](const std::shared_ptr<ov::Node>& op) -> bool {
        if (ov::is_type<ov::op::v0::Parameter>(op) ||
            ov::is_type<ov::op::v0::Result>(op) ||
            ov::is_type<op::Buffer>(op))
            return true;
        auto& rt = op->get_rt_info();
        auto outside_rt = rt.find("outside_loop");
        bool is_outside = false;
        // If rt info isn't setted it means that op should be inside loop by default
        if (outside_rt != rt.end()) {
            is_outside = outside_rt->second.as<bool>();
        }
        return is_outside;
    };

    for (auto iter = ops.begin(); iter < ops.end(); iter++) {
        const auto op = *iter;
        // Need to check for that op should be inside or outside loop
        if (op_is_outside_loop(op)) {
            continue;
        }

        // If we meet loopBegin or Brgemm, it means that all previous nodes from ordered body
        // should be in one body. It's like stop signal
        const auto& loop_begin = ov::as_type_ptr<op::LoopBegin>(op);
        const auto& brgemm = ov::as_type_ptr<op::Brgemm>(op);
        if (loop_begin || brgemm) {
            if (!body.empty()) {
                if (!body_results.empty()) {
                    wrap_body_by_loop(body, body_parameters, body_results);
                } else {
                    // If there aren't body results, it means that the current body ops are inputs of the next some operations in ordered_ops
                    // So this set of the current body ops is part of the future body loop.
                    // We should save them to add in body ops in the future
                    std::move(body.begin(), body.end(), std::back_inserter(body_remainder));
                }
            }

            // we should skip the next existing Loop body
            if (loop_begin) {
                const auto &loop_end = loop_begin->get_loop_end();
                iter = std::find(iter, ops.end(), loop_end);
            }

            // clear loop body to create the next
            body.clear();
            body_parameters.clear();
            body_results.clear();
        } else {
            add_missing_body_ops(op, body);
            add_body_parameters(op, body_parameters);
            add_body_results(op, body_results);

            body.push_back(op);
        }
    }

    if (!body.empty()) {
        wrap_body_by_loop(body, body_parameters, body_results);
    }
}

bool InsertLoops::run_on_model(const std::shared_ptr<ov::Model> &model) {
    RUN_ON_FUNCTION_SCOPE(InsertLoops);
    if (m_master_shape.is_dynamic())
        throw ngraph_error("InsertLoops doesn't support dynamic shapes yet");

    const auto inner_work_amount = utils::get_inner_dim(m_master_shape).get_length();
    const auto outer_work_amount = m_loop_depth == 2 ? utils::get_outer_dim(m_master_shape).get_length() : 1;

    auto ops = model->get_ordered_ops();
    ParameterVector commonParams = model->get_parameters();
    // Note that topological sort parses node arguments in reversed order, but results are added  - in direct order
    // So ve need to pass the reversed results to LoopEnd to keep the original traversal order in topological sorter
    const auto& orig_results = model->get_results();
    ResultVector commonResults(orig_results.rbegin(), orig_results.rend());
    std::vector<PartialShape> ioShapes;

    const auto& body_rt_info = model->get_rt_info();
    const auto& plugin_shapes = body_rt_info.find("PluginShapesOverride");
    if (plugin_shapes == body_rt_info.end()) {
        throw ngraph_error("InsertLoops requires PluginShapesOverride rt_info field");
    } else {
        const auto& new_shapes = plugin_shapes->second.as<std::vector<std::vector<size_t>>>();
        if (new_shapes.size() != commonResults.size() + commonParams.size())
            throw ngraph_error("InsertLoops got invalid number of plugin-overriden shapes");
        for (int i = 0; i < commonParams.size(); i++)
            ioShapes.emplace_back(new_shapes[i]);
        // reverse overriden_shapes for results since commonResults are reversed with respect to model->get_parameters()
        for (int i = 0; i < commonResults.size(); i++)
            ioShapes.emplace_back(new_shapes[new_shapes.size() - 1 - i]);
    }

    if (inner_work_amount > 0) {
        if (m_single_loop_body) {
            const auto apply_increments = InsertLoops::calculate_inner_apply_increments(m_master_shape, ioShapes);
            std::vector<int64_t> inner_finalization_offsets(ioShapes.size(), 0);
            if (outer_work_amount > 1) {
                inner_finalization_offsets = InsertLoops::calculate_finalization_offsets(m_master_shape, ioShapes);
            }
            const auto& inner_loop_begin = op::insertLoopBegin(commonParams);
            const auto& inner_loop_end = insertLoopEnd(commonResults, inner_loop_begin, inner_work_amount,
                                                       m_vector_size, apply_increments, inner_finalization_offsets);
            // set internal flag to enable scalar vs vector loop optimizations
            inner_loop_end->has_outer_loop = outer_work_amount > 1;
            // Due to features of topological sort, some Constants (Scalars) may appear right after Parameters in
            // sorted ops (so it's between Parameters and LoopBegin). Consequently, ScalarEmitters would be called
            // outside the Loop, and only the first Loop iteration would yield correct data (assuming the vector reg
            // assigned to scalar will get corrupted inside the loop body). To avoid such cases, we add control dependency
            // on LoopBegin to guarantee that the constants are executed inside the Loop.
            for (const auto& n : model->get_ordered_ops()) {
                if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(n))
                    c->add_control_dependency(inner_loop_begin);
                else if (n == inner_loop_begin)
                    break;
            }

            if (outer_work_amount > 1) {
                std::vector<bool> apply_increments = InsertLoops::calculate_outer_apply_increments(ioShapes);
                const auto& outer_loop_begin = op::insertLoopBegin(commonParams);
                op::insertLoopEnd(commonResults, outer_loop_begin, outer_work_amount, 1lu, apply_increments);
            }
        } else {
            insert_loops_explicitly(ops, m_vector_size);
        }
    }

    return true;
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
