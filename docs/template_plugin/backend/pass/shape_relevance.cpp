// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/shape_relevance.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"

using namespace ngraph;

//
// This pass refreshes the "is_relevant_to_shape" flag on each parameter. A parameter will be
// flagged as relevant to shapes if there is any path from that parameter to a shape-relevant
// input that does _not_ pass through a value-irrelevant input. For example:
//
//      N0[Parameter] N1[Parameter]
//       |            |
//       |            |
//       |            |
//       N2[v1::Reshape]
//
// N1 (but not N0) will be flagged as shape-relevant, because N1 feeds into the "shape" input
// of N2.
//
//      N0[Parameter] N1[Parameter]
//       |            |
//       |            N2[ShapeOf]
//       |            |
//       N3[v1::Reshape]
//
// Neither N0 nor N1 will be flagged as shape-relevant. (N1 does feed into the "shape" input of N3,
// but only via the value-irrelevant input of ShapeOf.)
//
bool pass::ShapeRelevance::run_on_model(const std::shared_ptr<Function>& f) {
    // TODO(amprocte): We are probably reinventing the wheel with the graph traversal here; the
    // reason is that we need to cut the traversal short in cases where input values are
    // irrelevant. See if there is a way to reduce this duplication.

    // Set of nodes that must be evaluated to determine the value of shape-relevant inputs.
    std::set<Node*> shape_determinants;

    // Step 1: Find root nodes (these are nodes with an output connected to a shape-relevant
    // input).
    for (auto& n : f->get_ops()) {
        for (auto& output : n->outputs()) {
            for (auto& input : output.get_target_inputs()) {
                if (input.get_is_relevant_to_shapes()) {
                    shape_determinants.insert(n.get());
                    break;
                }
            }
        }
    }

    // Step 2: Find all shape determinants. This is the transitive closure of R, where n1 R n2
    // iff there is a data flow edge from n2 to n1 and that data flow edge is not
    // value-irrelevant.
    bool changes_made = false;

    {
        std::list<Node*> to_visit{shape_determinants.begin(), shape_determinants.end()};
        std::set<Node*> already_visited;

        while (!to_visit.empty()) {
            auto node = to_visit.front();
            to_visit.pop_front();

            if (already_visited.count(node) > 0) {
                continue;
            }

            shape_determinants.insert(node);
            already_visited.insert(node);

            if (op::is_parameter(node)) {
                auto node_as_param = static_cast<op::Parameter*>(node);
                if (!node_as_param->is_relevant_to_shapes()) {
                    node_as_param->set_is_relevant_to_shapes(true);
                    changes_made = true;
                }
            }

            for (size_t i = 0; i < node->get_input_size(); i++) {
                if (!node->input(i).get_is_relevant_to_values()) {
                    continue;
                }
                auto source_node = node->get_input_node_ptr(i);
                if (already_visited.count(source_node) == 0) {
                    to_visit.push_front(source_node);
                }
            }
        }
    }

    return changes_made;
}
