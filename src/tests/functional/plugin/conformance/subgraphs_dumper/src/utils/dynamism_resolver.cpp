// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/dynamism_resolver.hpp"

namespace SubgraphsDumper {

bool has_dynamic_output(std::shared_ptr<ngraph::Node> n) {
    for (size_t i = 0; i < n->get_output_size(); i++) {
        if (n->get_output_partial_shape(i).is_dynamic()) {
            return true;
        }
    }
    return false;
}

void resolve_dynamic_shapes(const std::shared_ptr<ngraph::Function>& f) {
    const auto & f_ops = f->get_ordered_ops();
    if (std::all_of(f_ops.begin(), f_ops.end(),
                    [](std::shared_ptr<ngraph::Node> results) {
                        return !results->is_dynamic() && !has_dynamic_output(results); })) {
        return;
    }
    auto f_clone = ngraph::clone_function(*f);
    const auto & f_clone_ops = f_clone->get_ordered_ops();
    NGRAPH_CHECK(f_ops.size() == f_clone_ops.size(), "Unexpected get_ordered_ops method behaviour");

    for (size_t id = 0; id < f_ops.size(); ++id) {
        auto & op = f_ops[id];
        auto & clone_op = f_clone_ops[id];

        if (auto op_subgraph = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(op)) {
            resolve_dynamic_shapes(op_subgraph->get_function());
        }

        op->validate_and_infer_types();
        clone_op->validate_and_infer_types();

        // dynamic_to_static function converts dynamic dimensions to static using
        // upperbound (get_max_length) dimension value.
        auto dynamic_to_static = [&op](const ngraph::PartialShape & shape) -> ngraph::PartialShape {
            if (shape.is_static() || shape.rank().is_dynamic()) {
                return shape;
            }
            std::vector<ngraph::Dimension> out_shape;
            std::transform(std::begin(shape), std::end(shape),
                           std::back_inserter(out_shape),
                           [](const ngraph::Dimension& d) -> ngraph::Dimension {
                               return d.get_max_length();
                           });
            NGRAPH_CHECK(ngraph::PartialShape(out_shape).is_static(),
                         "Dynamic dimension cannot be resolved in ", op);
            return out_shape;
        };

        ngraph::OutputVector replacements(clone_op->get_output_size());
        if (!clone_op->constant_fold(replacements, clone_op->input_values())) {
            for (size_t output_id = 0; output_id < clone_op->get_output_size(); ++output_id) {
                clone_op->set_output_type(output_id, clone_op->output(output_id).get_element_type(),
                                          dynamic_to_static(clone_op->output(output_id).get_partial_shape()));
                op->set_output_type(output_id, clone_op->output(output_id).get_element_type(),
                                    clone_op->output(output_id).get_partial_shape());
            }
        } else {
            for (size_t output_id = 0; output_id < clone_op->get_output_size(); ++output_id) {
                op->set_output_type(output_id, replacements[output_id].get_element_type(),
                                    replacements[output_id].get_partial_shape());
            }

            for (size_t i = 0; i < replacements.size(); ++i) {
                auto node_output = clone_op->output(i);
                auto replacement = replacements.at(i);
                if (replacement.get_node_shared_ptr() && (node_output != replacement)) {
                    node_output.replace(replacement);
                }
            }
        }
    }
}

} // namespace SubgraphsDumper