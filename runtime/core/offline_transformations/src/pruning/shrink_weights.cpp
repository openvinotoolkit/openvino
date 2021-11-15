// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "pruning.hpp"
#include "mask_attribute.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/log.hpp>
#include <ngraph/ngraph.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ShrinkWeights, "ShrinkWeights", 0);

bool ngraph::pass::ShrinkWeights::run_on_function(std::shared_ptr<ngraph::Function> f) {
    int64_t reduced_weights_count{0};
    int64_t total_weights_count{0};
    for (const auto & node : f->get_ordered_ops()) {
        // calculate shape for every node in graph as the input shape may change
        // during Constant shrinking
        node->validate_and_infer_types();

        // TODO: constant can be shared across functions so we need to avoid consumers from other function
        auto const_node = std::dynamic_pointer_cast<opset6::Constant>(node);
        if (!const_node) continue;

        const auto & const_shape = const_node->get_shape();
        total_weights_count += shape_size(const_shape);

        auto mask = getMask(const_node->output(0));
        if (!mask) continue;

        auto last_output = const_node->output(0);
        auto consumers = last_output.get_target_inputs();

        if (mask->is_shape_like()) {
            // TODO: think about it
            auto res = const_node->get_shape_val();
            if (res.size() != mask->size()) {
                throw ngraph_error("Mask size (" + std::to_string(mask->size()) + ") is not equal to (" + std::to_string(res.size()) + ")");
            }
            for (size_t dim = 0; dim < mask->size(); ++dim) {
                res[dim] -= mask->at(dim).size();
            }
            auto new_const = opset6::Constant::create(const_node->get_element_type(), Shape{res.size()}, res);
            replace_node(const_node, new_const);
            NGRAPH_DEBUG << "Transform shape like (" << last_output.get_node()->get_friendly_name() << "): "
                         << const_node->get_shape_val() << " to " << new_const->get_shape_val() << std::endl;
            new_const->set_friendly_name(const_node->get_friendly_name());
        } else {
            for (size_t dim = 0; dim < mask->size(); ++dim) {
                const auto &dim_size = mask->at(dim).size();
                if (dim_size == 0) continue;
                // Broadcastable 1-size dimension shouldn't be shrank with mask
                if (const_node->get_shape().at(dim) == 1 && dim_size > 1) continue;

                // Convert dims that we want remove to dims that we need to keep
                std::vector<int64_t> dims_to_keep;
                for (size_t dim_value = 0; dim_value < const_shape[dim]; ++dim_value) {
                    if (!mask->at(dim).count(dim_value)) {
                        dims_to_keep.emplace_back(dim_value);
                    }
                }

                const auto & prev_shape = last_output.get_partial_shape();
                const auto & prev_name = last_output.get_node()->get_friendly_name();
                last_output = std::make_shared<opset6::Gather>(last_output,
                                                               opset6::Constant::create(element::i64, Shape{dims_to_keep.size()}, dims_to_keep),
                                                               opset6::Constant::create(element::i64, Shape{}, {dim}));
                NGRAPH_DEBUG << "Transform(" << prev_name << "): " << prev_shape << " to " << last_output.get_partial_shape();

                if (prev_shape.is_static() && last_output.get_partial_shape().is_static()) {
                    reduced_weights_count += shape_size(prev_shape.get_shape()) - shape_size(last_output.get_shape());
                } else {
                    NGRAPH_DEBUG << "[ WARNING ] Can not find the number of reduced elements due to dynamic shapes.";
                }
            }
            // Trying to fold sequence of Gather ops to avoid additional constant folding.
            if (auto folded_const = ngraph::get_constant_from_source(last_output)) {
                last_output = folded_const;
            }
            // as we insert Gather operations after Constant we need to reconnect all
            // Constant consumers to the latest Gather.
            for (auto consumer : consumers) {
                consumer.replace_source_output(last_output);
            }
        }
    }
    NGRAPH_DEBUG << "[ INFO ]   TOTAL WEIGHTS: " << total_weights_count << std::endl;
    NGRAPH_DEBUG << "[ INFO ] REDUCED WEIGHTS: " << reduced_weights_count << std::endl;
    return true;
}