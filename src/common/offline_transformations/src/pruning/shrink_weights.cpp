// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/log.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "mask_attribute.hpp"
#include "pruning.hpp"

template <typename T>
static std::string vec_to_str(const std::vector<T> m) {
    std::ostringstream out;
    out << "[ ";
    for (const auto& val : m)
        out << val << ' ';
    out << "]";
    return out.str();
}

static bool not_empty_mask(ngraph::Mask::Ptr mask) {
    return mask && !mask->all_dims_are_empty();
}

static bool is_static_reshape_op(std::shared_ptr<ov::Node> node) {
    auto reshape_node = std::dynamic_pointer_cast<ngraph::opset6::Reshape>(node);
    if (!reshape_node)
        return false;

    const auto input = reshape_node->input_value(0);
    const auto shape = reshape_node->input_value(1);
    if (input.get_partial_shape().is_dynamic() || shape.get_partial_shape().is_dynamic())
        return false;

    const auto output_shape_const_op = get_constant_from_source(shape);
    if (!output_shape_const_op)
        return false;

    const auto input_shape = input.get_shape();
    const auto output_shape = output_shape_const_op->cast_vector<int64_t>();
    const auto input_elems = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
    const auto output_elems = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
    if (output_elems <= 0 || input_elems == output_elems)
        return false;
    return true;
}

static bool maybe_adopt_reshape_node(std::shared_ptr<ov::Node> reshape, ngraph::Mask::Ptr mask) {
    const auto shape = reshape->input_value(1);
    const auto consumers = shape.get_node()->get_output_target_inputs(0);
    if (shape.get_node()->outputs().size() != 1 || consumers.size() != 1) {
        NGRAPH_DEBUG << "Adoptation for node " << shape.get_node()->get_friendly_name() << " is not supported.";
        return false;
    }

    auto sub_const_vector = std::vector<int64_t>();
    for (auto& dim : *mask.get())
        sub_const_vector.push_back(dim.size());

    const auto sub_const = ngraph::opset6::Constant::create(shape.get_element_type(), {mask->size()}, sub_const_vector);
    const auto sub = std::make_shared<ngraph::opset6::Subtract>(shape, sub_const);
    consumers.begin()->replace_source_output(sub);

    NGRAPH_DEBUG << "Adopting values in (" << shape.get_node()->get_friendly_name() << ")"
                 << " by substracting " << vec_to_str(sub_const_vector);
    return true;
}

bool ngraph::pass::ShrinkWeights::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    int64_t reduced_weights_count{0};
    int64_t total_weights_count{0};
    for (const auto& node : f->get_ordered_ops()) {
        // calculate shape for every node in graph as the input shape may change
        // during Constant shrinking
        auto mask = getMask(node->output(0));

#ifdef ENABLE_OPENVINO_DEBUG
        auto init_mask = getInitMask(node->output(0));
        if (!mask && init_mask)
            NGRAPH_DEBUG << "Mask was ruined for node:" << node->get_friendly_name() << "\nInit mask: " << *init_mask;
#endif
        if (is_static_reshape_op(node) && not_empty_mask(mask))
            if (!maybe_adopt_reshape_node(node, mask))
                continue;

        node->revalidate_and_infer_types();

        if (!mask)
            continue;

        // TODO: constant can be shared across functions so we need to avoid consumers from other function
        auto const_node = std::dynamic_pointer_cast<opset6::Constant>(node);
        if (!const_node)
            continue;

        const auto& const_shape = const_node->get_shape();
        total_weights_count += shape_size(const_shape);

#ifdef ENABLE_OPENVINO_DEBUG
        if (init_mask) {
            for (size_t dim = 0; dim < init_mask->size(); ++dim) {
                auto& dim_init_set = (*init_mask)[dim];
                auto& dim_current_set = (*mask)[dim];
                if (!dim_init_set.empty() && !std::includes(dim_current_set.begin(),
                                                            dim_current_set.end(),
                                                            dim_init_set.begin(),
                                                            dim_init_set.end())) {
                    NGRAPH_DEBUG << "Mask was ruined for node:" << const_node->get_friendly_name()
                                 << "\nInit mask: " << *init_mask << "\nCurrent mask: " << *mask;
                    break;
                }
            }
        }
#endif

        // Case mask should adjust value from constant instead of constant pruning
        if (mask->adjust_value() && !mask->all_dims_are_empty()) {
            std::vector<int64_t> new_const_value;
            auto value = const_node->cast_vector<int64_t>();
            for (size_t i = 0; i < mask->size(); i++) {
                const int64_t res = value[i] - mask->at(i).size();
                new_const_value.push_back((res > 0) ? res : value[i]);
            }

            const auto new_const =
                opset6::Constant::create(const_node->get_element_type(), const_node->get_shape(), new_const_value);
            new_const->set_friendly_name(const_node->get_friendly_name());
            ngraph::copy_runtime_info(const_node, new_const);
            ngraph::replace_node(const_node, new_const);

            NGRAPH_DEBUG << "Adjust value in (" << const_node->get_friendly_name() << "): " << vec_to_str(value)
                         << " to " << vec_to_str(new_const_value);
            continue;
        }
        auto last_output = const_node->output(0);
        auto consumers = last_output.get_target_inputs();

        if (mask->is_shape_like()) {
            // TODO: think about it
            auto res = const_node->get_shape_val();
            if (res.size() != mask->size()) {
                throw ngraph_error("Mask size (" + std::to_string(mask->size()) + ") is not equal to (" +
                                   std::to_string(res.size()) + ")");
            }
            for (size_t dim = 0; dim < mask->size(); ++dim) {
                res[dim] -= mask->at(dim).size();
            }
            auto new_const = opset6::Constant::create(const_node->get_element_type(), Shape{res.size()}, res);
            replace_node(const_node, new_const);
            NGRAPH_DEBUG << "Transform shape like (" << last_output.get_node()->get_friendly_name()
                         << "): " << const_node->get_shape_val() << " to " << new_const->get_shape_val() << std::endl;
            new_const->set_friendly_name(const_node->get_friendly_name());
        } else {
            for (size_t dim = 0; dim < mask->size(); ++dim) {
                const auto& dim_size = mask->at(dim).size();
                if (dim_size == 0)
                    continue;
                // Broadcastable 1-size dimension shouldn't be shrank with mask
                if (const_node->get_shape().at(dim) == 1 && dim_size > 1)
                    continue;

                // Convert dims that we want remove to dims that we need to keep
                std::vector<int64_t> dims_to_keep;
                for (size_t dim_value = 0; dim_value < const_shape[dim]; ++dim_value) {
                    if (!mask->at(dim).count(dim_value)) {
                        dims_to_keep.emplace_back(dim_value);
                    }
                }

                const auto& prev_shape = last_output.get_partial_shape();
                const auto& prev_name = last_output.get_node()->get_friendly_name();
                last_output = std::make_shared<opset6::Gather>(
                    last_output,
                    opset6::Constant::create(element::i64, Shape{dims_to_keep.size()}, dims_to_keep),
                    opset6::Constant::create(element::i64, Shape{}, {dim}));
                NGRAPH_DEBUG << "Transform(" << prev_name << "): " << prev_shape << " to "
                             << last_output.get_partial_shape();

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
