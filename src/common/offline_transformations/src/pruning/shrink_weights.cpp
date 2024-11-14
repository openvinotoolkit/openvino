// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "mask_attribute.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
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

static bool not_empty_mask(ov::Mask::Ptr mask) {
    return mask && !mask->all_dims_are_empty();
}

static bool is_static_reshape_op(std::shared_ptr<ov::Node> node) {
    auto reshape_node = std::dynamic_pointer_cast<ov::opset6::Reshape>(node);
    if (!reshape_node)
        return false;

    const auto input = reshape_node->input_value(0);
    const auto shape = reshape_node->input_value(1);

    if (input.get_partial_shape().is_dynamic() || shape.get_partial_shape().is_dynamic())
        return false;

    const auto output_shape_const_op = ov::util::get_constant_from_source(shape);
    if (!output_shape_const_op)
        return false;

    const auto& input_shape = input.get_shape();
    const auto& output_shape = output_shape_const_op->cast_vector<int64_t>();
    // below casts are needed due to VC warning C4244, literals are not enough in this case
    const int64_t input_elems =
        std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    const auto output_elems =
        std::accumulate(output_shape.begin(), output_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    return input_elems != output_elems;
}

static bool maybe_adopt_reshape_node(std::shared_ptr<ov::Node> reshape, ov::Mask::Ptr mask) {
    const auto shape = reshape->input_value(1);
    const auto consumers = shape.get_node()->get_output_target_inputs(0);
    if (shape.get_node()->outputs().size() != 1 || consumers.size() != 1) {
        OPENVINO_DEBUG("Adoptation for node ", shape.get_node()->get_friendly_name(), " is not supported.");
        return false;
    }

    const auto constant = ov::util::get_constant_from_source(shape);
    if (!constant) {
        return false;
    }
    const auto new_shape = constant->cast_vector<int64_t>();
    std::vector<int64_t> sub_const_vector;
    sub_const_vector.reserve(mask->size());
    bool all_zeros = true;
    for (size_t i = 0; i < mask->size(); i++) {
        if (new_shape[i] <= 0) {
            sub_const_vector.push_back(0);
        } else {
            all_zeros = all_zeros && mask->at(i).size() == 0;
            sub_const_vector.push_back(mask->at(i).size());
        }
    }

    if (all_zeros)
        return true;

    const auto sub_const = ov::opset6::Constant::create(shape.get_element_type(), {mask->size()}, sub_const_vector);
    const auto sub = std::make_shared<ov::opset6::Subtract>(shape, sub_const);
    consumers.begin()->replace_source_output(sub);
    copy_runtime_info(shape.get_node_shared_ptr(), {sub_const, sub});

    OPENVINO_DEBUG("Adopting values in (",
                   shape.get_node()->get_friendly_name(),
                   ")"
                   " by substracting ",
                   vec_to_str(sub_const_vector));
    return true;
}

static bool handle_variadic_split(const std::shared_ptr<ov::Node>& split) {
    const auto axis_node = ov::as_type<ov::opset6::Constant>(split->get_input_node_ptr(1));
    if (!axis_node)
        return false;

    const auto& input_shape = split->get_input_partial_shape(0);
    if (input_shape.rank().is_dynamic())
        return false;

    auto axis = axis_node->cast_vector<int64_t>()[0];
    if (axis < 0)
        axis += input_shape.size();

    if (input_shape[axis].is_dynamic())
        return false;

    const auto split_lengths_node = ov::as_type<ov::opset6::Constant>(split->get_input_node_ptr(2));
    if (!split_lengths_node)
        return false;
    const auto split_lengths = split_lengths_node->cast_vector<int64_t>();

    std::vector<int64_t> sub_values;
    bool sub_with_zero = true;

    // adjust split_lengths by size of the set for axis in mask
    for (size_t i = 0; i < split->get_output_size(); i++) {
        auto mask = ov::getMask(split->output(i));
        if (!mask)
            return false;
        auto set_size = mask->at(axis).size();
        if (split_lengths[i] == -1 && set_size > 0) {
            const auto& shape = split->get_output_partial_shape(i);
            sub_values.push_back(-1 - (shape[axis].get_length() - set_size));
        } else {
            sub_values.push_back(set_size);
        }
        sub_with_zero = sub_with_zero && sub_values.back() == 0;
    }

    if (sub_with_zero)
        return true;

    const auto& split_lengths_type = split_lengths_node->get_output_element_type(0);
    const auto sub_const = ov::opset6::Constant::create(split_lengths_type, {sub_values.size()}, sub_values);
    const auto sub = std::make_shared<ov::opset6::Subtract>(split->input_value(2), sub_const);
    copy_runtime_info(split->get_input_source_output(2).get_node_shared_ptr(), {sub_const, sub});
    split->input(2).replace_source_output(sub);

    return true;
}

static std::shared_ptr<ov::Node> handle_split(const std::shared_ptr<ov::Node>& split) {
    const auto axis_node = ov::as_type<ov::opset6::Constant>(split->get_input_node_ptr(1));
    if (!axis_node)
        return nullptr;

    const auto& input_shape = split->get_input_partial_shape(0);
    if (input_shape.rank().is_dynamic())
        return nullptr;

    auto axis = axis_node->cast_vector<int64_t>()[0];
    if (axis < 0)
        axis += input_shape.size();

    if (input_shape[axis].is_dynamic())
        return nullptr;

    std::vector<int64_t> split_lengths;
    bool equal_output_chunks = true;

    // create split_lengths array
    for (size_t i = 0; i < split->get_output_size(); i++) {
        auto mask = ov::getMask(split->output(i));
        if (!mask)
            return nullptr;
        auto set_size = mask->at(axis).size();
        const auto& shape = split->get_output_partial_shape(i);
        split_lengths.push_back(shape[axis].get_length() - set_size);
        equal_output_chunks = equal_output_chunks && split_lengths.back() == split_lengths[0];
    }

    if (equal_output_chunks)
        return split;

    const auto split_lengths_node =
        ov::opset6::Constant::create(ov::element::i64, {split_lengths.size()}, split_lengths);
    auto var_split =
        std::make_shared<ov::opset6::VariadicSplit>(split->input_value(0), split->input_value(1), split_lengths_node);
    var_split->set_friendly_name(split->get_friendly_name());
    ov::copy_runtime_info(split, var_split);
    ov::replace_node(split, var_split);

    return var_split;
}

bool ov::pass::ShrinkWeights::run_on_model(const std::shared_ptr<ov::Model>& f) {
#ifdef ENABLE_OPENVINO_DEBUG
    int64_t reduced_weights_count{0};
    int64_t total_weights_count{0};
#endif
    for (const auto& node : f->get_ordered_ops()) {
        // calculate shape for every node in graph as the input shape may change
        // during Constant shrinking
        auto mask = getMask(node->output(0));

#ifdef ENABLE_OPENVINO_DEBUG
        auto init_mask = getInitMask(node->output(0));
        if (!mask && init_mask)
            OPENVINO_DEBUG("Mask was ruined for node:", node->get_friendly_name(), "\nInit mask: ", *init_mask);
#endif
        if (is_static_reshape_op(node) && not_empty_mask(mask) &&
            !ov::op::util::is_constant(node->get_input_node_ptr(1)))
            if (!maybe_adopt_reshape_node(node, mask))
                continue;

        if (ov::is_type<opset6::VariadicSplit>(node) && !handle_variadic_split(node))
            continue;

        if (ov::is_type<opset6::Split>(node)) {
            auto split = handle_split(node);
            if (split)
                split->revalidate_and_infer_types();
            continue;
        }

        node->revalidate_and_infer_types();

        if (!mask)
            continue;

        // TODO: constant can be shared across functions so we need to avoid consumers from other function
        auto const_node = std::dynamic_pointer_cast<opset6::Constant>(node);
        if (!const_node)
            continue;

        const auto& const_shape = const_node->get_shape();
#ifdef ENABLE_OPENVINO_DEBUG
        total_weights_count += shape_size(const_shape);
#endif

#ifdef ENABLE_OPENVINO_DEBUG
        if (init_mask) {
            for (size_t dim = 0; dim < init_mask->size(); ++dim) {
                auto& dim_init_set = (*init_mask)[dim];
                auto& dim_current_set = (*mask)[dim];
                if (!dim_init_set.empty() && !std::includes(dim_current_set.begin(),
                                                            dim_current_set.end(),
                                                            dim_init_set.begin(),
                                                            dim_init_set.end())) {
                    OPENVINO_DEBUG("Mask was ruined for node: ",
                                   const_node->get_friendly_name(),
                                   "\nInit mask: ",
                                   *init_mask,
                                   "\nCurrent mask: ",
                                   *mask);
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
            ov::copy_runtime_info(const_node, new_const);
            ov::replace_node(const_node, new_const);

            OPENVINO_DEBUG("Adjust value in (",
                           const_node->get_friendly_name(),
                           "): ",
                           vec_to_str(value),
                           " to ",
                           vec_to_str(new_const_value));
            continue;
        }
        auto last_output = const_node->output(0);
        auto consumers = last_output.get_target_inputs();

        if (mask->is_shape_like()) {
            // TODO: think about it
            auto res = const_node->get_shape_val();
            if (res.size() != mask->size()) {
                OPENVINO_THROW("Mask size (" + std::to_string(mask->size()) + ") is not equal to (" +
                               std::to_string(res.size()) + ")");
            }
            for (size_t dim = 0; dim < mask->size(); ++dim) {
                res[dim] -= mask->at(dim).size();
            }
            auto new_const = opset6::Constant::create(const_node->get_element_type(), Shape{res.size()}, res);
            replace_node(const_node, new_const);
            copy_runtime_info(const_node, new_const);
            OPENVINO_DEBUG("Transform shape like (",
                           last_output.get_node()->get_friendly_name(),
                           "): ",
                           const_node->get_shape_val(),
                           " to ",
                           new_const->get_shape_val(),
                           "\n");
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

#ifdef ENABLE_OPENVINO_DEBUG
                const auto& prev_shape = last_output.get_partial_shape();
#endif

                last_output = std::make_shared<opset6::Gather>(
                    last_output,
                    opset6::Constant::create(element::i64, Shape{dims_to_keep.size()}, dims_to_keep),
                    opset6::Constant::create(element::i64, Shape{}, {dim}));
#ifdef ENABLE_OPENVINO_DEBUG
                const auto& prev_name = last_output.get_node()->get_friendly_name();
                OPENVINO_DEBUG("Transform(", prev_name, "): ", prev_shape, " to ", last_output.get_partial_shape());

                if (prev_shape.is_static() && last_output.get_partial_shape().is_static()) {
                    reduced_weights_count += shape_size(prev_shape.get_shape()) - shape_size(last_output.get_shape());
                } else {
                    OPENVINO_DEBUG("[ WARNING ] Can not find the number of reduced elements due to dynamic shapes.");
                }
#endif
            }
            // Trying to fold sequence of Gather ops to avoid additional constant folding.
            if (auto folded_const = ov::util::get_constant_from_source(last_output)) {
                last_output = folded_const;
            }
            // as we insert Gather operations after Constant we need to reconnect all
            // Constant consumers to the latest Gather.
            for (auto consumer : consumers) {
                consumer.replace_source_output(last_output);
            }
            copy_runtime_info(const_node, last_output.get_node_shared_ptr());
        }
    }
#ifdef ENABLE_OPENVINO_DEBUG
    OPENVINO_DEBUG("[ INFO ]   TOTAL WEIGHTS: ", total_weights_count, "\n");
    OPENVINO_DEBUG("[ INFO ] REDUCED WEIGHTS: ", reduced_weights_count, "\n");
#endif
    return true;
}
