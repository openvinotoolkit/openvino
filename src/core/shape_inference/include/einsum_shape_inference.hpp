// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/einsum.hpp"
#include "utils.hpp"
namespace ov {
namespace op {
namespace v7 {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Einsum* op, const std::vector<T>& input_shapes) {
    using DimType = typename T::value_type;

    // check that equation has correct format and extract input and output subscripts
    std::vector<std::string> input_subscripts;
    std::string output_subscript;
    Einsum::parse_equation(op->get_equation(), input_subscripts, output_subscript);

    // a number of input subscripts must match with a number of input tensors
    NODE_VALIDATION_CHECK(op,
                          input_subscripts.size() == input_shapes.size(),
                          "Equation must contain a number of subscripts equal to a number of Einsum inputs.");

    const auto output_labels = Einsum::extract_labels(output_subscript);
    const auto has_out_ellipsis = std::any_of(output_labels.begin(), output_labels.end(), [](std::string label) {
        return label == "...";
    });
    // create a dictionary with dimension sizes (or ranges in case of dynamic shapes) for each label
    // and check their compatibility in case of repeating labels
    std::unordered_map<std::string, TRShape> label_to_shape;
    for (size_t input_idx = 0; input_idx < input_shapes.size(); ++input_idx) {
        const auto& pshape = input_shapes[input_idx];
        const auto labels = Einsum::extract_labels(input_subscripts[input_idx]);
        const auto has_ellipsis = std::any_of(labels.begin(), labels.end(), [](std::string label) {
            return label == "...";
        });

        if (pshape.rank().is_static()) {
            size_t input_rank = pshape.size();
            // check that a rank is greater or equal to a number of labels
            // these numbers are always equal if there is no ellipsis in the subscript
            NODE_VALIDATION_CHECK(
                op,
                (input_rank >= (labels.size() - 1) && has_ellipsis) || (input_rank == labels.size() && !has_ellipsis),
                "Input rank must be greater or equal to a number of labels in the "
                "corresponding input subscript.");
            std::unordered_map<std::string, TRShape> single_input_label_to_shape;
            for (size_t label_ind = 0, dim_ind = 0; label_ind < labels.size() && dim_ind < input_rank; ++label_ind) {
                auto const& label = labels[label_ind];
                if (label.compare("...") == 0) {
                    size_t num_broadcasted_dims = input_rank - labels.size() + 1;
                    auto current_sub_pshape = T(std::vector<DimType>(pshape.begin() + dim_ind,
                                                                     pshape.begin() + dim_ind + num_broadcasted_dims));
                    if (label_to_shape.find(label) == label_to_shape.end()) {
                        label_to_shape[label] = std::move(current_sub_pshape);
                    } else {
                        bool is_broadcast_success = TRShape::broadcast_merge_into(label_to_shape[label],
                                                                                  current_sub_pshape,
                                                                                  op::AutoBroadcastType::NUMPY);
                        NODE_VALIDATION_CHECK(op,
                                              is_broadcast_success,
                                              "Input dimensions labeled with ellipsis for Einsum "
                                              "must be broadcastable.");
                    }
                    dim_ind += num_broadcasted_dims;
                } else {
                    // Check if repeated label dimension in single input are compatible
                    if (single_input_label_to_shape.find(label) == single_input_label_to_shape.end()) {
                        single_input_label_to_shape[label] = TRShape{pshape[dim_ind]};
                    } else {
                        NODE_VALIDATION_CHECK(
                            op,
                            TRShape::merge_into(single_input_label_to_shape[label], TRShape{pshape[dim_ind]}),
                            "Different input dimensions indicated by the repeated labels within single input node for "
                            "Einsum "
                            "must be compatible.");
                    }
                    ++dim_ind;
                }
            }
            for (auto label_it : single_input_label_to_shape) {
                // Repeated labels in single input node are eliminated, so we can merge them into the global
                // label_to_shape.
                if (label_to_shape.find(label_it.first) == label_to_shape.end()) {
                    label_to_shape[label_it.first] = label_it.second;
                } else {
                    NODE_VALIDATION_CHECK(op,
                                          TRShape::broadcast_merge_into(label_to_shape[label_it.first],
                                                                        label_it.second,
                                                                        op::AutoBroadcastType::NUMPY),
                                          "Different input dimensions indicated by the same labels for Einsum "
                                          "must be broadcastable.");
                }
            }
        } else {
            if (has_ellipsis && has_out_ellipsis) {
                // Shape has dynamic rank and ellipsis
                return {pshape};
            }
            for (auto const& label : labels) {
                if (label_to_shape.find(label) == label_to_shape.end()) {
                    label_to_shape[label] = ov::PartialShape{Dimension::dynamic()};
                }
            }
        }
    }

    // compute the output shape
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];

    for (auto const& output_label : output_labels) {
        if (output_label == "..." && label_to_shape.find(output_label) == label_to_shape.end()) {
            // Output labels may contain ellipsis that does not cover any dimensions.
            continue;
        }
        NODE_VALIDATION_CHECK(op,
                              label_to_shape.find(output_label) != label_to_shape.end(),
                              "Label in output subscript of Einsum equation must enter at least "
                              "one input subscript.");
        output_shape.insert(output_shape.end(),
                            label_to_shape[output_label].begin(),
                            label_to_shape[output_label].end());
    }
    return output_shapes;
}
}  // namespace v7
}  // namespace op
}  // namespace ov
