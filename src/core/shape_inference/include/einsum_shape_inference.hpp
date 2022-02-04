// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/einsum.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v7 {

template <class T>
void shape_infer(const Einsum* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    // check that equation has correct format and extract input and output subscripts
    std::vector<std::string> input_subscripts;
    std::string output_subscript;
    Einsum::parse_equation(op->get_equation(), input_subscripts, output_subscript);

    // a number of input subscripts must match with a number of input tensors
    NODE_VALIDATION_CHECK(op,
                          input_subscripts.size() == input_shapes.size(),
                          "Equation must contain a number of subscripts equal to a number of Einsum inputs.");
    NODE_VALIDATION_CHECK(op, output_shapes.size() == 1);

    // create a dictionary with dimension sizes (or ranges in case of dynamic shapes) for each label
    // and check their compatibility in case of repeating labels
    std::unordered_map<std::string, T> label_to_shape;

    for (size_t input_idx = 0; input_idx < input_shapes.size(); ++input_idx) {
        const auto& pshape = input_shapes[input_idx];
        std::vector<std::string> labels;
        labels = Einsum::extract_labels(input_subscripts[input_idx]);

        if (pshape.rank().is_static()) {
            size_t input_rank = pshape.size();
            // check that a rank is greater or equal to a number of labels
            // these numbers are always equal if there is no ellipsis in the subscript
            NODE_VALIDATION_CHECK(op,
                                  input_rank >= labels.size(),
                                  "Input rank must be greater or equal to a number of labels in the "
                                  "corresponding input subscript.");

            for (size_t label_ind = 0, dim_ind = 0; label_ind < labels.size() && dim_ind < input_rank; ++label_ind) {
                auto const& label = labels[label_ind];
                if (label.compare("...") == 0) {
                    size_t num_broadcasted_dims = input_rank - labels.size() + 1;
                    auto current_sub_pshape = T(std::vector<DimType>(pshape.begin() + dim_ind,
                                                                     pshape.begin() + dim_ind + num_broadcasted_dims));
                    if (label_to_shape.find(label) == label_to_shape.end()) {
                        label_to_shape[label] = current_sub_pshape;
                    } else {
                        bool is_broadcast_success = T::broadcast_merge_into(label_to_shape[label],
                                                                            current_sub_pshape,
                                                                            op::AutoBroadcastType::NUMPY);
                        NODE_VALIDATION_CHECK(op,
                                              is_broadcast_success,
                                              "Input dimensions labeled with ellipsis for Einsum "
                                              "must be broadcastable.");
                    }
                    dim_ind += num_broadcasted_dims;
                } else {
                    if (label_to_shape.find(label) == label_to_shape.end()) {
                        label_to_shape[label] = T{pshape[dim_ind]};
                    } else {
                        NODE_VALIDATION_CHECK(op,
                                              label_to_shape[label].compatible(T{pshape[label_ind]}),
                                              "Different input dimensions indicated by the same labels for Einsum "
                                              "must be compatible.");
                        OPENVINO_ASSERT(T::merge_into(label_to_shape[label], T{pshape[dim_ind]}));
                    }
                    ++dim_ind;
                }
            }
        } else {
            for (auto const& label : labels) {
                NODE_VALIDATION_CHECK(op,
                                      label != "...",
                                      "The subscript corresponding to a dynamic rank input must "
                                      "not contain ellipsis.");

                if (label_to_shape.find(label) == label_to_shape.end()) {
                    label_to_shape[label] = ov::PartialShape{Dimension::dynamic()};
                }
            }
        }
    }

    // compute the output shape
    std::vector<std::string> output_labels;
    output_labels = Einsum::extract_labels(output_subscript);
    auto& output_shape = output_shapes[0];

    output_shape.resize(0);

    for (auto const& output_label : output_labels) {
        NODE_VALIDATION_CHECK(op,
                              label_to_shape.find(output_label) != label_to_shape.end(),
                              "Label in output subscript of Einsum equation must enter at least "
                              "one input subscript.");
        output_shape.insert(output_shape.end(),
                            label_to_shape[output_label].begin(),
                            label_to_shape[output_label].end());
    }
}
}  // namespace v7
}  // namespace op
}  // namespace ov
