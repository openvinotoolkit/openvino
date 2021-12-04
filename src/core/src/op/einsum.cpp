// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/einsum.hpp"

#include <algorithm>
#include <cctype>
#include <ngraph/validation_util.hpp>
#include <string>
#include <unordered_map>

#include "itt.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v7::Einsum);

op::v7::Einsum::Einsum(const OutputVector& inputs, const std::string& equation) : Op(inputs), m_equation(equation) {
    // normalize input equation by removing extra white-spaces from the equation
    m_equation.erase(std::remove_if(m_equation.begin(), m_equation.end(), ::isspace), m_equation.end());

    constructor_validate_and_infer_types();
}

void op::v7::Einsum::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v7_Einsum_validate_and_infer_types);

    // check that Einsum operation has at least one input
    auto num_inputs = get_input_size();
    NODE_VALIDATION_CHECK(this, num_inputs > 0, "Einsum must have at least one input.");

    // check that all inputs have the same type and the type is numeric
    const auto& input_type_0 = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_type_0.is_real() || input_type_0.is_integral_number(),
                          "The input type for Einsum operation must be numeric.");
    for (size_t input_idx = 1; input_idx < num_inputs; ++input_idx) {
        const auto& input_type_i = get_input_element_type(input_idx);
        NODE_VALIDATION_CHECK(this,
                              input_type_0 == input_type_i,
                              "Inputs to Einsum operation must have the same type.");
    }

    // check that equation has correct format and extract input and output subscripts
    std::vector<std::string> input_subscripts;
    std::string output_subscript;
    parse_equation(m_equation, input_subscripts, output_subscript);

    // a number of input subscripts must match with a number of input tensors
    NODE_VALIDATION_CHECK(this,
                          input_subscripts.size() == num_inputs,
                          "Equation must contain a number of subscripts equal to a number of Einsum inputs.");

    // create a dictionary with dimension sizes (or ranges in case dynamic shapes) for each label
    // and check their compatibility in case repeating labels
    unordered_map<string, ov::PartialShape> label_to_shape;
    label_to_shape.clear();

    for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
        const auto& pshape = get_input_partial_shape(input_idx);
        std::vector<std::string> labels;
        labels = extract_labels(input_subscripts[input_idx]);

        if (pshape.rank().is_static()) {
            size_t input_rank = pshape.rank().get_length();
            // check that a rank is greater or equal to a number of labels
            // these numbers are always equal if there is no ellipsis in the subscript
            NODE_VALIDATION_CHECK(this,
                                  input_rank >= labels.size(),
                                  "Input rank must be greater or equal to a number of labels in the "
                                  "corresponding input subscript.");

            for (size_t label_ind = 0, dim_ind = 0; label_ind < labels.size() && dim_ind < input_rank; ++label_ind) {
                auto const& label = labels[label_ind];
                if (label.compare("...") == 0) {
                    size_t num_broadcasted_dims = input_rank - labels.size() + 1;
                    auto current_sub_pshape =
                        ov::PartialShape(std::vector<Dimension>(pshape.begin() + dim_ind,
                                                                pshape.begin() + dim_ind + num_broadcasted_dims));
                    if (label_to_shape.find(label) == label_to_shape.end()) {
                        label_to_shape[label] = current_sub_pshape;
                    } else {
                        bool is_broadcast_success =
                            ov::PartialShape::broadcast_merge_into(label_to_shape[label],
                                                                   current_sub_pshape,
                                                                   op::AutoBroadcastType::NUMPY);
                        NODE_VALIDATION_CHECK(this,
                                              is_broadcast_success,
                                              "Input dimensions labeled with ellipsis for Einsum "
                                              "must be broadcastable.");
                    }
                    dim_ind += num_broadcasted_dims;
                } else {
                    if (label_to_shape.find(label) == label_to_shape.end()) {
                        label_to_shape[label] = ov::PartialShape{pshape[dim_ind]};
                    } else {
                        NODE_VALIDATION_CHECK(this,
                                              label_to_shape[label].compatible(ov::PartialShape{pshape[label_ind]}),
                                              "Different input dimensions indicated by the same labels for Einsum "
                                              "must be compatible.");
                        ov::PartialShape::merge_into(label_to_shape[label], ov::PartialShape{pshape[dim_ind]});
                    }
                    ++dim_ind;
                }
            }
        } else {
            for (auto const& label : labels) {
                NODE_VALIDATION_CHECK(this,
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
    output_labels = extract_labels(output_subscript);
    std::vector<Dimension> output_pshape_vector;

    for (auto const& output_label : output_labels) {
        NODE_VALIDATION_CHECK(this,
                              label_to_shape.find(output_label) != label_to_shape.end(),
                              "Label in output subscript of Einsum equation must enter at least "
                              "one input subscript.");
        output_pshape_vector.insert(output_pshape_vector.end(),
                                    label_to_shape[output_label].begin(),
                                    label_to_shape[output_label].end());
    }
    set_output_type(0, input_type_0, ov::PartialShape(output_pshape_vector));
}

bool op::v7::Einsum::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v7_Einsum_visit_attributes);
    visitor.on_attribute("equation", m_equation);
    return true;
}

shared_ptr<Node> op::v7::Einsum::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v7_Einsum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v7::Einsum>(new_args, m_equation);
}
