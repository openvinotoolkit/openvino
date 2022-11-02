// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/einsum.hpp"

#include <algorithm>
#include <cctype>
#include <ngraph/validation_util.hpp>
#include <string>
#include <unordered_map>

#include "einsum_shape_inference.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v7::Einsum);

op::v7::Einsum::Einsum(const OutputVector& inputs, const std::string& equation) : Op(inputs), m_equation(equation) {
    // normalize input equation by removing extra white-spaces from the equation
    m_equation.erase(std::remove_if(m_equation.begin(), m_equation.end(), ::isspace), m_equation.end());

    constructor_validate_and_infer_types();
}

namespace {

/// \brief      Check that a subscript contains only alphabetic letters or
/// alphabetic letters with one ellipsis
///
/// \param      subscripts          A subscript to check its format
///
/// \param      is_ellipsis_met     Marker if ellipsis is met in the subscript
///
/// \return     true - correct subscript, false - otherwise
///
bool is_subscript_correct(const std::string& subscript, bool& is_ellipsis_met) {
    is_ellipsis_met = false;
    auto subscript_length = subscript.length();
    for (size_t ch_idx = 0; ch_idx < subscript_length; ++ch_idx) {
        if (is_ellipsis_met == false && ((subscript_length - ch_idx) > 2) &&
            (subscript.substr(ch_idx, 3).compare("...") == 0)) {
            // mark that ellipsis is met once
            is_ellipsis_met = true;

            // make additional increment since ellipsis consists of three dots.
            ch_idx += 2;
        } else if (std::isalpha(subscript[ch_idx]) == 0) {
            return false;
        }
    }

    return true;
}

/// \brief      Check if the given label is met in input subscripts excluding ones
/// specified by a vector excluded_indices
///
/// \param      input_subscripts         The vector of the input subscripts
/// \param      label_to_check           A label to check
/// \param      excluded_indices         A vector of input subscript indices to be excluded
///
/// \return     true - met, false - otherwise
///
bool is_label_elsewhere(const std::vector<std::string>& input_subscripts,
                        const std::string& label_to_check,
                        const std::vector<size_t>& excluded_indices) {
    for (size_t input_ind = 0; input_ind < input_subscripts.size(); ++input_ind) {
        const auto& input_subscript = input_subscripts[input_ind];
        // the subscript is checked only if its index is not in excluded indices list
        bool check_subscript =
            (std::find(excluded_indices.begin(), excluded_indices.end(), input_ind) == excluded_indices.end());
        if (check_subscript && input_subscript.find(label_to_check) != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace

void op::v7::Einsum::parse_equation(const std::string& equation,
                                    std::vector<std::string>& input_subscripts,
                                    std::string& output_subscript) {
    OV_OP_SCOPE(v7_Einsum_parse_equation);
    constexpr char ellipsis[] = "...";

    // split equation to input subscripts and an output subscript
    auto pos_output_delimeter = equation.find("->");
    auto input_subscripts_str = equation.substr(0, pos_output_delimeter);

    // split the input subscripts into a vector of input subscripts
    bool is_ellipsis_met = false;
    input_subscripts.clear();
    std::istringstream input;
    input.str(input_subscripts_str);
    for (std::string input_subscript; std::getline(input, input_subscript, ',');) {
        bool local_is_ellipsis_met = false;
        // check that input subscript contains only alphabetic letter or ellipsis
        NGRAPH_CHECK(is_subscript_correct(input_subscript, local_is_ellipsis_met),
                     "Input subscript of Einsum equation must consist of either only "
                     "alphabetic letters or alphabetic letters with one ellipsis.");

        // mark that ellipsis is met at least in one input subscript
        if (local_is_ellipsis_met) {
            is_ellipsis_met = true;
        }
        input_subscripts.push_back(input_subscript);
    }

    if (pos_output_delimeter == std::string::npos) {
        // equation is in implicit mode so recover output subscript
        output_subscript = "";
        for (size_t ind = 0; ind < input_subscripts.size(); ++ind) {
            auto const& input_subscript = input_subscripts[ind];
            for (auto const& label : extract_labels(input_subscript)) {
                if (label != ellipsis && (is_label_elsewhere(input_subscripts, label, {ind}) == false)) {
                    output_subscript += label;
                }
            }
        }
        std::sort(output_subscript.begin(), output_subscript.end());
        if (is_ellipsis_met) {
            output_subscript = "..." + output_subscript;
        }
    } else {
        output_subscript = equation.substr(pos_output_delimeter + 2);
        bool output_is_ellipsis_met = false;

        // check that the output subscript has the correct format
        NGRAPH_CHECK(is_subscript_correct(output_subscript, output_is_ellipsis_met),
                     "Output subscript of Einsum equation must consist of either only "
                     "alphabetic letters or alphabetic letters with one ellipsis.");

        // if the ellipsis is met in input subscripts, one ellipsis must be in the output subscript
        NGRAPH_CHECK(is_ellipsis_met == output_is_ellipsis_met,
                     "Output subscript of Einsum equation must contain one ellipsis if "
                     "ellipsis is met in any input subscript.");
    }
}

std::vector<std::string> op::v7::Einsum::extract_labels(const std::string& subscript) {
    OV_OP_SCOPE(v7_Einsum_extract_labels);

    std::vector<std::string> labels;
    labels.clear();
    auto subscript_length = subscript.length();
    for (size_t ch_idx = 0; ch_idx < subscript_length; ++ch_idx) {
        if (std::isalpha(subscript[ch_idx])) {
            labels.push_back(subscript.substr(ch_idx, 1));
        } else if (((subscript_length - ch_idx) > 2) && (subscript.substr(ch_idx, 3).compare("...") == 0)) {
            labels.push_back("...");
            // make additional increment since ellipsis consists of three dots.
            ch_idx += 2;
        } else {
            NGRAPH_CHECK(false, "Einsum equation has invalid label.");
        }
    }

    return labels;
}

void op::v7::Einsum::validate_and_infer_types() {
    OV_OP_SCOPE(v7_Einsum_validate_and_infer_types);

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

    std::vector<ov::PartialShape> input_shapes;
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape::dynamic()};
    for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
        input_shapes.push_back(get_input_partial_shape(input_idx));
    }

    shape_infer(this, input_shapes, output_shapes);

    set_output_type(0, input_type_0, output_shapes[0]);
}

bool op::v7::Einsum::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v7_Einsum_visit_attributes);
    visitor.on_attribute("equation", m_equation);
    return true;
}

shared_ptr<Node> op::v7::Einsum::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v7_Einsum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v7::Einsum>(new_args, m_equation);
}
