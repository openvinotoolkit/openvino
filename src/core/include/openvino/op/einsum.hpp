// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v7 {
/// \brief Einsum operation.
class OPENVINO_API Einsum : public Op {
public:
    OPENVINO_OP("Einsum", "opset7", op::Op, 7);
    BWDCMP_RTTI_DECLARATION;

    Einsum() = default;

    ///
    /// \brief      Constructs Einsum operation.
    ///
    /// \param      inputs        Input nodes on which Einsum operation performs
    /// contraction
    ///
    /// \param      equation      Einstein summation convention
    ///
    Einsum(const OutputVector& inputs, const std::string& equation);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \brief      Get an equation of Einsum operation
    ///
    /// \return     Einsum equation
    ///
    std::string get_equation() const {
        return m_equation;
    }

    /// \brief      Check correctness of equation format and extract input subscripts
    /// and output subscript
    ///
    /// \param      equation              Equation to be parsed and checked
    ///
    /// \param      input_subscripts      A vector of extracted input subscripts
    ///
    /// \param      output_subscript      An output subscript
    ///
    static void parse_equation(const std::string& equation,
                               std::vector<std::string>& input_subscripts,
                               std::string& output_subscript) {
        auto is_subscript_correct = [](const std::string& subscript, bool& is_ellipsis_met) {
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
        };

        auto is_label_elsewhere = [](const std::vector<std::string>& input_subscripts,
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
        };
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
            OPENVINO_ASSERT(is_subscript_correct(input_subscript, local_is_ellipsis_met),
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
            OPENVINO_ASSERT(is_subscript_correct(output_subscript, output_is_ellipsis_met),
                            "Output subscript of Einsum equation must consist of either only "
                            "alphabetic letters or alphabetic letters with one ellipsis.");

            // if the ellipsis is met in input subscripts, one ellipsis must be in the output subscript
            OPENVINO_ASSERT(is_ellipsis_met == output_is_ellipsis_met,
                            "Output subscript of Einsum equation must contain one ellipsis if "
                            "ellipsis is met in any input subscript.");
        }
    }

    /// \brief      Extract labels (from subscript) that can be alphabetic letters or
    /// ellipsis
    ///
    /// \param      subscript      Subscript
    ///
    /// \return     A vector of extracted labels from the input subscript in the order
    /// of appearence
    ///
    static std::vector<std::string> extract_labels(const std::string& subscript) {
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
                OPENVINO_ASSERT(false, "Einsum equation has invalid label.");
            }
        }

        return labels;
    }

private:
    std::string m_equation;
};
}  // namespace v7
}  // namespace op
}  // namespace ov
