// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/cre.hpp"

#include <functional>

#define CRE_EVAL_ASSERT(...) \
    OPENVINO_ASSERT_HELPER(::intel_npu::InvalidCRE, ::ov::AssertFailure::default_msg, __VA_ARGS__)

namespace {

const std::unordered_set<intel_npu::CREToken> BINARY_OPERATORS{intel_npu::CRE::AND, intel_npu::CRE::OR};
const std::unordered_set<intel_npu::CREToken> OPERATORS{intel_npu::CRE::AND, intel_npu::CRE::OR, intel_npu::CRE::NOT};

inline bool and_function(bool a, bool b) {
    return a && b;
}

inline bool or_function(bool a, bool b) {
    return a || b;
}

inline bool first_operand_function(bool /*a*/, bool b) {
    return b;
}

}  // namespace

namespace intel_npu {

void InvalidCRE::create(const char* file,
                        int line,
                        const char* check_string,
                        const std::string& context_info,
                        const std::string& explanation) {
    throw InvalidCRE(make_what(file, line, check_string, context_info, explanation));
}

CRE::CRE(const ov::log::Level log_level) : m_logger("CRE", log_level) {}

CRE::CRE(const std::vector<CREToken>& subexpression, const ov::log::Level log_level) : m_logger("CRE", log_level) {
    if (!subexpression.empty()) {
        m_subexpressions.push_back(subexpression);
    }
}

bool CRE::subexpression_already_registered(const std::vector<CREToken>& subexpression) const {
    for (const std::vector<CREToken>& registered_subexpression : m_subexpressions) {
        if (subexpression == registered_subexpression) {
            return true;
        }
    }

    return false;
}

void CRE::append_to_expression(const CREToken requirement_token) {
    OPENVINO_ASSERT(!RESERVED_TOKENS.count(requirement_token),
                    "Appending subexpressions should be done through the \"vector\" API");

    const std::vector<CREToken> subexpression{requirement_token};
    if (subexpression_already_registered(subexpression)) {
        m_logger.trace("CREToken %u was already registered", requirement_token);
        return;
    }

    m_subexpressions.push_back(subexpression);
    m_logger.trace("Appended token %u", requirement_token);
}

void CRE::append_to_expression(const std::vector<CREToken>& subexpression) {
    const size_t subexpression_size = subexpression.size();
    if (!subexpression_size) {
        return;
    }

    OPENVINO_ASSERT(!BINARY_OPERATORS.count(subexpression.at(0)), "Subexpressions cannot start with a binary operator");
    const CREToken last_token = subexpression.at(subexpression_size - 1);
    OPENVINO_ASSERT(!OPERATORS.count(last_token) && last_token != OPEN,
                    "The last token within a subexpression cannot be an operator nor open parrenthesis");

    const bool subexpression_enclosed = subexpression.at(0) == CRE::OPEN && last_token == CRE::CLOSE;
    std::vector<CREToken> maybe_enclosed_subexpression;

    // At least three tokens are required for a binary operator and its operands. In this case, parrenthesis are
    // required to ensure the correct operator precedence
    if (subexpression_size > 2 && !subexpression_enclosed) {
        maybe_enclosed_subexpression.push_back(CRE::OPEN);
    }
    maybe_enclosed_subexpression.insert(maybe_enclosed_subexpression.end(), subexpression.begin(), subexpression.end());

    if (subexpression_size > 2 && !subexpression_enclosed) {
        maybe_enclosed_subexpression.push_back(CRE::CLOSE);
    }

    if (subexpression_already_registered(subexpression)) {
        m_logger.trace("Subexpression already registered");
        return;
    }

    m_subexpressions.push_back(maybe_enclosed_subexpression);
    m_logger.trace("Appended subexpression");
}

size_t CRE::get_expression_length() const {
    if (m_subexpressions.empty()) {
        return 0;
    }

    size_t result = 0;
    for (const std::vector<CREToken>& subexpression : m_subexpressions) {
        result += subexpression.size();
    }

    // The "AND"s between subexpressions
    result += m_subexpressions.size() - 1;
    return result;
}

std::vector<CREToken> CRE::get_expression() const {
    if (m_subexpressions.empty()) {
        return {};
    }

    std::vector<CREToken> expression;
    size_t index = 0;
    for (const std::vector<CREToken>& subexpression : m_subexpressions) {
        if (index++ != 0) {
            // All subexpressions at depth level 0 are stitched together using ANDs by convention
            expression.push_back(CRE::AND);
        }
        expression.insert(expression.end(), subexpression.begin(), subexpression.end());
    }

    return expression;
}

bool CRE::empty() const {
    return m_subexpressions.empty();
}

void CRE::advance_iterator(std::vector<CREToken>::const_iterator& expression_iterator,
                           const std::vector<CREToken>::const_iterator& expression_end) const {
    CRE_EVAL_ASSERT(expression_iterator != expression_end, "The CRE ended unexpectedly");
    expression_iterator++;
}

bool CRE::end_condition(const std::vector<CREToken>::const_iterator& expression_iterator,
                        const std::vector<CREToken>::const_iterator& expression_end,
                        const Delimiter end_delimiter) const {
    switch (end_delimiter) {
    case Delimiter::PARRENTHESIS:
        return *expression_iterator == CLOSE;
    case Delimiter::SIZE:
        return expression_iterator == expression_end;
    default:
        // This is a logic error, not an "invalid CRE" one
        OPENVINO_THROW("Received an unknown value for the end condition delimiter");
    }
}

bool CRE::evaluate(
    std::vector<CREToken>::const_iterator& expression_iterator,
    const std::vector<CREToken>::const_iterator& expression_end,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& section_type_evaluators,
    const std::unordered_map<SectionID, SectionInstanceEvaluator>& section_type_instance_evaluators,
    const Delimiter end_delimiter,
    const bool skip_all_evaluations) const {
    std::function<bool(bool, bool)> logical_function = first_operand_function;
    bool result = true;
    bool negate = false;
    bool expect_binary_operator = false;
    bool at_least_one_iteration = false;
    bool skip_next_evaluation = false;
    bool subexpression_result;

    while (!end_condition(expression_iterator, expression_end, end_delimiter)) {
        CRE_EVAL_ASSERT(*expression_iterator != CLOSE, "Found a closed parrenthesis without any matching open token");
        at_least_one_iteration = true;

        // TODO comments
        switch (*expression_iterator) {
        case NOT:
            CRE_EVAL_ASSERT(!expect_binary_operator, "A \"NOT\" token was found when a binary operator was expected");
            negate = !negate;

            break;
        case OPEN:
            CRE_EVAL_ASSERT(!expect_binary_operator,
                            "An open parrenthesis was found when a binary operator was expected");
            // A subexpression is also an operand, and it should be followed by an operator
            expect_binary_operator = true;

            advance_iterator(expression_iterator, expression_end);
            // If the evaluation of the current operand is useless, then all children operands following this one are
            // also useless
            subexpression_result = evaluate(expression_iterator,
                                            expression_end,
                                            section_type_evaluators,
                                            section_type_instance_evaluators,
                                            Delimiter::PARRENTHESIS,
                                            skip_all_evaluations || skip_next_evaluation);
            CRE_EVAL_ASSERT(*expression_iterator == CLOSE,
                            "Expected a closed parrenthesis token during CRE evaluation. Received: ",
                            *expression_iterator);

            subexpression_result = negate ? !subexpression_result : subexpression_result;
            negate = false;

            result = logical_function(result, subexpression_result);
            break;
        case AND:
            CRE_EVAL_ASSERT(expect_binary_operator, "A binary operator was found when an operand was expected");
            expect_binary_operator = false;  // A binary operator should be followed by an operand

            logical_function = and_function;
            // No point in evaluating the next operand if the previous one yielded "false"
            skip_next_evaluation = result == false ? true : false;
            break;
        case OR:
            CRE_EVAL_ASSERT(expect_binary_operator, "A binary operator was found when an operand was expected");
            expect_binary_operator = false;  // A binary operator should be followed by an operand

            logical_function = or_function;
            // No point in evaluating the next operand if the previous one yielded "true"
            skip_next_evaluation = result == true ? true : false;
            break;
        default:
            // A section type (instance) token was found
            CRE_EVAL_ASSERT(!expect_binary_operator,
                            "A capability token was found when a binary operator was expected");
            expect_binary_operator = true;  // An operand should be followed by an operator

            if (!skip_all_evaluations && !skip_next_evaluation) {
                const SectionType section_type = *expression_iterator;
                bool operand = section_type_evaluators.count(section_type)
                                   ? section_type_evaluators.at(section_type)->check_support()
                                   : false;

                m_logger.trace("Section type %lu evaluated to %d", section_type, operand);

                if (operand) {
                    // Only if the section type evaluation succeeded, proceed to evaluate the section type instance if
                    // an instance ID is also found
                    expression_iterator++;

                    if (expression_iterator != expression_end && !RESERVED_TOKENS.count(*expression_iterator)) {
                        // Found a section type instance ID. The current section ID is supported only if the instance is
                        // supported
                        const SectionID section_id(section_type, *expression_iterator);
                        operand = section_type_evaluators.count(*expression_iterator)
                                      ? section_type_instance_evaluators.at(section_id).check_support()
                                      : true;

                        m_logger.trace("Section ID %s evaluated to %d", section_id, operand);
                    }
                    expression_iterator--;
                }

                operand = negate ? !operand : operand;

                result = logical_function(result, operand);
            }

            negate = false;
            break;
        }

        advance_iterator(expression_iterator, expression_end);
    }

    CRE_EVAL_ASSERT(at_least_one_iteration, "Cannot evaluate empty subexpressions");
    CRE_EVAL_ASSERT(expect_binary_operator,
                    "The CRE did not end with an operand. This means the final operator is missing its operand");

    return result;
}

bool CRE::check_compatibility(
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& section_type_evaluators,
    const std::unordered_map<SectionID, SectionInstanceEvaluator>& section_type_instance_evaluators) const {
    if (m_subexpressions.empty()) {
        return true;
    }

    const std::vector<CREToken> expression = get_expression();
    std::vector<CREToken>::const_iterator expression_iterator = expression.begin();
    const std::vector<CREToken>::const_iterator expression_end = expression.end();
    const bool result = evaluate(expression_iterator,
                                 expression_end,
                                 section_type_evaluators,
                                 section_type_instance_evaluators,
                                 Delimiter::SIZE);
    CRE_EVAL_ASSERT(expression_iterator == expression.end(),
                    "CRE evaluation ended before parsing the whole expression");

    m_logger.debug("Expression evaluated to %d", result);
    return result;
}

}  // namespace intel_npu
