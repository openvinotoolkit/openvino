// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/cre.hpp"

#include <functional>

#define CRE_EVAL_ASSERT(...) \
    OPENVINO_ASSERT_HELPER(::intel_npu::InvalidCRE, ::ov::AssertFailure::default_msg, __VA_ARGS__)

namespace {

using namespace intel_npu;

const std::unordered_set<CREToken> BINARY_OPERATORS{CRE::AND, CRE::OR};
const std::unordered_set<CREToken> OPERATORS{CRE::AND, CRE::OR, CRE::NOT};

constexpr char OPERAND_AND_RESERVED_TOKEN_SEPARATOR = '.';
constexpr char SECTION_TYPE_AND_INSTANCE_SEPARATOR = '_';

constexpr std::string_view AND_TOKEN_NAME = "AND";
constexpr std::string_view OR_TOKEN_NAME = "OR";
constexpr std::string_view NOT_TOKEN_NAME = "NOT";
constexpr std::string_view OPEN_TOKEN_NAME = "OPEN";
constexpr std::string_view CLOSE_TOKEN_NAME = "CLOSE";

constexpr std::string_view UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE = "Unsupported \"ov::CompatibilityCheck\" value";

/**
 * @brief Logical AND function between two "compatibility check" data types
 * @details Consider:
 *   * ov::CompatibilityCheck::SUPPORTED = TRUE
 *   * ov::CompatibilityCheck::UNSUPPORTED = FALSE
 *   * ov::CompatibilityCheck::NOT_APPLICABLE = UNKNOWN
 *
 * Then:
 *   * TRUE AND TRUE = TRUE
 *   * TRUE AND FALSE = FALSE
 *   * TRUE AND UNKNOWN = UNKNOWN
 *   * FALSE AND FALSE = FALSE
 *   * FALSE AND UNKNOWN = FALSE
 *   * UNKNOWN AND UNKNOWN = UNKNOWN
 */
ov::CompatibilityCheck and_function(const ov::CompatibilityCheck a, const ov::CompatibilityCheck b) {
    switch (a) {
    case ov::CompatibilityCheck::SUPPORTED: {
        switch (b) {
        case ov::CompatibilityCheck::SUPPORTED:
            return ov::CompatibilityCheck::SUPPORTED;
        case ov::CompatibilityCheck::UNSUPPORTED:
            return ov::CompatibilityCheck::UNSUPPORTED;
        case ov::CompatibilityCheck::NOT_APPLICABLE:
            return ov::CompatibilityCheck::NOT_APPLICABLE;
        default:
            OPENVINO_THROW(UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE);
        }
    }
    case ov::CompatibilityCheck::UNSUPPORTED: {
        switch (b) {
        case ov::CompatibilityCheck::SUPPORTED:
            return ov::CompatibilityCheck::UNSUPPORTED;
        case ov::CompatibilityCheck::UNSUPPORTED:
            return ov::CompatibilityCheck::UNSUPPORTED;
        case ov::CompatibilityCheck::NOT_APPLICABLE:
            return ov::CompatibilityCheck::UNSUPPORTED;
        default:
            OPENVINO_THROW(UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE);
        }
    }
    case ov::CompatibilityCheck::NOT_APPLICABLE: {
        switch (b) {
        case ov::CompatibilityCheck::SUPPORTED:
            return ov::CompatibilityCheck::NOT_APPLICABLE;
        case ov::CompatibilityCheck::UNSUPPORTED:
            return ov::CompatibilityCheck::UNSUPPORTED;
        case ov::CompatibilityCheck::NOT_APPLICABLE:
            return ov::CompatibilityCheck::NOT_APPLICABLE;
        default:
            OPENVINO_THROW(UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE);
        }
    }
    default: {
        OPENVINO_THROW(UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE);
    }
    }
}

/**
 * @brief Logical OR function between two "compatibility check" data types
 * @details Consider:
 *   * ov::CompatibilityCheck::SUPPORTED = TRUE
 *   * ov::CompatibilityCheck::UNSUPPORTED = FALSE
 *   * ov::CompatibilityCheck::NOT_APPLICABLE = UNKNOWN
 *
 * Then:
 *   * TRUE OR TRUE = TRUE
 *   * TRUE OR FALSE = TRUE
 *   * TRUE OR UNKNOWN = TRUE
 *   * FALSE OR FALSE = FALSE
 *   * FALSE OR UNKNOWN = UNKNOWN
 *   * UNKNOWN OR UNKNOWN = UNKNOWN
 */
ov::CompatibilityCheck or_function(const ov::CompatibilityCheck a, const ov::CompatibilityCheck b) {
    switch (a) {
    case ov::CompatibilityCheck::SUPPORTED: {
        switch (b) {
        case ov::CompatibilityCheck::SUPPORTED:
            return ov::CompatibilityCheck::SUPPORTED;
        case ov::CompatibilityCheck::UNSUPPORTED:
            return ov::CompatibilityCheck::SUPPORTED;
        case ov::CompatibilityCheck::NOT_APPLICABLE:
            return ov::CompatibilityCheck::SUPPORTED;
        default:
            OPENVINO_THROW(UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE);
        }
    }
    case ov::CompatibilityCheck::UNSUPPORTED: {
        switch (b) {
        case ov::CompatibilityCheck::SUPPORTED:
            return ov::CompatibilityCheck::SUPPORTED;
        case ov::CompatibilityCheck::UNSUPPORTED:
            return ov::CompatibilityCheck::UNSUPPORTED;
        case ov::CompatibilityCheck::NOT_APPLICABLE:
            return ov::CompatibilityCheck::NOT_APPLICABLE;
        default:
            OPENVINO_THROW(UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE);
        }
    }
    case ov::CompatibilityCheck::NOT_APPLICABLE: {
        switch (b) {
        case ov::CompatibilityCheck::SUPPORTED:
            return ov::CompatibilityCheck::SUPPORTED;
        case ov::CompatibilityCheck::UNSUPPORTED:
            return ov::CompatibilityCheck::NOT_APPLICABLE;
        case ov::CompatibilityCheck::NOT_APPLICABLE:
            return ov::CompatibilityCheck::NOT_APPLICABLE;
        default:
            OPENVINO_THROW(UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE);
        }
    }
    default: {
        OPENVINO_THROW(UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE);
    }
    }
}

/**
 * @brief Logical NOT function applied on a "compatibility check" data type
 * @details Consider:
 *   * ov::CompatibilityCheck::SUPPORTED = TRUE
 *   * ov::CompatibilityCheck::UNSUPPORTED = FALSE
 *   * ov::CompatibilityCheck::NOT_APPLICABLE = UNKNOWN
 *
 * Then:
 *   * NOT TRUE = FALSE
 *   * NOT FALSE = TRUE
 *   * NOT UNKNOWN = UNKNOWN
 */
ov::CompatibilityCheck not_function(const ov::CompatibilityCheck a) {
    switch (a) {
    case ov::CompatibilityCheck::SUPPORTED:
        return ov::CompatibilityCheck::UNSUPPORTED;
    case ov::CompatibilityCheck::UNSUPPORTED:
        return ov::CompatibilityCheck::SUPPORTED;
    case ov::CompatibilityCheck::NOT_APPLICABLE:
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    default:
        OPENVINO_THROW(UNSUPPORTED_COMPATIBILITY_CHECK_MESSAGE);
    }
}

/**
 * @note This function exists only to make the CRE evaluation function more compact
 * @returns The second argument
 */
ov::CompatibilityCheck first_operand_function(const ov::CompatibilityCheck /*a*/, const ov::CompatibilityCheck b) {
    return b;
}

ov::CompatibilityCheck bool_to_compatibility_check(const bool a) {
    return a ? ov::CompatibilityCheck::SUPPORTED : ov::CompatibilityCheck::UNSUPPORTED;
}

std::string reserved_token_to_string(const std::shared_ptr<CREToken> token) {
    const auto special_token = std::dynamic_pointer_cast<CRESpecialToken>(token);
    OPENVINO_ASSERT(special_token);

    switch (special_token->get_code()) {
    case CRESpecialTokenCode::AND: {
        return AND_TOKEN_NAME.data();
    }
    case CRESpecialTokenCode::OR: {
        return OR_TOKEN_NAME.data();
    }
    case CRESpecialTokenCode::OPEN: {
        return OPEN_TOKEN_NAME.data();
    }
    case CRESpecialTokenCode::CLOSE: {
        return CLOSE_TOKEN_NAME.data();
    }
    case CRESpecialTokenCode::NOT: {
        return NOT_TOKEN_NAME.data();
    }
    default: {
        OPENVINO_THROW("The given token is not a special one");
    }
    }
}

std::shared_ptr<CREToken> reserved_token_from_string(std::string_view token) {
    if (token == AND_TOKEN_NAME) {
        return CRE::AND;
    }
    if (token == OR_TOKEN_NAME) {
        return CRE::OR;
    }
    if (token == OPEN_TOKEN_NAME) {
        return CRE::OPEN;
    }
    if (token == CLOSE_TOKEN_NAME) {
        return CRE::CLOSE;
    }
    if (token == NOT_TOKEN_NAME) {
        return CRE::NOT;
    }
    return nullptr;
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

CRESpecialToken::CRESpecialToken(const CRESpecialTokenCode code) : m_code(code) {}

CRESpecialTokenCode CRESpecialToken::get_code() const {
    return m_code;
}

bool CRESpecialToken::operator==(const CRESpecialToken& other) const {
    return m_code == other.get_code();
}

bool is_cre_special_token(const std::shared_ptr<CREToken>& candidate) {
    return std::dynamic_pointer_cast<CRESpecialToken>(candidate) != nullptr;
}

CRE::CRE(const ov::log::Level log_level) : m_logger("CRE", log_level) {}

// TODO validation check inside ctor? or actually validation function, called in multiple other methods
CRE::CRE(const std::vector<std::shared_ptr<CREToken>>& subexpression, const ov::log::Level log_level)
    : m_logger("CRE", log_level) {
    if (!subexpression.empty()) {
        m_subexpressions.push_back(subexpression);
    }
}

bool CRE::subexpression_already_registered(const std::vector<std::shared_ptr<CREToken>>& subexpression) const {
    for (const std::vector<std::shared_ptr<CREToken>>& registered_subexpression : m_subexpressions) {
        if (subexpression == registered_subexpression) {
            return true;
        }
    }

    return false;
}

void CRE::append_to_expression(const std::shared_ptr<CREToken> requirement_token) {
    OPENVINO_ASSERT(!RESERVED_TOKENS.count(requirement_token),
                    "Appending subexpressions should be done through the \"vector\" API");

    const std::vector<std::shared_ptr<CREToken>> subexpression{requirement_token};
    if (subexpression_already_registered(subexpression)) {
        m_logger.trace("CREToken %u was already registered", requirement_token);
        return;
    }

    m_subexpressions.push_back(subexpression);
    m_logger.trace("Appended token %u", requirement_token);
}

void CRE::append_to_expression(const std::vector<std::shared_ptr<CREToken>>& subexpression) {
    const size_t subexpression_size = subexpression.size();
    if (!subexpression_size) {
        return;
    }

    OPENVINO_ASSERT(!BINARY_OPERATORS.count(subexpression.at(0)), "Subexpressions cannot start with a binary operator");
    const std::shared_ptr<CREToken> last_token = subexpression.at(subexpression_size - 1);
    OPENVINO_ASSERT(!OPERATORS.count(last_token) && last_token != OPEN,
                    "The last token within a subexpression cannot be an operator nor open parrenthesis");

    const bool subexpression_enclosed = subexpression.at(0) == CRE::OPEN && last_token == CRE::CLOSE;
    std::vector<std::shared_ptr<CREToken>> maybe_enclosed_subexpression;

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
    for (const std::vector<std::shared_ptr<CREToken>>& subexpression : m_subexpressions) {
        result += subexpression.size();
    }

    // The "AND"s between subexpressions
    result += m_subexpressions.size() - 1;
    return result;
}

std::vector<std::shared_ptr<CREToken>> CRE::get_expression() const {
    if (m_subexpressions.empty()) {
        return {};
    }

    std::vector<std::shared_ptr<CREToken>> expression;
    size_t index = 0;
    for (const std::vector<std::shared_ptr<CREToken>>& subexpression : m_subexpressions) {
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

void CRE::advance_iterator(std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_iterator,
                           const std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_end) const {
    CRE_EVAL_ASSERT(expression_iterator != expression_end, "The CRE ended unexpectedly");
    expression_iterator++;
}

bool CRE::end_condition(const std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_iterator,
                        const std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_end,
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

ov::CompatibilityCheck CRE::evaluate(
    std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_iterator,
    const std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_end,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& section_type_evaluators,
    const std::unordered_map<SectionID, SectionInstanceEvaluator>& section_instance_evaluators,
    const Delimiter end_delimiter,
    const bool skip_all_evaluations) const {
    std::function<ov::CompatibilityCheck(ov::CompatibilityCheck, ov::CompatibilityCheck)> logical_function =
        first_operand_function;
    ov::CompatibilityCheck result(ov::CompatibilityCheck::SUPPORTED);
    bool negate = false;
    bool expect_binary_operator = false;
    bool at_least_one_iteration = false;
    bool skip_next_evaluation = false;
    ov::CompatibilityCheck subexpression_result;

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
                                            section_instance_evaluators,
                                            Delimiter::PARRENTHESIS,
                                            skip_all_evaluations || skip_next_evaluation);
            CRE_EVAL_ASSERT(*expression_iterator == CLOSE,
                            "Expected a closed parrenthesis token during CRE evaluation. Received: ",
                            *expression_iterator);

            subexpression_result = negate ? not_function(subexpression_result) : subexpression_result;
            negate = false;

            result = logical_function(result, subexpression_result);
            break;
        case AND:
            CRE_EVAL_ASSERT(expect_binary_operator, "A binary operator was found when an operand was expected");
            expect_binary_operator = false;  // A binary operator should be followed by an operand

            logical_function = and_function;
            // No point in evaluating the next operand if the previous one yielded "false"
            skip_next_evaluation = result == ov::CompatibilityCheck::UNSUPPORTED ? true : false;
            break;
        case OR:
            CRE_EVAL_ASSERT(expect_binary_operator, "A binary operator was found when an operand was expected");
            expect_binary_operator = false;  // A binary operator should be followed by an operand

            logical_function = or_function;
            // No point in evaluating the next operand if the previous one yielded "true"
            skip_next_evaluation = result == ov::CompatibilityCheck::SUPPORTED ? true : false;
            break;
        default:
            // A section type (instance) token was found
            CRE_EVAL_ASSERT(!expect_binary_operator,
                            "A capability token was found when a binary operator was expected");
            expect_binary_operator = true;  // An operand should be followed by an operator

            if (!skip_all_evaluations && !skip_next_evaluation) {
                const SectionType section_type = *expression_iterator;
                ov::CompatibilityCheck operand = bool_to_compatibility_check(
                    section_type_evaluators.count(section_type) ? section_type_evaluators.at(section_type)->get_result()
                                                                : false);

                m_logger.trace("Section type %lu evaluated to %d", section_type, operand);

                if (operand != ov::CompatibilityCheck::UNSUPPORTED) {
                    // Only if the section type evaluation succeeded, proceed to evaluate the section type instance if
                    // an instance ID is also found
                    expression_iterator++;

                    if (expression_iterator != expression_end && !RESERVED_TOKENS.count(*expression_iterator)) {
                        // Found a section type instance ID. The current section ID is supported only if the instance is
                        // supported
                        const SectionID section_id = *expression_iterator;
                        operand = section_instance_evaluators.count(section_id)
                                      ? section_instance_evaluators.at(section_id).get_result()
                                      : ov::CompatibilityCheck::SUPPORTED;

                        m_logger.trace("Section ID %s evaluated to %d", section_id, operand);
                    }
                    expression_iterator--;
                }

                operand = negate ? not_function(operand) : operand;

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

ov::CompatibilityCheck CRE::check_compatibility(
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& section_type_evaluators,
    const std::unordered_map<SectionID, SectionInstanceEvaluator>& section_instance_evaluators) const {
    if (m_subexpressions.empty()) {
        return ov::CompatibilityCheck::SUPPORTED;
    }

    const std::vector<std::shared_ptr<CREToken>> expression = get_expression();
    std::vector<std::shared_ptr<CREToken>>::const_iterator expression_iterator = expression.begin();
    const std::vector<std::shared_ptr<CREToken>>::const_iterator expression_end = expression.end();
    const ov::CompatibilityCheck result = evaluate(expression_iterator,
                                                   expression_end,
                                                   section_type_evaluators,
                                                   section_instance_evaluators,
                                                   Delimiter::SIZE);
    CRE_EVAL_ASSERT(expression_iterator == expression.end(),
                    "CRE evaluation ended before parsing the whole expression");

    m_logger.debug("Expression evaluated to %d", result);
    return result;
}

std::string cre_to_string(const CRE cre) {
    // TODO validate the CRE
    const std::vector<std::shared_ptr<CREToken>> expression = cre.get_expression();
    std::string result("");

    bool is_first_token = true;
    for (const std::shared_ptr<CREToken> token : expression) {
        if (PREDEFINED_SECTION_TYPES.count(token)) {
            if (!is_first_token) {
                result += OPERAND_AND_RESERVED_TOKEN_SEPARATOR;
            }
            is_first_token = false;

            result += section_type_to_string(token);
            continue;
        }
        if (CRE::RESERVED_TOKENS.count(token)) {
            if (!is_first_token) {
                result += OPERAND_AND_RESERVED_TOKEN_SEPARATOR;
            }
            is_first_token = false;

            result += reserved_token_to_string(token);
            continue;
        }

        // Last case remaining: the token is a section ID following a section type
        OPENVINO_ASSERT(!is_first_token);
        result += SECTION_TYPE_AND_INSTANCE_SEPARATOR;
        result += std::to_string(token);
    }
}

CRE cre_from_string(std::string_view cre) {
    std::vector<std::shared_ptr<CREToken>> expression;
    std::string_view remaining = cre;

    while (true) {
        const size_t dot_location = remaining.find(OPERAND_AND_RESERVED_TOKEN_SEPARATOR);
        const std::string_view token_string = remaining.substr(0, dot_location);

        const std::shared_ptr<CREToken> reserved_token = reserved_token_from_string(token_string);
        if (reserved_token) {
            expression.push_back(reserved_token.value());
        } else {
            // The current substring should have the form "<section type name>_<id>"
            const auto [section_type, section_id] = section_type_and_id_from_string(token_string);
            expression.push_back(section_type);
            expression.push_back(section_id);
        }

        if (dot_location == std::string_view::npos) {
            break;
        }
        remaining = remaining.substr(dot_location + 1);
        OPENVINO_ASSERT(!remaining.empty(), "Trailing dot found while parsing the cre \"", cre, "\"");
    }

    return CRE(expression);
}

}  // namespace intel_npu
