// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/cre.hpp"

#include <functional>

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/icapability.hpp"
#include "intel_npu/common/itt.hpp"

#define CRE_EVAL_ASSERT(...) \
    OPENVINO_ASSERT_HELPER(::intel_npu::InvalidCRE, ::ov::AssertFailure::default_msg, __VA_ARGS__)

namespace {

const std::unordered_set<intel_npu::CRE::Token> BINARY_OPERATORS{intel_npu::CRE::AND, intel_npu::CRE::OR};
const std::unordered_set<intel_npu::CRE::Token> OPERATORS{intel_npu::CRE::AND, intel_npu::CRE::OR, intel_npu::CRE::NOT};

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

// TODO use the logger more after modifying the algorithm
CRE::CRE(const ov::log::Level log_level) : m_logger("CRE", log_level) {}

CRE::CRE(const std::vector<Token>& expression, const ov::log::Level log_level) : m_logger("CRE", log_level) {
    if (!expression.empty()) {
        m_subexpressions.push_back(expression);
    }
}

bool CRE::subexpression_already_registered(const std::vector<CRE::Token>& subexpression) const {
    for (const std::vector<CRE::Token>& registered_subexpression : m_subexpressions) {
        if (subexpression == registered_subexpression) {
            return true;
        }
    }

    return false;
}

void CRE::append_to_expression(const CRE::Token requirement_token) {
    OPENVINO_ASSERT(!RESERVED_TOKENS.count(requirement_token),
                    "Appending subexpressions should be done through the \"vector\" API");

    const std::vector<CRE::Token> subexpression{requirement_token};
    if (subexpression_already_registered(subexpression)) {
        m_logger.trace("Token %u was already registered", requirement_token);
        return;
    }

    m_subexpressions.push_back(subexpression);
    m_logger.trace("Appended token %u", requirement_token);
}

void CRE::append_to_expression(const std::vector<CRE::Token>& requirement_tokens) {
    const size_t subexpression_size = requirement_tokens.size();
    if (!subexpression_size) {
        return;
    }

    OPENVINO_ASSERT(!BINARY_OPERATORS.count(requirement_tokens.at(0)),
                    "Subexpressions cannot start with a binary operator");
    const CRE::Token last_token = requirement_tokens.at(subexpression_size - 1);
    OPENVINO_ASSERT(!OPERATORS.count(last_token) && last_token != OPEN,
                    "The last token within a subexpression cannot be an operator nor open parrenthesis");

    const bool subexpression_enclosed = requirement_tokens.at(0) == CRE::OPEN && last_token == CRE::CLOSE;
    std::vector<CRE::Token> subexpression;

    // At least three tokens are required for a binary operator and its operands. In this case, parrenthesis are
    // required to ensure the correct operator precedence
    if (subexpression_size > 2 && !subexpression_enclosed) {
        subexpression.push_back(CRE::OPEN);
    }
    subexpression.insert(subexpression.end(), requirement_tokens.begin(), requirement_tokens.end());

    if (subexpression_size > 2 && !subexpression_enclosed) {
        subexpression.push_back(CRE::CLOSE);
    }

    if (subexpression_already_registered(subexpression)) {
        m_logger.trace("Subexpression already registered");
        return;
    }

    m_subexpressions.push_back(subexpression);
    m_logger.trace("Appended subexpression");
}

size_t CRE::get_expression_length() const {
    if (m_subexpressions.empty()) {
        return 0;
    }

    size_t result = 0;
    for (const std::vector<Token>& subexpression : m_subexpressions) {
        result += subexpression.size();
    }

    // The "AND"s between subexpressions
    result += m_subexpressions.size() - 1;
    return result;
}

std::vector<CRE::Token> CRE::get_expression() const {
    if (m_subexpressions.empty()) {
        return {};
    }

    std::vector<CRE::Token> expression;
    size_t index = 0;
    for (const std::vector<CRE::Token>& subexpression : m_subexpressions) {
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

void CRE::advance_iterator(std::vector<Token>::const_iterator& expression_iterator,
                           const std::vector<Token>::const_iterator& expression_end) const {
    CRE_EVAL_ASSERT(expression_iterator != expression_end, "The CRE ended unexpectedly");
    expression_iterator++;
}

bool CRE::end_condition(const std::vector<Token>::const_iterator& expression_iterator,
                        const std::vector<Token>::const_iterator& expression_end,
                        const Delimiter end_delimiter) const {
    switch (end_delimiter) {
    case Delimiter::PARRENTHESIS:
        return *expression_iterator == CLOSE;
    case Delimiter::SIZE:
        return expression_iterator == expression_end;
    default:
        // Delimiter::NOT_CAPABILITY
        return RESERVED_TOKENS.count(*expression_iterator) || expression_iterator == expression_end;
    }
}

bool CRE::evaluate(std::vector<Token>::const_iterator& expression_iterator,
                   const std::vector<Token>::const_iterator& expression_end,
                   const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities,
                   const Delimiter end_delimiter) const {
    std::function<bool(bool, bool)> logical_function = first_operand_function;
    bool result = true;
    bool negate = false;
    bool expect_binary_operator = false;
    bool at_least_one_iteration = false;
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
            subexpression_result =
                evaluate(expression_iterator, expression_end, plugin_capabilities, Delimiter::PARRENTHESIS);
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
            break;
        case OR:
            CRE_EVAL_ASSERT(expect_binary_operator, "A binary operator was found when an operand was expected");
            expect_binary_operator = false;  // A binary operator should be followed by an operand

            logical_function = or_function;
            break;
        default:
            // A capability token was found
            CRE_EVAL_ASSERT(!expect_binary_operator,
                            "A capability token was found when a binary operator was expected");
            expect_binary_operator = true;  // An operand should be followed by an operator

            bool operand = plugin_capabilities.count(*expression_iterator)
                               ? plugin_capabilities.at(*expression_iterator)->check_support()
                               : false;
            operand = negate ? !operand : operand;
            negate = false;

            result = logical_function(result, operand);
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
    const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities) const {
    if (m_subexpressions.empty()) {
        return true;
    }

    const std::vector<Token> expression = get_expression();
    std::vector<Token>::const_iterator expression_iterator = expression.begin();
    const std::vector<Token>::const_iterator expression_end = expression.end();
    const bool result = evaluate(expression_iterator, expression_end, plugin_capabilities, Delimiter::SIZE);
    CRE_EVAL_ASSERT(expression_iterator == expression.end(),
                    "CRE evaluation ended before parsing the whole expression");
    return result;
}

CRESection::CRESection(const CRE& cre, const ov::log::Level log_level)
    : ISection(PredefinedSectionType::CRE),
      m_cre(cre),
      m_logger("CRESection", log_level) {}

void CRESection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CRESection::write");

    writer.write(m_cre.get_expression().data(), m_cre.get_expression_length() * sizeof(CRE::Token));

    m_logger.debug("%lu tokens written", m_cre.get_expression_length());
}

CRE CRESection::get_cre() const {
    return m_cre;
}

std::shared_ptr<ISection> CRESection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CRESection::read");
    Logger logger("CRESection", blob_reader.get_log_level());

    const size_t section_length = blob_reader.get_section_length();
    OPENVINO_ASSERT(section_length % sizeof(CRE::Token) == 0,
                    "Received a CRE section length that is not divisible by the CRE token size. Section length: ",
                    section_length,
                    ". CRE token size: ",
                    sizeof(CRE::Token));
    size_t number_of_tokens = section_length / sizeof(CRE::Token);
    OPENVINO_ASSERT(number_of_tokens != 0,
                    "Read \"0\" as the number of CRE tokens. This value is invalid since at least one token (the CRE "
                    "capability) is expected");

    logger.debug("Reading %lu tokens", number_of_tokens);

    std::vector<CRE::Token> tokens(number_of_tokens);
    blob_reader.copy_data_from_source(reinterpret_cast<char*>(tokens.data()), number_of_tokens * sizeof(CRE::Token));

    return std::make_shared<CRESection>(CRE(tokens, logger.level()), logger.level());
}

}  // namespace intel_npu
