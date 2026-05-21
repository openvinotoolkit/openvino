// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/cre.hpp"

#include <functional>

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/icapability.hpp"

#define CRE_EVAL_ASSERT(...) \
    OPENVINO_ASSERT_HELPER(::intel_npu::InvalidCRE, ::ov::AssertFailure::default_msg, __VA_ARGS__)

namespace {

const std::unordered_set<intel_npu::CRE::Token> OPERATORS{intel_npu::CRE::AND, intel_npu::CRE::OR};

inline bool and_function(bool a, bool b) {
    return a && b;
}

inline bool or_function(bool a, bool b) {
    return a || b;
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

CRE::CRE() : m_expression({CRE::AND}) {}

CRE::CRE(const std::vector<Token>& expression) : m_expression(expression) {}

void CRE::append_to_expression(const CRE::Token requirement_token) {
    OPENVINO_ASSERT(!RESERVED_TOKENS.count(requirement_token),
                    "Appending subexpressions should be done through the \"vector\" API");
    m_expression.push_back(requirement_token);
}

void CRE::append_to_expression(const std::vector<CRE::Token>& requirement_tokens) {
    m_expression.insert(m_expression.end(), requirement_tokens.begin(), requirement_tokens.end());
}

size_t CRE::get_expression_length() const {
    return m_expression.size();
}

std::vector<CRE::Token> CRE::get_expression() const {
    return m_expression;
}

void CRE::advance_iterator(std::vector<Token>::const_iterator& expression_iterator) {
    CRE_EVAL_ASSERT(expression_iterator != m_expression.end());
    expression_iterator++;
}

bool CRE::end_condition(const std::vector<Token>::const_iterator& expression_iterator, const Delimiter end_delimiter) {
    switch (end_delimiter) {
    case Delimiter::PARRENTHESIS:
        return *expression_iterator == CLOSE;
    case Delimiter::SIZE:
        return expression_iterator == m_expression.end();
    default:
        // Delimiter::NOT_CAPABILITY
        return RESERVED_TOKENS.count(*expression_iterator) || expression_iterator == m_expression.end();
    }
}

bool CRE::evaluate(std::vector<Token>::const_iterator& expression_iterator,
                   const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities,
                   const Delimiter end_delimiter) {
    std::function<bool(bool, bool)> logical_function;
    bool base, subexpression_result, negate;

    CRE_EVAL_ASSERT(*expression_iterator != CLOSE, "Found a closed parrenthesis without any matching open token");

    // TODO comments
    switch (*expression_iterator) {
    case NOT:
        negate = false;
        while (*expression_iterator == NOT) {
            negate = !negate;
            advance_iterator(expression_iterator);
            CRE_EVAL_ASSERT(expression_iterator != m_expression.end(), "NOT operator is missing its operand");
        }
        subexpression_result = evaluate(expression_iterator, plugin_capabilities, end_delimiter);
        return negate ? !subexpression_result : subexpression_result;
    case OPEN:
        advance_iterator(expression_iterator);
        subexpression_result = evaluate(expression_iterator, plugin_capabilities, Delimiter::PARRENTHESIS);
        CRE_EVAL_ASSERT(*expression_iterator == CLOSE);
        advance_iterator(expression_iterator);
        return subexpression_result;
    case AND:
        logical_function = and_function;
        base = true;
        break;
    case OR:
        logical_function = or_function;
        base = false;
        break;
    default:
        // A capability token was found
        const bool operand = plugin_capabilities.count(*expression_iterator)
                                 ? plugin_capabilities.at(*expression_iterator)->check_support()
                                 : false;
        advance_iterator(expression_iterator);
        return operand;
    }

    advance_iterator(expression_iterator);

    // Found an n-ary operator (AND or OR). This should be followed by n operands, n >= 1. One operand can be defined
    // as:
    //   * The ID of a capability
    //   * Open parrenthesis - subexpression - closed parrenthesis
    //   * Subexpression without parrenthesis (starts with an operator)
    subexpression_result = base;
    bool no_operands = true;
    while (!end_condition(expression_iterator, end_delimiter)) {
        no_operands = false;

        subexpression_result =
            logical_function(subexpression_result,
                             evaluate(expression_iterator, plugin_capabilities, Delimiter::NOT_CAPABILITY_ID));
    }

    CRE_EVAL_ASSERT(!no_operands, "At least one operator doesn't have any operand");

    return subexpression_result;
}

bool CRE::check_compatibility(const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities) {
    if (m_expression.empty()) {
        return true;
    }

    std::vector<Token>::const_iterator expression_iterator = m_expression.begin();
    const bool result = evaluate(expression_iterator, plugin_capabilities, Delimiter::SIZE);
    CRE_EVAL_ASSERT(expression_iterator == m_expression.end());
    return result;
}

CRESection::CRESection(const CRE& cre) : ISection(PredefinedSectionType::CRE), m_cre(cre) {}

void CRESection::write(const std::unique_ptr<BlobWriterInterface>& writer) {
    writer->write(m_cre.get_expression().data(), m_cre.get_expression_length() * sizeof(CRE::Token));
}

CRE CRESection::get_cre() const {
    return m_cre;
}

std::shared_ptr<ISection> CRESection::read(BlobReader* blob_reader, const size_t section_length) {
    size_t number_of_tokens = section_length / sizeof(CRE::Token);
    OPENVINO_ASSERT(number_of_tokens != 0);

    std::vector<CRE::Token> tokens(number_of_tokens);
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(tokens.data()), number_of_tokens * sizeof(CRE::Token));

    return std::make_shared<CRESection>(CRE(tokens));
}

}  // namespace intel_npu
