// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cre.hpp"

namespace {

constexpr intel_npu::ISection::SectionID CRE_SECTION_ID = 100;

const std::unordered_set<intel_npu::CRE::Token> OPERATORS{AND, OR};

inline bool and_function(bool a, bool b) {
    return a && b;
}

inline bool or_function(bool a, bool b) {
    return a || b;
}

}  // namespace

namespace intel_npu {

CRE::CRE() : m_expression({CRE::AND}) {}

size_t CRE::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(m_expression.data()), m_expression.size());
    return m_expression.size();
}

void CRE::append_to_expression(const CRE::Token requirement_token) {
    OPENVINO_ASSERT(!RESERVED_TOKENS.count(requirement_token),
                    "Appending subexpressions should be done through the \"vector\" API");
    m_expression.push_back(requirement_token);
}

void CRE::append_to_expression(const std::vector<CRE::Token>& requirement_tokens) {
    m_expression.insert(m_expression.end(), requirement_tokens.begin(), requirement_tokens.end());
}

size_t CRE::get_expression_length() {
    return m_expression.size();
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

// TODO: bound checking
bool CRE::evaluate(std::vector<Token>::const_iterator& expression_iterator,
                   const std::unordered_set<CRE::Token>& plugin_capabilities,
                   const Delimiter end_delimiter) {
    std::function<bool(bool, bool)> logical_function;
    bool base;

    // An operator is always expected first
    switch (*(expression_iterator++)) {
    case AND:
        logical_function = and_function;
        base = true;
        break;
    case OR:
        logical_function = or_function;
        base = false;
        break;
    default:
        OPENVINO_THROW_HELPER(InvalidCRE,
                              ov::Exception::default_msg,
                              "Received: ",
                              *(--expression_iterator),
                              " instead of an operator");
    }

    // Followed by n operands, n >= 2. One operand can be defined as:
    //   * The ID of a capability
    //   * Open parrenthesis - subexpression - closed parrenthesis
    //   * Subexpression without parrenthesis (starts with an operator)
    size_t n_operands = 0;
    while (!end_condition(expression_iterator, end_delimiter)) {
        ++n_operands;

        if (*expression_iterator == OPEN) {
            base =
                logical_function(base, evaluate(++expression_iterator, plugin_capabilities, Delimiter::PARRENTHESIS));
            OPENVINO_ASSERT(*(expression_iterator++) == CLOSE);
        } else if (*expression_iterator == CLOSE) {
            OPENVINO_THROW_HELPER(InvalidCRE,
                                  ov::Exception::default_msg,
                                  "Found a closed parrenthesis without any matching open token");
        } else if (OPERATORS.count(*expression_iterator)) {
            base = logical_function(base,
                                    evaluate(expression_iterator, plugin_capabilities, Delimiter::NOT_CAPABILITY_ID));
        } else {
            base = logical_function(base, plugin_capabilities.count(*(expression_iterator++)));
        }
    }

    if (n_operands < 2) {
        OPENVINO_THROW_HELPER(InvalidCRE,
                              ov::Exception::default_msg,
                              "At least one operator has less than two operands");
    }

    return base;
}

bool CRE::check_compatibility(const std::unordered_set<CRE::Token>& plugin_capabilities) {
    std::vector<Token>::const_iterator expression_iterator = m_expression.begin();
    return evaluate(expression_iterator, plugin_capabilities, Delimiter::SIZE);
}

CRESection::CRESection() : ISection(CRE_SECTION_ID) {}

void CRESection::append_to_expression(const CRE::Token requirement_token) {
    m_cre.append_to_expression(requirement_token);
}

void CRESection::append_to_expression(const std::vector<CRE::Token>& requirement_tokens) {
    m_cre.append_to_expression(requirement_tokens);
}

void CRESection::write(std::ostream& stream, BlobWriter* writer) {
    writer->cursor += m_cre.write(stream);
}

std::optional<uint64_t> CRESection::get_length() {
    return m_cre.get_expression_length();
}

}  // namespace intel_npu
