// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/cre.hpp"

#include <functional>

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/icapability.hpp"

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
    OPENVINO_ASSERT(expression_iterator != m_expression.end());
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
    bool base;

    // An operator is always expected first
    switch (*expression_iterator) {
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
                              *expression_iterator,
                              " instead of an operator");
    }

    advance_iterator(expression_iterator);

    // Followed by n operands, n >= 1. One operand can be defined as:
    //   * The ID of a capability
    //   * Open parrenthesis - subexpression - closed parrenthesis
    //   * Subexpression without parrenthesis (starts with an operator)
    bool no_operands = true;
    while (!end_condition(expression_iterator, end_delimiter)) {
        no_operands = false;

        if (*expression_iterator == OPEN) {
            advance_iterator(expression_iterator);
            base = logical_function(base, evaluate(expression_iterator, plugin_capabilities, Delimiter::PARRENTHESIS));
            OPENVINO_ASSERT(*expression_iterator == CLOSE);
            advance_iterator(expression_iterator);
        } else if (*expression_iterator == CLOSE) {
            OPENVINO_THROW_HELPER(InvalidCRE,
                                  ov::Exception::default_msg,
                                  "Found a closed parrenthesis without any matching open token");
        } else if (OPERATORS.count(*expression_iterator)) {
            base = logical_function(base,
                                    evaluate(expression_iterator, plugin_capabilities, Delimiter::NOT_CAPABILITY_ID));
        } else {
            const bool has_capability = plugin_capabilities.count(*expression_iterator)
                                            ? plugin_capabilities.at(*expression_iterator)->check_support()
                                            : false;

            base = logical_function(base, has_capability);
            advance_iterator(expression_iterator);
        }
    }

    if (no_operands) {
        OPENVINO_THROW_HELPER(InvalidCRE, ov::Exception::default_msg, "At least one operator doesn't have any operand");
    }

    return base;
}

bool CRE::check_compatibility(const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities) {
    if (m_expression.empty()) {
        return true;
    }
    if (m_expression.size() == 1) {
        return plugin_capabilities.count(m_expression.at(0));
    }

    std::vector<Token>::const_iterator expression_iterator = m_expression.begin();
    return evaluate(expression_iterator, plugin_capabilities, Delimiter::SIZE);
}

CRESection::CRESection(const CRE& cre) : ISection(PredefinedSectionType::CRE), m_cre(cre) {}

void CRESection::write(std::ostream& stream, BlobWriter* writer) {
    stream.write(reinterpret_cast<const char*>(m_cre.get_expression().data()),
                 m_cre.get_expression_length() * sizeof(CRE::Token));
}

CRE CRESection::get_cre() const {
    return m_cre;
}

std::shared_ptr<ISection> CRESection::read(BlobReader* blob_reader, const size_t section_length) {
    size_t number_of_tokens = section_length / sizeof(CRE::Token);
    OPENVINO_ASSERT(number_of_tokens > 0);
    CRE cre;

    // We expect the expression to start with "AND". The ctor also places this token at the beginning.
    CRE::Token token;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&token), sizeof(token));
    OPENVINO_ASSERT(token == CRE::AND);

    while (--number_of_tokens) {
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&token), sizeof(token));
        cre.append_to_expression(token);
    }

    return std::make_shared<CRESection>(cre);
}

}  // namespace intel_npu
