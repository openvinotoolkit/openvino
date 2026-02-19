// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

class ICapability;

class InvalidCRE : public ov::Exception {};

class CRE final {
public:
    using Token = uint16_t;

    // TODO add the "NOT" operator as well
    enum ReservedToken : Token { AND = 50000, OR = 50001, OPEN = 50002, CLOSE = 50003 };

    static inline const std::unordered_set<Token> RESERVED_TOKENS{ReservedToken::AND,
                                                                  ReservedToken::OR,
                                                                  ReservedToken::OPEN,
                                                                  ReservedToken::CLOSE};

    /**
     * @brief All capability codes known in advance. Past codes should be recorded here as well to help avoiding
     * code collision.
     */
    enum PredefinedCapabilityToken : Token {
        CRE_EVALUATION = 100,
        ELF_SCHEDULE = 101,
        BATCHING = 102,
        WEIGHTS_SEPARATION = 103
    };

    // TODO: separate set for the "static" ones?
    static inline const std::unordered_set<Token> DEFAULT_PLUGIN_CAPABILITIES_TOKENS{
        PredefinedCapabilityToken::CRE_EVALUATION,
        PredefinedCapabilityToken::ELF_SCHEDULE,
        PredefinedCapabilityToken::BATCHING,
        PredefinedCapabilityToken::WEIGHTS_SEPARATION};

    CRE();

    CRE(const std::vector<Token>& expression);

    /**
     * @brief Append a new token to the CRE, at depth-level 1. All tokens found at this depth-level are bound by a
     * logical "AND" operator.
     */
    void append_to_expression(const CRE::Token requirement_token);

    /**
     * @brief Append a new CRE subexpression to the CRE, at depth-level 1. All tokens found at this depth-level are
     * bound by a logical "AND" operator.
     */
    void append_to_expression(const std::vector<CRE::Token>& requirement_tokens);

    size_t get_expression_length() const;

    std::vector<Token> get_expression() const;

    /**
     * @brief Evaluates the expression against the given NPU plugin capabilities.
     * @details The plugin capabilities are evaluated in a lazy manner: the check support function is called only upon
     * encountering the corresponding CRE token.
     *
     * @param plugin_capabilities A mapping between CRE tokens and their (lazy) evaluators.
     */
    bool check_compatibility(const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities);

private:
    enum class Delimiter { PARRENTHESIS, SIZE, NOT_CAPABILITY_ID };

    void advance_iterator(std::vector<Token>::const_iterator& expression_iterator);

    bool end_condition(const std::vector<Token>::const_iterator& expression_iterator, const Delimiter end_delimiter);

    bool evaluate(std::vector<Token>::const_iterator& expression_iterator,
                  const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities,
                  const Delimiter end_delimiter);

    std::vector<Token> m_expression;
};

class CRESection final : public ISection {
public:
    CRESection(const CRE& cre);

    void write(std::ostream& stream, BlobWriter* writer) override;

    CRE get_cre() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

private:
    CRE m_cre;
};

}  // namespace intel_npu
