// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

class ISectionTypeEvaluator;

class InvalidCRE final : public ov::AssertFailure {
public:
    [[noreturn]] static void create(const char* file,
                                    int line,
                                    const char* check_string,
                                    const std::string& context_info,
                                    const std::string& explanation);

protected:
    explicit InvalidCRE(const std::string& what_arg) : ov::AssertFailure(what_arg) {}
};

class CRE final {
public:
    using Token = uint16_t;

    enum ReservedToken : Token { AND = 65400, OR = 65401, OPEN = 65402, CLOSE = 65403, NOT = 65404 };

    static inline const std::unordered_set<Token> RESERVED_TOKENS{ReservedToken::AND,
                                                                  ReservedToken::OR,
                                                                  ReservedToken::OPEN,
                                                                  ReservedToken::CLOSE,
                                                                  ReservedToken::NOT};

    CRE(const ov::log::Level log_level = ov::log::Level::WARNING);

    CRE(const std::vector<Token>& expression, const ov::log::Level log_level = ov::log::Level::WARNING);

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

    bool empty() const;

    /**
     * @brief Evaluates the expression against the given NPU plugin capabilities.
     * @details The plugin capabilities are evaluated in a lazy manner: the check support function is called only upon
     * encountering the corresponding CRE token.
     *
     * @param section_type_evaluators A mapping between CRE tokens and their (lazy) evaluators.
     */
    bool check_compatibility(
        const std::unordered_map<CRE::Token, std::shared_ptr<ISectionTypeEvaluator>>& section_type_evaluators) const;

private:
    enum class Delimiter { PARRENTHESIS, SIZE, NOT_CAPABILITY_ID };

    bool subexpression_already_registered(const std::vector<Token>& subexpression) const;

    void advance_iterator(std::vector<Token>::const_iterator& expression_iterator,
                          const std::vector<Token>::const_iterator& expression_end) const;

    bool end_condition(const std::vector<Token>::const_iterator& expression_iterator,
                       const std::vector<Token>::const_iterator& expression_end,
                       const Delimiter end_delimiter) const;

    /**
     * @brief Evaluates a subexpression from left to right.
     * @details The evaluation starts from the position where the iterator was left at. The end of the subexpression is
     * determined based on the given type of delimiter.
     *
     * The parent of the current subexpression might have determined that all evaluations within this subexpression have
     * no impact on the final result. If that is the case, then "skip_all_evaluations" should be set to true, and
     * operand evaluation will be skipped to save some resources.
     * @param expression_iterator The cursor corresponding to the expression that is being evaluated. The initial value
     * indicates the start of the subexpression.
     * @param expression_end Points towards the end of the whole expression.
     * @param section_type_evaluators
     * @param end_delimiter The type of delimiter that is used for judging the end of the subexpression.
     * @param skip_all_evaluations If set to "true", all operand evaluations wihtin this subexpressions will be skipped.
     * However, CRE validity checks will still be performed.
     */
    bool evaluate(std::vector<Token>::const_iterator& expression_iterator,
                  const std::vector<Token>::const_iterator& expression_end,
                  const std::unordered_map<CRE::Token, std::shared_ptr<ISectionTypeEvaluator>>& section_type_evaluators,
                  const Delimiter end_delimiter,
                  const bool skip_all_evaluations = false) const;

    std::vector<std::vector<Token>> m_subexpressions;

    Logger m_logger;
};

}  // namespace intel_npu
