// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include "intel_npu/common/cre_token.hpp"
#include "intel_npu/common/isection.hpp"
#include "intel_npu/common/isection_type_evaluator.hpp"
#include "intel_npu/common/section_id.hpp"
#include "intel_npu/common/section_type.hpp"
#include "intel_npu/common/single_section_instance_evaluator.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

/**
 * @brief Exception used to indicate an invalid expression. Logic errors or other type of failures should not be
 * represented by this.
 */
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

// TODO double check it's fine to have no predetermined value (these are not stored)
enum class CRESpecialTokenCode { AND, OR, NOT, OPEN, CLOSE };

/**
 * @brief All tokens that are not operands. This set is made by operators and parrenthesis.
 */
class CRESpecialToken final : public CREToken {
public:
    CRESpecialToken(const CRESpecialTokenCode code);

    CRESpecialTokenCode get_code() const;

    bool operator==(const CRESpecialToken& other) const;

    bool operator!=(const CRESpecialToken& other) const;

private:
    CRESpecialTokenCode m_code;
};

bool is_cre_special_token(const std::shared_ptr<CREToken>& candidate);

/**
 * @brief Handles the construction, validation, evaluation and serialization of the Compatibility Requirements
 * Expression.
 * @details The CRE is used to evaluate the high-level compatibility requirements *between* sections types and/or
 * instances using a logical expression. This evaluation happens during model import or when a compatibility descriptor
 * is validated using the "Plugin::get_property" API call.
 */
class CRE final {
public:
    CRE(const ov::log::Level log_level = ov::log::Level::WARNING);

    CRE(const std::vector<std::shared_ptr<CREToken>>& subexpression,
        const ov::log::Level log_level = ov::log::Level::WARNING);

    /**
     * @brief Append a new token to the CRE, at depth-level 1. All tokens found at this depth-level are bound by a
     * logical "AND" operator.
     */
    void append_to_expression(const std::shared_ptr<CREToken> requirement_token);

    /**
     * @brief Append a new CRE subexpression to the CRE, at depth-level 1. All tokens found at this depth-level are
     * bound by a logical "AND" operator.
     */
    void append_to_expression(const std::vector<std::shared_ptr<CREToken>>& subexpression);

    size_t get_expression_length() const;

    std::vector<std::shared_ptr<CREToken>> get_expression() const;

    bool empty() const;

    /**
     * @brief Evaluates the expression using the given evaluators for section types and instances.
     * @details The support for both section types and instances is evaluated in a lazy manner: the check support
     * function is called only upon encountering the corresponding CRE token.
     *
     * @param section_type_evaluators A mapping between section types and their (lazy) evaluators.
     * @param section_instance_evaluators A mapping between section IDs and their (lazy) evaluators.
     */
    ov::CompatibilityCheck check_compatibility(
        const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& section_type_evaluators,
        const std::unordered_map<SectionID, SingleSectionInstanceEvaluator>& section_instance_evaluators) const;

    /**
     * @brief Serializes the CRE.
     * @note This "human-readable" form is meant to be used:
     *   1. Inside the runtime requirements section, as part of a compiled model and
     *   2. As part of the compatibility descriptor returned to the user via the "Plugin::get_property" API call.
     */
    std::string to_string() const;

    /**
     * @brief Deserializes the CRE string into internal token codes.
     */
    static CRE from_string(const std::string_view cre, const ov::log::Level log_level = ov::log::Level::WARNING);

    // TODO reconsider these
    // Some "globals" for convenience
    static inline const auto AND_PTR = std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND);
    static inline const auto OR_PTR = std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR);
    static inline const auto NOT_PTR = std::make_shared<CRESpecialToken>(CRESpecialTokenCode::NOT);
    static inline const auto OPEN_PTR = std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OPEN);
    static inline const auto CLOSE_PTR = std::make_shared<CRESpecialToken>(CRESpecialTokenCode::CLOSE);

    static inline const CRESpecialToken AND = *AND_PTR;
    static inline const CRESpecialToken OR = *OR_PTR;
    static inline const CRESpecialToken NOT = *NOT_PTR;
    static inline const CRESpecialToken OPEN = *OPEN_PTR;
    static inline const CRESpecialToken CLOSE = *CLOSE_PTR;

private:
    enum class Delimiter { PARRENTHESIS, SIZE };

    /**
     * @brief Checks if the given expression forms a valid CRE.
     * @details The easiest way to verify this is to check if the expression can be evaluated successfully, regardless
     * of result. This is what this function does.
     */
    bool is_expression_valid(const std::vector<std::shared_ptr<CREToken>>& expression) const;

    bool subexpression_already_registered(const std::vector<std::shared_ptr<CREToken>>& subexpression) const;

    void advance_iterator(std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_iterator,
                          const std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_end) const;

    bool end_condition(const std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_iterator,
                       const std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_end,
                       const Delimiter end_delimiter) const;

    /**
     * @brief Evaluates a subexpression from left to right.
     * @details The evaluation starts from the position where the iterator was left at. The end of the subexpression is
     * determined based on the given type of delimiter.
     *
     * The parent of the current subexpression might have determined that all evaluations within this subexpression have
     * no impact on the final result. If that is the case, then "skip_all_evaluations" should be set to true, and
     * operand evaluation will be skipped to save some resources (unless "force_all_evaluations" was also set to true).
     *
     * All operands can take one of two forms: a SectionType alone, or a SectionType followed by a SectionID. The given
     * section type evaluators are used as part of the evaluation process of all operands. The instance evaluators are
     * the second layer of operand evalution, used only when a SectionID is present and the type evaluation succeeded.
     * @param expression_iterator The cursor corresponding to the expression that is being evaluated. The initial value
     * indicates the start of the subexpression.
     * @param expression_end Points towards the end of the whole expression.
     * @param section_type_evaluators Entities used to evaluate whether or not the current software supports the
     * corresponding SectionType.
     * @param section_instance_evaluators Used to evaluate the operands that contain a SectionID.
     * @param end_delimiter The type of delimiter that is used for judging the end of the subexpression.
     * @param skip_all_evaluations If set to "true", all operand evaluations wihtin this subexpressions will be skipped
     * (unless "force_all_evaluations" was also set to true). However, some validity checks will still be performed.
     * @param force_all_evaluations Forces all evaluations to be performed. This flag has a higher priority than
     * "skip_all_evaluations". This is meant to be used when validating the whole CRE is desired.
     */
    ov::CompatibilityCheck evaluate(
        std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_iterator,
        const std::vector<std::shared_ptr<CREToken>>::const_iterator& expression_end,
        const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& section_type_evaluators,
        const std::unordered_map<SectionID, SingleSectionInstanceEvaluator>& section_instance_evaluators,
        const Delimiter end_delimiter,
        const bool skip_all_evaluations = false,
        const bool force_all_evaluations = false) const;

    /**
     * @note Stitched together using "AND"s, these subexpression form the whole expression.
     */
    std::vector<std::vector<std::shared_ptr<CREToken>>> m_subexpressions;

    Logger m_logger;
};

}  // namespace intel_npu
