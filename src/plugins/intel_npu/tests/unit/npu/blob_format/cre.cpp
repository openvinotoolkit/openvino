// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/cre.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/section_type.hpp"
#include "intel_npu/common/supported_section_type_evaluator.hpp"

using namespace intel_npu;
using testing::_;

// Name, expression, supported types, unsupported instances, instances of unknown support
using CREParams = std::tuple<std::vector<std::shared_ptr<CREToken>>,
                             std::vector<SectionTypeCode>,
                             std::vector<uint16_t>,
                             std::vector<uint16_t>,
                             ov::CompatibilityCheck>;

namespace {

constexpr std::string_view STRING_THAT_EVALUATES_TO_SUPPORTED = "1";
constexpr std::string_view STRING_THAT_EVALUATES_TO_UNSUPPORTED = "";
constexpr std::string_view STRING_THAT_EVALUATES_TO_UNKNOWN = "0";
const std::string TEST_NAME_FIELDS_SEPARATOR = "__";
constexpr char VALUES_SEPARATOR = ',';

const auto UNKNOWN_TOKEN = std::make_shared<SectionType>(SectionTypeCode::UNKNOWN);
const auto RUNTIME_REQUIREMENTS_TOKEN = std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS);
const auto ELF_MAIN_SCHEDULE_TOKEN = std::make_shared<SectionType>(SectionTypeCode::ELF_MAIN_SCHEDULE);
const auto ELF_INIT_SCHEDULES_TOKEN = std::make_shared<SectionType>(SectionTypeCode::ELF_INIT_SCHEDULES);
const auto DYNAMIC_SCHEDULE_TOKEN = std::make_shared<SectionType>(SectionTypeCode::DYNAMIC_SCHEDULE);
const auto IO_LAYOUTS_TOKEN = std::make_shared<SectionType>(SectionTypeCode::IO_LAYOUTS);
const auto BATCH_SIZE_TOKEN = std::make_shared<SectionType>(SectionTypeCode::BATCH_SIZE);
const auto ENCRYPTED_SCHEDULES_FLAG_TOKEN = std::make_shared<SectionType>(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG);
const auto COMPILER_VERSION_TOKEN = std::make_shared<SectionType>(SectionTypeCode::COMPILER_VERSION);

const auto ID_0_TOKEN = std::make_shared<SectionID>(0);
const auto ID_1_TOKEN = std::make_shared<SectionID>(1);
const auto ID_2_TOKEN = std::make_shared<SectionID>(2);
const auto LAST_ID_TOKEN = std::make_shared<SectionID>(std::numeric_limits<uint16_t>::max());

constexpr SectionTypeCode RUNTIME_REQUIREMENTS_CODE = SectionTypeCode::RUNTIME_REQUIREMENTS;
constexpr SectionTypeCode ELF_MAIN_SCHEDULE_CODE = SectionTypeCode::ELF_MAIN_SCHEDULE;
constexpr SectionTypeCode ELF_INIT_SCHEDULES_CODE = SectionTypeCode::ELF_INIT_SCHEDULES;
constexpr SectionTypeCode BATCH_SIZE_CODE = SectionTypeCode::BATCH_SIZE;

// TODO could use gtests "MOCK"
class MockTypeEvaluator : public ISectionTypeEvaluator {
public:
    MockTypeEvaluator(const bool result) : ISectionTypeEvaluator(), m_result(result) {}

private:
    bool evaluate() const override {
        return m_result;
    }

    bool m_result;
};

class MockInstanceEvaluator : public ISectionInstanceEvaluator {
public:
    ov::CompatibilityCheck evaluate(std::string_view runtime_requirements) const override {
        if (runtime_requirements == STRING_THAT_EVALUATES_TO_SUPPORTED) {
            return ov::CompatibilityCheck::SUPPORTED;
        }
        if (runtime_requirements == STRING_THAT_EVALUATES_TO_UNSUPPORTED) {
            return ov::CompatibilityCheck::UNSUPPORTED;
        }
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }
};

const auto MOCK_INSTANCE_EVALUATOR = std::make_shared<MockInstanceEvaluator>();

CRE make_complex_cre_1() {
    return CRE({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR),
                std::make_shared<SectionType>(SectionTypeCode::ELF_MAIN_SCHEDULE),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::NOT),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OPEN),
                std::make_shared<SectionType>(SectionTypeCode::ELF_INIT_SCHEDULES),
                std::make_shared<SectionID>(2),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND),
                std::make_shared<SectionType>(SectionTypeCode::DYNAMIC_SCHEDULE),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::NOT),
                std::make_shared<SectionType>(SectionTypeCode::IO_LAYOUTS),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OPEN),
                std::make_shared<SectionType>(SectionTypeCode::BATCH_SIZE),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::CLOSE),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND),
                std::make_shared<SectionType>(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::CLOSE),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR),
                std::make_shared<SectionType>(SectionTypeCode::COMPILER_VERSION),
                std::make_shared<SectionID>(0)});
}

CRE make_complex_cre_2() {
    return CRE({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND),
                std::make_shared<SectionType>(SectionTypeCode::ELF_MAIN_SCHEDULE),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::NOT),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OPEN),
                std::make_shared<SectionType>(SectionTypeCode::DYNAMIC_SCHEDULE),
                std::make_shared<SectionID>(2),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND),
                std::make_shared<SectionType>(SectionTypeCode::ELF_INIT_SCHEDULES),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::NOT),
                std::make_shared<SectionType>(SectionTypeCode::IO_LAYOUTS),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OPEN),
                std::make_shared<SectionType>(SectionTypeCode::BATCH_SIZE),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::CLOSE),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND),
                std::make_shared<SectionType>(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::CLOSE),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR),
                std::make_shared<SectionType>(SectionTypeCode::COMPILER_VERSION),
                std::make_shared<SectionID>(0)});
}

CREParams make_test_params(const std::vector<std::shared_ptr<CREToken>>& expression,
                           const ov::CompatibilityCheck expected_result,
                           const std::vector<SectionTypeCode>& supported_section_types = {},
                           const std::vector<uint16_t>& unsupported_section_instances = {},
                           const std::vector<uint16_t>& section_instances_unknown_support = {}) {
    return CREParams(expression,
                     supported_section_types,
                     unsupported_section_instances,
                     section_instances_unknown_support,
                     expected_result);
}

CREParams make_test_params_invalid_expression(const std::vector<std::shared_ptr<CREToken>>& expression) {
    return make_test_params(expression, ov::CompatibilityCheck::UNSUPPORTED);
}

std::vector<CREParams> generate_invalid_test_cases(
    const std::vector<std::vector<std::shared_ptr<CREToken>>>& expression) {
    std::vector<CREParams> result;
    result.resize(expression.size());
    std::transform(expression.begin(), expression.end(), result.begin(), make_test_params_invalid_expression);
    return result;
}

}  // namespace

using CRETests = ::testing::Test;

TEST_F(CRETests, EmptyCtor) {
    CRE cre;
    ASSERT_TRUE(cre.empty());
    ASSERT_EQ(cre.get_expression_length(), 0);
    ASSERT_EQ(cre.get_expression(), std::vector<std::shared_ptr<CREToken>>{});
}

TEST_F(CRETests, NonEmptyCtor) {
    CRE cre({BATCH_SIZE_TOKEN});
    ASSERT_FALSE(cre.empty());
    ASSERT_EQ(cre.get_expression_length(), 1);
    ASSERT_EQ(cre.get_expression(), std::vector<std::shared_ptr<CREToken>>{BATCH_SIZE_TOKEN});
}

TEST_F(CRETests, CtorAcceptsAllValidTokenTypes) {
    const std::vector<std::shared_ptr<CREToken>> expression{ELF_MAIN_SCHEDULE_TOKEN,
                                                            ID_1_TOKEN,
                                                            CRE::AND_PTR,
                                                            BATCH_SIZE_TOKEN,
                                                            CRE::OR_PTR,
                                                            CRE::NOT_PTR,
                                                            CRE::OPEN_PTR,
                                                            ELF_INIT_SCHEDULES_TOKEN,
                                                            ID_0_TOKEN,
                                                            CRE::CLOSE_PTR};
    CRE cre(expression);
    ASSERT_FALSE(cre.empty());
    ASSERT_EQ(cre.get_expression_length(), expression.size());
    ASSERT_EQ(cre.get_expression(), expression);
}

TEST_F(CRETests, NonEmptyAppendToExpression) {
    CRE cre1({BATCH_SIZE_TOKEN});
    cre1.append_to_expression(ELF_MAIN_SCHEDULE_TOKEN);
    std::vector<std::shared_ptr<CREToken>> reference{BATCH_SIZE_TOKEN, CRE::AND_PTR, ELF_MAIN_SCHEDULE_TOKEN};

    ASSERT_FALSE(cre1.empty());
    ASSERT_EQ(cre1.get_expression_length(), reference.size());
    ASSERT_EQ(cre1.get_expression(), reference);

    CRE cre2;
    cre2.append_to_expression({BATCH_SIZE_TOKEN, CRE::AND_PTR, ELF_MAIN_SCHEDULE_TOKEN});
    ASSERT_FALSE(cre2.empty());
    ASSERT_EQ(cre2.get_expression_length(), reference.size());
    ASSERT_EQ(cre2.get_expression(), reference);
}

TEST_F(CRETests, EmptyAppendToExpression) {
    CRE cre1({BATCH_SIZE_TOKEN});
    cre1.append_to_expression(std::vector<std::shared_ptr<CREToken>>{});
    std::vector<std::shared_ptr<CREToken>> reference{BATCH_SIZE_TOKEN};

    ASSERT_FALSE(cre1.empty());
    ASSERT_EQ(cre1.get_expression_length(), reference.size());
    ASSERT_EQ(cre1.get_expression(), reference);

    CRE cre2;
    cre2.append_to_expression(std::vector<std::shared_ptr<CREToken>>{});
    ASSERT_TRUE(cre2.empty());
}

TEST_F(CRETests, AppendInvalidExpression) {
    CRE cre;
    OV_EXPECT_THROW(cre.append_to_expression(CRE::AND_PTR), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(CRE::OR_PTR), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(CRE::NOT_PTR), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(CRE::OPEN_PTR), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(CRE::CLOSE_PTR), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(ID_0_TOKEN), InvalidCRE, _);

    OV_EXPECT_THROW(cre.append_to_expression(std::vector<std::shared_ptr<CREToken>>{CRE::AND_PTR}), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(std::vector<std::shared_ptr<CREToken>>{CRE::OR_PTR}), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(std::vector<std::shared_ptr<CREToken>>{CRE::NOT_PTR}), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(std::vector<std::shared_ptr<CREToken>>{CRE::OPEN_PTR}), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(std::vector<std::shared_ptr<CREToken>>{CRE::CLOSE_PTR}), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression(std::vector<std::shared_ptr<CREToken>>{ID_0_TOKEN}), InvalidCRE, _);
    OV_EXPECT_THROW(cre.append_to_expression({ELF_MAIN_SCHEDULE_TOKEN,
                                              CRE::AND_PTR,
                                              BATCH_SIZE_TOKEN,
                                              CRE::OR_PTR,
                                              ELF_INIT_SCHEDULES_TOKEN,
                                              CRE::CLOSE_PTR}),
                    InvalidCRE,
                    _);
}

TEST_F(CRETests, AppendingAvoidsRedundancy) {
    const std::vector<std::shared_ptr<CREToken>> expression{ELF_MAIN_SCHEDULE_TOKEN,
                                                            ID_1_TOKEN,
                                                            CRE::AND_PTR,
                                                            BATCH_SIZE_TOKEN,
                                                            CRE::OR_PTR,
                                                            CRE::NOT_PTR,
                                                            CRE::OPEN_PTR,
                                                            ELF_INIT_SCHEDULES_TOKEN,
                                                            ID_0_TOKEN,
                                                            CRE::CLOSE_PTR};
    const std::vector<std::shared_ptr<CREToken>> same_expression{
        std::make_shared<SectionType>(SectionTypeCode::ELF_MAIN_SCHEDULE),
        std::make_shared<SectionID>(1),
        std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND),
        BATCH_SIZE_TOKEN,
        CRE::OR_PTR,
        CRE::NOT_PTR,
        CRE::OPEN_PTR,
        ELF_INIT_SCHEDULES_TOKEN,
        ID_0_TOKEN,
        CRE::CLOSE_PTR};

    CRE cre(expression);
    cre.append_to_expression(same_expression);
    ASSERT_EQ(cre.get_expression(), expression);
    cre.append_to_expression(same_expression);
    ASSERT_EQ(cre.get_expression(), expression);

    cre = CRE({BATCH_SIZE_TOKEN});
    cre.append_to_expression(BATCH_SIZE_TOKEN);
    ASSERT_EQ(cre.get_expression(), std::vector<std::shared_ptr<CREToken>>{BATCH_SIZE_TOKEN});
    cre.append_to_expression(BATCH_SIZE_TOKEN);
    ASSERT_EQ(cre.get_expression(), std::vector<std::shared_ptr<CREToken>>{BATCH_SIZE_TOKEN});
}

TEST_F(CRETests, AppendAndBrackets) {
    CRE cre;
    cre.append_to_expression(std::vector<std::shared_ptr<CREToken>>{BATCH_SIZE_TOKEN});
    ASSERT_EQ(cre.get_expression(), std::vector<std::shared_ptr<CREToken>>{BATCH_SIZE_TOKEN});

    cre.append_to_expression(std::vector<std::shared_ptr<CREToken>>{ELF_MAIN_SCHEDULE_TOKEN});
    std::vector<std::shared_ptr<CREToken>> reference{BATCH_SIZE_TOKEN,
                                                     CRE::AND_PTR,
                                                     CRE::OPEN_PTR,
                                                     ELF_MAIN_SCHEDULE_TOKEN,
                                                     CRE::CLOSE_PTR};
    ASSERT_EQ(cre.get_expression(), reference);

    cre.append_to_expression(std::vector<std::shared_ptr<CREToken>>{ELF_MAIN_SCHEDULE_TOKEN});
    ASSERT_EQ(cre.get_expression(), reference);

    cre.append_to_expression(
        std::vector<std::shared_ptr<CREToken>>{CRE::OPEN_PTR, ELF_INIT_SCHEDULES_TOKEN, CRE::CLOSE_PTR});
    reference = std::vector<std::shared_ptr<CREToken>>{BATCH_SIZE_TOKEN,
                                                       CRE::AND_PTR,
                                                       CRE::OPEN_PTR,
                                                       ELF_MAIN_SCHEDULE_TOKEN,
                                                       CRE::CLOSE_PTR,
                                                       CRE::AND_PTR,
                                                       CRE::OPEN_PTR,
                                                       CRE::OPEN_PTR,
                                                       ELF_INIT_SCHEDULES_TOKEN,
                                                       CRE::CLOSE_PTR,
                                                       CRE::CLOSE_PTR};
    ASSERT_EQ(cre.get_expression(), reference);
}

TEST_F(CRETests, EqualityOperatorsOnEmptyExpression) {
    ASSERT_TRUE(CRE() == CRE());
    ASSERT_FALSE(CRE() != CRE());

    ASSERT_FALSE(CRE() == CRE({BATCH_SIZE_TOKEN}));
    ASSERT_TRUE(CRE() != CRE({BATCH_SIZE_TOKEN}));
}

TEST_F(CRETests, EqualityOperatorsOnSimpleExpressions) {
    CRE cre1({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS)});
    CRE cre2({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS)});
    ASSERT_TRUE(cre1 == cre2);
    ASSERT_FALSE(cre1 != cre2);

    cre2 = CRE({std::make_shared<SectionType>(SectionTypeCode::BATCH_SIZE)});
    ASSERT_FALSE(cre1 == cre2);
    ASSERT_TRUE(cre1 != cre2);

    cre2 = CRE({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND),
                std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS)});
    ASSERT_FALSE(cre1 == cre2);
    ASSERT_TRUE(cre1 != cre2);

    cre1 = CRE({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND),
                std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS)});
    ASSERT_TRUE(cre1 == cre2);
    ASSERT_FALSE(cre1 != cre2);

    cre2 = CRE({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS),
                std::make_shared<CRESpecialToken>(CRESpecialTokenCode::OR),
                std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS)});
    ASSERT_FALSE(cre1 == cre2);
    ASSERT_TRUE(cre1 != cre2);

    cre1 = CRE({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS), std::make_shared<SectionID>(0)});
    cre2 = CRE({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS), std::make_shared<SectionID>(0)});
    ASSERT_TRUE(cre1 == cre2);
    ASSERT_FALSE(cre1 != cre2);

    cre2 = CRE({std::make_shared<SectionType>(SectionTypeCode::RUNTIME_REQUIREMENTS), std::make_shared<SectionID>(1)});
    ASSERT_FALSE(cre1 == cre2);
    ASSERT_TRUE(cre1 != cre2);
}

TEST_F(CRETests, EqualityOperatorsOnComplexExpressions) {
    const CRE cre1 = make_complex_cre_1();
    ASSERT_TRUE(cre1 == cre1);
    ASSERT_FALSE(cre1 == make_complex_cre_2());
    ASSERT_TRUE(cre1 == make_complex_cre_1());
}

TEST_F(CRETests, ToStringEmpty) {
    CRE cre;
    ASSERT_EQ(cre.to_string(), "");
}

TEST_F(CRETests, ToStringSingleToken) {
    CRE cre({RUNTIME_REQUIREMENTS_TOKEN});
    ASSERT_EQ(cre.to_string(), "RUNTIME_REQUIREMENTS");

    cre = CRE({ELF_MAIN_SCHEDULE_TOKEN});
    ASSERT_EQ(cre.to_string(), "ELF_MAIN_SCHEDULE");

    cre = CRE({ELF_INIT_SCHEDULES_TOKEN});
    ASSERT_EQ(cre.to_string(), "ELF_INIT_SCHEDULES");

    cre = CRE({DYNAMIC_SCHEDULE_TOKEN});
    ASSERT_EQ(cre.to_string(), "DYNAMIC_SCHEDULE");

    cre = CRE({IO_LAYOUTS_TOKEN});
    ASSERT_EQ(cre.to_string(), "IO_LAYOUTS");

    cre = CRE({BATCH_SIZE_TOKEN});
    ASSERT_EQ(cre.to_string(), "BATCH_SIZE");

    cre = CRE({ENCRYPTED_SCHEDULES_FLAG_TOKEN});
    ASSERT_EQ(cre.to_string(), "ENCRYPTED_SCHEDULES_FLAG");

    cre = CRE({COMPILER_VERSION_TOKEN});
    ASSERT_EQ(cre.to_string(), "COMPILER_VERSION");
}

TEST_F(CRETests, ToStringAllTokensOneSubexpression) {
    CRE cre({
        RUNTIME_REQUIREMENTS_TOKEN,
        CRE::OR_PTR,
        ELF_MAIN_SCHEDULE_TOKEN,
        CRE::AND_PTR,
        CRE::NOT_PTR,
        CRE::OPEN_PTR,
        ELF_INIT_SCHEDULES_TOKEN,
        ID_2_TOKEN,
        CRE::AND_PTR,
        DYNAMIC_SCHEDULE_TOKEN,
        CRE::OR_PTR,
        CRE::NOT_PTR,
        IO_LAYOUTS_TOKEN,
        CRE::OR_PTR,
        CRE::OPEN_PTR,
        BATCH_SIZE_TOKEN,
        CRE::CLOSE_PTR,
        CRE::AND_PTR,
        ENCRYPTED_SCHEDULES_FLAG_TOKEN,
        CRE::CLOSE_PTR,
        CRE::OR_PTR,
        COMPILER_VERSION_TOKEN,
        ID_0_TOKEN,
    });
    ASSERT_EQ(
        cre.to_string(),
        "RUNTIME_REQUIREMENTS.OR.ELF_MAIN_SCHEDULE.AND.NOT.OPEN.ELF_INIT_SCHEDULES_2.AND.DYNAMIC_SCHEDULE.OR.NOT.IO_"
        "LAYOUTS.OR.OPEN.BATCH_SIZE.CLOSE.AND.ENCRYPTED_SCHEDULES_FLAG.CLOSE.OR.COMPILER_VERSION_0");
}

TEST_F(CRETests, ToStringAllTokensMultipleSubexpression) {
    CRE cre({
        RUNTIME_REQUIREMENTS_TOKEN,
        CRE::OR_PTR,
        ELF_MAIN_SCHEDULE_TOKEN,
    });
    cre.append_to_expression({ELF_INIT_SCHEDULES_TOKEN,
                              ID_2_TOKEN,
                              CRE::AND_PTR,
                              DYNAMIC_SCHEDULE_TOKEN,
                              CRE::OR_PTR,
                              CRE::NOT_PTR,
                              IO_LAYOUTS_TOKEN,
                              CRE::OR_PTR,
                              CRE::OPEN_PTR,
                              BATCH_SIZE_TOKEN,
                              CRE::CLOSE_PTR,
                              CRE::AND_PTR,
                              ENCRYPTED_SCHEDULES_FLAG_TOKEN});
    cre.append_to_expression({COMPILER_VERSION_TOKEN, ID_0_TOKEN});
    ASSERT_EQ(cre.to_string(),
              "RUNTIME_REQUIREMENTS.OR.ELF_MAIN_SCHEDULE.AND.OPEN.ELF_INIT_SCHEDULES_2.AND.DYNAMIC_SCHEDULE.OR.NOT.IO_"
              "LAYOUTS.OR.OPEN.BATCH_SIZE.CLOSE.AND.ENCRYPTED_SCHEDULES_FLAG.CLOSE.AND.OPEN.COMPILER_VERSION_0.CLOSE");
}

TEST_F(CRETests, UnknownToString) {
    CRE cre({UNKNOWN_TOKEN});
    OV_EXPECT_THROW(cre.to_string(), ov::Exception, _);

    cre = CRE({ELF_MAIN_SCHEDULE_TOKEN});
    cre.append_to_expression(UNKNOWN_TOKEN);
    OV_EXPECT_THROW(cre.to_string(), ov::Exception, _);
}

TEST_F(CRETests, FromEmptyString) {
    CRE cre = CRE::from_string("");
    ASSERT_TRUE(cre.empty());
}

TEST_F(CRETests, FromStringSingleValidToken) {
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTS") == CRE({RUNTIME_REQUIREMENTS_TOKEN}));
    ASSERT_TRUE(CRE::from_string("ELF_MAIN_SCHEDULE") == CRE({ELF_MAIN_SCHEDULE_TOKEN}));
    ASSERT_TRUE(CRE::from_string("ELF_INIT_SCHEDULES") == CRE({ELF_INIT_SCHEDULES_TOKEN}));
    ASSERT_TRUE(CRE::from_string("DYNAMIC_SCHEDULE") == CRE({DYNAMIC_SCHEDULE_TOKEN}));
    ASSERT_TRUE(CRE::from_string("IO_LAYOUTS") == CRE({IO_LAYOUTS_TOKEN}));
    ASSERT_TRUE(CRE::from_string("BATCH_SIZE") == CRE({BATCH_SIZE_TOKEN}));
    ASSERT_TRUE(CRE::from_string("ENCRYPTED_SCHEDULES_FLAG") == CRE({ENCRYPTED_SCHEDULES_FLAG_TOKEN}));
    ASSERT_TRUE(CRE::from_string("COMPILER_VERSION") == CRE({COMPILER_VERSION_TOKEN}));
}

TEST_F(CRETests, FromStringSectionIDs) {
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTS_0") == CRE({RUNTIME_REQUIREMENTS_TOKEN, ID_0_TOKEN}));
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTS_" + std::to_string(std::numeric_limits<uint16_t>::max())) ==
                CRE({RUNTIME_REQUIREMENTS_TOKEN, LAST_ID_TOKEN}));
}

TEST_F(CRETests, FromStringOperators) {
    ASSERT_TRUE(CRE::from_string("NOT.RUNTIME_REQUIREMENTS") == CRE({CRE::NOT_PTR, RUNTIME_REQUIREMENTS_TOKEN}));
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTS.AND.RUNTIME_REQUIREMENTS") ==
                CRE({RUNTIME_REQUIREMENTS_TOKEN, CRE::AND_PTR, RUNTIME_REQUIREMENTS_TOKEN}));
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTS.OR.RUNTIME_REQUIREMENTS") ==
                CRE({RUNTIME_REQUIREMENTS_TOKEN, CRE::OR_PTR, RUNTIME_REQUIREMENTS_TOKEN}));
    ASSERT_TRUE(CRE::from_string("OPEN.RUNTIME_REQUIREMENTS.CLOSE") ==
                CRE({CRE::OPEN_PTR, RUNTIME_REQUIREMENTS_TOKEN, CRE::CLOSE_PTR}));
}

TEST_F(CRETests, FromStringComplexExpression) {
    CRE result = CRE::from_string(
        "RUNTIME_REQUIREMENTS.OR.ELF_MAIN_SCHEDULE.AND.NOT.OPEN.ELF_INIT_SCHEDULES_2.AND.DYNAMIC_SCHEDULE.OR.NOT.IO_"
        "LAYOUTS.OR.OPEN.BATCH_SIZE.CLOSE.AND.ENCRYPTED_SCHEDULES_FLAG.CLOSE.OR.COMPILER_VERSION_0");
    CRE reference({
        RUNTIME_REQUIREMENTS_TOKEN,
        CRE::OR_PTR,
        ELF_MAIN_SCHEDULE_TOKEN,
        CRE::AND_PTR,
        CRE::NOT_PTR,
        CRE::OPEN_PTR,
        ELF_INIT_SCHEDULES_TOKEN,
        ID_2_TOKEN,
        CRE::AND_PTR,
        DYNAMIC_SCHEDULE_TOKEN,
        CRE::OR_PTR,
        CRE::NOT_PTR,
        IO_LAYOUTS_TOKEN,
        CRE::OR_PTR,
        CRE::OPEN_PTR,
        BATCH_SIZE_TOKEN,
        CRE::CLOSE_PTR,
        CRE::AND_PTR,
        ENCRYPTED_SCHEDULES_FLAG_TOKEN,
        CRE::CLOSE_PTR,
        CRE::OR_PTR,
        COMPILER_VERSION_TOKEN,
        ID_0_TOKEN,
    });

    ASSERT_TRUE(result == reference);
}

TEST_F(CRETests, UnknownFromString) {
    ASSERT_TRUE(CRE::from_string("RUNTIMEREQUIREMENTS") == CRE({UNKNOWN_TOKEN}));
    ASSERT_TRUE(CRE::from_string("UNKNOWN") == CRE({UNKNOWN_TOKEN}));
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTS0") == CRE({UNKNOWN_TOKEN}));
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTS_0a0") == CRE({UNKNOWN_TOKEN}));
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTS_BATCH_SIZE") == CRE({UNKNOWN_TOKEN}));
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTSBATCH_SIZE") == CRE({UNKNOWN_TOKEN}));
    ASSERT_TRUE(CRE::from_string("RUNTIMEREQUIREMENTS") == CRE({UNKNOWN_TOKEN}));
    ASSERT_TRUE(CRE::from_string("RUNTIME_REQUIREMENTS_-1") == CRE({UNKNOWN_TOKEN}));
}

TEST_F(CRETests, FromInvalidStrings) {
    OV_EXPECT_THROW(CRE::from_string("0"), ov::Exception, _);
    OV_EXPECT_THROW(CRE::from_string("AND"), InvalidCRE, _);
    OV_EXPECT_THROW(CRE::from_string("NOT"), InvalidCRE, _);
    OV_EXPECT_THROW(
        CRE::from_string("RUNTIME_REQUIREMENTS_" + std::to_string(std::numeric_limits<uint16_t>::max() + 1)),
        ov::Exception,
        _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS.0"), ov::Exception, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS_0.0"), ov::Exception, _);
    OV_EXPECT_THROW(CRE::from_string(".RUNTIME_REQUIREMENTS"), ov::Exception, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS."), ov::Exception, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS.OP.BATCH_SIZE"), InvalidCRE, _);
    OV_EXPECT_THROW(CRE::from_string("NOT..RUNTIME_REQUIREMENTS"), ov::Exception, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS.BATCH_SIZE"), InvalidCRE, _);
    OV_EXPECT_THROW(CRE::from_string("0.BATCH_SIZE"), ov::Exception, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS.OR.0"), ov::Exception, _);
    OV_EXPECT_THROW(CRE::from_string("OPEN.RUNTIME_REQUIREMENTS"), InvalidCRE, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS.CLOSE"), InvalidCRE, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS.NOT.BATCH_SIZE"), InvalidCRE, _);
    OV_EXPECT_THROW(CRE::from_string("OR.BATCH_SIZE"), InvalidCRE, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS.OPEN.BATCH_SIZE.CLOSE"), InvalidCRE, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS.AND"), InvalidCRE, _);
    OV_EXPECT_THROW(CRE::from_string("RUNTIME_REQUIREMENTS.NOT"), InvalidCRE, _);
}

TEST_F(CRETests, ToStringFromStringChain) {
    CRE cre({
        RUNTIME_REQUIREMENTS_TOKEN,
        CRE::OR_PTR,
        ELF_MAIN_SCHEDULE_TOKEN,
        CRE::AND_PTR,
        CRE::NOT_PTR,
        CRE::OPEN_PTR,
        ELF_INIT_SCHEDULES_TOKEN,
        ID_2_TOKEN,
        CRE::AND_PTR,
        DYNAMIC_SCHEDULE_TOKEN,
        CRE::OR_PTR,
        CRE::NOT_PTR,
        IO_LAYOUTS_TOKEN,
        CRE::OR_PTR,
        CRE::OPEN_PTR,
        BATCH_SIZE_TOKEN,
        CRE::CLOSE_PTR,
        CRE::AND_PTR,
        ENCRYPTED_SCHEDULES_FLAG_TOKEN,
        CRE::CLOSE_PTR,
        CRE::OR_PTR,
        COMPILER_VERSION_TOKEN,
        ID_0_TOKEN,
    });
    ASSERT_TRUE(cre == CRE::from_string(cre.to_string()));
}

TEST_F(CRETests, SectionTypeEvaluation) {
    CRE cre({BATCH_SIZE_TOKEN});
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> section_type_evaluators;

    section_type_evaluators.emplace(BATCH_SIZE_CODE, std::make_shared<MockTypeEvaluator>(true));
    cre.check_compatibility(section_type_evaluators, {});
    ASSERT_TRUE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_EQ(section_type_evaluators.at(BATCH_SIZE_CODE)->get_result(), true);

    section_type_evaluators[BATCH_SIZE_CODE] = std::make_shared<MockTypeEvaluator>(false);
    cre.check_compatibility(section_type_evaluators, {});
    ASSERT_TRUE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_EQ(section_type_evaluators.at(BATCH_SIZE_CODE)->get_result(), false);
}

TEST_F(CRETests, SectionIdEvaluation) {
    CRE cre({BATCH_SIZE_TOKEN, ID_0_TOKEN});
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> section_type_evaluators;
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> section_instance_evaluators;
    section_instance_evaluators.emplace(
        SectionID(0),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));

    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_FALSE(section_instance_evaluators.at(SectionID(0)).evaluated());

    section_type_evaluators.emplace(BATCH_SIZE_CODE, std::make_shared<MockTypeEvaluator>(true));
    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(0)).evaluated());
    ASSERT_EQ(section_instance_evaluators.at(SectionID(0)).get_result(), ov::CompatibilityCheck::NOT_APPLICABLE);

    section_instance_evaluators.erase(SectionID(0));
    section_instance_evaluators.emplace(
        SectionID(0),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_SUPPORTED));
    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(0)).evaluated());
    ASSERT_EQ(section_instance_evaluators.at(SectionID(0)).get_result(), ov::CompatibilityCheck::SUPPORTED);

    section_instance_evaluators.erase(SectionID(0));
    section_instance_evaluators.emplace(
        SectionID(0),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNSUPPORTED));
    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(0)).evaluated());
    ASSERT_EQ(section_instance_evaluators.at(SectionID(0)).get_result(), ov::CompatibilityCheck::UNSUPPORTED);
}

TEST_F(CRETests, AndShallowEvaluation) {
    CRE cre({RUNTIME_REQUIREMENTS_TOKEN, ID_1_TOKEN, CRE::AND_PTR, BATCH_SIZE_TOKEN, ID_0_TOKEN});
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> section_type_evaluators;
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> section_instance_evaluators;

    section_type_evaluators.emplace(RUNTIME_REQUIREMENTS_CODE, std::make_shared<MockTypeEvaluator>(false));
    section_type_evaluators.emplace(BATCH_SIZE_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_instance_evaluators.emplace(
        SectionID(0),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));

    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_type_evaluators.at(RUNTIME_REQUIREMENTS_CODE)->evaluated());
    ASSERT_FALSE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_FALSE(section_instance_evaluators.at(SectionID(0)).evaluated());

    section_type_evaluators[RUNTIME_REQUIREMENTS_CODE] = std::make_shared<MockTypeEvaluator>(true);
    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_type_evaluators.at(RUNTIME_REQUIREMENTS_CODE)->evaluated());
    ASSERT_TRUE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(0)).evaluated());

    section_type_evaluators[RUNTIME_REQUIREMENTS_CODE] = std::make_shared<MockTypeEvaluator>(true);
    section_type_evaluators[BATCH_SIZE_CODE] = std::make_shared<MockTypeEvaluator>(true);
    section_instance_evaluators.erase(SectionID(0));
    section_instance_evaluators.emplace(
        SectionID(0),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));
    section_instance_evaluators.emplace(
        SectionID(1),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));
    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_type_evaluators.at(RUNTIME_REQUIREMENTS_CODE)->evaluated());
    ASSERT_TRUE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(0)).evaluated());
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(1)).evaluated());
}

TEST_F(CRETests, OrShallowEvaluation) {
    CRE cre({RUNTIME_REQUIREMENTS_TOKEN, ID_1_TOKEN, CRE::OR_PTR, BATCH_SIZE_TOKEN, ID_0_TOKEN});
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> section_type_evaluators;
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> section_instance_evaluators;

    section_type_evaluators.emplace(RUNTIME_REQUIREMENTS_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_type_evaluators.emplace(BATCH_SIZE_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_instance_evaluators.emplace(
        SectionID(0),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));

    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_type_evaluators.at(RUNTIME_REQUIREMENTS_CODE)->evaluated());
    ASSERT_FALSE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_FALSE(section_instance_evaluators.at(SectionID(0)).evaluated());

    section_type_evaluators[RUNTIME_REQUIREMENTS_CODE] = std::make_shared<MockTypeEvaluator>(false);
    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_type_evaluators.at(RUNTIME_REQUIREMENTS_CODE)->evaluated());
    ASSERT_TRUE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(0)).evaluated());

    section_type_evaluators[RUNTIME_REQUIREMENTS_CODE] = std::make_shared<MockTypeEvaluator>(true);
    section_type_evaluators[BATCH_SIZE_CODE] = std::make_shared<MockTypeEvaluator>(true);
    section_instance_evaluators.erase(SectionID(0));
    section_instance_evaluators.emplace(
        SectionID(0),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));
    section_instance_evaluators.emplace(
        SectionID(1),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));
    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_type_evaluators.at(RUNTIME_REQUIREMENTS_CODE)->evaluated());
    ASSERT_TRUE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(0)).evaluated());
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(1)).evaluated());
}

TEST_F(CRETests, AndDeepEvaluation) {
    // RR AND NOT (BS0 AND (EMS1)) AND EIS
    // Only EMS1 should not evaluate
    CRE cre({RUNTIME_REQUIREMENTS_TOKEN,
             CRE::AND_PTR,
             CRE::NOT_PTR,
             CRE::OPEN_PTR,
             BATCH_SIZE_TOKEN,
             ID_0_TOKEN,
             CRE::AND_PTR,
             CRE::OPEN_PTR,
             ELF_MAIN_SCHEDULE_TOKEN,
             ID_1_TOKEN,
             CRE::CLOSE_PTR,
             CRE::CLOSE_PTR,
             CRE::AND_PTR,
             ELF_INIT_SCHEDULES_TOKEN});

    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> section_type_evaluators;
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> section_instance_evaluators;
    section_type_evaluators.emplace(RUNTIME_REQUIREMENTS_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_type_evaluators.emplace(BATCH_SIZE_CODE, std::make_shared<MockTypeEvaluator>(false));
    section_type_evaluators.emplace(ELF_MAIN_SCHEDULE_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_type_evaluators.emplace(ELF_INIT_SCHEDULES_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_instance_evaluators.emplace(
        SectionID(0),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));
    section_instance_evaluators.emplace(
        SectionID(1),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));

    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_type_evaluators.at(RUNTIME_REQUIREMENTS_CODE)->evaluated());
    ASSERT_TRUE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_FALSE(section_type_evaluators.at(ELF_MAIN_SCHEDULE_CODE)->evaluated());
    ASSERT_TRUE(section_type_evaluators.at(ELF_INIT_SCHEDULES_CODE)->evaluated());
    ASSERT_FALSE(section_instance_evaluators.at(SectionID(0)).evaluated());
    ASSERT_FALSE(section_instance_evaluators.at(SectionID(1)).evaluated());
}

TEST_F(CRETests, OrDeepEvaluation) {
    // RR AND (BS0 OR (EMS1)) AND EIS
    // Only EMS1 should not evaluate
    CRE cre({RUNTIME_REQUIREMENTS_TOKEN,
             CRE::AND_PTR,
             CRE::OPEN_PTR,
             BATCH_SIZE_TOKEN,
             ID_0_TOKEN,
             CRE::OR_PTR,
             CRE::OPEN_PTR,
             ELF_MAIN_SCHEDULE_TOKEN,
             ID_1_TOKEN,
             CRE::CLOSE_PTR,
             CRE::CLOSE_PTR,
             CRE::AND_PTR,
             ELF_INIT_SCHEDULES_TOKEN});

    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> section_type_evaluators;
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> section_instance_evaluators;
    section_type_evaluators.emplace(RUNTIME_REQUIREMENTS_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_type_evaluators.emplace(BATCH_SIZE_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_type_evaluators.emplace(ELF_MAIN_SCHEDULE_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_type_evaluators.emplace(ELF_INIT_SCHEDULES_CODE, std::make_shared<MockTypeEvaluator>(true));
    section_instance_evaluators.emplace(
        SectionID(0),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_SUPPORTED));
    section_instance_evaluators.emplace(
        SectionID(1),
        SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));

    cre.check_compatibility(section_type_evaluators, section_instance_evaluators);
    ASSERT_TRUE(section_type_evaluators.at(RUNTIME_REQUIREMENTS_CODE)->evaluated());
    ASSERT_TRUE(section_type_evaluators.at(BATCH_SIZE_CODE)->evaluated());
    ASSERT_FALSE(section_type_evaluators.at(ELF_MAIN_SCHEDULE_CODE)->evaluated());
    ASSERT_TRUE(section_type_evaluators.at(ELF_INIT_SCHEDULES_CODE)->evaluated());
    ASSERT_TRUE(section_instance_evaluators.at(SectionID(0)).evaluated());
    ASSERT_FALSE(section_instance_evaluators.at(SectionID(1)).evaluated());
}

class CREEvaluationTests : public ::testing::TestWithParam<CREParams> {
protected:
    void SetUp() override {
        std::vector<SectionTypeCode> supported_section_types;
        std::vector<uint16_t> unsupported_section_instances;
        std::vector<uint16_t> section_instances_unknown_support;
        std::tie(expression,
                 supported_section_types,
                 unsupported_section_instances,
                 section_instances_unknown_support,
                 expected_result) = GetParam();

        for (const auto code : supported_section_types) {
            section_type_evaluators[SectionType(code)] = SupportedSectionTypeEvaluator::get_instance();
        }
        for (const auto id : unsupported_section_instances) {
            section_instance_evaluators.emplace(
                SectionID(id),
                SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNSUPPORTED));
        }
        for (const auto id : section_instances_unknown_support) {
            section_instance_evaluators.emplace(
                SectionID(id),
                SingleSectionInstanceEvaluator(MOCK_INSTANCE_EVALUATOR, STRING_THAT_EVALUATES_TO_UNKNOWN));
        }
    }

    std::vector<std::shared_ptr<CREToken>> expression;
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> section_type_evaluators;
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> section_instance_evaluators;
    ov::CompatibilityCheck expected_result;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CREParams> obj) {
        const auto& [expression,
                     supported_section_types,
                     unsupported_section_instances,
                     section_instances_unknown_support,
                     expected_result] = obj.param;

        std::string expression_string;
        std::string supported_section_types_string;
        std::string unsupported_section_instances_string;
        std::string section_instances_unknown_support_string;
        std::string result_string =
            std::string(TEST_NAME_FIELDS_SEPARATOR) +
            (expected_result == ov::CompatibilityCheck::SUPPORTED
                 ? "supported"
                 : (expected_result == ov::CompatibilityCheck::UNSUPPORTED ? "unsupported" : "not_applicable"));

        for (const auto& token : expression) {
            expression_string += VALUES_SEPARATOR;

            if (is_section_type(token) &&
                std::dynamic_pointer_cast<SectionType>(token)->get_code() == SectionTypeCode::UNKNOWN) {
                expression_string += "UNKNOWN";
                continue;
            }
            expression_string += token->to_string();
        }

        for (const auto code : supported_section_types) {
            supported_section_types_string += VALUES_SEPARATOR;
            supported_section_types_string += SectionType(code).to_string();
        }
        for (const auto id : unsupported_section_instances) {
            unsupported_section_instances_string += VALUES_SEPARATOR;
            unsupported_section_instances_string += SectionID(id).to_string();
        }
        for (const auto id : section_instances_unknown_support) {
            section_instances_unknown_support_string += VALUES_SEPARATOR;
            section_instances_unknown_support_string += SectionID(id).to_string();
        }

        if (!expression_string.empty()) {
            expression_string = "expression=" + expression_string.substr(1);
        }
        if (!supported_section_types_string.empty()) {
            supported_section_types_string =
                TEST_NAME_FIELDS_SEPARATOR + "supported_types=" + supported_section_types_string.substr(1);
        }
        if (!unsupported_section_instances_string.empty()) {
            unsupported_section_instances_string =
                TEST_NAME_FIELDS_SEPARATOR + "unsupported_instances=" + unsupported_section_instances_string.substr(1);
        }
        if (!section_instances_unknown_support_string.empty()) {
            section_instances_unknown_support_string =
                TEST_NAME_FIELDS_SEPARATOR + "unknown_instances=" + section_instances_unknown_support_string.substr(1);
        }

        return expression_string + supported_section_types_string + unsupported_section_instances_string +
               section_instances_unknown_support_string + result_string;
    }
};

using ValidExpression = CREEvaluationTests;

TEST_P(ValidExpression, check_compatibility) {
    EXPECT_EQ(CRE(expression).check_compatibility(section_type_evaluators, section_instance_evaluators),
              expected_result);
}

using InvalidExpression = CREEvaluationTests;

TEST_P(InvalidExpression, check_compatibility) {
    OV_EXPECT_THROW(CRE{expression}, InvalidCRE, _);
}

const std::vector<std::shared_ptr<CREToken>> expression_1{};

// RR0
const std::vector<std::shared_ptr<CREToken>> expression_2{RUNTIME_REQUIREMENTS_TOKEN, ID_0_TOKEN};

// EMS
const std::vector<std::shared_ptr<CREToken>> expression_3{ELF_MAIN_SCHEDULE_TOKEN};

/*
           AND
          /   \
       *ELF*  *BT*
*/
const std::vector<std::shared_ptr<CREToken>> expression_4{ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, BATCH_SIZE_TOKEN};

/*
              AND
           /   |   \
        *ELF* *BT* *WS*
*/
const std::vector<std::shared_ptr<CREToken>> expression_5{ELF_MAIN_SCHEDULE_TOKEN,
                                                          CRE::AND_PTR,
                                                          BATCH_SIZE_TOKEN,
                                                          CRE::AND_PTR,
                                                          ELF_INIT_SCHEDULES_TOKEN};

/*
            AND
           /   \
        *ELF*  OR
              /  \
           *BT*  *WS*
*/
const std::vector<std::shared_ptr<CREToken>> expression_6{ELF_MAIN_SCHEDULE_TOKEN,
                                                          CRE::AND_PTR,
                                                          CRE::OPEN_PTR,
                                                          BATCH_SIZE_TOKEN,
                                                          CRE::OR_PTR,
                                                          ELF_INIT_SCHEDULES_TOKEN,
                                                          CRE::CLOSE_PTR};

/*
            OR
          /    \
        *ELF*  AND
              /   \
           *BT*   *WS*
*/
const std::vector<std::shared_ptr<CREToken>> expression_7{ELF_MAIN_SCHEDULE_TOKEN,
                                                          CRE::OR_PTR,
                                                          CRE::OPEN_PTR,
                                                          BATCH_SIZE_TOKEN,
                                                          CRE::AND_PTR,
                                                          ELF_INIT_SCHEDULES_TOKEN,
                                                          CRE::CLOSE_PTR};

/*
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                    /  \    /  \
                 *BT* *WS* *WS* *BT*
*/
const std::vector<std::shared_ptr<CREToken>> expression_8{ELF_MAIN_SCHEDULE_TOKEN,
                                                          CRE::AND_PTR,
                                                          CRE::OPEN_PTR,
                                                          BATCH_SIZE_TOKEN,
                                                          CRE::OR_PTR,
                                                          ELF_INIT_SCHEDULES_TOKEN,
                                                          CRE::CLOSE_PTR,
                                                          CRE::AND_PTR,
                                                          CRE::OPEN_PTR,
                                                          ELF_INIT_SCHEDULES_TOKEN,
                                                          CRE::OR_PTR,
                                                          BATCH_SIZE_TOKEN,
                                                          CRE::CLOSE_PTR};

/*
                  ____ OR ____
                /             \
               /               \
              /                 \
         __ AND __             _ OR _
        /    |    \          /   |    \
      *ELF* *WS* *BT*     *ELF* *WS*  AND
                                     /   \
                                   *ELF* *BT*
*/
const std::vector<std::shared_ptr<CREToken>> expression_9{CRE::OPEN_PTR,
                                                          ELF_MAIN_SCHEDULE_TOKEN,
                                                          CRE::AND_PTR,
                                                          ELF_INIT_SCHEDULES_TOKEN,
                                                          CRE::AND_PTR,
                                                          BATCH_SIZE_TOKEN,
                                                          CRE::CLOSE_PTR,
                                                          CRE::OR_PTR,
                                                          CRE::OPEN_PTR,
                                                          ELF_MAIN_SCHEDULE_TOKEN,
                                                          CRE::OR_PTR,
                                                          ELF_INIT_SCHEDULES_TOKEN,
                                                          CRE::OR_PTR,
                                                          CRE::OPEN_PTR,
                                                          ELF_MAIN_SCHEDULE_TOKEN,
                                                          CRE::AND_PTR,
                                                          BATCH_SIZE_TOKEN,
                                                          CRE::CLOSE_PTR,
                                                          CRE::CLOSE_PTR};

/*
                  ____ OR ____
                /             \
               /               \
              /                 \
         __ OR __             _ AND _
        /   |    \           /   |   \
      AND  *WS* *ELF*     *ELF* *WS* *BT*
     /   \
   *BT* *ELF*
*/
// expression_9 but with reversed leaves
const std::vector<std::shared_ptr<CREToken>> expression_10{CRE::OPEN_PTR,
                                                           CRE::OPEN_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::AND_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::CLOSE_PTR,
                                                           CRE::OR_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::OR_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::CLOSE_PTR,
                                                           CRE::OR_PTR,
                                                           CRE::OPEN_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::AND_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::CLOSE_PTR};

/*
             AND
            /   \
         *ELF*   OR
                /   \
             *WS*   OR
                   /   \
                  AND   *BT*
                 /  \
             *ELF* *WS*
*/
const std::vector<std::shared_ptr<CREToken>> expression_12{ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,
                                                           CRE::OPEN_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::OR_PTR,
                                                           CRE::OPEN_PTR,
                                                           CRE::OPEN_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::CLOSE_PTR,
                                                           CRE::OR_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::CLOSE_PTR,
                                                           CRE::CLOSE_PTR};

/*
              AND
           /   |   \
        *ELF* *BT* *ELF*
*/
const std::vector<std::shared_ptr<CREToken>> expression_13{ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::AND_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN};

/*
    NOT
     |
   *ELF*
*/
const std::vector<std::shared_ptr<CREToken>> expression_14{CRE::NOT_PTR, ELF_MAIN_SCHEDULE_TOKEN};

/*
              AND
           /   |   \
        ~ELF  ~BT  *WS*
*/
const std::vector<std::shared_ptr<CREToken>> expression_16{CRE::NOT_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,
                                                           CRE::NOT_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::AND_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN};

/*
            AND
           /   \
        ~ELF  ~OR
              /  \
           *BT*  ~WS
*/
const std::vector<std::shared_ptr<CREToken>> expression_17{CRE::NOT_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,
                                                           CRE::NOT_PTR,
                                                           CRE::OPEN_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::OR_PTR,
                                                           CRE::NOT_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::CLOSE_PTR};

/*
      NOT
       |
      AND
     /   \
  *ELF*  *BT*
*/
const std::vector<std::shared_ptr<CREToken>> expression_18 =
    {CRE::NOT_PTR, CRE::OPEN_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, BATCH_SIZE_TOKEN, CRE::CLOSE_PTR};

/*
    AND
    /  \
~ELF  *BT*
*/
const std::vector<std::shared_ptr<CREToken>> expression_15{CRE::NOT_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,
                                                           BATCH_SIZE_TOKEN};

/*
                    NOT
                     |
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                    /  \    /  \
                 ~BT  *WS* *WS* ~BT
*/
const std::vector<std::shared_ptr<CREToken>> expression_19{CRE::NOT_PTR,
                                                           CRE::OPEN_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,
                                                           CRE::OPEN_PTR,
                                                           CRE::NOT_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::OR_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::CLOSE_PTR,
                                                           CRE::AND_PTR,
                                                           CRE::OPEN_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::OR_PTR,
                                                           CRE::NOT_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::CLOSE_PTR,
                                                           CRE::CLOSE_PTR};

/*
                  _ OR _
                /        \
               /          \
             NOT           \
              |             \
              OR             OR
            /    \         /    \
         *ELF*  *BT*    *ELF*   *WS*
*/
const std::vector<std::shared_ptr<CREToken>> expression_20{CRE::NOT_PTR,
                                                           CRE::OPEN_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::OR_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::CLOSE_PTR,
                                                           CRE::OR_PTR,
                                                           CRE::OPEN_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::OR_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::CLOSE_PTR};

/*
                NOT
                 |
                NOT
                 |
                NOT
                 |
              _ AND _
             /       \
            /         \
          ~ELF        OR
                    /    \
                  ~BT    ~WS
*/
const std::vector<std::shared_ptr<CREToken>> expression_21{CRE::NOT_PTR,
                                                           CRE::NOT_PTR,
                                                           CRE::NOT_PTR,
                                                           CRE::OPEN_PTR,
                                                           CRE::NOT_PTR,
                                                           ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,
                                                           CRE::OPEN_PTR,
                                                           CRE::NOT_PTR,
                                                           BATCH_SIZE_TOKEN,
                                                           CRE::OR_PTR,
                                                           CRE::NOT_PTR,
                                                           ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::CLOSE_PTR,
                                                           CRE::CLOSE_PTR};

// (EMS)
const std::vector<std::shared_ptr<CREToken>> expression_22{CRE::OPEN_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR};

// ((NOT EMS))
const std::vector<std::shared_ptr<CREToken>> expression_23 =
    {CRE::OPEN_PTR, CRE::OPEN_PTR, CRE::NOT_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR, CRE::CLOSE_PTR};

// UNK
const std::vector<std::shared_ptr<CREToken>> expression_24 = {UNKNOWN_TOKEN};

// BS AND (UNK)
const std::vector<std::shared_ptr<CREToken>> expression_25 = {BATCH_SIZE_TOKEN,
                                                              CRE::AND_PTR,
                                                              CRE::OPEN_PTR,
                                                              UNKNOWN_TOKEN,
                                                              CRE::CLOSE_PTR};

// UNK2
const std::vector<std::shared_ptr<CREToken>> expression_27{UNKNOWN_TOKEN, ID_2_TOKEN};

// NOT UNK
const std::vector<std::shared_ptr<CREToken>> expression_28{CRE::NOT_PTR, UNKNOWN_TOKEN};

// NOT BS1
const std::vector<std::shared_ptr<CREToken>> expression_29{CRE::NOT_PTR, BATCH_SIZE_TOKEN, ID_1_TOKEN};

// BS AND UNK
const std::vector<std::shared_ptr<CREToken>> expression_30{BATCH_SIZE_TOKEN, CRE::AND_PTR, UNKNOWN_TOKEN};

// BS OR UNK
const std::vector<std::shared_ptr<CREToken>> expression_31{BATCH_SIZE_TOKEN, CRE::OR_PTR, UNKNOWN_TOKEN};

// BS0 AND RR1
const std::vector<std::shared_ptr<CREToken>> expression_32{BATCH_SIZE_TOKEN,
                                                           ID_0_TOKEN,
                                                           CRE::AND_PTR,
                                                           RUNTIME_REQUIREMENTS_TOKEN,
                                                           ID_1_TOKEN};

// BS0 OR RR1
const std::vector<std::shared_ptr<CREToken>> expression_33{BATCH_SIZE_TOKEN,
                                                           ID_0_TOKEN,
                                                           CRE::OR_PTR,
                                                           RUNTIME_REQUIREMENTS_TOKEN,
                                                           ID_1_TOKEN};

/*
                    NOT
                     |
                ___ AND _____
               /     |        \
           *ELF*     OR        OR
                    /  \      /  \
                 ~BT0  UNK2 *WS* ~BT1

    NOT (EMS AND (NOT BS0 OR UNK2) AND (EIS OR NOT BS1))
*/
const std::vector<std::shared_ptr<CREToken>> expression_34{CRE::NOT_PTR,     CRE::OPEN_PTR,  ELF_MAIN_SCHEDULE_TOKEN,
                                                           CRE::AND_PTR,     CRE::OPEN_PTR,  CRE::NOT_PTR,
                                                           BATCH_SIZE_TOKEN, ID_0_TOKEN,     CRE::OR_PTR,
                                                           UNKNOWN_TOKEN,    ID_2_TOKEN,     CRE::CLOSE_PTR,
                                                           CRE::AND_PTR,     CRE::OPEN_PTR,  ELF_INIT_SCHEDULES_TOKEN,
                                                           CRE::OR_PTR,      CRE::NOT_PTR,   BATCH_SIZE_TOKEN,
                                                           ID_1_TOKEN,       CRE::CLOSE_PTR, CRE::CLOSE_PTR};

std::vector<std::vector<std::shared_ptr<CREToken>>> invalid_expressions{
    // missing both operands for the OR operator
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, CRE::OPEN_PTR, CRE::OR_PTR, CRE::CLOSE_PTR},

    // Missing only the first operand for the OR operator
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, CRE::OPEN_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::OR_PTR, CRE::CLOSE_PTR},

    // Missing only the second operand for the OR operator
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, CRE::OPEN_PTR, CRE::OR_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR},

    // missing closed parenthesis
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, CRE::OPEN_PTR, BATCH_SIZE_TOKEN, CRE::OR_PTR, ELF_INIT_SCHEDULES_TOKEN},

    // missing open parenthesis
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, BATCH_SIZE_TOKEN, CRE::OR_PTR, ELF_INIT_SCHEDULES_TOKEN, CRE::CLOSE_PTR},

    /*
                    ___ AND ___
                   /     |      \
               *ELF*     OR      OR
                         |      /  \
                              *WS* *BT*
    */
    // missing operands for the first OR operator
    {ELF_MAIN_SCHEDULE_TOKEN,
     CRE::AND_PTR,
     CRE::OPEN_PTR,
     CRE::OR_PTR,
     CRE::CLOSE_PTR,
     CRE::OPEN_PTR,
     ELF_INIT_SCHEDULES_TOKEN,
     CRE::OR_PTR,
     BATCH_SIZE_TOKEN,
     CRE::CLOSE_PTR},

    // missing operands for nested operators
    {CRE::OPEN_PTR, CRE::OR_PTR, CRE::OPEN_PTR, CRE::OR_PTR, CRE::CLOSE_PTR, CRE::CLOSE_PTR, CRE::AND_PTR},

    // NOT missing operand
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, CRE::NOT_PTR},

    // chained NOTs with no operand
    {CRE::NOT_PTR, CRE::NOT_PTR},

    // NOT missing operand before CLOSE
    {CRE::OPEN_PTR, CRE::NOT_PTR, CRE::CLOSE_PTR},

    // missing operand
    {CRE::AND_PTR},

    // too many operands
    {CRE::NOT_PTR, ELF_MAIN_SCHEDULE_TOKEN, BATCH_SIZE_TOKEN},

    // missing CLOSE
    {CRE::OPEN_PTR, CRE::OPEN_PTR, CRE::NOT_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR},

    // missing OPEN
    {CRE::OPEN_PTR, CRE::NOT_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR, CRE::CLOSE_PTR},

    // Empty parrentheses cannot play the role of an operand
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, CRE::OPEN_PTR, CRE::CLOSE_PTR},

    // The subexpression is just "NOT". The operand is missing
    {CRE::OPEN_PTR, CRE::NOT_PTR, CRE::CLOSE_PTR, ELF_MAIN_SCHEDULE_TOKEN},

    // AND has too many operands
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, ELF_MAIN_SCHEDULE_TOKEN, ELF_MAIN_SCHEDULE_TOKEN},

    // OR has too many operands
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::OR_PTR, ELF_MAIN_SCHEDULE_TOKEN, ELF_MAIN_SCHEDULE_TOKEN},

    // No operator to tie the two tokens
    {ELF_MAIN_SCHEDULE_TOKEN, ELF_MAIN_SCHEDULE_TOKEN},

    // No operator to tie the two subexpressions
    {CRE::OPEN_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR, CRE::OPEN_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR},

    // "OR" cannot replace an operand
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::OR_PTR, CRE::OR_PTR},

    // Can't start with a binary operator
    {CRE::OR_PTR},

    // Section ID's should always follow section types
    {ID_0_TOKEN},

    // Wrong order
    {ID_0_TOKEN, BATCH_SIZE_TOKEN},

    // Missing section type
    {BATCH_SIZE_TOKEN, CRE::AND_PTR, ID_0_TOKEN},

    // NOT used as binary
    {BATCH_SIZE_TOKEN, CRE::NOT_PTR, RUNTIME_REQUIREMENTS_TOKEN},

    // Open bracket right after operand; missing operator
    {BATCH_SIZE_TOKEN, CRE::OPEN_PTR, RUNTIME_REQUIREMENTS_TOKEN, CRE::CLOSE_PTR},

    // The first one should have been a section type
    {ID_0_TOKEN, ID_2_TOKEN},

    // Only one ID per section type is allowed
    {BATCH_SIZE_TOKEN, ID_0_TOKEN, ID_2_TOKEN},

    // Only one ID per section type is allowed
    {BATCH_SIZE_TOKEN, CRE::OR_PTR, BATCH_SIZE_TOKEN, ID_0_TOKEN, ID_2_TOKEN},
};

std::vector<CREParams> invalid_test_cases = generate_invalid_test_cases(invalid_expressions);

// clang-format off
std::vector<CREParams> valid_test_cases{
    make_test_params(expression_1, ov::CompatibilityCheck::SUPPORTED),

    make_test_params(expression_2, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {}),
    make_test_params(expression_2, ov::CompatibilityCheck::UNSUPPORTED, {}, {0}, {}),
    make_test_params(expression_2, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {0}),
    make_test_params(expression_2, ov::CompatibilityCheck::SUPPORTED, {RUNTIME_REQUIREMENTS_CODE}, {}, {}),
    make_test_params(expression_2, ov::CompatibilityCheck::UNSUPPORTED, {RUNTIME_REQUIREMENTS_CODE}, {0}, {}),
    make_test_params(expression_2, ov::CompatibilityCheck::NOT_APPLICABLE, {RUNTIME_REQUIREMENTS_CODE}, {}, {0}),
    make_test_params(expression_2, ov::CompatibilityCheck::SUPPORTED, {RUNTIME_REQUIREMENTS_CODE}, {1}, {}),
    make_test_params(expression_2, ov::CompatibilityCheck::SUPPORTED, {RUNTIME_REQUIREMENTS_CODE}, {}, {1}),

    make_test_params(expression_3, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_3, ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_4, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_4, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_4, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),

    make_test_params(expression_5, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_5, ov::CompatibilityCheck::UNSUPPORTED,
    {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_5,
    ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),

    make_test_params(expression_6, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_6, ov::CompatibilityCheck::SUPPORTED,
    {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_6,
    ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}), make_test_params(expression_6,
    ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_6,
    ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_7, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_7, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_7, ov::CompatibilityCheck::SUPPORTED,
    {ELF_MAIN_SCHEDULE_CODE}), make_test_params(expression_7, ov::CompatibilityCheck::UNSUPPORTED,
    {ELF_INIT_SCHEDULES_CODE}),

    make_test_params(expression_8, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_8, ov::CompatibilityCheck::SUPPORTED,
    {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_8,
    ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}), make_test_params(expression_8,
    ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),

    make_test_params(expression_9, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_9, ov::CompatibilityCheck::SUPPORTED,
    {ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_9, ov::CompatibilityCheck::SUPPORTED,
    {ELF_MAIN_SCHEDULE_CODE}), make_test_params(expression_9, ov::CompatibilityCheck::UNSUPPORTED,
    {BATCH_SIZE_CODE}), make_test_params(expression_9, ov::CompatibilityCheck::UNSUPPORTED),

    // should have the same behavior as expression_9
    make_test_params(expression_10, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_10, ov::CompatibilityCheck::SUPPORTED,
    {ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_10, ov::CompatibilityCheck::SUPPORTED,
    {ELF_MAIN_SCHEDULE_CODE}), make_test_params(expression_10, ov::CompatibilityCheck::UNSUPPORTED,
    {BATCH_SIZE_CODE}), make_test_params(expression_10, ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_12, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_12, ov::CompatibilityCheck::SUPPORTED,
    {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_12,
    ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}), make_test_params(expression_12,
    ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_12,
    ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),

    make_test_params(expression_13, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),

    make_test_params(expression_14, ov::CompatibilityCheck::SUPPORTED, {ELF_INIT_SCHEDULES_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_14, ov::CompatibilityCheck::SUPPORTED),
    make_test_params(expression_14, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_14, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),

    make_test_params(expression_15, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_15, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_15, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE}),
    make_test_params(expression_15, ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_16, ov::CompatibilityCheck::SUPPORTED, {ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_16, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_16, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_16, ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_17, ov::CompatibilityCheck::SUPPORTED, {ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_17, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_17, ov::CompatibilityCheck::UNSUPPORTED,
    {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_17,
    ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),

    make_test_params(expression_18, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_18, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_18, ov::CompatibilityCheck::SUPPORTED),

    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED, {ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE}),
    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED),
    make_test_params(expression_19, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE,
    ELF_INIT_SCHEDULES_CODE}),

    make_test_params(expression_20, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_20, ov::CompatibilityCheck::SUPPORTED,
    {ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_20, ov::CompatibilityCheck::SUPPORTED),
    make_test_params(expression_20, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),

    make_test_params(expression_21, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE,
    ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_21, ov::CompatibilityCheck::SUPPORTED,
    {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_21,
    ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_21,
    ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}), make_test_params(expression_21,
    ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}), make_test_params(expression_21,
    ov::CompatibilityCheck::UNSUPPORTED, {ELF_INIT_SCHEDULES_CODE}), make_test_params(expression_21,
    ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}), make_test_params(expression_21,
    ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_22, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_23, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),

    make_test_params(expression_24, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(expression_24, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),

    make_test_params(expression_25, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),

    make_test_params(expression_27, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {}),
    make_test_params(expression_27, ov::CompatibilityCheck::UNSUPPORTED, {}, {2}, {}),
    make_test_params(expression_27, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {2}),

    make_test_params(expression_28, ov::CompatibilityCheck::SUPPORTED),

    make_test_params(expression_29, ov::CompatibilityCheck::SUPPORTED, {}, {}, {}),
    make_test_params(expression_29, ov::CompatibilityCheck::SUPPORTED, {}, {1}, {}),
    make_test_params(expression_29, ov::CompatibilityCheck::SUPPORTED, {}, {}, {1}),
    make_test_params(expression_29, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {}, {}),
    make_test_params(expression_29, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE}, {1}, {}),
    make_test_params(expression_29, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE}, {}, {1}),

    make_test_params(expression_30, ov::CompatibilityCheck::UNSUPPORTED, {}),
    make_test_params(expression_30, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),

    make_test_params(expression_31, ov::CompatibilityCheck::UNSUPPORTED, {}),
    make_test_params(expression_31, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE}),

    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {}, {0}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {0}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {}, {1}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {1}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {}, {0, 1}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {}, {0}, {1}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {0, 1}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {}, {1}, {0}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {0}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {}, {0}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {1}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {}, {1}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {0, 1}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {0}, {1}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {}, {0, 1}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {1}, {0}),
    make_test_params(expression_32, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {0}, {}), 
    make_test_params(expression_32, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {}, {0}), 
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {1}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {}, {1}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {0, 1}, {}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {0}, {1}),
    make_test_params(expression_32, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {}, {0, 1}),
    make_test_params(expression_32, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {1}, {0}),

    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {}, {0}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {0}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {}, {1}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {1}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {}, {0, 1}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {}, {0}, {1}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {}, {}, {0, 1}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {}, {1}, {0}),
    make_test_params(expression_33, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE}, {}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {0}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE}, {}, {0}),
    make_test_params(expression_33, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE}, {1}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE}, {}, {1}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {0, 1}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}, {0}, {1}),
    make_test_params(expression_33, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE}, {}, {0, 1}),
    make_test_params(expression_33, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE}, {1}, {0}),
    make_test_params(expression_33, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {0}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {}, {0}),
    make_test_params(expression_33, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {1}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {}, {1}),
    make_test_params(expression_33, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {0, 1}, {}),
    make_test_params(expression_33, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {0}, {1}),
    make_test_params(expression_33, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {}, {0, 1}),
    make_test_params(expression_33, ov::CompatibilityCheck::NOT_APPLICABLE, {BATCH_SIZE_CODE, RUNTIME_REQUIREMENTS_CODE}, {1}, {0}),

    // NOT (EMS AND (NOT BS0 OR UNK2) AND (EIS OR NOT BS1))
    make_test_params(expression_34, ov::CompatibilityCheck::SUPPORTED, {}, {}, {}),
    make_test_params(expression_34,
                 ov::CompatibilityCheck::SUPPORTED,
                 {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE},
                 {},
                 {}),
    make_test_params(expression_34,
                 ov::CompatibilityCheck::UNSUPPORTED,
                 {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE},
                 {0, 1, 2},
                 {}),
    make_test_params(expression_34,
                 ov::CompatibilityCheck::NOT_APPLICABLE,
                 {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE},
                 {},
                 {0, 1, 2}),
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(CRE,
                         ValidExpression,
                         ::testing::ValuesIn(valid_test_cases),
                         CREEvaluationTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CRE,
                         InvalidExpression,
                         ::testing::ValuesIn(invalid_test_cases),
                         CREEvaluationTests::getTestCaseName);
