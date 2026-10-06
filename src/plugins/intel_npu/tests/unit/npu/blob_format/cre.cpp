// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/cre.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "intel_npu/common/section_type.hpp"
#include "intel_npu/common/supported_section_type_evaluator.hpp"

#define MAKE_PARAM(expr, result, ...) \
    std::make_tuple(std::string(#expr), expr, std::vector<uint16_t>{__VA_ARGS__}, result)

using namespace intel_npu;

namespace {

constexpr std::string_view STRING_THAT_EVALUATES_TO_UNSUPPORTED = "";
constexpr std::string_view STRING_THAT_EVALUATES_TO_UNKNOWN = "0";
const std::string TEST_NAME_FIELDS_SEPARATOR = "__";

}  // namespace

class MockInstanceEvaluator : public ISectionInstanceEvaluator {
public:
    ov::CompatibilityCheck evaluate(std::string_view runtime_requirements) const override {
        return runtime_requirements.empty() ? ov::CompatibilityCheck::UNSUPPORTED
                                            : ov::CompatibilityCheck::NOT_APPLICABLE;
    }
};

// Name, expression, supported types, unsupported instances, instances of unknown support
using CREParams = std::tuple<std::string,
                             std::vector<std::shared_ptr<CREToken>>,
                             std::vector<SectionTypeCode>,
                             std::vector<uint16_t>,
                             std::vector<uint16_t>,
                             ov::CompatibilityCheck>;

class CREEvaluationTests : public ::testing::TestWithParam<CREParams> {
protected:
    void SetUp() override {
        std::string name;
        std::vector<std::shared_ptr<CREToken>> expression;
        std::vector<SectionTypeCode> supported_section_types;
        std::vector<uint16_t> unsupported_section_instances;
        std::vector<uint16_t> section_instances_unknown_support;
        std::tie(name,
                 expression,
                 supported_section_types,
                 unsupported_section_instances,
                 section_instances_unknown_support,
                 expected_result) = GetParam();

        cre = CRE(expression);

        for (const auto code : supported_section_types) {
            section_type_evaluators[SectionType(code)] = SupportedSectionTypeEvaluator::get_instance();
        }
        for (const auto id : unsupported_section_instances) {
            section_instance_evaluators[SectionID(id)] =
                SingleSectionInstanceEvaluator(std::make_shared<MockInstanceEvaluator>(),
                                               STRING_THAT_EVALUATES_TO_UNSUPPORTED);
        }
        for (const auto id : section_instances_unknown_support) {
            section_instance_evaluators[SectionID(id)] =
                SingleSectionInstanceEvaluator(std::make_shared<MockInstanceEvaluator>(),
                                               STRING_THAT_EVALUATES_TO_UNKNOWN);
        }
    }

    CRE cre;
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> section_type_evaluators;
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> section_instance_evaluators;
    ov::CompatibilityCheck expected_result;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CREParams> obj) {
        const auto& [name,
                     expression,
                     supported_section_types,
                     unsupported_section_instances,
                     section_instances_unknown_support,
                     expected_result] = obj.param;

        std::string supported_section_types_string = "supported_types=";
        std::string unsupported_section_instances_string = "unsupported_instances=";
        std::string section_instances_unknown_support_string = "unknown_instances=";
        std::string result_string =
            expected_result == ov::CompatibilityCheck::SUPPORTED
                ? "supported"
                : (expected_result == ov::CompatibilityCheck::UNSUPPORTED ? "unsupported" : "not_applicable");

        return name + TEST_NAME_FIELDS_SEPARATOR + supported_section_types_string + TEST_NAME_FIELDS_SEPARATOR +
               unsupported_section_instances_string + TEST_NAME_FIELDS_SEPARATOR +
               section_instances_unknown_support_string + TEST_NAME_FIELDS_SEPARATOR + result_string;
    }
};

using ValidExpression = CREEvaluationTests;

TEST_P(ValidExpression, check_compatibility) {
    EXPECT_EQ(cre.check_compatibility(section_type_evaluators, section_instance_evaluators), expected_result);
}

using InvalidExpression = CREEvaluationTests;

TEST_P(InvalidExpression, check_compatibility) {
    EXPECT_THROW(cre.check_compatibility(section_type_evaluators, section_instance_evaluators), InvalidCRE);
}

using CREAppendSingleToken = ::testing::Test;

TEST_F(CREAppendSingleToken, AppendValidTokenUpdatesExpression) {
    CRE cre;
    cre.append_to_expression(SectionTypeCode::ELF_MAIN_SCHEDULE);
    EXPECT_EQ(cre.get_expression_length(), 1);
    EXPECT_EQ(cre.get_expression(), (std::vector<CREToken>{SectionTypeCode::ELF_MAIN_SCHEDULE}));
}

TEST_F(CREAppendSingleToken, AppendMultipleValidTokensAccumulates) {
    CRE cre;
    cre.append_to_expression(SectionTypeCode::ELF_MAIN_SCHEDULE);
    cre.append_to_expression(SectionTypeCode::BATCH_SIZE);
    cre.append_to_expression(SectionTypeCode::ELF_INIT_SCHEDULES);
    EXPECT_EQ(cre.get_expression_length(), 5);
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CREToken>{SectionTypeCode::ELF_MAIN_SCHEDULE,
                                     CRE::AND,
                                     SectionTypeCode::BATCH_SIZE,
                                     CRE::AND,
                                     SectionTypeCode::ELF_INIT_SCHEDULES}));
}

TEST_F(CREAppendSingleToken, AppendReservedTokenThrows) {
    CRE cre;
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::AND));
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::OR));
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::OPEN));
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::CLOSE));
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::NOT));
}

TEST_F(CREAppendSingleToken, BuildsEvaluableAndExpression) {
    CRE cre;
    cre.append_to_expression(SectionTypeCode::ELF_MAIN_SCHEDULE);
    cre.append_to_expression(SectionTypeCode::BATCH_SIZE);

    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> caps;
    caps[SectionTypeCode::ELF_MAIN_SCHEDULE] =
        std::make_shared<SupportedSectionTypeEvaluator>(SectionTypeCode::ELF_MAIN_SCHEDULE);
    caps[SectionTypeCode::BATCH_SIZE] = std::make_shared<SupportedSectionTypeEvaluator>(SectionTypeCode::BATCH_SIZE);
    EXPECT_TRUE(cre.check_compatibility(caps));

    caps.erase(SectionTypeCode::BATCH_SIZE);
    EXPECT_FALSE(cre.check_compatibility(caps));
}

using CREAppendToken = ::testing::Test;

TEST_F(CREAppendToken, AppendEmptyVector) {
    CRE cre;
    cre.append_to_expression(std::vector<CREToken>{});
    EXPECT_EQ(cre.get_expression_length(), 0);
    EXPECT_EQ(cre.get_expression(), (std::vector<CREToken>{}));
}

TEST_F(CREAppendToken, AppendSubexpressionTokens) {
    CRE cre;
    cre.append_to_expression(std::vector<CREToken>{CRE::OPEN,
                                                   SectionTypeCode::BATCH_SIZE,
                                                   CRE::OR,
                                                   SectionTypeCode::ELF_INIT_SCHEDULES,
                                                   CRE::CLOSE});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CREToken>{CRE::OPEN,
                                     SectionTypeCode::BATCH_SIZE,
                                     CRE::OR,
                                     SectionTypeCode::ELF_INIT_SCHEDULES,
                                     CRE::CLOSE}));

    cre.append_to_expression(std::vector<CREToken>{CRE::OPEN, SectionTypeCode::BATCH_SIZE, CRE::CLOSE});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CREToken>{CRE::OPEN,
                                     SectionTypeCode::BATCH_SIZE,
                                     CRE::OR,
                                     SectionTypeCode::ELF_INIT_SCHEDULES,
                                     CRE::CLOSE,
                                     CRE::AND,
                                     CRE::OPEN,
                                     SectionTypeCode::BATCH_SIZE,
                                     CRE::CLOSE}));
}

/**
 * @brief Upon appending a subexpression, the CRE will add parrethesis automatically if the subexpression is longer
 than
 * two tokens (to include a valid binary operator) and the expression is not already enclosed.
 */
TEST_F(CREAppendToken, AppendSubexpressionAddsParrentheses) {
    CRE cre;
    cre.append_to_expression(
        std::vector<CREToken>{SectionTypeCode::BATCH_SIZE, CRE::OR, SectionTypeCode::ELF_INIT_SCHEDULES});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CREToken>{CRE::OPEN,
                                     SectionTypeCode::BATCH_SIZE,
                                     CRE::OR,
                                     SectionTypeCode::ELF_INIT_SCHEDULES,
                                     CRE::CLOSE}));

    cre = {};
    cre.append_to_expression(
        std::vector<CREToken>{CRE::OPEN, SectionTypeCode::BATCH_SIZE, CRE::OR, SectionTypeCode::ELF_INIT_SCHEDULES});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CREToken>{CRE::OPEN,
                                     CRE::OPEN,
                                     SectionTypeCode::BATCH_SIZE,
                                     CRE::OR,
                                     SectionTypeCode::ELF_INIT_SCHEDULES,
                                     CRE::CLOSE}));

    cre = {};
    cre.append_to_expression(
        std::vector<CREToken>{SectionTypeCode::BATCH_SIZE, CRE::OR, SectionTypeCode::ELF_INIT_SCHEDULES, CRE::CLOSE});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CREToken>{CRE::OPEN,
                                     SectionTypeCode::BATCH_SIZE,
                                     CRE::OR,
                                     SectionTypeCode::ELF_INIT_SCHEDULES,
                                     CRE::CLOSE,
                                     CRE::CLOSE}));
}

/**
 * @brief Parretheses are not necessary if the subexpression has less than three tokens (no valid binary operator can
 be
 * there) or if the subexpression is already enclosed.
 */
TEST_F(CREAppendToken, AppendSubexpressionWithoutParrentheses) {
    CRE cre;
    cre.append_to_expression(std::vector<CREToken>{CRE::NOT, SectionTypeCode::BATCH_SIZE});
    EXPECT_EQ(cre.get_expression(), (std::vector<CREToken>{CRE::NOT, SectionTypeCode::BATCH_SIZE}));

    cre = {};
    cre.append_to_expression(std::vector<CREToken>{CRE::OPEN, CRE::NOT, SectionTypeCode::BATCH_SIZE, CRE::CLOSE});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CREToken>{CRE::OPEN, CRE::NOT, SectionTypeCode::BATCH_SIZE, CRE::CLOSE}));
}

/**
 * @brief The CRE code should be able to detect duplicate subexpressions (relative to depth level 0) and avoid
 inserting
 * copies.
 */
TEST_F(CREAppendToken, AvoidAppendingDuplicates) {
    CRE cre;
    cre.append_to_expression(SectionTypeCode::BATCH_SIZE);
    cre.append_to_expression(SectionTypeCode::BATCH_SIZE);
    EXPECT_EQ(cre.get_expression(), (std::vector<CREToken>{SectionTypeCode::BATCH_SIZE}));

    cre = {};
    cre.append_to_expression(std::vector<CREToken>{CRE::OPEN,
                                                   CRE::NOT,
                                                   SectionTypeCode::BATCH_SIZE,
                                                   CRE::OR,
                                                   SectionTypeCode::ELF_INIT_SCHEDULES,
                                                   CRE::CLOSE});
    cre.append_to_expression(std::vector<CREToken>{CRE::OPEN,
                                                   CRE::NOT,
                                                   SectionTypeCode::BATCH_SIZE,
                                                   CRE::OR,
                                                   SectionTypeCode::ELF_INIT_SCHEDULES,
                                                   CRE::CLOSE});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CREToken>{CRE::OPEN,
                                     CRE::NOT,
                                     SectionTypeCode::BATCH_SIZE,
                                     CRE::OR,
                                     SectionTypeCode::ELF_INIT_SCHEDULES,
                                     CRE::CLOSE}));
}

TEST_F(CREAppendToken, MixedAppend) {
    CRE cre;
    cre.append_to_expression(SectionTypeCode::ELF_MAIN_SCHEDULE);
    cre.append_to_expression(std::vector<CREToken>{CRE::OPEN,
                                                   SectionTypeCode::BATCH_SIZE,
                                                   CRE::OR,
                                                   SectionTypeCode::ELF_INIT_SCHEDULES,
                                                   CRE::CLOSE});

    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> caps;
    caps[SectionTypeCode::ELF_MAIN_SCHEDULE] =
        std::make_shared<SupportedSectionTypeEvaluator>(SectionTypeCode::ELF_MAIN_SCHEDULE);
    caps[SectionTypeCode::BATCH_SIZE] = std::make_shared<SupportedSectionTypeEvaluator>(SectionTypeCode::BATCH_SIZE);
    EXPECT_TRUE(cre.check_compatibility(caps));

    caps.erase(SectionTypeCode::ELF_MAIN_SCHEDULE);
    EXPECT_FALSE(cre.check_compatibility(caps));
}

class CREOperandsEvaluation : public ::testing::Test {
protected:
    void SetUp() override {
        cap_1 = std::make_shared<MockCapability>(MockTypes::MOCK_1);
        cap_2 = std::make_shared<MockCapability>(MockTypes::MOCK_2);
        cap_3 = std::make_shared<MockCapability>(MockTypes::MOCK_3);

        caps[MockTypes::MOCK_1] = cap_1;
        caps[MockTypes::MOCK_2] = cap_2;
        caps[MockTypes::MOCK_3] = cap_3;
    }

    std::shared_ptr<MockCapability> cap_1;
    std::shared_ptr<MockCapability> cap_2;
    std::shared_ptr<MockCapability> cap_3;
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> caps;
};

TEST_F(CREOperandsEvaluation, Depth0ORs) {
    EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, evaluate()).Times(0);

    CRE cre({MockTypes::MOCK_1, CRE::OR, MockTypes::MOCK_2, CRE::OR, MockTypes::MOCK_2});

    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, Depth0ANDs) {
    EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, evaluate()).Times(0);

    CRE cre({CRE::NOT, MockTypes::MOCK_1, CRE::AND, MockTypes::MOCK_2, CRE::AND, MockTypes::MOCK_2});

    EXPECT_FALSE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, Depth0AllEvaluate) {
    EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, evaluate()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_3, evaluate()).Times(1).WillOnce(::testing::Return(true));

    CRE cre({CRE::NOT, MockTypes::MOCK_1, CRE::OR, MockTypes::MOCK_2, CRE::AND, MockTypes::MOCK_3});

    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, ORFollowedByAND) {
    EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, evaluate()).Times(0);

    CRE cre(
        {CRE::NOT, CRE::OPEN, MockTypes::MOCK_1, CRE::OR, MockTypes::MOCK_2, CRE::CLOSE, CRE::AND, MockTypes::MOCK_2});

    EXPECT_FALSE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, Depth1NotEvaluated) {
    EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, evaluate()).Times(0);
    EXPECT_CALL(*cap_3, evaluate()).Times(0);

    CRE cre({MockTypes::MOCK_1, CRE::OR, CRE::OPEN, MockTypes::MOCK_2, CRE::AND, MockTypes::MOCK_3, CRE::CLOSE});

    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, Depth2NotEvaluated) {
    EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, evaluate()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_3, evaluate()).Times(0);

    CRE cre({CRE::NOT,
             MockTypes::MOCK_1,
             CRE::OR,
             CRE::OPEN,
             CRE::NOT,
             MockTypes::MOCK_2,
             CRE::AND,
             CRE::OPEN,
             MockTypes::MOCK_3,
             CRE::CLOSE,
             CRE::CLOSE});

    EXPECT_FALSE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, AllDepthNotEvaluated) {
    EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, evaluate()).Times(0);
    EXPECT_CALL(*cap_3, evaluate()).Times(0);

    CRE cre({CRE::NOT,
             MockTypes::MOCK_1,
             CRE::AND,
             CRE::OPEN,
             MockTypes::MOCK_2,
             CRE::AND,
             CRE::OPEN,
             MockTypes::MOCK_3,
             CRE::CLOSE,
             CRE::CLOSE});

    EXPECT_FALSE(cre.check_compatibility(caps));
}

const std::vector<CREToken> expression_1 = {};

const std::vector<CREToken> expression_3 = {SectionTypeCode::ELF_MAIN_SCHEDULE};

/*
           AND
          /   \
       *ELF*  *BT*
*/
const std::vector<CREToken> expression_4 = {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, SectionTypeCode::BATCH_SIZE};

/*
              AND
           /   |   \
        *ELF* *BT* *WS*
*/
const std::vector<CREToken> expression_5 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::AND,
                                            SectionTypeCode::ELF_INIT_SCHEDULES};

/*
            AND
           /   \
        *ELF*  OR
              /  \
           *BT*  *WS*
*/
const std::vector<CREToken> expression_6 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            CRE::OPEN,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::OR,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::CLOSE};

/*
            OR
          /    \
        *ELF*  AND
              /   \
           *BT*   *WS*
*/
const std::vector<CREToken> expression_7 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::OR,
                                            CRE::OPEN,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::AND,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::CLOSE};

/*
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                    /  \    /  \
                 *BT* *WS* *WS* *BT*
*/
const std::vector<CREToken> expression_8 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            CRE::OPEN,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::OR,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::CLOSE,
                                            CRE::AND,
                                            CRE::OPEN,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::OR,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::CLOSE};

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
const std::vector<CREToken> expression_9 = {CRE::OPEN,
                                            SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::AND,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::CLOSE,
                                            CRE::OR,
                                            CRE::OPEN,
                                            SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::OR,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::OR,
                                            CRE::OPEN,
                                            SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::CLOSE,
                                            CRE::CLOSE};

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
const std::vector<CREToken> expression_10 = {CRE::OPEN,
                                             CRE::OPEN,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::CLOSE,
                                             CRE::OR,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::OR,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::CLOSE,
                                             CRE::OR,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::AND,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::CLOSE};

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
const std::vector<CREToken> expression_12 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::OR,
                                             CRE::OPEN,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::CLOSE,
                                             CRE::OR,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::CLOSE,
                                             CRE::CLOSE};

/*
              AND
           /   |   \
        *ELF* *BT* *ELF*
*/
const std::vector<CREToken> expression_13 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE};

/*
    NOT
     |
   *ELF*
*/
const std::vector<CREToken> expression_14 = {CRE::NOT, SectionTypeCode::ELF_MAIN_SCHEDULE};

/*
              AND
           /   |   \
        ~ELF  ~BT  *WS*
*/
const std::vector<CREToken> expression_16 = {CRE::NOT,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::NOT,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_INIT_SCHEDULES};

/*
            AND
           /   \
        ~ELF  ~OR
              /  \
           *BT*  ~WS
*/
const std::vector<CREToken> expression_17 = {CRE::NOT,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::NOT,
                                             CRE::OPEN,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::OR,
                                             CRE::NOT,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::CLOSE};

/*
      NOT
       |
      AND
     /   \
  *ELF*  *BT*
*/
const std::vector<CREToken> expression_18 =
    {CRE::NOT, CRE::OPEN, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, SectionTypeCode::BATCH_SIZE, CRE::CLOSE};

/*
    AND
    /  \
~ELF  *BT*
*/
const std::vector<CREToken> expression_15 = {CRE::NOT,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             SectionTypeCode::BATCH_SIZE};

/*
                    NOT
                     |
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                    /  \    /  \
                 ~BT  *WS* *WS* ~BT
*/
const std::vector<CREToken> expression_19 = {CRE::NOT,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::OPEN,
                                             CRE::NOT,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::OR,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::CLOSE,
                                             CRE::AND,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::OR,
                                             CRE::NOT,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::CLOSE,
                                             CRE::CLOSE};

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
const std::vector<CREToken> expression_20 = {CRE::NOT,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::OR,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::CLOSE,
                                             CRE::OR,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::OR,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::CLOSE};

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
const std::vector<CREToken> expression_21 = {CRE::NOT,
                                             CRE::NOT,
                                             CRE::NOT,
                                             CRE::OPEN,
                                             CRE::NOT,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::OPEN,
                                             CRE::NOT,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::OR,
                                             CRE::NOT,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::CLOSE,
                                             CRE::CLOSE};

const std::vector<CREToken> expression_22 = {CRE::OPEN, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::CLOSE};

const std::vector<CREToken> expression_23 =
    {CRE::OPEN, CRE::OPEN, CRE::NOT, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::CLOSE, CRE::CLOSE};

// missing both operands for the OR operator
const std::vector<CREToken> invalid_expression_1 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                    CRE::AND,
                                                    CRE::OPEN,
                                                    CRE::OR,
                                                    CRE::CLOSE};

// Missing only the first operand for the OR operator
const std::vector<CREToken> invalid_expression_15 =
    {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, CRE::OPEN, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::OR, CRE::CLOSE};

// Missing only the second operand for the OR operator
const std::vector<CREToken> invalid_expression_16 =
    {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, CRE::OPEN, CRE::OR, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::CLOSE};

// missing closed parenthesis
const std::vector<CREToken> invalid_expression_2 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                    CRE::AND,
                                                    CRE::OPEN,
                                                    SectionTypeCode::BATCH_SIZE,
                                                    CRE::OR,
                                                    SectionTypeCode::ELF_INIT_SCHEDULES};

// missing open parenthesis
const std::vector<CREToken> invalid_expression_3 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                    CRE::AND,
                                                    SectionTypeCode::BATCH_SIZE,
                                                    CRE::OR,
                                                    SectionTypeCode::ELF_INIT_SCHEDULES,
                                                    CRE::CLOSE};

/*
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                     |      /  \
                     0    *WS* *BT*
*/
// missing operand for the first OR operator
const std::vector<CREToken> invalid_expression_4 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                    CRE::AND,
                                                    CRE::OPEN,
                                                    CRE::OR,
                                                    CRE::CLOSE,
                                                    CRE::OPEN,
                                                    SectionTypeCode::ELF_INIT_SCHEDULES,
                                                    CRE::OR,
                                                    SectionTypeCode::BATCH_SIZE,
                                                    CRE::CLOSE};

// missing operands for nested operators
const std::vector<CREToken> invalid_expression_5 =
    {CRE::OPEN, CRE::OR, CRE::OPEN, CRE::OR, CRE::CLOSE, CRE::CLOSE, CRE::AND};

// NOT missing operand
const std::vector<CREToken> invalid_expression_6 = {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, CRE::NOT};

// chained NOTs with no operand
const std::vector<CREToken> invalid_expression_7 = {CRE::NOT, CRE::NOT};

// NOT missing operand before CLOSE
const std::vector<CREToken> invalid_expression_8 = {CRE::OPEN, CRE::NOT, CRE::CLOSE};

// missing operand
const std::vector<CREToken> invalid_expression_9 = {CRE::AND};

// too many operands
const std::vector<CREToken> invalid_expression_10 = {CRE::NOT,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     SectionTypeCode::BATCH_SIZE};

// missing CLOSE
const std::vector<CREToken> invalid_expression_11 = {CRE::OPEN,
                                                     CRE::OPEN,
                                                     CRE::NOT,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::CLOSE};

// missing OPEN
const std::vector<CREToken> invalid_expression_12 = {CRE::OPEN,
                                                     CRE::NOT,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::CLOSE,
                                                     CRE::CLOSE};

// Empty parrentheses cannot play the role of an operand
const std::vector<CREToken> invalid_expression_13 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::AND,
                                                     CRE::OPEN,
                                                     CRE::CLOSE};

// The subexpression is just "NOT". The operand is missing
const std::vector<CREToken> invalid_expression_14 = {CRE::OPEN,
                                                     CRE::NOT,
                                                     CRE::CLOSE,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE};

// AND has too many operands
const std::vector<CREToken> invalid_expression_17 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::AND,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE};

// OR has too many operands
const std::vector<CREToken> invalid_expression_18 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::OR,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE};

// No operator to tie the two tokens
const std::vector<CREToken> invalid_expression_19 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE};

// No operator to tie the two subexpressions
const std::vector<CREToken> invalid_expression_20 = {CRE::OPEN,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::CLOSE,
                                                     CRE::OPEN,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::CLOSE};

// "OR" cannot replace an operand
const std::vector<CREToken> invalid_expression_21 = {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::OR, CRE::OR};

std::vector<CREParams> valid_test_cases = {
    MAKE_PARAM(expression_1, true),

    MAKE_PARAM(expression_3, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_3, false),

    MAKE_PARAM(expression_4, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_4, false, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_4, false, SectionTypeCode::BATCH_SIZE),

    MAKE_PARAM(expression_5,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_5, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_5, false, SectionTypeCode::ELF_MAIN_SCHEDULE),

    MAKE_PARAM(expression_6,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_6, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_6, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_6, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_6, false),

    MAKE_PARAM(expression_7,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_7, true, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_7, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_7, false, SectionTypeCode::ELF_INIT_SCHEDULES),

    MAKE_PARAM(expression_8,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_8, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_8, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_8, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),

    MAKE_PARAM(expression_9,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_9, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_9, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_9, false, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_9, false),

    // should have the same behavior as expression_9
    MAKE_PARAM(expression_10,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_10, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_10, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_10, false, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_10, false),

    MAKE_PARAM(expression_12,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_12, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_12, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_12, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_12, false, SectionTypeCode::ELF_MAIN_SCHEDULE),

    MAKE_PARAM(expression_13, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),

    MAKE_PARAM(expression_14, true, SectionTypeCode::ELF_INIT_SCHEDULES, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_14, true),
    MAKE_PARAM(expression_14, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_14, false, SectionTypeCode::ELF_MAIN_SCHEDULE),

    MAKE_PARAM(expression_15, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_15, false, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_15, true, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_15, false),

    MAKE_PARAM(expression_16, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_16,
               false,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_16, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_16, false),

    MAKE_PARAM(expression_17, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_17,
               false,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_17, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_17, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),

    MAKE_PARAM(expression_18, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_18, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_18, true),

    MAKE_PARAM(expression_19, true, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_19, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_19, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_19, true, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_19, true),
    MAKE_PARAM(expression_19, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),

    MAKE_PARAM(expression_20,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_20, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_20, true),
    MAKE_PARAM(expression_20, false, SectionTypeCode::BATCH_SIZE),

    MAKE_PARAM(expression_21,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_21, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_21, true, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_21, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_21, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_21, false, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_21, false, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_21, false),

    MAKE_PARAM(expression_22, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_23, false, SectionTypeCode::ELF_MAIN_SCHEDULE),
};

INSTANTIATE_TEST_SUITE_P(CRE,
                         ValidExpression,
                         ::testing::ValuesIn(valid_test_cases),
                         CREEvaluationTests::getTestCaseName);

std::vector<CREParams> invalid_test_cases = {
    MAKE_PARAM(invalid_expression_1, false),  MAKE_PARAM(invalid_expression_2, false),
    MAKE_PARAM(invalid_expression_3, false),  MAKE_PARAM(invalid_expression_4, false),
    MAKE_PARAM(invalid_expression_5, false),  MAKE_PARAM(invalid_expression_6, false),
    MAKE_PARAM(invalid_expression_7, false),  MAKE_PARAM(invalid_expression_8, false),
    MAKE_PARAM(invalid_expression_9, false),  MAKE_PARAM(invalid_expression_10, false),
    MAKE_PARAM(invalid_expression_11, false), MAKE_PARAM(invalid_expression_12, false),
    MAKE_PARAM(invalid_expression_13, false), MAKE_PARAM(invalid_expression_14, false),
    MAKE_PARAM(invalid_expression_15, false), MAKE_PARAM(invalid_expression_16, false),
    MAKE_PARAM(invalid_expression_17, false), MAKE_PARAM(invalid_expression_18, false),
    MAKE_PARAM(invalid_expression_19, false), MAKE_PARAM(invalid_expression_20, false),
    MAKE_PARAM(invalid_expression_21, false),
};

INSTANTIATE_TEST_SUITE_P(CRE,
                         InvalidExpression,
                         ::testing::ValuesIn(invalid_test_cases),
                         CREEvaluationTests::getTestCaseName);
