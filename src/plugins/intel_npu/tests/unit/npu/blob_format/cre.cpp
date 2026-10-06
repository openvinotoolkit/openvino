// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/cre.hpp"

#include <gtest/gtest.h>

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

class MockInstanceEvaluator : public ISectionInstanceEvaluator {
public:
    ov::CompatibilityCheck evaluate(std::string_view runtime_requirements) const override {
        return runtime_requirements.empty() ? ov::CompatibilityCheck::UNSUPPORTED
                                            : ov::CompatibilityCheck::NOT_APPLICABLE;
    }
};

namespace {

constexpr std::string_view STRING_THAT_EVALUATES_TO_UNSUPPORTED = "";
constexpr std::string_view STRING_THAT_EVALUATES_TO_UNKNOWN = "0";
const std::string TEST_NAME_FIELDS_SEPARATOR = "__";
constexpr char VALUES_SEPARATOR = ',';

const auto ELF_MAIN_SCHEDULE_TOKEN = std::make_shared<SectionType>(SectionTypeCode::ELF_MAIN_SCHEDULE);
const auto ELF_INIT_SCHEDULES_TOKEN = std::make_shared<SectionType>(SectionTypeCode::ELF_INIT_SCHEDULES);
const auto BATCH_SIZE_TOKEN = std::make_shared<SectionType>(SectionTypeCode::BATCH_SIZE);

const auto ID_0_TOKEN = std::make_shared<SectionID>(0);
const auto ID_1_TOKEN = std::make_shared<SectionID>(1);
const auto ID_2_TOKEN = std::make_shared<SectionID>(2);

constexpr SectionTypeCode ELF_MAIN_SCHEDULE_CODE = SectionTypeCode::ELF_MAIN_SCHEDULE;
constexpr SectionTypeCode ELF_INIT_SCHEDULES_CODE = SectionTypeCode::ELF_INIT_SCHEDULES;
constexpr SectionTypeCode BATCH_SIZE_CODE = SectionTypeCode::BATCH_SIZE;

const auto mock_evaluator = std::make_shared<MockInstanceEvaluator>();

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

}  // namespace

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
                SingleSectionInstanceEvaluator(mock_evaluator, STRING_THAT_EVALUATES_TO_UNSUPPORTED));
        }
        for (const auto id : section_instances_unknown_support) {
            section_instance_evaluators.emplace(
                SectionID(id),
                SingleSectionInstanceEvaluator(mock_evaluator, STRING_THAT_EVALUATES_TO_UNKNOWN));
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

// using CREAppendSingleToken = ::testing::Test;

// TEST_F(CREAppendSingleToken, AppendValidTokenUpdatesExpression) {
//     CRE cre;
//     cre.append_to_expression(ELF_MAIN_SCHEDULE_TOKEN);
//     EXPECT_EQ(cre.get_expression_length(), 1);
//     EXPECT_EQ(cre.get_expression(), (std::vector<CREToken>{ELF_MAIN_SCHEDULE_TOKEN}));
// }

// TEST_F(CREAppendSingleToken, AppendMultipleValidTokensAccumulates) {
//     CRE cre;
//     cre.append_to_expression(ELF_MAIN_SCHEDULE_TOKEN);
//     cre.append_to_expression(BATCH_SIZE_TOKEN);
//     cre.append_to_expression(ELF_INIT_SCHEDULES_TOKEN);
//     EXPECT_EQ(cre.get_expression_length(), 5);
//     EXPECT_EQ(cre.get_expression(),
//               (std::vector<CREToken>{ELF_MAIN_SCHEDULE_TOKEN,
//                                      CRE::AND_PTR,
//                                      BATCH_SIZE_TOKEN,
//                                      CRE::AND_PTR,
//                                      ELF_INIT_SCHEDULES_TOKEN}));
// }

// TEST_F(CREAppendSingleToken, AppendReservedTokenThrows) {
//     CRE cre;
//     EXPECT_ANY_THROW(cre.append_to_expression(CRE::AND_PTR));
//     EXPECT_ANY_THROW(cre.append_to_expression(CRE::OR_PTR));
//     EXPECT_ANY_THROW(cre.append_to_expression(CRE::OPEN_PTR));
//     EXPECT_ANY_THROW(cre.append_to_expression(CRE::CLOSE_PTR));
//     EXPECT_ANY_THROW(cre.append_to_expression(CRE::NOT_PTR));
// }

// TEST_F(CREAppendSingleToken, BuildsEvaluableAndExpression) {
//     CRE cre;
//     cre.append_to_expression(ELF_MAIN_SCHEDULE_TOKEN);
//     cre.append_to_expression(BATCH_SIZE_TOKEN);

//     std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> caps;
//     caps[ELF_MAIN_SCHEDULE_TOKEN] = std::make_shared<SupportedSectionTypeEvaluator>(ELF_MAIN_SCHEDULE_TOKEN);
//     caps[BATCH_SIZE_TOKEN] = std::make_shared<SupportedSectionTypeEvaluator>(BATCH_SIZE_TOKEN);
//     EXPECT_TRUE(cre.check_compatibility(caps));

//     caps.erase(BATCH_SIZE_TOKEN);
//     EXPECT_FALSE(cre.check_compatibility(caps));
// }

// using CREAppendToken = ::testing::Test;

// TEST_F(CREAppendToken, AppendEmptyVector) {
//     CRE cre;
//     cre.append_to_expression(std::vector<CREToken>{});
//     EXPECT_EQ(cre.get_expression_length(), 0);
//     EXPECT_EQ(cre.get_expression(), (std::vector<CREToken>{}));
// }

// TEST_F(CREAppendToken, AppendSubexpressionTokens) {
//     CRE cre;
//     cre.append_to_expression(
//         std::vector<CREToken>{CRE::OPEN_PTR, BATCH_SIZE_TOKEN, CRE::OR_PTR, ELF_INIT_SCHEDULES_TOKEN,
//         CRE::CLOSE_PTR});
//     EXPECT_EQ(cre.get_expression(),
//               (std::vector<CREToken>{CRE::OPEN_PTR,
//                                      BATCH_SIZE_TOKEN,
//                                      CRE::OR_PTR,
//                                      ELF_INIT_SCHEDULES_TOKEN,
//                                      CRE::CLOSE_PTR}));

//     cre.append_to_expression(std::vector<CREToken>{CRE::OPEN_PTR, BATCH_SIZE_TOKEN, CRE::CLOSE_PTR});
//     EXPECT_EQ(cre.get_expression(),
//               (std::vector<CREToken>{CRE::OPEN_PTR,
//                                      BATCH_SIZE_TOKEN,
//                                      CRE::OR_PTR,
//                                      ELF_INIT_SCHEDULES_TOKEN,
//                                      CRE::CLOSE_PTR,
//                                      CRE::AND_PTR,
//                                      CRE::OPEN_PTR,
//                                      BATCH_SIZE_TOKEN,
//                                      CRE::CLOSE_PTR}));
// }

// /**
//  * @brief Upon appending a subexpression, the CRE will add parrethesis automatically if the subexpression is longer
//  than
//  * two tokens (to include a valid binary operator) and the expression is not already enclosed.
//  */
// TEST_F(CREAppendToken, AppendSubexpressionAddsParrentheses) {
//     CRE cre;
//     cre.append_to_expression(std::vector<CREToken>{BATCH_SIZE_TOKEN, CRE::OR_PTR, ELF_INIT_SCHEDULES_TOKEN});
//     EXPECT_EQ(cre.get_expression(),
//               (std::vector<CREToken>{CRE::OPEN_PTR,
//                                      BATCH_SIZE_TOKEN,
//                                      CRE::OR_PTR,
//                                      ELF_INIT_SCHEDULES_TOKEN,
//                                      CRE::CLOSE_PTR}));

//     cre{};
//     cre.append_to_expression(
//         std::vector<CREToken>{CRE::OPEN_PTR, BATCH_SIZE_TOKEN, CRE::OR_PTR, ELF_INIT_SCHEDULES_TOKEN});
//     EXPECT_EQ(cre.get_expression(),
//               (std::vector<CREToken>{CRE::OPEN_PTR,
//                                      CRE::OPEN_PTR,
//                                      BATCH_SIZE_TOKEN,
//                                      CRE::OR_PTR,
//                                      ELF_INIT_SCHEDULES_TOKEN,
//                                      CRE::CLOSE_PTR}));

//     cre{};
//     cre.append_to_expression(
//         std::vector<CREToken>{BATCH_SIZE_TOKEN, CRE::OR_PTR, ELF_INIT_SCHEDULES_TOKEN, CRE::CLOSE_PTR});
//     EXPECT_EQ(cre.get_expression(),
//               (std::vector<CREToken>{CRE::OPEN_PTR,
//                                      BATCH_SIZE_TOKEN,
//                                      CRE::OR_PTR,
//                                      ELF_INIT_SCHEDULES_TOKEN,
//                                      CRE::CLOSE_PTR,
//                                      CRE::CLOSE_PTR}));
// }

// /**
//  * @brief Parretheses are not necessary if the subexpression has less than three tokens (no valid binary operator can
//  be
//  * there) or if the subexpression is already enclosed.
//  */
// TEST_F(CREAppendToken, AppendSubexpressionWithoutParrentheses) {
//     CRE cre;
//     cre.append_to_expression(std::vector<CREToken>{CRE::NOT_PTR, BATCH_SIZE_TOKEN});
//     EXPECT_EQ(cre.get_expression(), (std::vector<CREToken>{CRE::NOT_PTR, BATCH_SIZE_TOKEN}));

//     cre{};
//     cre.append_to_expression(std::vector<CREToken>{CRE::OPEN_PTR, CRE::NOT_PTR, BATCH_SIZE_TOKEN, CRE::CLOSE_PTR});
//     EXPECT_EQ(cre.get_expression(),
//               (std::vector<CREToken>{CRE::OPEN_PTR, CRE::NOT_PTR, BATCH_SIZE_TOKEN, CRE::CLOSE_PTR}));
// }

// /**
//  * @brief The CRE code should be able to detect duplicate subexpressions (relative to depth level 0) and avoid
//  inserting
//  * copies.
//  */
// TEST_F(CREAppendToken, AvoidAppendingDuplicates) {
//     CRE cre;
//     cre.append_to_expression(BATCH_SIZE_TOKEN);
//     cre.append_to_expression(BATCH_SIZE_TOKEN);
//     EXPECT_EQ(cre.get_expression(), (std::vector<CREToken>{BATCH_SIZE_TOKEN}));

//     cre{};
//     cre.append_to_expression(std::vector<CREToken>{CRE::OPEN_PTR,
//                                                    CRE::NOT_PTR,
//                                                    BATCH_SIZE_TOKEN,
//                                                    CRE::OR_PTR,
//                                                    ELF_INIT_SCHEDULES_TOKEN,
//                                                    CRE::CLOSE_PTR});
//     cre.append_to_expression(std::vector<CREToken>{CRE::OPEN_PTR,
//                                                    CRE::NOT_PTR,
//                                                    BATCH_SIZE_TOKEN,
//                                                    CRE::OR_PTR,
//                                                    ELF_INIT_SCHEDULES_TOKEN,
//                                                    CRE::CLOSE_PTR});
//     EXPECT_EQ(cre.get_expression(),
//               (std::vector<CREToken>{CRE::OPEN_PTR,
//                                      CRE::NOT_PTR,
//                                      BATCH_SIZE_TOKEN,
//                                      CRE::OR_PTR,
//                                      ELF_INIT_SCHEDULES_TOKEN,
//                                      CRE::CLOSE_PTR}));
// }

// TEST_F(CREAppendToken, MixedAppend) {
//     CRE cre;
//     cre.append_to_expression(ELF_MAIN_SCHEDULE_TOKEN);
//     cre.append_to_expression(
//         std::vector<CREToken>{CRE::OPEN_PTR, BATCH_SIZE_TOKEN, CRE::OR_PTR, ELF_INIT_SCHEDULES_TOKEN,
//         CRE::CLOSE_PTR});

//     std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> caps;
//     caps[ELF_MAIN_SCHEDULE_TOKEN] = std::make_shared<SupportedSectionTypeEvaluator>(ELF_MAIN_SCHEDULE_TOKEN);
//     caps[BATCH_SIZE_TOKEN] = std::make_shared<SupportedSectionTypeEvaluator>(BATCH_SIZE_TOKEN);
//     EXPECT_TRUE(cre.check_compatibility(caps));

//     caps.erase(ELF_MAIN_SCHEDULE_TOKEN);
//     EXPECT_FALSE(cre.check_compatibility(caps));
// }

// class CREOperandsEvaluation : public ::testing::Test {
// protected:
//     void SetUp() override {
//         cap_1 = std::make_shared<MockCapability>(MockTypes::MOCK_1);
//         cap_2 = std::make_shared<MockCapability>(MockTypes::MOCK_2);
//         cap_3 = std::make_shared<MockCapability>(MockTypes::MOCK_3);

//         caps[MockTypes::MOCK_1] = cap_1;
//         caps[MockTypes::MOCK_2] = cap_2;
//         caps[MockTypes::MOCK_3] = cap_3;
//     }

//     std::shared_ptr<MockCapability> cap_1;
//     std::shared_ptr<MockCapability> cap_2;
//     std::shared_ptr<MockCapability> cap_3;
//     std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> caps;
// };

// TEST_F(CREOperandsEvaluation, Depth0ORs) {
//     EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
//     EXPECT_CALL(*cap_2, evaluate()).Times(0);

//     CRE cre({MockTypes::MOCK_1, CRE::OR_PTR, MockTypes::MOCK_2, CRE::OR_PTR, MockTypes::MOCK_2});

//     EXPECT_TRUE(cre.check_compatibility(caps));
// }

// TEST_F(CREOperandsEvaluation, Depth0ANDs) {
//     EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
//     EXPECT_CALL(*cap_2, evaluate()).Times(0);

//     CRE cre({CRE::NOT_PTR, MockTypes::MOCK_1, CRE::AND_PTR, MockTypes::MOCK_2, CRE::AND_PTR, MockTypes::MOCK_2});

//     EXPECT_FALSE(cre.check_compatibility(caps));
// }

// TEST_F(CREOperandsEvaluation, Depth0AllEvaluate) {
//     EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
//     EXPECT_CALL(*cap_2, evaluate()).Times(1).WillOnce(::testing::Return(true));
//     EXPECT_CALL(*cap_3, evaluate()).Times(1).WillOnce(::testing::Return(true));

//     CRE cre({CRE::NOT_PTR, MockTypes::MOCK_1, CRE::OR_PTR, MockTypes::MOCK_2, CRE::AND_PTR, MockTypes::MOCK_3});

//     EXPECT_TRUE(cre.check_compatibility(caps));
// }

// TEST_F(CREOperandsEvaluation, ORFollowedByAND) {
//     EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
//     EXPECT_CALL(*cap_2, evaluate()).Times(0);

//     CRE cre({CRE::NOT_PTR,
//              CRE::OPEN_PTR,
//              MockTypes::MOCK_1,
//              CRE::OR_PTR,
//              MockTypes::MOCK_2,
//              CRE::CLOSE_PTR,
//              CRE::AND_PTR,
//              MockTypes::MOCK_2});

//     EXPECT_FALSE(cre.check_compatibility(caps));
// }

// TEST_F(CREOperandsEvaluation, Depth1NotEvaluated) {
//     EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
//     EXPECT_CALL(*cap_2, evaluate()).Times(0);
//     EXPECT_CALL(*cap_3, evaluate()).Times(0);

//     CRE cre({MockTypes::MOCK_1,
//              CRE::OR_PTR,
//              CRE::OPEN_PTR,
//              MockTypes::MOCK_2,
//              CRE::AND_PTR,
//              MockTypes::MOCK_3,
//              CRE::CLOSE_PTR});

//     EXPECT_TRUE(cre.check_compatibility(caps));
// }

// TEST_F(CREOperandsEvaluation, Depth2NotEvaluated) {
//     EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
//     EXPECT_CALL(*cap_2, evaluate()).Times(1).WillOnce(::testing::Return(true));
//     EXPECT_CALL(*cap_3, evaluate()).Times(0);

//     CRE cre({CRE::NOT_PTR,
//              MockTypes::MOCK_1,
//              CRE::OR_PTR,
//              CRE::OPEN_PTR,
//              CRE::NOT_PTR,
//              MockTypes::MOCK_2,
//              CRE::AND_PTR,
//              CRE::OPEN_PTR,
//              MockTypes::MOCK_3,
//              CRE::CLOSE_PTR,
//              CRE::CLOSE_PTR});

//     EXPECT_FALSE(cre.check_compatibility(caps));
// }

// TEST_F(CREOperandsEvaluation, AllDepthNotEvaluated) {
//     EXPECT_CALL(*cap_1, evaluate()).Times(1).WillOnce(::testing::Return(true));
//     EXPECT_CALL(*cap_2, evaluate()).Times(0);
//     EXPECT_CALL(*cap_3, evaluate()).Times(0);

//     CRE cre({CRE::NOT_PTR,
//              MockTypes::MOCK_1,
//              CRE::AND_PTR,
//              CRE::OPEN_PTR,
//              MockTypes::MOCK_2,
//              CRE::AND_PTR,
//              CRE::OPEN_PTR,
//              MockTypes::MOCK_3,
//              CRE::CLOSE_PTR,
//              CRE::CLOSE_PTR});

//     EXPECT_FALSE(cre.check_compatibility(caps));
// }

const std::vector<std::shared_ptr<CREToken>> expression_1{};

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

const std::vector<std::shared_ptr<CREToken>> expression_22{CRE::OPEN_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR};

const std::vector<std::shared_ptr<CREToken>> expression_23 =
    {CRE::OPEN_PTR, CRE::OPEN_PTR, CRE::NOT_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR, CRE::CLOSE_PTR};

// missing both operands for the OR operator
const std::vector<std::shared_ptr<CREToken>> invalid_expression_1{ELF_MAIN_SCHEDULE_TOKEN,
                                                                  CRE::AND_PTR,
                                                                  CRE::OPEN_PTR,
                                                                  CRE::OR_PTR,
                                                                  CRE::CLOSE_PTR};

// Missing only the first operand for the OR operator
const std::vector<std::shared_ptr<CREToken>> invalid_expression_15 =
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, CRE::OPEN_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::OR_PTR, CRE::CLOSE_PTR};

// Missing only the second operand for the OR operator
const std::vector<std::shared_ptr<CREToken>> invalid_expression_16 =
    {ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, CRE::OPEN_PTR, CRE::OR_PTR, ELF_MAIN_SCHEDULE_TOKEN, CRE::CLOSE_PTR};

// missing closed parenthesis
const std::vector<std::shared_ptr<CREToken>> invalid_expression_2{ELF_MAIN_SCHEDULE_TOKEN,
                                                                  CRE::AND_PTR,
                                                                  CRE::OPEN_PTR,
                                                                  BATCH_SIZE_TOKEN,
                                                                  CRE::OR_PTR,
                                                                  ELF_INIT_SCHEDULES_TOKEN};

// missing open parenthesis
const std::vector<std::shared_ptr<CREToken>> invalid_expression_3{ELF_MAIN_SCHEDULE_TOKEN,
                                                                  CRE::AND_PTR,
                                                                  BATCH_SIZE_TOKEN,
                                                                  CRE::OR_PTR,
                                                                  ELF_INIT_SCHEDULES_TOKEN,
                                                                  CRE::CLOSE_PTR};

/*
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                     |      /  \
                     0    *WS* *BT*
*/
// missing operand for the first OR operator
const std::vector<std::shared_ptr<CREToken>> invalid_expression_4{ELF_MAIN_SCHEDULE_TOKEN,
                                                                  CRE::AND_PTR,
                                                                  CRE::OPEN_PTR,
                                                                  CRE::OR_PTR,
                                                                  CRE::CLOSE_PTR,
                                                                  CRE::OPEN_PTR,
                                                                  ELF_INIT_SCHEDULES_TOKEN,
                                                                  CRE::OR_PTR,
                                                                  BATCH_SIZE_TOKEN,
                                                                  CRE::CLOSE_PTR};

// missing operands for nested operators
const std::vector<std::shared_ptr<CREToken>> invalid_expression_5 =
    {CRE::OPEN_PTR, CRE::OR_PTR, CRE::OPEN_PTR, CRE::OR_PTR, CRE::CLOSE_PTR, CRE::CLOSE_PTR, CRE::AND_PTR};

// NOT missing operand
const std::vector<std::shared_ptr<CREToken>> invalid_expression_6{ELF_MAIN_SCHEDULE_TOKEN, CRE::AND_PTR, CRE::NOT_PTR};

// chained NOTs with no operand
const std::vector<std::shared_ptr<CREToken>> invalid_expression_7{CRE::NOT_PTR, CRE::NOT_PTR};

// NOT missing operand before CLOSE
const std::vector<std::shared_ptr<CREToken>> invalid_expression_8{CRE::OPEN_PTR, CRE::NOT_PTR, CRE::CLOSE_PTR};

// missing operand
const std::vector<std::shared_ptr<CREToken>> invalid_expression_9{CRE::AND_PTR};

// too many operands
const std::vector<std::shared_ptr<CREToken>> invalid_expression_10{CRE::NOT_PTR,
                                                                   ELF_MAIN_SCHEDULE_TOKEN,
                                                                   BATCH_SIZE_TOKEN};

// missing CLOSE
const std::vector<std::shared_ptr<CREToken>> invalid_expression_11{CRE::OPEN_PTR,
                                                                   CRE::OPEN_PTR,
                                                                   CRE::NOT_PTR,
                                                                   ELF_MAIN_SCHEDULE_TOKEN,
                                                                   CRE::CLOSE_PTR};

// missing OPEN
const std::vector<std::shared_ptr<CREToken>> invalid_expression_12{CRE::OPEN_PTR,
                                                                   CRE::NOT_PTR,
                                                                   ELF_MAIN_SCHEDULE_TOKEN,
                                                                   CRE::CLOSE_PTR,
                                                                   CRE::CLOSE_PTR};

// Empty parrentheses cannot play the role of an operand
const std::vector<std::shared_ptr<CREToken>> invalid_expression_13{ELF_MAIN_SCHEDULE_TOKEN,
                                                                   CRE::AND_PTR,
                                                                   CRE::OPEN_PTR,
                                                                   CRE::CLOSE_PTR};

// The subexpression is just "NOT". The operand is missing
const std::vector<std::shared_ptr<CREToken>> invalid_expression_14{CRE::OPEN_PTR,
                                                                   CRE::NOT_PTR,
                                                                   CRE::CLOSE_PTR,
                                                                   ELF_MAIN_SCHEDULE_TOKEN};

// AND has too many operands
const std::vector<std::shared_ptr<CREToken>> invalid_expression_17{ELF_MAIN_SCHEDULE_TOKEN,
                                                                   CRE::AND_PTR,
                                                                   ELF_MAIN_SCHEDULE_TOKEN,
                                                                   ELF_MAIN_SCHEDULE_TOKEN};

// OR has too many operands
const std::vector<std::shared_ptr<CREToken>> invalid_expression_18{ELF_MAIN_SCHEDULE_TOKEN,
                                                                   CRE::OR_PTR,
                                                                   ELF_MAIN_SCHEDULE_TOKEN,
                                                                   ELF_MAIN_SCHEDULE_TOKEN};

// No operator to tie the two tokens
const std::vector<std::shared_ptr<CREToken>> invalid_expression_19{ELF_MAIN_SCHEDULE_TOKEN, ELF_MAIN_SCHEDULE_TOKEN};

// No operator to tie the two subexpressions
const std::vector<std::shared_ptr<CREToken>> invalid_expression_20{CRE::OPEN_PTR,
                                                                   ELF_MAIN_SCHEDULE_TOKEN,
                                                                   CRE::CLOSE_PTR,
                                                                   CRE::OPEN_PTR,
                                                                   ELF_MAIN_SCHEDULE_TOKEN,
                                                                   CRE::CLOSE_PTR};

// "OR" cannot replace an operand
const std::vector<std::shared_ptr<CREToken>> invalid_expression_21{ELF_MAIN_SCHEDULE_TOKEN, CRE::OR_PTR, CRE::OR_PTR};

std::vector<CREParams> valid_test_cases{
    make_test_params(expression_1, ov::CompatibilityCheck::SUPPORTED),

    make_test_params(expression_3, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_3, ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_4, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_4, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_4, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),

    make_test_params(expression_5,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_5,
                     ov::CompatibilityCheck::UNSUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_5, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),

    make_test_params(expression_6,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_6,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_6, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_6, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_6, ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_7,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_7, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_7, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_7, ov::CompatibilityCheck::UNSUPPORTED, {ELF_INIT_SCHEDULES_CODE}),

    make_test_params(expression_8,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_8,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_8, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_8, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),

    make_test_params(expression_9,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_9, ov::CompatibilityCheck::SUPPORTED, {ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_9, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_9, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),
    make_test_params(expression_9, ov::CompatibilityCheck::UNSUPPORTED),

    // should have the same behavior as expression_9
    make_test_params(expression_10,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_10, ov::CompatibilityCheck::SUPPORTED, {ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_10, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_10, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),
    make_test_params(expression_10, ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_12,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_12,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_12, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_12, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_12, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),

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
    make_test_params(expression_16,
                     ov::CompatibilityCheck::UNSUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_16, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_16, ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_17, ov::CompatibilityCheck::SUPPORTED, {ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_17,
                     ov::CompatibilityCheck::UNSUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_17,
                     ov::CompatibilityCheck::UNSUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_17, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),

    make_test_params(expression_18, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_18, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_18, ov::CompatibilityCheck::SUPPORTED),

    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED, {ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE}),
    make_test_params(expression_19, ov::CompatibilityCheck::SUPPORTED),
    make_test_params(expression_19,
                     ov::CompatibilityCheck::UNSUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}),

    make_test_params(expression_20,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_20, ov::CompatibilityCheck::SUPPORTED, {ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_20, ov::CompatibilityCheck::SUPPORTED),
    make_test_params(expression_20, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),

    make_test_params(expression_21,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_21,
                     ov::CompatibilityCheck::SUPPORTED,
                     {ELF_MAIN_SCHEDULE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_21, ov::CompatibilityCheck::SUPPORTED, {BATCH_SIZE_CODE, ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_21, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE, BATCH_SIZE_CODE}),
    make_test_params(expression_21, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_21, ov::CompatibilityCheck::UNSUPPORTED, {ELF_INIT_SCHEDULES_CODE}),
    make_test_params(expression_21, ov::CompatibilityCheck::UNSUPPORTED, {BATCH_SIZE_CODE}),
    make_test_params(expression_21, ov::CompatibilityCheck::UNSUPPORTED),

    make_test_params(expression_22, ov::CompatibilityCheck::SUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
    make_test_params(expression_23, ov::CompatibilityCheck::UNSUPPORTED, {ELF_MAIN_SCHEDULE_CODE}),
};

INSTANTIATE_TEST_SUITE_P(CRE,
                         ValidExpression,
                         ::testing::ValuesIn(valid_test_cases),
                         CREEvaluationTests::getTestCaseName);

std::vector<CREParams> invalid_test_cases{
    make_test_params(invalid_expression_1, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_2, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_3, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_4, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_5, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_6, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_7, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_8, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_9, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_10, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_11, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_12, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_13, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_14, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_15, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_16, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_17, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_18, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_19, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_20, ov::CompatibilityCheck::UNSUPPORTED),
    make_test_params(invalid_expression_21, ov::CompatibilityCheck::UNSUPPORTED),
};

INSTANTIATE_TEST_SUITE_P(CRE,
                         InvalidExpression,
                         ::testing::ValuesIn(invalid_test_cases),
                         CREEvaluationTests::getTestCaseName);
