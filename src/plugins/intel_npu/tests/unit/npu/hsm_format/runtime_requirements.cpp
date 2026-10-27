// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/runtime_requirements.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "utils.hpp"

using namespace intel_npu;
using testing::_;

namespace {

const auto MOCK_INSTANCE_EVALUATOR = std::make_shared<MockInstanceEvaluator>();
const auto MOCK_TYPE_EVALUATOR_TRUE = std::make_shared<MockTypeEvaluator>(true);
const auto MOCK_TYPE_EVALUATOR_FALSE = std::make_shared<MockTypeEvaluator>(false);

const auto ELF_MAIN_SCHEDULE_PTR = std::make_shared<SectionType>(SectionTypeCode::ELF_MAIN_SCHEDULE);
const auto BATCH_SIZE_PTR = std::make_shared<SectionType>(SectionTypeCode::BATCH_SIZE);
const auto COMPILER_VERSION_PTR = std::make_shared<SectionType>(SectionTypeCode::COMPILER_VERSION);
const auto ID_0_PTR = std::make_shared<SectionID>(0);
const auto ID_2_PTR = std::make_shared<SectionID>(2);

}  // namespace

using RuntimeRequirementsTest = testing::Test;

TEST_F(RuntimeRequirementsTest, CtorAndBasicGetters) {
    RuntimeRequirements runtime_requirements({}, CRE(), {});
    ASSERT_TRUE(runtime_requirements.get_sections_requirements().empty());
    ASSERT_TRUE(runtime_requirements.get_cre().empty());
    ASSERT_TRUE(runtime_requirements.get_section_id_to_type_mapping().empty());

    const std::map<SectionID, std::string> sections_requirements{{SectionID(0), "reqs"}};
    const CRE cre({BATCH_SIZE_PTR});
    const std::unordered_map<SectionID, SectionType> section_id_to_type{
        {SectionID(0), SectionType(SectionTypeCode::BATCH_SIZE)}};

    runtime_requirements = RuntimeRequirements(sections_requirements, cre, section_id_to_type);
    ASSERT_EQ(runtime_requirements.get_sections_requirements(), sections_requirements);
    ASSERT_EQ(runtime_requirements.get_cre(), cre);
    ASSERT_EQ(runtime_requirements.get_section_id_to_type_mapping(), section_id_to_type);
}

TEST_F(RuntimeRequirementsTest, UnknownTypeInEvaluators) {
    RuntimeRequirements runtime_requirements({}, CRE(), {});
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> type_evaluators{
        {SectionType(SectionTypeCode::UNKNOWN), MOCK_TYPE_EVALUATOR_FALSE}};
    OV_EXPECT_THROW(runtime_requirements.get_compatibility_check_result(type_evaluators, {}), ov::Exception, _);

    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> instance_evaluators{
        {SectionType(SectionTypeCode::UNKNOWN), MOCK_INSTANCE_EVALUATOR}};
    OV_EXPECT_THROW(runtime_requirements.get_compatibility_check_result({}, instance_evaluators), ov::Exception, _);
}

// TODO atm there can be a mismatch between the type-ID info within the CRE & the map
// TODO CRE to check that one section ID corresponds to a single type?
TEST_F(RuntimeRequirementsTest, MissingSectionID) {
    RuntimeRequirements runtime_requirements({{SectionID(0), "reqs"}},
                                             CRE(),
                                             {{SectionID(1), SectionType(SectionTypeCode::BATCH_SIZE)}});
    OV_EXPECT_THROW(runtime_requirements.get_compatibility_check_result({}, {}), ov::Exception, _);
}

TEST_F(RuntimeRequirementsTest, NoEvaluationResultBeforeEvaluation) {
    RuntimeRequirements runtime_requirements({}, CRE(), {});
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE), std::nullopt);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::UNKNOWN), std::nullopt);
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::nullopt);

    const std::unordered_map<SectionID, SectionType> section_id_to_type{
        {SectionID(0), SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE)}};
    const CRE cre({ELF_MAIN_SCHEDULE_PTR, ID_0_PTR});
    runtime_requirements =
        RuntimeRequirements({{SectionID(0), STRING_THAT_EVALUATES_TO_SUPPORTED.data()}}, cre, section_id_to_type);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE), std::nullopt);
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::nullopt);
}

TEST_F(RuntimeRequirementsTest, EvaluationResultWithEvaluators) {
    const std::unordered_map<SectionID, SectionType> section_id_to_type{
        {SectionID(0), SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE)}};
    const CRE cre({ELF_MAIN_SCHEDULE_PTR, ID_0_PTR});
    RuntimeRequirements runtime_requirements({{SectionID(0), STRING_THAT_EVALUATES_TO_SUPPORTED.data()}},
                                             cre,
                                             section_id_to_type);

    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> type_evaluators{
        {SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE), MOCK_TYPE_EVALUATOR_FALSE}};
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> instance_evaluators{
        {SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE), MOCK_INSTANCE_EVALUATOR}};
    ASSERT_EQ(runtime_requirements.get_compatibility_check_result(type_evaluators, instance_evaluators),
              ov::CompatibilityCheck::UNSUPPORTED);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE),
              std::make_optional(false));
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::nullopt);

    type_evaluators[SectionTypeCode::ELF_MAIN_SCHEDULE] = MOCK_TYPE_EVALUATOR_TRUE;
    ASSERT_EQ(runtime_requirements.get_compatibility_check_result(type_evaluators, instance_evaluators),
              ov::CompatibilityCheck::SUPPORTED);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE),
              std::make_optional(true));
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::make_optional(true));

    runtime_requirements =
        RuntimeRequirements({{SectionID(0), STRING_THAT_EVALUATES_TO_UNSUPPORTED.data()}}, cre, section_id_to_type);
    ASSERT_EQ(runtime_requirements.get_compatibility_check_result(type_evaluators, instance_evaluators),
              ov::CompatibilityCheck::UNSUPPORTED);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE),
              std::make_optional(true));
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::make_optional(false));

    runtime_requirements =
        RuntimeRequirements({{SectionID(0), STRING_THAT_EVALUATES_TO_UNKNOWN.data()}}, cre, section_id_to_type);
    ASSERT_EQ(runtime_requirements.get_compatibility_check_result(type_evaluators, instance_evaluators),
              ov::CompatibilityCheck::NOT_APPLICABLE);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE),
              std::make_optional(true));
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::nullopt);
}

TEST_F(RuntimeRequirementsTest, EvaluationResultWithMissingEvaluators) {
    const std::unordered_map<SectionID, SectionType> section_id_to_type{
        {SectionID(0), SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE)}};
    const CRE cre({ELF_MAIN_SCHEDULE_PTR, ID_0_PTR});
    RuntimeRequirements runtime_requirements({{SectionID(0), STRING_THAT_EVALUATES_TO_UNKNOWN.data()}},
                                             cre,
                                             section_id_to_type);

    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> type_evaluators{
        {SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE), MOCK_TYPE_EVALUATOR_TRUE}};
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> instance_evaluators{
        {SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE), MOCK_INSTANCE_EVALUATOR}};

    ASSERT_EQ(runtime_requirements.get_compatibility_check_result(type_evaluators, {}),
              ov::CompatibilityCheck::SUPPORTED);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE),
              std::make_optional(true));
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::nullopt);

    ASSERT_EQ(runtime_requirements.get_compatibility_check_result({}, instance_evaluators),
              ov::CompatibilityCheck::UNSUPPORTED);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE), std::nullopt);
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::nullopt);

    ASSERT_EQ(runtime_requirements.get_compatibility_check_result({}, {}), ov::CompatibilityCheck::UNSUPPORTED);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE), std::nullopt);
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::nullopt);
}

TEST_F(RuntimeRequirementsTest, MoreComplexEvaluation) {
    const std::map<SectionID, std::string> sections_requirements{
        {SectionID(0), STRING_THAT_EVALUATES_TO_SUPPORTED.data()},
        {SectionID(2), STRING_THAT_EVALUATES_TO_SUPPORTED.data()}};
    const std::unordered_map<SectionID, SectionType> section_id_to_type{
        {SectionID(0), SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE)},
        {SectionID(2), SectionType(SectionTypeCode::BATCH_SIZE)}};
    const CRE cre(
        {BATCH_SIZE_PTR, ID_2_PTR, CRE::OR_PTR, COMPILER_VERSION_PTR, CRE::AND_PTR, ELF_MAIN_SCHEDULE_PTR, ID_0_PTR});
    RuntimeRequirements runtime_requirements(sections_requirements, cre, section_id_to_type);

    // Need different instances of "false mock evaluators" to check which one has evaluated
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> type_evaluators{
        {SectionType(SectionTypeCode::BATCH_SIZE), std::make_shared<MockTypeEvaluator>(true)},
        {SectionType(SectionTypeCode::COMPILER_VERSION), std::make_shared<MockTypeEvaluator>(false)},
        {SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE), std::make_shared<MockTypeEvaluator>(false)}};
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> instance_evaluators{
        {SectionType(SectionTypeCode::BATCH_SIZE), MOCK_INSTANCE_EVALUATOR},
        {SectionType(SectionTypeCode::COMPILER_VERSION), MOCK_INSTANCE_EVALUATOR},
        {SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE), MOCK_INSTANCE_EVALUATOR}};

    ASSERT_EQ(runtime_requirements.get_compatibility_check_result(type_evaluators, instance_evaluators),
              ov::CompatibilityCheck::UNSUPPORTED);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::BATCH_SIZE), std::make_optional(true));
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::COMPILER_VERSION), std::nullopt);
    ASSERT_EQ(runtime_requirements.get_type_evaluation_result(SectionTypeCode::ELF_MAIN_SCHEDULE),
              std::make_optional(false));
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(0), std::nullopt);
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(1), std::nullopt);
    ASSERT_EQ(runtime_requirements.get_instance_evaluation_result(2), std::make_optional(true));
}
