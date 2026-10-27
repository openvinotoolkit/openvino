// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/runtime_requirements.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "utils.hpp"

namespace {}

using namespace intel_npu;
using testing::_;

using RuntimeRequirementsTest = testing::Test;

TEST_F(RuntimeRequirementsTest, CtorAndBasicGetters) {
    RuntimeRequirements runtime_requirements({}, CRE(), {});
    ASSERT_TRUE(runtime_requirements.get_sections_requirements().empty());
    ASSERT_TRUE(runtime_requirements.get_cre().empty());
    ASSERT_TRUE(runtime_requirements.get_section_id_to_type_mapping().empty());

    const std::map<SectionID, std::string> sections_requirements{{SectionID(0), "reqs"}};
    const CRE cre({std::make_shared<SectionType>(SectionTypeCode::BATCH_SIZE)});
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
        {SectionType(SectionTypeCode::UNKNOWN), std::make_shared<MockTypeEvaluator>(false)}};
    OV_EXPECT_THROW(runtime_requirements.get_compatibility_check_result(type_evaluators, {}), ov::Exception, _);

    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> instance_evaluators{
        {SectionType(SectionTypeCode::UNKNOWN), std::make_shared<MockInstanceEvaluator>()}};
    OV_EXPECT_THROW(runtime_requirements.get_compatibility_check_result({}, instance_evaluators), ov::Exception, _);
}
