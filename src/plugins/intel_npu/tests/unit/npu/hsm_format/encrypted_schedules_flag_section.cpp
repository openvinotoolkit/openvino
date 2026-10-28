// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/encrypted_schedules_flag_section.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/section_type.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "utils.hpp"

using namespace intel_npu;

class EncryptedSchedulesFlagSectionTest : public testing::TestWithParam<bool> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<bool>& obj) {
        return obj.param ? "true" : "false";
    }

protected:
    void SetUp() override {
        encrypted_flag = GetParam();
        section = std::make_shared<EncryptedSchedulesFlagSection>(encrypted_flag);
    }

    bool encrypted_flag;
    std::shared_ptr<EncryptedSchedulesFlagSection> section;
    std::stringstream stream;
};

TEST_P(EncryptedSchedulesFlagSectionTest, CtorSetsTheRightValue) {
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG));
    ASSERT_EQ(section->get_flag(), encrypted_flag);
}

TEST_P(EncryptedSchedulesFlagSectionTest, CompatibilityReqsSubexpression) {
    const std::vector<std::shared_ptr<CREToken>> requirements =
        section->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 1);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG));
}

TEST_P(EncryptedSchedulesFlagSectionTest, NullIndividualReqs) {
    ASSERT_FALSE(section->get_individual_compatibility_requirements().has_value());
}

TEST_P(EncryptedSchedulesFlagSectionTest, Write) {
    section->encode(create_generic_section_sink(stream));

    bool parsed_flag;
    ASSERT_TRUE(stream.good());
    stream.read(reinterpret_cast<char*>(&parsed_flag), sizeof(parsed_flag));
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(parsed_flag, encrypted_flag);
}

TEST_P(EncryptedSchedulesFlagSectionTest, Read) {
    ov::runtime::hsm::ManifestEntry manifest_entry =
        create_manifest_entry_of_type(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG, true);
    std::memcpy(&manifest_entry.inline_bytes, &encrypted_flag, sizeof(encrypted_flag));

    auto read_section = EncryptedSchedulesFlagSection::decode(ov::runtime::hsm::Section(manifest_entry));
    ASSERT_TRUE(read_section);
    auto casted_section = std::dynamic_pointer_cast<EncryptedSchedulesFlagSection>(read_section.value());
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_flag(), encrypted_flag);
}

TEST_P(EncryptedSchedulesFlagSectionTest, WriteRead) {
    section->encode(create_generic_section_sink(stream));

    const std::string buffer = stream.str();
    ov::runtime::hsm::ManifestEntry manifest_entry =
        create_manifest_entry_of_type(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG, true);
    std::memcpy(&manifest_entry.inline_bytes, buffer.data(), buffer.size());

    auto read_section = EncryptedSchedulesFlagSection::decode(ov::runtime::hsm::Section(manifest_entry));
    ASSERT_TRUE(read_section);
    auto casted_section = std::dynamic_pointer_cast<EncryptedSchedulesFlagSection>(read_section.value());
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_flag(), encrypted_flag);
}

TEST_P(EncryptedSchedulesFlagSectionTest, WrongTagToDecoder) {
    ov::runtime::hsm::ManifestEntry manifest_entry = create_manifest_entry_of_type(SectionTypeCode::BATCH_SIZE, true);
    std::memcpy(&manifest_entry.inline_bytes, &encrypted_flag, sizeof(encrypted_flag));

    OV_EXPECT_THROW(EncryptedSchedulesFlagSection::decode(ov::runtime::hsm::Section(manifest_entry)),
                    ov::Exception,
                    testing::_);
}

TEST_P(EncryptedSchedulesFlagSectionTest, NotInline) {
    ov::runtime::hsm::ManifestEntry manifest_entry =
        create_manifest_entry_of_type(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG, false);
    std::memcpy(&manifest_entry.inline_bytes, &encrypted_flag, sizeof(encrypted_flag));

    OV_EXPECT_THROW(EncryptedSchedulesFlagSection::decode(ov::runtime::hsm::Section(manifest_entry)),
                    ov::Exception,
                    testing::_);
}

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         EncryptedSchedulesFlagSectionTest,
                         testing::ValuesIn(std::vector<bool>{true, false}),
                         EncryptedSchedulesFlagSectionTest::getTestCaseName);
