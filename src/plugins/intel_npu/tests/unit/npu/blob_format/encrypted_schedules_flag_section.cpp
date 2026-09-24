// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/common/blob_reader_interface.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/encrypted_schedules_flag_section.hpp"
#include "intel_npu/common/section_type.hpp"
#include "utils.hpp"

using namespace intel_npu;

class EncryptedSchedulesFlagSectionUnitTests : public testing::TestWithParam<bool> {
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

TEST_P(EncryptedSchedulesFlagSectionUnitTests, CtorSetsTheRightValue) {
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG));
    ASSERT_EQ(section->get_flag(), encrypted_flag);
}

TEST_P(EncryptedSchedulesFlagSectionUnitTests, CompatibilityReqsSubexpression) {
    const std::vector<std::shared_ptr<CREToken>> requirements =
        section->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 1);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG));
}

TEST_P(EncryptedSchedulesFlagSectionUnitTests, NullIndividualReqs) {
    ASSERT_FALSE(section->get_individual_compatibility_requirements().has_value());
}

TEST_P(EncryptedSchedulesFlagSectionUnitTests, Write) {
    BlobWriterInterface writer = ov::unit_test::intel_npu::create_default_writer_interface(stream);
    section->write(writer);

    bool parsed_flag;
    ASSERT_TRUE(stream.good());
    stream.read(reinterpret_cast<char*>(&parsed_flag), sizeof(parsed_flag));
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(parsed_flag, encrypted_flag);
}

TEST_P(EncryptedSchedulesFlagSectionUnitTests, Read) {
    ov::Tensor tensor(ov::element::u8, ov::Shape{sizeof(encrypted_flag)}, &encrypted_flag);
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, sizeof(encrypted_flag), 0, sizeof(encrypted_flag));

    auto read_section = EncryptedSchedulesFlagSection::read(reader);
    auto casted_section = std::dynamic_pointer_cast<EncryptedSchedulesFlagSection>(read_section);
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_flag(), encrypted_flag);
}

TEST_P(EncryptedSchedulesFlagSectionUnitTests, WriteRead) {
    BlobWriterInterface writer = ov::unit_test::intel_npu::create_default_writer_interface(stream);
    section->write(writer);

    const std::string buffer = stream.str();
    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, stream.tellp(), 0, stream.tellp());

    auto read_section = EncryptedSchedulesFlagSection::read(reader);
    auto casted_section = std::dynamic_pointer_cast<EncryptedSchedulesFlagSection>(read_section);
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_flag(), encrypted_flag);
}

TEST_P(EncryptedSchedulesFlagSectionUnitTests, InvalidSectionLength) {
    std::vector<uint8_t> dummy(0xFFFF, 0xFF);
    ov::Tensor tensor(ov::element::u8, ov::Shape{dummy.size()}, const_cast<uint8_t*>(dummy.data()));
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size() + 1);
    ASSERT_ANY_THROW(EncryptedSchedulesFlagSection::read(reader));
}

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         EncryptedSchedulesFlagSectionUnitTests,
                         testing::ValuesIn(std::vector<bool>{true, false}),
                         EncryptedSchedulesFlagSectionUnitTests::getTestCaseName);
