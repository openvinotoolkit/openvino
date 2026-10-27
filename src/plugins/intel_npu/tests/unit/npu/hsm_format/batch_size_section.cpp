// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_size_section.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/section_type.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "utils.hpp"

using namespace intel_npu;

class BatchSizeSectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        batch_size = 0xDEADBEEF;
        // TODO right log level?
        section = std::make_shared<BatchSizeSection>(batch_size);
    }

    int64_t batch_size;
    std::shared_ptr<BatchSizeSection> section;
    std::stringstream stream;
};

TEST_F(BatchSizeSectionTest, CtorSetsTheRightValue) {
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::BATCH_SIZE));
    ASSERT_EQ(section->get_batch_size(), batch_size);
}

TEST_F(BatchSizeSectionTest, CompatibilityReqsSubexpression) {
    const std::vector<std::shared_ptr<CREToken>> requirements =
        section->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 1);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::BATCH_SIZE));
}

TEST_F(BatchSizeSectionTest, NullIndividualReqs) {
    ASSERT_FALSE(section->get_individual_compatibility_requirements().has_value());
}

TEST_F(BatchSizeSectionTest, Write) {
    section->encode(ov::unit_test::intel_npu::create_generic_section_sink(stream));

    int64_t parsed_batch_size;
    ASSERT_TRUE(stream.good());
    stream.read(reinterpret_cast<char*>(&parsed_batch_size), sizeof(parsed_batch_size));
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(parsed_batch_size, batch_size);
}

TEST_F(BatchSizeSectionTest, Read) {
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::BATCH_SIZE, true);
    std::memcpy(&manifest_entry.inline_bytes, &batch_size, sizeof(batch_size));

    auto read_section = BatchSizeSection::decode(ov::runtime::hsm::Section(manifest_entry));
    ASSERT_TRUE(read_section.has_value());
    auto casted_section = std::dynamic_pointer_cast<BatchSizeSection>(read_section.value());
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_batch_size(), batch_size);
}

TEST_F(BatchSizeSectionTest, WriteRead) {
    section->encode(ov::unit_test::intel_npu::create_generic_section_sink(stream));

    const std::string buffer = stream.str();
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::BATCH_SIZE, true);
    std::memcpy(&manifest_entry.inline_bytes, buffer.data(), buffer.size());

    auto read_section = BatchSizeSection::decode(ov::runtime::hsm::Section(manifest_entry));
    ASSERT_TRUE(read_section.has_value());
    auto casted_section = std::dynamic_pointer_cast<BatchSizeSection>(read_section.value());
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_batch_size(), batch_size);
}

TEST_F(BatchSizeSectionTest, WrongTagToDecoder) {
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::COMPILER_VERSION, true);
    std::memcpy(&manifest_entry.inline_bytes, &batch_size, sizeof(batch_size));

    OV_EXPECT_THROW(BatchSizeSection::decode(ov::runtime::hsm::Section(manifest_entry)), ov::Exception, testing::_);
}

TEST_F(BatchSizeSectionTest, NotInline) {
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::BATCH_SIZE, false);
    std::memcpy(&manifest_entry.inline_bytes, &batch_size, sizeof(batch_size));

    OV_EXPECT_THROW(BatchSizeSection::decode(ov::runtime::hsm::Section(manifest_entry)), ov::Exception, testing::_);
}
