// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_version_section.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/section_type.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "utils.hpp"

using namespace intel_npu;

class CompilerVersionSectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        compiler_version = 0xDEADBEEF;
        // TODO right log level?
        section = std::make_shared<CompilerVersionSection>(compiler_version);
    }

    int32_t compiler_version;
    std::shared_ptr<CompilerVersionSection> section;
    std::stringstream stream;
};

TEST_F(CompilerVersionSectionTest, CtorSetsTheRightValue) {
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::COMPILER_VERSION));
    ASSERT_EQ(section->get_compiler_version(), compiler_version);
}

TEST_F(CompilerVersionSectionTest, EmptyCompatibilityReqsSubexpression) {
    const std::vector<std::shared_ptr<CREToken>> requirements =
        section->get_compatibility_requirements_subexpression({});
    ASSERT_TRUE(requirements.empty());
}

TEST_F(CompilerVersionSectionTest, NullIndividualReqs) {
    ASSERT_FALSE(section->get_individual_compatibility_requirements().has_value());
}

TEST_F(CompilerVersionSectionTest, Write) {
    section->encode(create_generic_section_sink(stream));

    int32_t parsed_compiler_version;
    ASSERT_TRUE(stream.good());
    stream.read(reinterpret_cast<char*>(&parsed_compiler_version), sizeof(parsed_compiler_version));
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(parsed_compiler_version, compiler_version);
}

TEST_F(CompilerVersionSectionTest, Read) {
    ov::runtime::hsm::ManifestEntry manifest_entry =
        create_manifest_entry_of_type(SectionTypeCode::COMPILER_VERSION, true);
    std::memcpy(&manifest_entry.inline_bytes, &compiler_version, sizeof(compiler_version));

    auto read_section = CompilerVersionSection::decode(ov::runtime::hsm::Section(manifest_entry));
    ASSERT_TRUE(read_section.has_value());
    auto casted_section = std::dynamic_pointer_cast<CompilerVersionSection>(read_section.value());
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_compiler_version(), compiler_version);
}

TEST_F(CompilerVersionSectionTest, WriteRead) {
    section->encode(create_generic_section_sink(stream));

    const std::string buffer = stream.str();
    ov::runtime::hsm::ManifestEntry manifest_entry =
        create_manifest_entry_of_type(SectionTypeCode::COMPILER_VERSION, true);
    std::memcpy(&manifest_entry.inline_bytes, buffer.data(), buffer.size());

    auto read_section = CompilerVersionSection::decode(ov::runtime::hsm::Section(manifest_entry));
    ASSERT_TRUE(read_section);
    auto casted_section = std::dynamic_pointer_cast<CompilerVersionSection>(read_section.value());
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_compiler_version(), compiler_version);
}

TEST_F(CompilerVersionSectionTest, WrongTagToDecoder) {
    ov::runtime::hsm::ManifestEntry manifest_entry = create_manifest_entry_of_type(SectionTypeCode::BATCH_SIZE, true);
    std::memcpy(&manifest_entry.inline_bytes, &compiler_version, sizeof(compiler_version));

    OV_EXPECT_THROW(CompilerVersionSection::decode(ov::runtime::hsm::Section(manifest_entry)),
                    ov::Exception,
                    testing::_);
}

TEST_F(CompilerVersionSectionTest, NotInline) {
    ov::runtime::hsm::ManifestEntry manifest_entry =
        create_manifest_entry_of_type(SectionTypeCode::COMPILER_VERSION, false);
    std::memcpy(&manifest_entry.inline_bytes, &compiler_version, sizeof(compiler_version));

    OV_EXPECT_THROW(CompilerVersionSection::decode(ov::runtime::hsm::Section(manifest_entry)),
                    ov::Exception,
                    testing::_);
}
