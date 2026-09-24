// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_version_section.hpp"

#include <gtest/gtest.h>

#include "intel_npu/common/blob_reader_interface.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/section_type.hpp"
#include "utils.hpp"

using namespace intel_npu;

class CompilerVersionSectionUnitTests : public ::testing::Test {
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

TEST_F(CompilerVersionSectionUnitTests, CtorSetsTheRightValue) {
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::COMPILER_VERSION));
    ASSERT_EQ(section->get_compiler_version(), compiler_version);
}

TEST_F(CompilerVersionSectionUnitTests, EmptyCompatibilityReqsSubexpression) {
    const std::vector<std::shared_ptr<CREToken>> requirements =
        section->get_compatibility_requirements_subexpression({});
    ASSERT_TRUE(requirements.empty());
}

TEST_F(CompilerVersionSectionUnitTests, NullIndividualReqs) {
    ASSERT_FALSE(section->get_individual_compatibility_requirements().has_value());
}

TEST_F(CompilerVersionSectionUnitTests, Write) {
    BlobWriterInterface writer = ov::unit_test::intel_npu::create_default_writer_interface(stream);
    section->write(writer);

    int32_t parsed_compiler_version;
    ASSERT_TRUE(stream.good());
    stream.read(reinterpret_cast<char*>(&parsed_compiler_version), sizeof(parsed_compiler_version));
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(parsed_compiler_version, compiler_version);
}

TEST_F(CompilerVersionSectionUnitTests, Read) {
    ov::Tensor tensor(ov::element::u8, ov::Shape{sizeof(compiler_version)}, &compiler_version);
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, sizeof(compiler_version), 0, sizeof(compiler_version));

    auto read_section = CompilerVersionSection::read(reader);
    auto casted_section = std::dynamic_pointer_cast<CompilerVersionSection>(read_section);
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_compiler_version(), compiler_version);
}

TEST_F(CompilerVersionSectionUnitTests, WriteRead) {
    BlobWriterInterface writer = ov::unit_test::intel_npu::create_default_writer_interface(stream);
    section->write(writer);

    const std::string buffer = stream.str();
    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, stream.tellp(), 0, stream.tellp());

    auto read_section = CompilerVersionSection::read(reader);
    auto casted_section = std::dynamic_pointer_cast<CompilerVersionSection>(read_section);
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_compiler_version(), compiler_version);
}

TEST_F(CompilerVersionSectionUnitTests, InvalidSectionLength) {
    std::vector<uint8_t> dummy(0xFFFF, 0xFF);
    ov::Tensor tensor(ov::element::u8, ov::Shape{dummy.size()}, const_cast<uint8_t*>(dummy.data()));
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size() - 1);
    ASSERT_ANY_THROW(CompilerVersionSection::read(reader));
}
