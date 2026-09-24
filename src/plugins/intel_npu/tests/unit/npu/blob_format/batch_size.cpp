// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "batch_size_section.hpp"
#include "intel_npu/common/blob_reader_interface.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/section_type.hpp"
#include "utils.hpp"

using namespace intel_npu;

class BatchSizeSectionUnitTests : public ::testing::Test {
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

TEST_F(BatchSizeSectionUnitTests, CtorSetsTheRightValue) {
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::BATCH_SIZE));
    ASSERT_EQ(section->get_batch_size(), batch_size);
}

TEST_F(BatchSizeSectionUnitTests, CompatibilityReqsSubexpression) {
    const std::vector<std::shared_ptr<CREToken>> requirements =
        section->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 1);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::BATCH_SIZE));
}

TEST_F(BatchSizeSectionUnitTests, NullIndividualReqs) {
    ASSERT_FALSE(section->get_individual_compatibility_requirements().has_value());
}

TEST_F(BatchSizeSectionUnitTests, Write) {
    BlobWriterInterface writer = ov::unit_test::intel_npu::create_default_writer_interface(stream);
    section->write(writer);

    int64_t parsed_batch_size;
    ASSERT_TRUE(stream.good());
    stream.read(reinterpret_cast<char*>(&parsed_batch_size), sizeof(parsed_batch_size));
    ASSERT_TRUE(stream.good());
    EXPECT_EQ(parsed_batch_size, batch_size);
}

TEST_F(BatchSizeSectionUnitTests, Read) {
    ov::Tensor tensor(ov::element::u8, ov::Shape{sizeof(batch_size)}, &batch_size);
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, sizeof(batch_size), 0, sizeof(batch_size));

    auto read_section = BatchSizeSection::read(reader);
    auto casted_section = std::dynamic_pointer_cast<BatchSizeSection>(read_section);
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_batch_size(), batch_size);
}

TEST_F(BatchSizeSectionUnitTests, WriteRead) {
    BlobWriterInterface writer = ov::unit_test::intel_npu::create_default_writer_interface(stream);
    section->write(writer);

    const std::string buffer = stream.str();
    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, stream.tellp(), 0, stream.tellp());

    auto read_section = BatchSizeSection::read(reader);
    auto casted_section = std::dynamic_pointer_cast<BatchSizeSection>(read_section);
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_batch_size(), batch_size);
}

TEST_F(BatchSizeSectionUnitTests, InvalidSectionLength) {
    std::vector<uint8_t> dummy(0xFFFF, 0xFF);
    ov::Tensor tensor(ov::element::u8, ov::Shape{dummy.size()}, const_cast<uint8_t*>(dummy.data()));
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size() - 1);
    ASSERT_ANY_THROW(BatchSizeSection::read(reader));
}
