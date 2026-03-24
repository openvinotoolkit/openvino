// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "batch_size_section.hpp"
#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/static_capability.hpp"
#include "io_layouts_section.hpp"

using namespace intel_npu;

namespace {
void prepare_writer(const std::shared_ptr<BlobWriter>& blobWriter,
                    const std::vector<CRE::Token>& expression,
                    const std::vector<std::shared_ptr<ISection>> sections) {
    blobWriter->append_compatibility_requirement(expression);
    // not sure if this is good enough to work
    for (const auto& section : sections) {
        blobWriter->register_section(section);
    }
}

// should we unit test the reader with precompiled compatibility blobs?
void reader_register_sections(const std::shared_ptr<BlobReader>& blobReader,
                              const std::vector<uint16_t> section_types,
                              std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& capabilities) {
    for (const auto& token : section_types) {
        capabilities[token] = std::make_shared<StaticCapability>(token);
    }
    // what if we have a different reader?
    blobReader->register_reader(PredefinedSectionType::BATCH_SIZE, BatchSizeSection::read);
    blobReader->register_reader(PredefinedSectionType::IO_LAYOUTS, IOLayoutsSection::read);
}

std::pair<std::string, std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>> make_simple_blob(
    int64_t batch_size) {
    BlobWriter writer;
    writer.register_section(std::make_shared<BatchSizeSection>(batch_size));
    std::stringstream stream;
    writer.write(stream);
    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[CRE::CRE_EVALUATION] = std::make_shared<StaticCapability>(CRE::CRE_EVALUATION);
    return {stream.str(), std::move(caps)};
}
}  // namespace

using WriterReaderParams = std::tuple<std::vector<CRE::Token>, std::vector<uint16_t>>;

// TODO: test with BlobWriter(BlobReader)

// should we randomize the order of serialized sections?
class WriterReaderUnitTests : public ::testing::TestWithParam<WriterReaderParams> {
protected:
    const int64_t BATCH = 0xDEADBEEF;
    const std::vector<ov::Layout> INPUT_LAYOUTS = {ov::Layout("NCHW"), ov::Layout("NHW")};
    const std::vector<ov::Layout> OUTPUT_LAYOUTS = {ov::Layout("NHWC")};

    void SetUp() override {
        std::tie(expression, section_types) = GetParam();

        batch_section = std::make_shared<BatchSizeSection>(BATCH);
        io_section = std::make_shared<IOLayoutsSection>(INPUT_LAYOUTS, OUTPUT_LAYOUTS);

        writer = std::make_shared<BlobWriter>();
        prepare_writer(writer, expression, {batch_section, io_section});
        writer->write(stream);

        buffer = stream.str();
        tensor = ov::Tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
        reader = std::make_shared<BlobReader>(tensor);
        reader_register_sections(reader, section_types, capabilities);
    }

    std::vector<CRE::Token> expression;
    std::vector<uint16_t> section_types;

    std::shared_ptr<BlobWriter> writer;
    std::shared_ptr<BlobReader> reader;
    std::shared_ptr<BatchSizeSection> batch_section;
    std::shared_ptr<IOLayoutsSection> io_section;
    std::stringstream stream;
    std::string buffer;
    ov::Tensor tensor;
    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> capabilities;
};

using AllSections = WriterReaderUnitTests;

// all sections except ELF
TEST_P(AllSections, WriteRead) {
    ASSERT_NO_THROW(reader->read(capabilities));

    auto read_batch =
        std::dynamic_pointer_cast<BatchSizeSection>(reader->retrieve_first_section(PredefinedSectionType::BATCH_SIZE));
    ASSERT_TRUE(read_batch);
    EXPECT_EQ(read_batch->get_batch_size(), batch_section->get_batch_size());

    auto read_io =
        std::dynamic_pointer_cast<IOLayoutsSection>(reader->retrieve_first_section(PredefinedSectionType::IO_LAYOUTS));
    ASSERT_TRUE(read_io);
    EXPECT_EQ(read_io->get_input_layouts(), io_section->get_input_layouts());
    EXPECT_EQ(read_io->get_output_layouts(), io_section->get_output_layouts());
}

using IncompatibleCRE = WriterReaderUnitTests;

TEST_P(IncompatibleCRE, ReadThrows) {
    ASSERT_ANY_THROW(reader->read(capabilities));
}

using WriterReaderEdgeCases = ::testing::Test;

TEST_F(WriterReaderEdgeCases, CorruptedMagicByte) {
    auto [blob, caps] = make_simple_blob(0xDEADBEEF);
    blob[0] = 'X';
    ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()}, blob.data());
    BlobReader reader(tensor);
    reader.register_reader(PredefinedSectionType::BATCH_SIZE, BatchSizeSection::read);
    ASSERT_ANY_THROW(reader.read(caps));
}

TEST_F(WriterReaderEdgeCases, CorruptedFormatVersion) {
    constexpr size_t MAGIC_BYTES_SIZE = 5;  // 'OVNPU'
    auto [blob, caps] = make_simple_blob(0xDEADBEEF);
    blob[MAGIC_BYTES_SIZE] = static_cast<char>(~static_cast<unsigned char>(blob[MAGIC_BYTES_SIZE]));
    ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()}, blob.data());
    BlobReader reader(tensor);
    reader.register_reader(PredefinedSectionType::BATCH_SIZE, BatchSizeSection::read);
    ASSERT_ANY_THROW(reader.read(caps));
}

// offsets table given to writer_2 from reader_1 leads to failing OffsetsTable::add_entry()
// BATCHING is found already being present in the map, so the assertion fails
TEST_F(WriterReaderEdgeCases, ReExportRoundTrip) {
    constexpr int64_t BATCH = 0xDEADBEEF;
    const std::vector<ov::Layout> input_layouts = {ov::Layout("NCHW")};
    const std::vector<ov::Layout> output_layouts = {ov::Layout("NHWC")};

    BlobWriter writer_1;
    writer_1.register_section(std::make_shared<BatchSizeSection>(BATCH));
    writer_1.register_section(std::make_shared<IOLayoutsSection>(input_layouts, output_layouts));
    std::stringstream stream_1;
    writer_1.write(stream_1);
    std::string buffer_1 = stream_1.str();

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[CRE::CRE_EVALUATION] = std::make_shared<StaticCapability>(CRE::CRE_EVALUATION);

    ov::Tensor tensor_1(ov::element::u8, ov::Shape{buffer_1.size()}, buffer_1.data());
    auto reader_1 = std::make_shared<BlobReader>(tensor_1);
    reader_1->register_reader(PredefinedSectionType::BATCH_SIZE, BatchSizeSection::read);
    reader_1->register_reader(PredefinedSectionType::IO_LAYOUTS, IOLayoutsSection::read);
    ASSERT_NO_THROW(reader_1->read(caps));

    BlobWriter writer_2(reader_1);
    std::stringstream stream_2;
    writer_2.write(stream_2);
    std::string buf2 = stream_2.str();

    ov::Tensor tensor_2(ov::element::u8, ov::Shape{buf2.size()}, buf2.data());
    auto reader_2 = std::make_shared<BlobReader>(tensor_2);
    reader_2->register_reader(PredefinedSectionType::BATCH_SIZE, BatchSizeSection::read);
    reader_2->register_reader(PredefinedSectionType::IO_LAYOUTS, IOLayoutsSection::read);
    ASSERT_NO_THROW(reader_2->read(caps));

    auto read_batch = std::dynamic_pointer_cast<BatchSizeSection>(
        reader_2->retrieve_first_section(PredefinedSectionType::BATCH_SIZE));
    ASSERT_TRUE(read_batch);
    EXPECT_EQ(read_batch->get_batch_size(), BATCH);

    auto read_io = std::dynamic_pointer_cast<IOLayoutsSection>(
        reader_2->retrieve_first_section(PredefinedSectionType::IO_LAYOUTS));
    ASSERT_TRUE(read_io);
    EXPECT_EQ(read_io->get_input_layouts(), input_layouts);
    EXPECT_EQ(read_io->get_output_layouts(), output_layouts);
}

// the reader is unable to read the first section
// npu_region_size from reader is not initialized?
TEST_F(WriterReaderEdgeCases, MultipleSectionsSameType) {
    constexpr int64_t BATCH_A = 0xDEADBEEF;
    constexpr int64_t BATCH_B = 0xCAFEBABE;

    BlobWriter writer;
    writer.register_section(std::make_shared<BatchSizeSection>(BATCH_A));
    writer.register_section(std::make_shared<BatchSizeSection>(BATCH_B));
    std::stringstream stream;
    writer.write(stream);
    std::string buffer = stream.str();

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobReader reader(tensor);
    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[CRE::CRE_EVALUATION] = std::make_shared<StaticCapability>(CRE::CRE_EVALUATION);
    reader.register_reader(PredefinedSectionType::BATCH_SIZE, BatchSizeSection::read);
    ASSERT_NO_THROW(reader.read(caps));

    auto sections = reader.retrieve_sections_same_type(PredefinedSectionType::BATCH_SIZE);
    ASSERT_TRUE(sections.has_value());
    ASSERT_EQ(sections->size(), 2);

    ASSERT_TRUE(sections->count(0));
    ASSERT_TRUE(sections->count(1));
    EXPECT_EQ(std::dynamic_pointer_cast<BatchSizeSection>(sections->at(0))->get_batch_size(), BATCH_A);
    EXPECT_EQ(std::dynamic_pointer_cast<BatchSizeSection>(sections->at(1))->get_batch_size(), BATCH_B);
}

TEST_F(WriterReaderEdgeCases, UnknownSectionSkipped) {
    auto [blob, caps] = make_simple_blob(99);
    ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()}, blob.data());
    BlobReader reader(tensor);

    // intentionally not registering batch size
    ASSERT_NO_THROW(reader.read(caps));
    EXPECT_EQ(reader.retrieve_first_section(PredefinedSectionType::BATCH_SIZE), nullptr);
}
