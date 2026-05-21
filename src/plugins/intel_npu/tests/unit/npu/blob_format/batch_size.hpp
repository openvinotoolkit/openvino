// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "batch_size_section.hpp"
#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "utils.hpp"

using namespace intel_npu;

class BatchSizeSectionUnitTests : public ::testing::Test {
protected:
    void SetUp() override {
        batch_size = 0xDEADBEEF;
        section = std::make_shared<BatchSizeSection>(batch_size);
        writer = ov::unit_test::intel_npu::create_default_writer_interface();
    }

    int64_t batch_size;
    std::shared_ptr<BatchSizeSection> section;
    std::unique_ptr<BlobWriterInterface> writer;
    std::stringstream stream;
};

TEST_F(BatchSizeSectionUnitTests, WriteRead) {
    section->write(writer);

    const std::string buffer = stream.str();
    ov::Tensor source_tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobReader reader(source_tensor);

    auto read_section = BatchSizeSection::read(&reader, stream.tellp());
    auto casted_section = std::dynamic_pointer_cast<BatchSizeSection>(read_section);
    ASSERT_TRUE(casted_section);
    EXPECT_EQ(casted_section->get_batch_size(), batch_size);
}

TEST_F(BatchSizeSectionUnitTests, InvalidSectionLength) {
    std::vector<uint8_t> dummy(0xFFFF, 0xFF);
    ov::Tensor source(ov::element::u8, ov::Shape{dummy.size()}, const_cast<uint8_t*>(dummy.data()));
    BlobReader reader(source);
    ASSERT_ANY_THROW(BatchSizeSection::read(&reader, source.get_byte_size() - 1));
}
