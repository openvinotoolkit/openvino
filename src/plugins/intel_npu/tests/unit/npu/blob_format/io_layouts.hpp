// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "io_layouts_section.hpp"
#include "openvino/core/layout.hpp"
#include "utils.hpp"

using namespace intel_npu;

using IOLayoutsParams = std::tuple<std::vector<ov::Layout>, std::vector<ov::Layout>>;

class IOLayoutsSectionUnitTests : public ::testing::TestWithParam<IOLayoutsParams> {
protected:
    void SetUp() override {
        std::vector<ov::Layout> input_layouts;
        std::vector<ov::Layout> output_layouts;
        std::tie(input_layouts, output_layouts) = GetParam();

        section = std::make_shared<IOLayoutsSection>(input_layouts, output_layouts);
        writer = ov::unit_test::intel_npu::create_default_writer_interface(stream);
    }

    std::shared_ptr<IOLayoutsSection> section;
    std::shared_ptr<BlobReader> reader;
    std::unique_ptr<BlobWriterInterface> writer;
    std::stringstream stream;
};

using ValidLayouts = IOLayoutsSectionUnitTests;

TEST_P(ValidLayouts, WriteRead) {
    section->write(writer);

    const std::string buffer = stream.str();
    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    reader = std::make_shared<BlobReader>(tensor);

    auto read_section = section->read(reader.get(), stream.tellp());

    auto layouts_result = std::dynamic_pointer_cast<IOLayoutsSection>(read_section);
    ASSERT_TRUE(layouts_result);
    EXPECT_EQ(layouts_result->get_input_layouts(), section->get_input_layouts());
    EXPECT_EQ(layouts_result->get_output_layouts(), section->get_output_layouts());
}

using IOLayoutsSectionRead = ::testing::Test;

TEST_F(IOLayoutsSectionRead, TooSmallSectionLength) {
    std::vector<uint8_t> dummy(0xFFFF, 0xFF);
    ov::Tensor tensor(ov::element::u8, ov::Shape{dummy.size()}, const_cast<uint8_t*>(dummy.data()));
    BlobReader reader(tensor);
    ASSERT_ANY_THROW(IOLayoutsSection::read(&reader, sizeof(uint64_t) - 1));
}

TEST_F(IOLayoutsSectionRead, LessLayoutsThanExpected) {
    IOLayoutsSection real_section({ov::Layout("NCHW"), ov::Layout("NHWC")}, {ov::Layout("NCHW"), ov::Layout("NHWC")});
    std::stringstream stream;
    auto writer = ov::unit_test::intel_npu::create_default_writer_interface(stream);
    real_section.write(writer);
    std::string buffer = stream.str();

    // overwrite total number of layouts
    const uint64_t fake_count = 20;
    std::memcpy(buffer.data(), &fake_count, sizeof(fake_count));
    std::memcpy(buffer.data() + sizeof(uint64_t), &fake_count, sizeof(fake_count));

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobReader reader(tensor);
    ASSERT_ANY_THROW(IOLayoutsSection::read(&reader, buffer.size()));
}

TEST_F(IOLayoutsSectionRead, InvalidLayout) {
    IOLayoutsSection valid_section({ov::Layout("N")}, {});
    std::stringstream stream;
    auto writer = ov::unit_test::intel_npu::create_default_writer_interface(stream);
    valid_section.write(writer);
    std::string buffer = stream.str();

    // overwrite input layout '[N]' with '[%]'
    const size_t layout_offset = sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint16_t);
    ASSERT_GE(buffer.size(), layout_offset + 3);
    buffer[layout_offset] = '[';
    buffer[layout_offset + 1] = '%';
    buffer[layout_offset + 2] = ']';

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobReader reader(tensor);

    std::shared_ptr<ISection> read_section;
    ASSERT_NO_THROW(read_section = IOLayoutsSection::read(&reader, buffer.size()));

    const auto layouts_result = std::dynamic_pointer_cast<IOLayoutsSection>(read_section);
    ASSERT_TRUE(layouts_result);
    ASSERT_EQ(layouts_result->get_input_layouts().size(), 1);
    EXPECT_EQ(layouts_result->get_input_layouts()[0], ov::Layout());
    EXPECT_TRUE(layouts_result->get_output_layouts().empty());
}
