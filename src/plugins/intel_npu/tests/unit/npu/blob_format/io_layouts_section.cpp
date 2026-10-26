// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "io_layouts_section.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "utils.hpp"

namespace {

constexpr size_t LAYOUTS_COUNT_SIZE = sizeof(uint32_t);

}

using namespace intel_npu;

using IOLayoutsParams = std::tuple<std::vector<ov::Layout>, std::vector<ov::Layout>>;

class IOLayoutsSectionTest : public ::testing::TestWithParam<IOLayoutsParams> {
protected:
    void SetUp() override {
        std::vector<ov::Layout> input_layouts;
        std::vector<ov::Layout> output_layouts;
        std::tie(input_layouts, output_layouts) = GetParam();

        section = std::make_shared<IOLayoutsSection>(input_layouts, output_layouts);
    }

    std::shared_ptr<IOLayoutsSection> section;
    std::stringstream stream;
};

using ValidLayouts = IOLayoutsSectionTest;

TEST_P(ValidLayouts, WriteRead) {
    section->encode(ov::unit_test::intel_npu::create_generic_section_sink(stream));

    const std::string buffer = stream.str();
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::IO_LAYOUTS);
    manifest_entry.size = buffer.size();

    auto read_section = section->decode(ov::runtime::hsm::Section(
        manifest_entry,
        ov::util::MemoryView(reinterpret_cast<const std::byte*>(buffer.data()), buffer.size())));
    ASSERT_TRUE(read_section);
    auto layouts_result = std::dynamic_pointer_cast<IOLayoutsSection>(read_section.value());
    ASSERT_TRUE(layouts_result);
    EXPECT_EQ(layouts_result->get_input_layouts(), section->get_input_layouts());
    EXPECT_EQ(layouts_result->get_output_layouts(), section->get_output_layouts());
}

using IOLayoutsSectionRead = ::testing::Test;

TEST_F(IOLayoutsSectionRead, CtorHasTheRightAttributes) {
    const std::vector<ov::Layout> input_layouts{ov::Layout("NCHW"), ov::Layout("NHWC")};
    const std::vector<ov::Layout> output_layouts{ov::Layout("AWER"), ov::Layout("WC")};
    IOLayoutsSection section(input_layouts, output_layouts);

    ASSERT_EQ(section.get_type(), SectionType(SectionTypeCode::IO_LAYOUTS));
    ASSERT_EQ(section.get_input_layouts(), input_layouts);
    ASSERT_EQ(section.get_output_layouts(), output_layouts);
}

TEST_F(IOLayoutsSectionRead, EmptyCompatibilityReqsSubexpression) {
    IOLayoutsSection section({}, {});
    const std::vector<std::shared_ptr<CREToken>> requirements =
        section.get_compatibility_requirements_subexpression({});
    ASSERT_TRUE(requirements.empty());
}

TEST_F(IOLayoutsSectionRead, NullIndividualReqs) {
    IOLayoutsSection section({}, {});
    ASSERT_FALSE(section.get_individual_compatibility_requirements().has_value());
}

TEST_F(IOLayoutsSectionRead, TooSmallSectionLength) {
    std::vector<uint8_t> payload(7, 0);
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::IO_LAYOUTS);
    manifest_entry.size = payload.size();

    auto read_section = IOLayoutsSection::decode(ov::runtime::hsm::Section(
        manifest_entry,
        ov::util::MemoryView(reinterpret_cast<const std::byte*>(payload.data()), payload.size())));
    ASSERT_FALSE(read_section);
}

TEST_F(IOLayoutsSectionRead, WrongTagToDecoder) {
    std::vector<uint8_t> payload(8, 0);
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::BATCH_SIZE);
    manifest_entry.size = payload.size();

    OV_EXPECT_THROW(IOLayoutsSection::decode(ov::runtime::hsm::Section(
                        manifest_entry,
                        ov::util::MemoryView(reinterpret_cast<const std::byte*>(payload.data()), payload.size()))),
                    ov::Exception,
                    testing::_);
}

TEST_F(IOLayoutsSectionRead, InlineTagToDecoder) {
    std::vector<uint8_t> payload(8, 0);
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::IO_LAYOUTS, true);
    std::memcpy(&manifest_entry.inline_bytes, payload.data(), payload.size());

    OV_EXPECT_THROW(IOLayoutsSection::decode(ov::runtime::hsm::Section(manifest_entry)), ov::Exception, testing::_);
}

TEST_F(IOLayoutsSectionRead, LessLayoutsThanExpected) {
    IOLayoutsSection real_section({ov::Layout("NCHW"), ov::Layout("NHWC")}, {ov::Layout("NCHW"), ov::Layout("NHWC")});
    std::stringstream stream;
    real_section.encode(ov::unit_test::intel_npu::create_generic_section_sink(stream));
    std::string buffer = stream.str();

    // overwrite both numbers of layouts
    const uint32_t fake_count = 20;
    std::memcpy(buffer.data(), &fake_count, sizeof(fake_count));
    std::memcpy(buffer.data() + LAYOUTS_COUNT_SIZE, &fake_count, sizeof(fake_count));

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size());
    ASSERT_ANY_THROW(IOLayoutsSection::read(reader));
}

TEST_F(IOLayoutsSectionRead, MoreLayoutsThanExpected) {
    IOLayoutsSection real_section({ov::Layout("NCHW"), ov::Layout("NHWC")}, {ov::Layout("NCHW"), ov::Layout("NHWC")});
    std::stringstream stream;
    real_section.encode(ov::unit_test::intel_npu::create_generic_section_sink(stream));
    std::string buffer = stream.str();

    // overwrite both numbers of layouts
    const uint32_t fake_count = 1;
    std::memcpy(buffer.data(), &fake_count, sizeof(fake_count));
    std::memcpy(buffer.data() + LAYOUTS_COUNT_SIZE, &fake_count, sizeof(fake_count));

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size());
    ASSERT_ANY_THROW(IOLayoutsSection::read(reader));
}

TEST_F(IOLayoutsSectionRead, InvalidLayoutNoThrow) {
    IOLayoutsSection valid_section({ov::Layout("N")}, {});
    std::stringstream stream;
    valid_section.encode(ov::unit_test::intel_npu::create_generic_section_sink(stream));
    std::string buffer = stream.str();

    // overwrite input layout '[N]' with '[%]'
    const size_t layout_offset = 2 * LAYOUTS_COUNT_SIZE + sizeof(uint16_t);
    ASSERT_GE(buffer.size(), layout_offset + 3);
    buffer[layout_offset] = '[';
    buffer[layout_offset + 1] = '%';
    buffer[layout_offset + 2] = ']';

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size());

    std::shared_ptr<ISection> read_section;
    ASSERT_NO_THROW(read_section = IOLayoutsSection::read(reader));

    const auto layouts_result = std::dynamic_pointer_cast<IOLayoutsSection>(read_section);
    ASSERT_TRUE(layouts_result);
    ASSERT_EQ(layouts_result->get_input_layouts().size(), 1);
    EXPECT_EQ(layouts_result->get_input_layouts()[0], ov::Layout());
    EXPECT_TRUE(layouts_result->get_output_layouts().empty());
}

TEST_F(IOLayoutsSectionRead, AlteredLayoutSizeTooBig) {
    IOLayoutsSection real_section({ov::Layout("NCHW")}, {});
    std::stringstream stream;
    real_section.encode(ov::unit_test::intel_npu::create_generic_section_sink(stream));
    std::string buffer = stream.str();

    // overwrite both numbers of layouts
    const uint16_t fake_layout_size = 5;
    std::memcpy(buffer.data() + 2 * LAYOUTS_COUNT_SIZE, &fake_layout_size, sizeof(fake_layout_size));

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size());
    ASSERT_ANY_THROW(IOLayoutsSection::read(reader));
}

INSTANTIATE_TEST_SUITE_P(
    IOLayoutsSectionTest,
    ValidLayouts,
    ::testing::Values(
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW")}, std::vector<ov::Layout>{ov::Layout("NCHW")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NHWC")}, std::vector<ov::Layout>{ov::Layout("NHWC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NC")}, std::vector<ov::Layout>{ov::Layout("NC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NH")}, std::vector<ov::Layout>{ov::Layout("NH")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NHW")}, std::vector<ov::Layout>{ov::Layout("NHW")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NWH")}, std::vector<ov::Layout>{ov::Layout("NWH")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCDHW")}, std::vector<ov::Layout>{ov::Layout("NDHWC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW"), ov::Layout("NHWC")},
                        std::vector<ov::Layout>{ov::Layout("NCHW"), ov::Layout("NHWC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW"), ov::Layout("NHW")},
                        std::vector<ov::Layout>{ov::Layout("NC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW"),
                                                ov::Layout("NHWC"),
                                                ov::Layout("NHW"),
                                                ov::Layout("NWH"),
                                                ov::Layout("NC"),
                                                ov::Layout("NH"),
                                                ov::Layout("NCDHW")},
                        std::vector<ov::Layout>{ov::Layout("NCHW"),
                                                ov::Layout("NHWC"),
                                                ov::Layout("NHW"),
                                                ov::Layout("NWH"),
                                                ov::Layout("NC"),
                                                ov::Layout("NH"),
                                                ov::Layout("NDHWC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("?N...C?"),
                                                ov::Layout("[?, N, CHANNELS, ?, ?, Custom_dim_name]")},
                        std::vector<ov::Layout>{ov::Layout("...3?456"), ov::Layout::scalar()}),
        std::make_tuple(std::vector<ov::Layout>{}, std::vector<ov::Layout>{ov::Layout("NCHW")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW")}, std::vector<ov::Layout>{}),
        std::make_tuple(std::vector<ov::Layout>{}, std::vector<ov::Layout>{})));
