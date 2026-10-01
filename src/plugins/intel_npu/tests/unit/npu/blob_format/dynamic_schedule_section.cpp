// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "compiler_schedules_sections.hpp"
#include "intel_npu/common/blob_reader_interface.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/section_type.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/util/codec_xor.hpp"
#include "utils.hpp"

using namespace intel_npu;

namespace {

constexpr std::string_view COMPILER_SCHEDULE_CONTENT = "dummy";
constexpr std::string_view ELF_BLOB_TYPE_SERIALIZED_VALUE = "\x00";
constexpr std::string_view LLVM_BLOB_TYPE_SERIALIZED_VALUE = "\x01";
constexpr std::string_view BYTECODE_BLOB_TYPE_SERIALIZED_VALUE = "\x02";
constexpr size_t FIRST_REGISTERED_SECTION_ID = 0;
constexpr size_t SECOND_REGISTERED_SECTION_ID = 1;

std::shared_ptr<DynamicScheduleSection> create_section_using_dummy_content() {
    return std::make_shared<DynamicScheduleSection>(

        ov::Tensor(ov::element::Type_t::u8, {COMPILER_SCHEDULE_CONTENT.size()}, COMPILER_SCHEDULE_CONTENT.data()),
        BlobType::BYTECODE);
}

}  // namespace

using testing::_;

using DynamicScheduleSectionTest = ::testing::Test;

TEST_F(DynamicScheduleSectionTest, NullGraphCtor) {
    OV_EXPECT_THROW(DynamicScheduleSection(nullptr), ov::Exception, _);
}

TEST_F(DynamicScheduleSectionTest, NullSetGraph) {
    std::shared_ptr<DynamicScheduleSection> section = create_section_using_dummy_content();
    OV_EXPECT_THROW(section->set_graph(nullptr), ov::Exception, _);
}

TEST_F(DynamicScheduleSectionTest, CtorSetsTheRightValue) {
    std::shared_ptr<DynamicScheduleSection> section = create_section_using_dummy_content();
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::DYNAMIC_SCHEDULE));
    ASSERT_EQ(section->get_blob_type(), BlobType::BYTECODE);
    ASSERT_EQ(section->get_schedule().get_element_type(), ov::element::Type_t::u8);
    ASSERT_EQ(section->get_schedule().get_shape(), ov::Shape({COMPILER_SCHEDULE_CONTENT.size()}));
    ASSERT_EQ(std::memcmp(static_cast<const ov::Tensor&>(section->get_schedule()).data(),
                          COMPILER_SCHEDULE_CONTENT.data(),
                          COMPILER_SCHEDULE_CONTENT.size()),
              0);

    // Test using an empty schedule too
    section =
        std::make_shared<DynamicScheduleSection>(ov::Tensor(ov::element::Type_t::u8, ov::Shape({0}), (void*)nullptr),
                                                 BlobType::LLVM);
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::DYNAMIC_SCHEDULE));
    ASSERT_EQ(section->get_blob_type(), BlobType::LLVM);
    ASSERT_EQ(section->get_schedule().get_element_type(), ov::element::Type_t::u8);
    ASSERT_EQ(section->get_schedule().get_shape(), ov::Shape({0}));
    ASSERT_EQ(section->get_schedule().get_byte_size(), 0);
}

TEST_F(DynamicScheduleSectionTest, CtorELFBlobTypeRejected) {
    OV_EXPECT_THROW(
        DynamicScheduleSection(ov::Tensor(ov::element::Type_t::u8, ov::Shape({0}), (void*)nullptr), BlobType::ELF),
        ov::Exception,
        _);
}

TEST_F(DynamicScheduleSectionTest, CompatibilityReqsSubexpression) {
    std::shared_ptr<DynamicScheduleSection> section1 = create_section_using_dummy_content();
    OV_EXPECT_THROW(section1->get_compatibility_requirements_subexpression({}), ov::Exception, _);

    BlobWriter writer;
    writer.register_section(section1);
    std::vector<std::shared_ptr<CREToken>> requirements = section1->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 2);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_TRUE(is_section_id(requirements.at(1)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::DYNAMIC_SCHEDULE));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionID>(requirements.at(1)).get(), FIRST_REGISTERED_SECTION_ID);

    std::shared_ptr<DynamicScheduleSection> section2 = create_section_using_dummy_content();
    writer.register_section(section2);
    requirements = section2->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 2);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_TRUE(is_section_id(requirements.at(1)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::DYNAMIC_SCHEDULE));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionID>(requirements.at(1)).get(), SECOND_REGISTERED_SECTION_ID);
}

TEST_F(DynamicScheduleSectionTest, InvalidStateWrite) {
    const std::shared_ptr<DynamicScheduleSection> section = create_section_using_dummy_content();
    std::ostringstream stream;
    BlobWriterInterface writer(stream, 0);
    // Invalid state. The attribute should have been a graph
    OV_EXPECT_THROW(section->write(writer), ov::Exception, _);
}

TEST_F(DynamicScheduleSectionTest, DecryptThrowsIfMissingCallback) {
    const std::shared_ptr<DynamicScheduleSection> section = create_section_using_dummy_content();
    OV_EXPECT_THROW(section->decrypt(nullptr), ov::Exception, _);
}

TEST_F(DynamicScheduleSectionTest, WorkingDecryption) {
    const std::shared_ptr<DynamicScheduleSection> section = create_section_using_dummy_content();
    section->decrypt(ov::util::codec_xor);

    const std::string decrypted_schedule =
        std::string(section->get_schedule().data<char>(), section->get_schedule().get_byte_size());
    const std::string reference = ov::util::codec_xor(std::string(COMPILER_SCHEDULE_CONTENT));

    // Take into account page alignment
    if (reference.size() % utils::STANDARD_PAGE_SIZE == 0) {
        ASSERT_EQ(decrypted_schedule, reference);
    } else {
        ASSERT_EQ(decrypted_schedule.size(), utils::align_size_to_standard_page_size(reference.size()));

        const std::string padding(decrypted_schedule.size() - reference.size(), 0);
        ASSERT_EQ(decrypted_schedule, reference + padding);
    }
}

TEST_F(DynamicScheduleSectionTest, ReadELFBlobType) {
    std::string section_content(ELF_BLOB_TYPE_SERIALIZED_VALUE);
    section_content += "\x00\x00";  // padding
    section_content += "0";         // dummy compiler schedule

    ov::Tensor tensor(ov::element::u8, ov::Shape{section_content.size()}, section_content.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size());
    OV_EXPECT_THROW(DynamicScheduleSection::read(reader), ov::Exception, _);

    std::istringstream stream(section_content);
    source = BlobSource(stream);
    reader = BlobReaderInterface(source, 0, section_content.size(), 0, section_content.size());
    OV_EXPECT_THROW(DynamicScheduleSection::read(reader), ov::Exception, _);
}

TEST_F(DynamicScheduleSectionTest, ReadingATooSmallSection) {
    std::string section_content = std::string(LLVM_BLOB_TYPE_SERIALIZED_VALUE);
    section_content += "0";  // dummy compiler schedule

    ov::Tensor tensor(ov::element::u8, ov::Shape{section_content.size()}, section_content.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size());
    OV_EXPECT_THROW(DynamicScheduleSection::read(reader), ov::Exception, _);

    std::istringstream stream(section_content);
    source = BlobSource(stream);
    reader = BlobReaderInterface(source, 0, section_content.size(), 0, section_content.size());
    OV_EXPECT_THROW(DynamicScheduleSection::read(reader), ov::Exception, _);
}

TEST_F(DynamicScheduleSectionTest, ReadPaddingTooBig) {
    std::string section_content = std::string(LLVM_BLOB_TYPE_SERIALIZED_VALUE);
    section_content += "\x00\x06";  // padding
    section_content += "dummy";     // dummy compiler schedule

    ov::Tensor tensor(ov::element::u8, ov::Shape{section_content.size()}, section_content.data());
    BlobSource source(tensor);
    BlobReaderInterface reader(source, 0, tensor.get_byte_size(), 0, tensor.get_byte_size());
    OV_EXPECT_THROW(DynamicScheduleSection::read(reader), ov::Exception, _);

    std::istringstream stream(section_content);
    source = BlobSource(stream);
    reader = BlobReaderInterface(source, 0, section_content.size(), 0, section_content.size());
    OV_EXPECT_THROW(DynamicScheduleSection::read(reader), ov::Exception, _);
}

class DynamicScheduleSectionReadTest : public testing::TestWithParam<std::tuple<bool, size_t, bool, bool>> {
public:
    DynamicScheduleSectionReadTest() : source(stream), reader(source, 0, 0, 0, 0) {}

    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<bool, size_t, bool, bool>>& obj) {
        bool llvm_blob_type, tensor_source, empty_schedule;
        size_t padding_size;
        std::tie(llvm_blob_type, padding_size, tensor_source, empty_schedule) = obj.param;

        return std::string(llvm_blob_type ? "llvm" : "bytecode") + "_" + std::to_string(padding_size) + "_" +
               (tensor_source ? "tensor" : "stream") + "_" + (empty_schedule ? "empty" : "not_empty");
    }

protected:
    void SetUp() override {
        uint16_t padding_size;
        bool empty_schedule;
        std::tie(is_llvm_blob_type, padding_size, is_tensor_source, empty_schedule) = GetParam();

        section_content = is_llvm_blob_type ? LLVM_BLOB_TYPE_SERIALIZED_VALUE : BYTECODE_BLOB_TYPE_SERIALIZED_VALUE;
        section_content += std::string(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
        section_content += std::string(padding_size, 0);
        after_padding_offset = section_content.size();
        if (!empty_schedule) {
            section_content += "dummy";
        }

        if (is_tensor_source) {
            tensor = ov::Tensor(ov::element::Type_t::u8, {section_content.size()}, section_content.data());
            source = BlobSource(tensor);
        } else {
            stream = std::istringstream(section_content);
            source = BlobSource(stream);
        }

        const auto options = std::make_shared<OptionsDesc>();
        FilteredConfig empty_config(options);
        reader = BlobReaderInterface(source, 0, section_content.size(), 0, section_content.size(), empty_config);
    }

    bool is_llvm_blob_type;
    bool is_tensor_source;
    std::string section_content;
    size_t after_padding_offset;
    ov::Tensor tensor;
    std::istringstream stream;
    BlobSource source;
    BlobReaderInterface reader;
};

TEST_P(DynamicScheduleSectionReadTest, SuccessfulRead) {
    auto read_section = DynamicScheduleSection::read(reader);
    auto casted_section = std::dynamic_pointer_cast<DynamicScheduleSection>(read_section);
    ASSERT_TRUE(casted_section);

    // Check that the cursor was left at the end of the section. This may be an indicator that the whole section has
    // been read.
    ASSERT_EQ(reader.get_offset_relative_to_current_section(), section_content.size());
    ASSERT_EQ(casted_section->get_blob_type(), is_llvm_blob_type ? BlobType::LLVM : BlobType::BYTECODE);

    const std::string parsed_content(casted_section->get_schedule().data<char>(),
                                     casted_section->get_schedule().get_byte_size());
    // The padding should be skipped by the parser
    ASSERT_EQ(parsed_content, std::string(section_content.begin() + after_padding_offset, section_content.end()));

    if (is_tensor_source) {
        // The parsed content should point towards the original buffer (past the padding region). This implies no copies
        // have been performed, and page aligment has been preserved.
        ASSERT_EQ(casted_section->get_schedule().data(), section_content.data() + after_padding_offset);
    } else {
        // Stream case: a page aligned buffer should have been allocated for the parsed content
        ASSERT_EQ(reinterpret_cast<size_t>(casted_section->get_schedule().data()) % utils::STANDARD_PAGE_SIZE, 0);
    }
}

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         DynamicScheduleSectionReadTest,
                         testing::Combine(testing::ValuesIn(std::vector<bool>{true, false}),
                                          testing::ValuesIn(std::vector<size_t>{0,
                                                                                1,
                                                                                utils::STANDARD_PAGE_SIZE - 1,
                                                                                utils::STANDARD_PAGE_SIZE,
                                                                                utils::STANDARD_PAGE_SIZE + 1}),
                                          testing::ValuesIn(std::vector<bool>{true, false}),
                                          testing::ValuesIn(std::vector<bool>{true, false})),
                         DynamicScheduleSectionReadTest::getTestCaseName);
