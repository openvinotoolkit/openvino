// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedules_sections.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/blob_reader_interface.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/section_type.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/util/codec_xor.hpp"
#include "utils.hpp"

using namespace intel_npu;

namespace {

constexpr std::string_view COMPILER_SCHEDULE_CONTENT = "dummy";
constexpr size_t FIRST_REGISTERED_SECTION_ID = 0;
constexpr size_t SECOND_REGISTERED_SECTION_ID = 1;

std::shared_ptr<ELFMainScheduleSection> create_section_using_dummy_content() {
    return std::make_shared<ELFMainScheduleSection>(
        ov::Tensor(ov::element::Type_t::u8, {COMPILER_SCHEDULE_CONTENT.size()}, COMPILER_SCHEDULE_CONTENT.data()));
}

}  // namespace

using testing::_;

using ELFMainScheduleSectionTest = ::testing::Test;

TEST_F(ELFMainScheduleSectionTest, NullGraphCtor) {
    OV_EXPECT_THROW(ELFMainScheduleSection(nullptr), ov::Exception, _);
}

TEST_F(ELFMainScheduleSectionTest, NullSetGraph) {
    std::shared_ptr<ELFMainScheduleSection> section = create_section_using_dummy_content();
    OV_EXPECT_THROW(section->set_graph(nullptr), ov::Exception, _);
}

TEST_F(ELFMainScheduleSectionTest, CtorSetsTheRightValue) {
    std::shared_ptr<ELFMainScheduleSection> section = create_section_using_dummy_content();
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE));
    ASSERT_EQ(section->get_schedule().get_element_type(), ov::element::Type_t::u8);
    ASSERT_EQ(section->get_schedule().get_shape(), ov::Shape({COMPILER_SCHEDULE_CONTENT.size()}));
    ASSERT_EQ(std::memcmp(static_cast<const ov::Tensor&>(section->get_schedule()).data(),
                          COMPILER_SCHEDULE_CONTENT.data(),
                          COMPILER_SCHEDULE_CONTENT.size()),
              0);
}

TEST_F(ELFMainScheduleSectionTest, CompatibilityReqsSubexpression) {
    std::shared_ptr<ELFMainScheduleSection> section1 = create_section_using_dummy_content();
    OV_EXPECT_THROW(section1->get_compatibility_requirements_subexpression({}), ov::Exception, _);

    BlobWriter writer;
    writer.register_section(section1);
    std::vector<std::shared_ptr<CREToken>> requirements = section1->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 2);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_TRUE(is_section_id(requirements.at(1)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionID>(requirements.at(1)).get(), FIRST_REGISTERED_SECTION_ID);

    std::shared_ptr<ELFMainScheduleSection> section2 = create_section_using_dummy_content();
    writer.register_section(section2);
    requirements = section2->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 2);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_TRUE(is_section_id(requirements.at(1)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionID>(requirements.at(1)).get(), SECOND_REGISTERED_SECTION_ID);
}

TEST_F(ELFMainScheduleSectionTest, InvalidStateWrite) {
    const std::shared_ptr<ELFMainScheduleSection> section = create_section_using_dummy_content();
    std::ostringstream stream;
    BlobWriterInterface writer(stream, 0);
    // Invalid state. The attribute should have been a graph
    OV_EXPECT_THROW(section->write(writer), ov::Exception, _);
}

TEST_F(ELFMainScheduleSectionTest, DecryptThrowsIfMissingCallback) {
    const std::shared_ptr<ELFMainScheduleSection> section = create_section_using_dummy_content();
    OV_EXPECT_THROW(section->decrypt(nullptr), ov::Exception, _);
}

TEST_F(ELFMainScheduleSectionTest, WorkingDecryption) {
    const std::shared_ptr<ELFMainScheduleSection> section = create_section_using_dummy_content();
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

// TEST_F(ELFMainScheduleSectionTest, Read) {
//     ov::Tensor tensor(ov::element::u8, ov::Shape{sizeof(batch_size)}, &batch_size);
//     BlobSource source(tensor);
//     BlobReaderInterface reader(source, 0, sizeof(batch_size), 0, sizeof(batch_size));

//     auto read_section = BatchSizeSection::read(reader);
//     auto casted_section = std::dynamic_pointer_cast<BatchSizeSection>(read_section);
//     ASSERT_TRUE(casted_section);
//     EXPECT_EQ(casted_section->get_batch_size(), batch_size);
// }
