// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "compiler_schedules_sections.hpp"
#include "intel_npu/common/blob_sections_registry.hpp"
#include "intel_npu/common/section_type.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/util/codec_xor.hpp"
#include "utils.hpp"

using namespace intel_npu;

namespace {

constexpr std::array<std::string_view, 2> COMPILER_SCHEDULES_CONTENT{"dummy1", "dumy2"};
constexpr uint16_t NO_INIT_SCHEDULE_SERIALIZED_VALUE = 0;
constexpr uint16_t TWO_INIT_SCHEDULES_SERIALIZED_VALUE = 2;
constexpr size_t FIRST_REGISTERED_SECTION_ID = 0;
constexpr size_t SECOND_REGISTERED_SECTION_ID = 1;

std::shared_ptr<ELFInitSchedulesSection> create_section_using_dummy_content() {
    return std::make_shared<ELFInitSchedulesSection>(
        std::vector<ov::Tensor>{ov::Tensor(ov::element::Type_t::u8,
                                           {COMPILER_SCHEDULES_CONTENT.at(0).size()},
                                           COMPILER_SCHEDULES_CONTENT.at(0).data()),
                                ov::Tensor(ov::element::Type_t::u8,
                                           {COMPILER_SCHEDULES_CONTENT.at(1).size()},
                                           COMPILER_SCHEDULES_CONTENT.at(1).data())});
}

}  // namespace

using testing::_;

using ELFInitSchedulesSectionTest = ::testing::Test;

TEST_F(ELFInitSchedulesSectionTest, NullGraphCtor) {
    OV_EXPECT_THROW(ELFInitSchedulesSection(nullptr), ov::Exception, _);
}

TEST_F(ELFInitSchedulesSectionTest, NullSetGraph) {
    std::shared_ptr<ELFInitSchedulesSection> section = create_section_using_dummy_content();
    OV_EXPECT_THROW(section->set_graph(nullptr), ov::Exception, _);
}

TEST_F(ELFInitSchedulesSectionTest, CtorSetsTheRightValue) {
    std::shared_ptr<ELFInitSchedulesSection> section = create_section_using_dummy_content();
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::ELF_INIT_SCHEDULES));
    ASSERT_EQ(section->get_schedules().size(), 2);
    ASSERT_EQ(section->get_schedules().at(0).get_element_type(), ov::element::Type_t::u8);
    ASSERT_EQ(section->get_schedules().at(0).get_shape(), ov::Shape({COMPILER_SCHEDULES_CONTENT.at(0).size()}));
    ASSERT_EQ(std::memcmp(static_cast<const ov::Tensor&>(section->get_schedules().at(0)).data(),
                          COMPILER_SCHEDULES_CONTENT.at(0).data(),
                          COMPILER_SCHEDULES_CONTENT.at(0).size()),
              0);
    ASSERT_EQ(section->get_schedules().at(1).get_element_type(), ov::element::Type_t::u8);
    ASSERT_EQ(section->get_schedules().at(1).get_shape(), ov::Shape({COMPILER_SCHEDULES_CONTENT.at(1).size()}));
    ASSERT_EQ(std::memcmp(static_cast<const ov::Tensor&>(section->get_schedules().at(1)).data(),
                          COMPILER_SCHEDULES_CONTENT.at(1).data(),
                          COMPILER_SCHEDULES_CONTENT.at(1).size()),
              0);

    // Empty vector
    section = std::make_shared<ELFInitSchedulesSection>(std::vector<ov::Tensor>{});
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::ELF_INIT_SCHEDULES));
    ASSERT_EQ(section->get_schedules().size(), 0);

    // Test using an empty schedule too
    section = std::make_shared<ELFInitSchedulesSection>(
        std::vector<ov::Tensor>{ov::Tensor(ov::element::Type_t::u8, ov::Shape({0}), (void*)nullptr)});
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::ELF_INIT_SCHEDULES));
    ASSERT_EQ(section->get_schedules().size(), 1);
    ASSERT_EQ(section->get_schedules().at(0).get_element_type(), ov::element::Type_t::u8);
    ASSERT_EQ(section->get_schedules().at(0).get_shape(), ov::Shape({0}));
    ASSERT_EQ(section->get_schedules().at(0).get_byte_size(), 0);
}

TEST_F(ELFInitSchedulesSectionTest, CompatibilityReqsSubexpression) {
    std::shared_ptr<ELFInitSchedulesSection> section1 = create_section_using_dummy_content();
    OV_EXPECT_THROW(section1->get_compatibility_requirements_subexpression({}), ov::Exception, _);

    BlobSectionsRegistry registry;
    registry.register_section(section1);
    std::vector<std::shared_ptr<CREToken>> requirements = section1->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 2);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_TRUE(is_section_id(requirements.at(1)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::ELF_INIT_SCHEDULES));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionID>(requirements.at(1)).get(), FIRST_REGISTERED_SECTION_ID);

    std::shared_ptr<ELFInitSchedulesSection> section2 = create_section_using_dummy_content();
    registry.register_section(section2);
    requirements = section2->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 2);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_TRUE(is_section_id(requirements.at(1)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::ELF_INIT_SCHEDULES));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionID>(requirements.at(1)).get(), SECOND_REGISTERED_SECTION_ID);
}

TEST_F(ELFInitSchedulesSectionTest, InvalidStateWrite) {
    const std::shared_ptr<ELFInitSchedulesSection> section = create_section_using_dummy_content();
    std::ostringstream stream;
    // Invalid state. The attribute should have been a graph
    OV_EXPECT_THROW(section->encode(ov::unit_test::intel_npu::create_generic_section_sink(stream)), ov::Exception, _);
}

TEST_F(ELFInitSchedulesSectionTest, DecryptThrowsIfMissingCallback) {
    const std::shared_ptr<ELFInitSchedulesSection> section = create_section_using_dummy_content();
    OV_EXPECT_THROW(section->decrypt(nullptr), ov::Exception, _);
}

TEST_F(ELFInitSchedulesSectionTest, WorkingDecryption) {
    // Empty section scenario first
    std::shared_ptr<ELFInitSchedulesSection> section =
        std::make_shared<ELFInitSchedulesSection>(std::vector<ov::Tensor>{});
    OV_ASSERT_NO_THROW(section->decrypt(ov::util::codec_xor));

    section = create_section_using_dummy_content();
    OV_ASSERT_NO_THROW(section->decrypt(ov::util::codec_xor));

    for (size_t schedule_index = 0; schedule_index < section->get_schedules().size(); ++schedule_index) {
        const ov::Tensor schedule = section->get_schedules().at(schedule_index);
        const std::string decrypted_schedule = std::string(schedule.data<char>(), schedule.get_byte_size());
        const std::string reference = ov::util::codec_xor(std::string(COMPILER_SCHEDULES_CONTENT.at(schedule_index)));

        // Take into account page alignment
        if (reference.size() % utils::STANDARD_PAGE_SIZE == 0) {
            ASSERT_EQ(decrypted_schedule, reference);
        } else {
            ASSERT_EQ(decrypted_schedule.size(), utils::align_size_to_standard_page_size(reference.size()));

            const std::string padding(decrypted_schedule.size() - reference.size(), 0);
            ASSERT_EQ(decrypted_schedule, reference + padding);
        }
    }
}

TEST_F(ELFInitSchedulesSectionTest, ReadingATooSmallSection) {
    const std::string section_content = "0";
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    auto read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(
        manifest_entry,
        ov::util::MemoryView(reinterpret_cast<const std::byte*>(section_content.data()), section_content.size())));
    ASSERT_FALSE(read_section);

    std::istringstream stream(section_content);
    manifest_entry = ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(manifest_entry, stream, 0));
    ASSERT_FALSE(read_section);
}

TEST_F(ELFInitSchedulesSectionTest, ReadNumberOfInitsTooBig) {
    std::string section_content("dummy");                                   // the content of init schedule 1
    section_content += std::string("\x05\x00\x00\x00\x00\x00\x00\x00", 8);  // the size of init schedule 1
    section_content += std::string("\x02\x00", 2);                          // number of inits

    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    auto read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(
        manifest_entry,
        ov::util::MemoryView(reinterpret_cast<const std::byte*>(section_content.data()), section_content.size())));
    ASSERT_FALSE(read_section);

    std::istringstream stream(section_content);
    manifest_entry = ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(manifest_entry, stream, 0));
    ASSERT_FALSE(read_section);
}

TEST_F(ELFInitSchedulesSectionTest, ReadNumberOfInitsTooSmall) {
    std::string section_content("dummy");                                   // the content of init schedule 1
    section_content += std::string("\x05\x00\x00\x00\x00\x00\x00\x00", 8);  // the size of init schedule 1
    section_content += std::string("\x00\x00", 2);                          // number of inits

    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    auto read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(
        manifest_entry,
        ov::util::MemoryView(reinterpret_cast<const std::byte*>(section_content.data()), section_content.size())));
    ASSERT_FALSE(read_section);

    std::istringstream stream(section_content);
    manifest_entry = ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(manifest_entry, stream, 0));
    ASSERT_FALSE(read_section);
}

// TODO test that checks page alignment using the OV HSM writer/reader
// TODO compare the coverage with the old branch
TEST_F(ELFInitSchedulesSectionTest, ReadInitSizeTooBig) {
    std::string section_content("dummy");                                   // the content of init schedule 1
    section_content += std::string("\x06\x00\x00\x00\x00\x00\x00\x00", 8);  // the size of init schedule 1
    section_content += std::string("\x01\x00", 2);                          // number of inits

    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    auto read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(
        manifest_entry,
        ov::util::MemoryView(reinterpret_cast<const std::byte*>(section_content.data()), section_content.size())));
    ASSERT_FALSE(read_section);

    std::istringstream stream(section_content);
    manifest_entry = ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(manifest_entry, stream, 0));
    ASSERT_FALSE(read_section);
}

TEST_F(ELFInitSchedulesSectionTest, ReadIntegerOverflow) {
    std::string section_content("dummy");                                   // the content of init schedule 1
    section_content += "dummy";                                             // the content of init schedule 2
    section_content += std::string("\x05\x00\x00\x00\x00\x00\x00\x00", 8);  // the size of init schedule 1
    section_content += std::string("\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF", 8);  // the size of init schedule 2
    section_content += std::string("\x02\x00", 2);                          // number of inits

    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    auto read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(
        manifest_entry,
        ov::util::MemoryView(reinterpret_cast<const std::byte*>(section_content.data()), section_content.size())));
    ASSERT_FALSE(read_section);

    std::istringstream stream(section_content);
    manifest_entry = ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(manifest_entry, stream, 0));
    ASSERT_FALSE(read_section);
}

TEST_F(ELFInitSchedulesSectionTest, ReadEmptySchedule) {
    std::shared_ptr<ELFInitSchedulesSection> casted_section;

    std::string section_content("\x00\x00\x00\x00\x00\x00\x00\x00", 8);  // the size of init schedule 1
    section_content += std::string("\x01\x00", 2);                       // number of inits

    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    auto read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(
        manifest_entry,
        ov::util::MemoryView(reinterpret_cast<const std::byte*>(section_content.data()), section_content.size())));
    ASSERT_TRUE(read_section);
    casted_section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(read_section.value());
    ASSERT_TRUE(casted_section);
    ASSERT_EQ(casted_section->get_schedules().size(), 1);
    ASSERT_EQ(casted_section->get_schedules().at(0).get_byte_size(), 0);

    std::istringstream stream(section_content);
    manifest_entry = ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
    manifest_entry.size = section_content.size();

    read_section = ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(manifest_entry, stream, 0));
    ASSERT_TRUE(read_section);
    casted_section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(read_section.value());
    ASSERT_TRUE(casted_section);
    ASSERT_EQ(casted_section->get_schedules().size(), 1);
    ASSERT_EQ(casted_section->get_schedules().at(0).get_byte_size(), 0);
}

TEST_F(ELFInitSchedulesSectionTest, WrongTagToDecoder) {
    std::vector<uint8_t> payload(2, 0);
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::BATCH_SIZE);
    manifest_entry.size = payload.size();

    OV_EXPECT_THROW(ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(
                        manifest_entry,
                        ov::util::MemoryView(reinterpret_cast<const std::byte*>(payload.data()), payload.size()))),
                    ov::Exception,
                    testing::_);
}

TEST_F(ELFInitSchedulesSectionTest, InlineTagToDecoder) {
    std::vector<uint8_t> payload(2, 0);
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES, true);
    std::memcpy(&manifest_entry.inline_bytes, payload.data(), payload.size());

    OV_EXPECT_THROW(ELFInitSchedulesSection::decode(ov::runtime::hsm::Section(manifest_entry)),
                    ov::Exception,
                    testing::_);
}

class ELFInitSchedulesSectionReadTest : public testing::TestWithParam<std::tuple<bool, bool>> {
public:
    ELFInitSchedulesSectionReadTest() : section({}) {}

    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<bool, bool>>& obj) {
        bool tensor_source;
        bool empty_schedule;
        std::tie(tensor_source, empty_schedule) = obj.param;

        return std::string(tensor_source ? "tensor" : "stream") + "_" + (empty_schedule ? "empty" : "not_empty");
    }

protected:
    void SetUp() override {
        std::tie(is_tensor_source, is_empty) = GetParam();

        if (is_empty) {
            section_content = std::string(reinterpret_cast<const char*>(&NO_INIT_SCHEDULE_SERIALIZED_VALUE),
                                          sizeof(NO_INIT_SCHEDULE_SERIALIZED_VALUE));
        } else {
            section_content = COMPILER_SCHEDULES_CONTENT.at(0);
            after_schedule1_offset = section_content.size();
            section_content += COMPILER_SCHEDULES_CONTENT.at(1);
            after_schedule2_offset = section_content.size();

            uint64_t size_of_init_schedule_1 = COMPILER_SCHEDULES_CONTENT.at(0).size();
            uint64_t size_of_init_schedule_2 = COMPILER_SCHEDULES_CONTENT.at(1).size();
            section_content +=
                std::string(reinterpret_cast<char*>(&size_of_init_schedule_1), sizeof(size_of_init_schedule_1));
            section_content +=
                std::string(reinterpret_cast<char*>(&size_of_init_schedule_2), sizeof(size_of_init_schedule_2));

            section_content += std::string(reinterpret_cast<const char*>(&TWO_INIT_SCHEDULES_SERIALIZED_VALUE),
                                           sizeof(TWO_INIT_SCHEDULES_SERIALIZED_VALUE));
        }

        ov::runtime::hsm::ManifestEntry manifest_entry =
            ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
        manifest_entry.size = section_content.size();

        if (is_tensor_source) {
            section = ov::runtime::hsm::Section(
                manifest_entry,
                ov::util::MemoryView(reinterpret_cast<std::byte*>(section_content.data()), section_content.size()));
        } else {
            stream = std::istringstream(section_content);
            section = ov::runtime::hsm::Section(manifest_entry, stream, 0);
        }
    }

    bool is_tensor_source;
    bool is_empty;
    std::string section_content;
    size_t after_schedule1_offset;
    size_t after_schedule2_offset;
    std::istringstream stream;
    ov::runtime::hsm::Section section;
};

TEST_P(ELFInitSchedulesSectionReadTest, SuccessfulRead) {
    auto read_section = ELFInitSchedulesSection::decode(section);
    ASSERT_TRUE(read_section);
    auto casted_section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(read_section.value());
    ASSERT_TRUE(casted_section);

    if (is_empty) {
        ASSERT_EQ(casted_section->get_schedules().size(), 0);
    } else {
        const ov::Tensor tensor1 = casted_section->get_schedules().at(0);
        const ov::Tensor tensor2 = casted_section->get_schedules().at(1);

        const std::string parsed_schedule1(tensor1.data<char>(), tensor1.get_byte_size());
        const std::string parsed_schedule2(tensor2.data<char>(), tensor2.get_byte_size());
        ASSERT_EQ(parsed_schedule1,
                  std::string(section_content.begin(), section_content.begin() + after_schedule1_offset));
        ASSERT_EQ(parsed_schedule2,
                  std::string(section_content.begin() + after_schedule1_offset,
                              section_content.begin() + after_schedule2_offset));

        if (is_tensor_source) {
            // The parsed content should point towards the original buffer. This implies no
            // copies have been performed, and page aligment has been preserved.
            ASSERT_EQ(tensor1.data(), section_content.data());
            ASSERT_EQ(tensor2.data(), section_content.data() + after_schedule1_offset);
        } else {
            // Stream case: a page aligned buffer should have been allocated for the parsed content
            ASSERT_EQ(reinterpret_cast<size_t>(tensor1.data()) % utils::STANDARD_PAGE_SIZE, 0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         ELFInitSchedulesSectionReadTest,
                         testing::Combine(testing::ValuesIn(std::vector<bool>{true, false}),
                                          testing::ValuesIn(std::vector<bool>{true, false})),
                         ELFInitSchedulesSectionReadTest::getTestCaseName);
