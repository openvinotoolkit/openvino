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

constexpr std::string_view COMPILER_SCHEDULE_CONTENT = "dummy";
constexpr size_t FIRST_REGISTERED_SECTION_ID = 0;
constexpr size_t SECOND_REGISTERED_SECTION_ID = 1;

std::shared_ptr<DynamicScheduleSection> create_section_using_dummy_content() {
    return std::make_shared<DynamicScheduleSection>(
        ov::Tensor(ov::element::Type_t::u8, {COMPILER_SCHEDULE_CONTENT.size()}, COMPILER_SCHEDULE_CONTENT.data()));
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
    ASSERT_EQ(section->get_schedule().get_element_type(), ov::element::Type_t::u8);
    ASSERT_EQ(section->get_schedule().get_shape(), ov::Shape({COMPILER_SCHEDULE_CONTENT.size()}));
    ASSERT_EQ(std::memcmp(static_cast<const ov::Tensor&>(section->get_schedule()).data(),
                          COMPILER_SCHEDULE_CONTENT.data(),
                          COMPILER_SCHEDULE_CONTENT.size()),
              0);

    // Test using an empty schedule too
    // TODO LLVM was replaced. make sure there is no test redundancy
    section =
        std::make_shared<DynamicScheduleSection>(ov::Tensor(ov::element::Type_t::u8, ov::Shape({0}), (void*)nullptr));
    ASSERT_EQ(section->get_type(), SectionType(SectionTypeCode::DYNAMIC_SCHEDULE));
    ASSERT_EQ(section->get_schedule().get_element_type(), ov::element::Type_t::u8);
    ASSERT_EQ(section->get_schedule().get_shape(), ov::Shape({0}));
    ASSERT_EQ(section->get_schedule().get_byte_size(), 0);
}

TEST_F(DynamicScheduleSectionTest, CompatibilityReqsSubexpression) {
    std::shared_ptr<DynamicScheduleSection> section1 = create_section_using_dummy_content();
    OV_EXPECT_THROW(section1->get_compatibility_requirements_subexpression({}), ov::Exception, _);

    BlobSectionsRegistry registry;
    registry.register_section(section1);
    std::vector<std::shared_ptr<CREToken>> requirements = section1->get_compatibility_requirements_subexpression({});
    ASSERT_EQ(requirements.size(), 2);
    ASSERT_TRUE(is_section_type(requirements.at(0)));
    ASSERT_TRUE(is_section_id(requirements.at(1)));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionType>(requirements.at(0)).get(),
              SectionType(SectionTypeCode::DYNAMIC_SCHEDULE));
    ASSERT_EQ(*std::dynamic_pointer_cast<SectionID>(requirements.at(1)).get(), FIRST_REGISTERED_SECTION_ID);

    std::shared_ptr<DynamicScheduleSection> section2 = create_section_using_dummy_content();
    registry.register_section(section2);
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
    // Invalid state. The attribute should have been a graph
    // TODO consider adding error messages to asserts
    OV_EXPECT_THROW(section->encode(ov::unit_test::intel_npu::create_generic_section_sink(stream)), ov::Exception, _);
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

TEST_F(DynamicScheduleSectionTest, WrongTagToDecoder) {
    std::vector<uint8_t> payload(1, 0);
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::BATCH_SIZE);
    manifest_entry.size = payload.size();

    OV_EXPECT_THROW(DynamicScheduleSection::decode(ov::runtime::hsm::Section(
                        manifest_entry,
                        ov::util::MemoryView(reinterpret_cast<const std::byte*>(payload.data()), payload.size()))),
                    ov::Exception,
                    testing::_);
}

TEST_F(DynamicScheduleSectionTest, InlineTagToDecoder) {
    std::vector<uint8_t> payload(1, 0);
    ov::runtime::hsm::ManifestEntry manifest_entry =
        ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::DYNAMIC_SCHEDULE, true);
    std::memcpy(&manifest_entry.inline_bytes, payload.data(), payload.size());

    OV_EXPECT_THROW(DynamicScheduleSection::decode(ov::runtime::hsm::Section(manifest_entry)),
                    ov::Exception,
                    testing::_);
}

class DynamicScheduleSectionReadTest : public testing::TestWithParam<std::tuple<bool, bool>> {
public:
    DynamicScheduleSectionReadTest() : section({}) {}

    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<bool, bool>>& obj) {
        bool tensor_source, empty_schedule;
        std::tie(tensor_source, empty_schedule) = obj.param;

        return std::string(tensor_source ? "tensor" : "stream") + "_" + (empty_schedule ? "empty" : "not_empty");
    }

protected:
    void SetUp() override {
        bool empty_schedule;
        std::tie(is_tensor_source, empty_schedule) = GetParam();
        section_content = empty_schedule ? "" : "dummy";

        ov::runtime::hsm::ManifestEntry manifest_entry =
            ov::unit_test::intel_npu::create_manifest_entry_of_type(SectionTypeCode::DYNAMIC_SCHEDULE);
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
    std::string section_content;
    std::istringstream stream;
    ov::runtime::hsm::Section section;
};

// TODO consider uniting with ELFMainSchedule tests
TEST_P(DynamicScheduleSectionReadTest, SuccessfulRead) {
    auto read_section = DynamicScheduleSection::decode(section);
    ASSERT_TRUE(read_section);
    auto casted_section = std::dynamic_pointer_cast<DynamicScheduleSection>(read_section.value());
    ASSERT_TRUE(casted_section);

    const ov::Tensor tensor = casted_section->get_schedule();
    const std::string parsed_content(tensor.data<char>(), tensor.get_byte_size());
    ASSERT_EQ(parsed_content, section_content);

    if (is_tensor_source) {
        if (!section_content.empty()) {
            // The parsed content should point towards the original buffer (past the padding region). This implies no
            // copies have been performed, and page aligment has been preserved.
            ASSERT_EQ(tensor.data(), section_content.data());
        }
    } else {
        // Stream case: a page aligned buffer should have been allocated for the parsed content
        ASSERT_EQ(reinterpret_cast<size_t>(tensor.data()) % utils::STANDARD_PAGE_SIZE, 0);
    }
}

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         DynamicScheduleSectionReadTest,
                         testing::Combine(testing::ValuesIn(std::vector<bool>{true, false}),
                                          testing::ValuesIn(std::vector<bool>{true, false})),
                         DynamicScheduleSectionReadTest::getTestCaseName);
