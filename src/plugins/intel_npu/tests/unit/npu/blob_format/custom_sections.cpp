// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/common/static_capability.hpp"
#include "mocks/mock_sections.hpp"

using namespace intel_npu;

namespace {

std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> make_caps() {
    return {{CRE::CRE_EVALUATION, std::make_shared<StaticCapability>(CRE::CRE_EVALUATION)}};
}

void compare_aligned_elements(const std::string& buffer, const std::vector<double>& values) {
    for (double value : values) {
        bool found = false;
        for (size_t i = 0; i + sizeof(double) <= buffer.size(); i++) {
            if (std::memcmp(buffer.data() + i, &value, sizeof(double)) == 0) {
                EXPECT_EQ(i % MockSection_2::ALIGNMENT, 0)
                    << "Element " << value << " at buffer offset " << i << " is not 64-byte aligned";
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Element " << value << " was not found in the serialized blob";
    }
}

}  // namespace

constexpr double VALUE = 0xCAFEBABE;

const std::vector<double> VALUES = {0xDEADBEEF, 0xDEAFBEEF, 0xDEADC0DE, 0xDEADF00D, 0xDEADCAFE};

// should we also use CREs in these tests?

TEST(MockSection1, WriteRead) {
    BlobWriter writer;
    writer.register_section(std::make_shared<MockSection_1>(VALUE));
    std::stringstream stream;
    writer.write(stream);
    const std::string buffer = stream.str();

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobReader reader(tensor);
    reader.register_reader(MockTypes::MOCK_1, MockSection_1::read);
    reader.read(make_caps());

    auto result = std::dynamic_pointer_cast<MockSection_1>(reader.retrieve_first_section(MockTypes::MOCK_1));
    ASSERT_TRUE(result);
    EXPECT_DOUBLE_EQ(result->get_value(), VALUE);
}

TEST(MockSection2, WriteRead) {
    BlobWriter writer;
    writer.register_section(std::make_shared<MockSection_2>(VALUES));
    std::stringstream stream;
    writer.write(stream);
    const std::string buffer = stream.str();

    compare_aligned_elements(buffer, VALUES);

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobReader reader(tensor);
    reader.register_reader(MockTypes::MOCK_2, MockSection_2::read);
    reader.read(make_caps());

    auto result = std::dynamic_pointer_cast<MockSection_2>(reader.retrieve_first_section(MockTypes::MOCK_2));
    ASSERT_TRUE(result);
    EXPECT_EQ(result->get_values(), VALUES);
}

TEST(MockSection2, WriteReadEmpty) {
    const std::vector<double> empty_values = {};

    BlobWriter writer;
    writer.register_section(std::make_shared<MockSection_2>(empty_values));
    std::stringstream stream;
    writer.write(stream);
    const std::string buffer = stream.str();

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobReader reader(tensor);
    reader.register_reader(MockTypes::MOCK_2, MockSection_2::read);
    reader.read(make_caps());

    auto result = std::dynamic_pointer_cast<MockSection_2>(reader.retrieve_first_section(MockTypes::MOCK_2));
    ASSERT_TRUE(result);
    EXPECT_TRUE(result->get_values().empty());
}

TEST(MockSection3, WriteRead) {
    BlobWriter writer;
    writer.register_section(std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(VALUE),
                                                            std::make_shared<MockSection_2>(VALUES)));
    std::stringstream stream;
    writer.write(stream);
    const std::string buffer = stream.str();

    compare_aligned_elements(buffer, VALUES);

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobReader reader(tensor);
    reader.register_reader(MockTypes::MOCK_3, MockSection_3::read);
    reader.read(make_caps());

    auto result = std::dynamic_pointer_cast<MockSection_3>(reader.retrieve_first_section(MockTypes::MOCK_3));
    ASSERT_TRUE(result);

    compare_aligned_elements(buffer, VALUES);

    auto [value, values] = result->get_values();
    EXPECT_DOUBLE_EQ(value, VALUE);
    EXPECT_EQ(values, VALUES);
}

TEST(MockSections, GetROITensors) {
    BlobWriter writer;
    writer.register_section(std::make_shared<MockSection_1>(VALUE));
    writer.register_section(std::make_shared<MockSection_2>(VALUES));
    std::stringstream stream;
    writer.write(stream);
    const std::string buffer = stream.str();

    std::vector<uint8_t> data(buffer.begin(), buffer.end());
    ov::Tensor tensor(ov::element::u8, ov::Shape{data.size()}, data.data());
    BlobReader reader(tensor);
    reader.register_reader(MockTypes::MOCK_1, MockSection_1::read);
    reader.register_reader(MockTypes::MOCK_2, MockSection_2::read);
    reader.read(make_caps());

    auto offsets_section = std::dynamic_pointer_cast<OffsetsTableSection>(
        reader.retrieve_first_section(PredefinedSectionType::OFFSETS_TABLE));
    ASSERT_TRUE(offsets_section);
    auto table = offsets_section->get_table();

    auto offset_1 = table.lookup_offset(SectionID(MockTypes::MOCK_1, 0)).value();
    auto length_1 = table.lookup_length(SectionID(MockTypes::MOCK_1, 0)).value();
    auto offset_2 = table.lookup_offset(SectionID(MockTypes::MOCK_2, 0)).value();
    auto length_2 = table.lookup_length(SectionID(MockTypes::MOCK_2, 0)).value();
    ASSERT_TRUE(offset_1 && length_1 && offset_2 && length_2);

    reader.move_cursor_to_relative_position(offset_1);
    auto roi_1 = reader.get_roi_tensor(length_1);

    reader.move_cursor_to_relative_position(offset_2);
    auto roi_2 = reader.get_roi_tensor(length_2);

    EXPECT_EQ(roi_1.data<uint8_t>(), tensor.data<uint8_t>() + offset_1);
    EXPECT_EQ(roi_1.get_byte_size(), length_1);
    EXPECT_EQ(roi_2.data<uint8_t>(), tensor.data<uint8_t>() + offset_2);
    EXPECT_EQ(roi_2.get_byte_size(), length_2);
}

//  _______________________________________________
// │ header                                        │
// |_______________________________________________|
// │ MockSection_3                                 │
// |_______________________________________________|
// │ MockSectionWithTable                          │
// │   - embedded table (MOCK_3x1, MOCK_2x2)       │
// │   - reachable[0]: MockSection_3               │
// │   - reachable[1]: MockSection_2               │
// │   - reachable[2]: MockSection_2               │
// |_______________________________________________|
// │ MockSection_2                                 │
// |_______________________________________________|
TEST(MockSectionWithTable, WriteRead) {
    constexpr double REACHABLE_VALUE_A = 0x7327AC3D;
    const std::vector<double> VALUES_B = {0xC00FFEEE, 0xFEEDBEEF};
    const std::vector<double> VALUES_C = {0x0FFCFFEE, 0xBEEFCAFE, 0xFEEDF00D};
    const std::vector<double> VALUES_D = {0xF00DCAFE, 0xF00DFACE};

    auto section_1 = std::make_shared<MockSection_1>(VALUE);
    auto section_2 = std::make_shared<MockSection_2>(VALUES);
    auto reacheable_section_1 = std::make_shared<MockSection_1>(REACHABLE_VALUE_A);

    BlobWriter writer;
    writer.register_section(std::make_shared<MockSection_3>(section_1, section_2));
    writer.register_section(std::make_shared<MockSectionWithTable>(
        section_1,
        std::vector<std::shared_ptr<ISection>>{std::make_shared<MockSection_3>(reacheable_section_1, section_2),
                                               std::make_shared<MockSection_2>(VALUES_B),
                                               std::make_shared<MockSection_2>(VALUES_C)}));
    writer.register_section(std::make_shared<MockSection_2>(VALUES_D));
    std::stringstream stream;
    writer.write(stream);
    const std::string buffer = stream.str();

    compare_aligned_elements(buffer, VALUES);
    compare_aligned_elements(buffer, VALUES_B);
    compare_aligned_elements(buffer, VALUES_C);
    compare_aligned_elements(buffer, VALUES_D);

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, buffer.data());
    BlobReader reader(tensor);
    reader.register_reader(MockTypes::MOCK_2, MockSection_2::read);
    reader.register_reader(MockTypes::MOCK_3, MockSection_3::read);
    reader.register_reader(MockTypes::MOCK_WITH_TABLE, MockSectionWithTable::read);
    reader.read(make_caps());

    auto result =
        std::dynamic_pointer_cast<MockSectionWithTable>(reader.retrieve_first_section(MockTypes::MOCK_WITH_TABLE));
    ASSERT_TRUE(result);

    ASSERT_TRUE(result->get_section_1());
    EXPECT_DOUBLE_EQ(result->get_section_1()->get_value(), VALUE);

    // all three entries are present in the embedded table
    const auto& reachables = result->get_reachables();
    ASSERT_EQ(reachables.size(), 3);

    // first and only instance of MockSection_3
    auto instance_3 = reachables.find(SectionID(MockTypes::MOCK_3, 0));
    ASSERT_NE(instance_3, reachables.end());
    auto reachable_section_3 = std::dynamic_pointer_cast<MockSection_3>(instance_3->second);
    ASSERT_TRUE(reachable_section_3);
    auto [val_a, vec_a] = reachable_section_3->get_values();
    EXPECT_DOUBLE_EQ(val_a, REACHABLE_VALUE_A);
    EXPECT_EQ(vec_a, VALUES);

    // first instance of MockSection_2
    auto instance_2_0 = reachables.find(SectionID(MockTypes::MOCK_2, 0));
    ASSERT_NE(instance_2_0, reachables.end());
    auto reachable_section_2_0 = std::dynamic_pointer_cast<MockSection_2>(instance_2_0->second);
    ASSERT_TRUE(reachable_section_2_0);
    EXPECT_EQ(reachable_section_2_0->get_values(), VALUES_B);

    // second instance of MockSection_2
    auto instance_2_1 = reachables.find(SectionID(MockTypes::MOCK_2, 1));
    ASSERT_NE(instance_2_1, reachables.end());
    auto reachable_section_2_1 = std::dynamic_pointer_cast<MockSection_2>(instance_2_1->second);
    ASSERT_TRUE(reachable_section_2_1);
    EXPECT_EQ(reachable_section_2_1->get_values(), VALUES_C);

    // MockSection_3 in the main blob (before custom table)
    auto section_3 = std::dynamic_pointer_cast<MockSection_3>(reader.retrieve_first_section(MockTypes::MOCK_3));
    ASSERT_TRUE(section_3);
    auto [value, values] = section_3->get_values();
    EXPECT_DOUBLE_EQ(value, VALUE);
    EXPECT_EQ(values, VALUES);

    // MockSection_2 in the main blob (after custom table)
    auto main_section_2 = std::dynamic_pointer_cast<MockSection_2>(reader.retrieve_first_section(MockTypes::MOCK_2));
    ASSERT_TRUE(main_section_2);
    EXPECT_EQ(main_section_2->get_values(), VALUES_D);
}
