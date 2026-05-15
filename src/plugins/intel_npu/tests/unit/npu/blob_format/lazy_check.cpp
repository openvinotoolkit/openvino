// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include "intel_npu/common/static_capability.hpp"
#include "mocks/mock_capabilities.hpp"

namespace {
constexpr double VALID_VALUE_1 = VALID_THRESHOLD - 1;
constexpr double VALID_VALUE_2 = VALID_THRESHOLD - 2;
constexpr double VALID_VALUE_3 = VALID_THRESHOLD - 3;
constexpr double INVALID_VALUE_1 = VALID_THRESHOLD + 1;
constexpr double INVALID_VALUE_2 = 0xDEADBEEF;

std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> make_caps() {
    return {{CRE::CRE_EVALUATION, std::make_shared<StaticCapability>(CRE::CRE_EVALUATION)}};
}

std::shared_ptr<MockSectionWithTable> write_read_section_with_table(std::shared_ptr<MockSection_1> section_1,
                                                                    std::vector<std::shared_ptr<ISection>> reachables) {
    BlobWriter writer;
    writer.register_section(std::make_shared<MockSectionWithTable>(std::move(section_1), std::move(reachables)));
    std::stringstream stream;
    writer.write(stream);
    const std::string buffer = stream.str();

    ov::Tensor tensor(ov::element::u8, ov::Shape{buffer.size()}, const_cast<char*>(buffer.data()));
    BlobReader reader(tensor);
    reader.register_reader(MockTypes::MOCK_2, MockSection_2::read);
    reader.register_reader(MockTypes::MOCK_3, MockSection_3::read);
    reader.register_reader(MockTypes::MOCK_WITH_TABLE, MockSectionWithTable::read);
    reader.read(make_caps());

    return std::dynamic_pointer_cast<MockSectionWithTable>(reader.retrieve_first_section(MockTypes::MOCK_WITH_TABLE));
}

}  // namespace

TEST(MockCapability_1, ValidValue) {
    MockCapability_1 cap(std::make_shared<MockSection_1>(VALID_VALUE_1));
    EXPECT_TRUE(cap.check_support());
}

TEST(MockCapability_1, InvalidValue) {
    MockCapability_1 cap(std::make_shared<MockSection_1>(INVALID_VALUE_1));
    EXPECT_FALSE(cap.check_support());
}

TEST(MockCapability_2, ValidValues) {
    MockCapability_2 cap(
        std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_1, VALID_VALUE_2, VALID_VALUE_3}));
    EXPECT_TRUE(cap.check_support());
}

TEST(MockCapability_2, EmptyVector) {
    MockCapability_2 cap(std::make_shared<MockSection_2>(std::vector<double>{}));
    EXPECT_TRUE(cap.check_support());
}

TEST(MockCapability_2, InvalidValues) {
    MockCapability_2 cap(
        std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_1, INVALID_VALUE_1, INVALID_VALUE_2}));
    EXPECT_FALSE(cap.check_support());
}

TEST(MockCapability_3, BothSectionsValid) {
    auto section_3 = std::make_shared<MockSection_3>(
        std::make_shared<MockSection_1>(VALID_VALUE_1),
        std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_2, VALID_VALUE_3}));
    MockCapability_3 cap(section_3);
    EXPECT_TRUE(cap.check_support());
}

TEST(MockCapability_3, Section1Invalid) {
    auto section_3 =
        std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(INVALID_VALUE_1),
                                        std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_2}));
    MockCapability_3 cap(section_3);
    EXPECT_FALSE(cap.check_support());
}

TEST(MockCapability_3, Section2Invalid) {
    auto section_3 =
        std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(VALID_VALUE_1),
                                        std::make_shared<MockSection_2>(std::vector<double>{INVALID_VALUE_1}));
    MockCapability_3 cap(section_3);
    EXPECT_FALSE(cap.check_support());
}

TEST(MockCapability_3, BothSectionsInvalid) {
    auto section_3 =
        std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(INVALID_VALUE_1),
                                        std::make_shared<MockSection_2>(std::vector<double>{INVALID_VALUE_1}));
    MockCapability_3 cap(section_3);
    EXPECT_FALSE(cap.check_support());
}

TEST(CapabilityMemoization, CalledTwiceChecksSupportOnce) {
    auto cap = std::make_shared<MockCapability>(MockTypes::MOCK_1);
    EXPECT_CALL(*cap, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_TRUE(cap->check_support());
    // second call should return m_supported without calling lazy_check_support() again
    EXPECT_TRUE(cap->check_support());
}

TEST(CapabilityMemoization, StoresNegativeResult) {
    auto cap = std::make_shared<MockCapability>(MockTypes::MOCK_1);
    EXPECT_CALL(*cap, lazy_check_support()).Times(1).WillOnce(::testing::Return(false));
    EXPECT_FALSE(cap->check_support());
    EXPECT_FALSE(cap->check_support());
}

TEST(EvaluatorLazyCheck, ValidSections) {
    auto section_1 = std::make_shared<MockSection_1>(VALID_VALUE_1);
    auto section_2 = std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_2});
    auto section_3 = std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(VALID_VALUE_1),
                                                     std::make_shared<MockSection_2>(std::vector<double>{}));

    auto result = write_read_section_with_table(section_1, {section_2, section_3});
    ASSERT_TRUE(result);

    const auto& parsed = result->get_reachables();
    auto parsed_s1 = result->get_section_1();
    auto parsed_s2 = std::dynamic_pointer_cast<MockSection_2>(parsed.at(SectionID(MockTypes::MOCK_2, 0)));
    auto parsed_s3 = std::dynamic_pointer_cast<MockSection_3>(parsed.at(SectionID(MockTypes::MOCK_3, 0)));

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[MockTypes::MOCK_1] = std::make_shared<MockCapability_1>(parsed_s1);
    caps[MockTypes::MOCK_2] = std::make_shared<MockCapability_2>(parsed_s2);
    caps[MockTypes::MOCK_3] = std::make_shared<MockCapability_3>(parsed_s3);

    CRE cre;
    cre.append_to_expression(MockTypes::MOCK_1);
    cre.append_to_expression(MockTypes::MOCK_2);
    cre.append_to_expression(MockTypes::MOCK_3);

    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST(EvaluatorLazyCheck, InvalidSection2) {
    auto section_1 = std::make_shared<MockSection_1>(VALID_VALUE_1);
    auto section_2 = std::make_shared<MockSection_2>(std::vector<double>{INVALID_VALUE_1});
    auto section_3 = std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(VALID_VALUE_1),
                                                     std::make_shared<MockSection_2>(std::vector<double>{}));

    auto result = write_read_section_with_table(section_1, {section_2, section_3});
    ASSERT_TRUE(result);

    const auto& parsed = result->get_reachables();
    auto parsed_s1 = result->get_section_1();
    auto parsed_s2 = std::dynamic_pointer_cast<MockSection_2>(parsed.at(SectionID(MockTypes::MOCK_2, 0)));
    auto parsed_s3 = std::dynamic_pointer_cast<MockSection_3>(parsed.at(SectionID(MockTypes::MOCK_3, 0)));

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[MockTypes::MOCK_1] = std::make_shared<MockCapability_1>(parsed_s1);
    caps[MockTypes::MOCK_2] = std::make_shared<MockCapability_2>(parsed_s2);
    caps[MockTypes::MOCK_3] = std::make_shared<MockCapability_3>(parsed_s3);

    CRE cre;
    cre.append_to_expression(MockTypes::MOCK_1);
    cre.append_to_expression(MockTypes::MOCK_2);
    cre.append_to_expression(MockTypes::MOCK_3);

    EXPECT_FALSE(cre.check_compatibility(caps));
}

TEST(EvaluatorLazyCheck, MemoizationPreventsRedundantQuerying) {
    auto cap = std::make_shared<MockCapability>(MockTypes::MOCK_1);
    EXPECT_CALL(*cap, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));

    CRE cre;
    cre.append_to_expression(MockTypes::MOCK_1);

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[MockTypes::MOCK_1] = cap;

    EXPECT_TRUE(cre.check_compatibility(caps));
    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST(EvaluatorLazyCheck, CapabilityAbsentFromSupportedMap) {
    CRE cre;
    cre.append_to_expression(MockTypes::MOCK_2);

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    EXPECT_FALSE(cre.check_compatibility(caps));
}

TEST(EvaluatorLazyCheck, SupportedCapabilityMissingFromCREIsNotQueried) {
    auto cap_1 = std::make_shared<MockCapability>(MockTypes::MOCK_1);
    EXPECT_CALL(*cap_1, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    auto cap_2 = std::make_shared<MockCapability>(MockTypes::MOCK_2);
    EXPECT_CALL(*cap_2, lazy_check_support()).Times(0);

    CRE cre;
    cre.append_to_expression(MockTypes::MOCK_1);

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[MockTypes::MOCK_1] = cap_1;
    caps[MockTypes::MOCK_2] = cap_2;

    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST(DriverCapabilityCheck, DriverQueriedOncePerCapability) {
    MockDriver driver;
    EXPECT_CALL(driver, supports_section(MockTypes::MOCK_2)).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(driver, supports_section(MockTypes::MOCK_3)).Times(1).WillOnce(::testing::Return(true));

    CRE cre;
    cre.append_to_expression(MockTypes::MOCK_2);
    cre.append_to_expression(MockTypes::MOCK_3);

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[MockTypes::MOCK_2] = std::make_shared<DriverCapability>(MockTypes::MOCK_2, driver);
    caps[MockTypes::MOCK_3] = std::make_shared<DriverCapability>(MockTypes::MOCK_3, driver);

    EXPECT_TRUE(cre.check_compatibility(caps));
    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST(DriverCapabilityCheck, DriverRejectionPropagates) {
    MockDriver driver;
    EXPECT_CALL(driver, supports_section(MockTypes::MOCK_2)).WillOnce(::testing::Return(false));
    EXPECT_CALL(driver, supports_section(MockTypes::MOCK_3)).WillRepeatedly(::testing::Return(true));

    CRE cre;
    cre.append_to_expression(MockTypes::MOCK_2);
    cre.append_to_expression(MockTypes::MOCK_3);

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[MockTypes::MOCK_2] = std::make_shared<DriverCapability>(MockTypes::MOCK_2, driver);
    caps[MockTypes::MOCK_3] = std::make_shared<DriverCapability>(MockTypes::MOCK_3, driver);

    EXPECT_FALSE(cre.check_compatibility(caps));
}

TEST(DriverCapabilityCheck, SupportedCapabilityMissingFromCREDriverIsNotQueried) {
    MockDriver driver;
    EXPECT_CALL(driver, supports_section(MockTypes::MOCK_2)).Times(0);
    EXPECT_CALL(driver, supports_section(MockTypes::MOCK_3)).Times(1).WillOnce(::testing::Return(true));

    CRE cre;
    cre.append_to_expression(MockTypes::MOCK_3);

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[MockTypes::MOCK_2] = std::make_shared<DriverCapability>(MockTypes::MOCK_2, driver);
    caps[MockTypes::MOCK_3] = std::make_shared<DriverCapability>(MockTypes::MOCK_3, driver);

    EXPECT_TRUE(cre.check_compatibility(caps));
}
