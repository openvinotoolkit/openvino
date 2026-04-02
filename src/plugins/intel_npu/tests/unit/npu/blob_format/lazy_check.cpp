// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include "intel_npu/common/static_capability.hpp"
#include "mocks/mock_sections.hpp"

namespace {
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

constexpr double VALID_VALUE_1 = VALID_THRESHOLD - 1;
constexpr double VALID_VALUE_2 = VALID_THRESHOLD - 2;
constexpr double VALID_VALUE_3 = VALID_THRESHOLD - 3;
;

constexpr double INVALID_VALUE_1 = VALID_THRESHOLD + 1;
constexpr double INVALID_VALUE_2 = 0xDEADBEEF;
}  // namespace

TEST(MockSection1LazyCheck, ValidValue) {
    MockSection_1 section(VALID_VALUE_1);
    EXPECT_TRUE(section.lazy_check());
}

TEST(MockSection1LazyCheck, InvalidValue) {
    MockSection_1 section(VALID_THRESHOLD + 1);
    EXPECT_FALSE(section.lazy_check());
}

TEST(MockSection2LazyCheck, ValidValues) {
    MockSection_2 section({VALID_VALUE_1, VALID_VALUE_2, VALID_VALUE_3});
    EXPECT_TRUE(section.lazy_check());
}

TEST(MockSection2LazyCheck, EmptyVector) {
    MockSection_2 section({});
    EXPECT_TRUE(section.lazy_check());
}

TEST(MockSection2LazyCheck, InvalidValues) {
    MockSection_2 section({VALID_VALUE_1, INVALID_VALUE_1, INVALID_VALUE_2});
    EXPECT_FALSE(section.lazy_check());
}

TEST(MockSection3LazyCheck, ValidSubsections) {
    auto section = std::make_shared<MockSection_3>(
        std::make_shared<MockSection_1>(VALID_VALUE_1),
        std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_2, VALID_VALUE_3}));
    EXPECT_TRUE(section->lazy_check());
}

TEST(MockSection3LazyCheck, InvalidSection1) {
    auto section = std::make_shared<MockSection_3>(
        std::make_shared<MockSection_1>(INVALID_VALUE_1),
        std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_2, VALID_VALUE_3}));
    EXPECT_FALSE(section->lazy_check());
}

TEST(MockSection3LazyCheck, InvalidSection2) {
    auto section =
        std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(VALID_VALUE_1),
                                        std::make_shared<MockSection_2>(std::vector<double>{INVALID_VALUE_1}));
    EXPECT_FALSE(section->lazy_check());
}

TEST(MockSection3LazyCheck, InvalidSections) {
    auto section =
        std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(INVALID_VALUE_1),
                                        std::make_shared<MockSection_2>(std::vector<double>{INVALID_VALUE_1}));
    EXPECT_FALSE(section->lazy_check());
}

TEST(MockSectionWithTableLazyCheck, InvalidSection1FailsCheckBeforeProbe) {
    // section_1 is invalid, so the driver probe must not be called at all
    ::testing::MockFunction<bool(SectionType)> mock_driver;
    EXPECT_CALL(mock_driver, Call(::testing::_)).Times(0);

    MockSectionWithTable section(std::make_shared<MockSection_1>(INVALID_VALUE_1),
                                 std::vector<std::shared_ptr<ISection>>{});
    section.set_driver_probe(mock_driver.AsStdFunction());

    EXPECT_FALSE(section.lazy_check());
}

TEST(MockSectionWithTableLazyCheck, NoReachablesAndNullProbe) {
    MockSectionWithTable section(std::make_shared<MockSection_1>(VALID_VALUE_1),
                                 std::vector<std::shared_ptr<ISection>>{});
    EXPECT_TRUE(section.lazy_check());
}

TEST(MockSectionWithTableLazyCheck, NoReachablesWithProbe) {
    ::testing::MockFunction<bool(SectionType)> mock_driver;
    EXPECT_CALL(mock_driver, Call(::testing::_)).Times(0);

    MockSectionWithTable section(std::make_shared<MockSection_1>(VALID_VALUE_1),
                                 std::vector<std::shared_ptr<ISection>>{});
    section.set_driver_probe(mock_driver.AsStdFunction());

    EXPECT_TRUE(section.lazy_check());
}

TEST(MockSectionWithTableLazyCheck, ProbeCalledOncePerUniqueType) {
    // one instance of mock_3 and two of mock_2: so probe should be called exactly twice
    ::testing::MockFunction<bool(SectionType)> mock_driver;
    EXPECT_CALL(mock_driver, Call(MockTypes::MOCK_2)).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(mock_driver, Call(MockTypes::MOCK_3)).Times(1).WillOnce(::testing::Return(true));

    auto section_1 = std::make_shared<MockSection_1>(VALID_VALUE_1);
    auto section_2_0 = std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_2});
    auto section_2_1 = std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_3});
    auto section_3 = std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(VALID_VALUE_1),
                                                     std::make_shared<MockSection_2>(std::vector<double>{}));

    auto result = write_read_section_with_table(section_1, {section_2_0, section_2_1, section_3});
    ASSERT_TRUE(result);
    result->set_driver_probe(mock_driver.AsStdFunction());
    EXPECT_TRUE(result->lazy_check());
}

TEST(MockSectionWithTableLazyCheck, ProbeReturnsFalseForOneTypeFails) {
    ::testing::MockFunction<bool(SectionType)> mock_driver;
    EXPECT_CALL(mock_driver, Call(MockTypes::MOCK_2)).WillOnce(::testing::Return(false));
    EXPECT_CALL(mock_driver, Call(MockTypes::MOCK_3)).WillRepeatedly(::testing::Return(true));

    auto section_1 = std::make_shared<MockSection_1>(VALID_VALUE_1);
    auto section_2 = std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_2});
    auto section_3 = std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(VALID_VALUE_1),
                                                     std::make_shared<MockSection_2>(std::vector<double>{}));

    auto result = write_read_section_with_table(section_1, {section_2, section_3});
    ASSERT_TRUE(result);
    result->set_driver_probe(mock_driver.AsStdFunction());
    EXPECT_FALSE(result->lazy_check());
}

// should we add another mock sectio which has the driver interrogation as a must?
TEST(MockSectionWithTableLazyCheck, NullProbeDoesNotCauseFailure) {
    auto section_1 = std::make_shared<MockSection_1>(VALID_VALUE_1);
    auto section_2 = std::make_shared<MockSection_2>(std::vector<double>{VALID_VALUE_2});
    auto section_3 = std::make_shared<MockSection_3>(std::make_shared<MockSection_1>(VALID_VALUE_1),
                                                     std::make_shared<MockSection_2>(std::vector<double>{}));

    auto result = write_read_section_with_table(section_1, {section_2, section_3});
    ASSERT_TRUE(result);
    EXPECT_TRUE(result->lazy_check());
}
