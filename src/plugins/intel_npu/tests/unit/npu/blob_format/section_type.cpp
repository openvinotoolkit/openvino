// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/section_type.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/cre.hpp"
#include "intel_npu/common/section_id.hpp"

namespace {

constexpr std::string_view RUNTIME_REQUIREMENTS_SECTION_NAME = "RUNTIME_REQUIREMENTS";
constexpr std::string_view MANIFEST_SECTION_NAME = "MANIFEST";
constexpr std::string_view ELF_MAIN_SCHEDULE_SECTION_NAME = "ELF_MAIN_SCHEDULE";
constexpr std::string_view ELF_INIT_SCHEDULES_SECTION_NAME = "ELF_INIT_SCHEDULES";
constexpr std::string_view DYNAMIC_SCHEDULE_SECTION_NAME = "DYNAMIC_SCHEDULE";
constexpr std::string_view IO_LAYOUTS_SECTION_NAME = "IO_LAYOUTS";
constexpr std::string_view BATCH_SIZE_SECTION_NAME = "BATCH_SIZE";
constexpr std::string_view ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME = "ENCRYPTED_SCHEDULES_FLAG";
constexpr std::string_view COMPILER_VERSION_SECTION_NAME = "COMPILER_VERSION";

constexpr std::string_view RUNTIME_REQUIREMENTS_SECTION_NAME_LOWER = "runtime_requirements";
constexpr std::string_view MANIFEST_SECTION_NAME_LOWER = "manifest";
constexpr std::string_view ELF_MAIN_SCHEDULE_SECTION_NAME_LOWER = "elf_main_schedule";
constexpr std::string_view ELF_INIT_SCHEDULES_SECTION_NAME_LOWER = "elf_init_schedules";
constexpr std::string_view DYNAMIC_SCHEDULE_SECTION_NAME_LOWER = "dynamic_schedule";
constexpr std::string_view IO_LAYOUTS_SECTION_NAME_LOWER = "io_layouts";
constexpr std::string_view BATCH_SIZE_SECTION_NAME_LOWER = "batch_size";
constexpr std::string_view ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME_LOWER = "encrypted_schedules_flag";
constexpr std::string_view COMPILER_VERSION_SECTION_NAME_LOWER = "compiler_version";

}  // namespace

using namespace intel_npu;

using testing::_;
using SectionTypeTest = ::testing::Test;

TEST_F(SectionTypeTest, CreateAllSectionTypeUsingCorrectCode) {
    ASSERT_EQ(SectionType(SectionTypeCode::UNKNOWN).get_code(), SectionTypeCode::UNKNOWN);
    ASSERT_EQ(SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS).get_code(), SectionTypeCode::RUNTIME_REQUIREMENTS);
    ASSERT_EQ(SectionType(SectionTypeCode::MANIFEST).get_code(), SectionTypeCode::MANIFEST);
    ASSERT_EQ(SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE).get_code(), SectionTypeCode::ELF_MAIN_SCHEDULE);
    ASSERT_EQ(SectionType(SectionTypeCode::ELF_INIT_SCHEDULES).get_code(), SectionTypeCode::ELF_INIT_SCHEDULES);
    ASSERT_EQ(SectionType(SectionTypeCode::DYNAMIC_SCHEDULE).get_code(), SectionTypeCode::DYNAMIC_SCHEDULE);
    ASSERT_EQ(SectionType(SectionTypeCode::IO_LAYOUTS).get_code(), SectionTypeCode::IO_LAYOUTS);
    ASSERT_EQ(SectionType(SectionTypeCode::BATCH_SIZE).get_code(), SectionTypeCode::BATCH_SIZE);
    ASSERT_EQ(SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG).get_code(),
              SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG);
    ASSERT_EQ(SectionType(SectionTypeCode::COMPILER_VERSION).get_code(), SectionTypeCode::COMPILER_VERSION);
}

TEST_F(SectionTypeTest, EqualOperator) {
    ASSERT_TRUE(SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS) ==
                SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS));
    ASSERT_FALSE(SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS) == SectionType(SectionTypeCode::MANIFEST));
}

TEST_F(SectionTypeTest, DifferentOperator) {
    ASSERT_TRUE(SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS) != SectionType(SectionTypeCode::MANIFEST));
    ASSERT_FALSE(SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS) !=
                 SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS));
}

TEST_F(SectionTypeTest, LowerOperator) {
    ASSERT_TRUE(SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS) < SectionType(SectionTypeCode::MANIFEST));
    ASSERT_FALSE(SectionType(SectionTypeCode::MANIFEST) < SectionType(SectionTypeCode::MANIFEST));
    ASSERT_FALSE(SectionType(SectionTypeCode::MANIFEST) < SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS));
}

TEST_F(SectionTypeTest, ToStringAllKnownTypes) {
    ASSERT_EQ(SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS).to_string(), RUNTIME_REQUIREMENTS_SECTION_NAME);
    ASSERT_EQ(SectionType(SectionTypeCode::MANIFEST).to_string(), MANIFEST_SECTION_NAME);
    ASSERT_EQ(SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE).to_string(), ELF_MAIN_SCHEDULE_SECTION_NAME);
    ASSERT_EQ(SectionType(SectionTypeCode::ELF_INIT_SCHEDULES).to_string(), ELF_INIT_SCHEDULES_SECTION_NAME);
    ASSERT_EQ(SectionType(SectionTypeCode::DYNAMIC_SCHEDULE).to_string(), DYNAMIC_SCHEDULE_SECTION_NAME);
    ASSERT_EQ(SectionType(SectionTypeCode::IO_LAYOUTS).to_string(), IO_LAYOUTS_SECTION_NAME);
    ASSERT_EQ(SectionType(SectionTypeCode::BATCH_SIZE).to_string(), BATCH_SIZE_SECTION_NAME);
    ASSERT_EQ(SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG).to_string(),
              ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME);
    ASSERT_EQ(SectionType(SectionTypeCode::COMPILER_VERSION).to_string(), COMPILER_VERSION_SECTION_NAME);
}

TEST_F(SectionTypeTest, ToStringUnKnownType) {
    OV_EXPECT_THROW(SectionType(SectionTypeCode::UNKNOWN).to_string(), ov::Exception, _);
}

TEST_F(SectionTypeTest, ValidFromStringUpperCased) {
    ASSERT_EQ(SectionType::from_string(RUNTIME_REQUIREMENTS_SECTION_NAME),
              SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS));
    ASSERT_EQ(SectionType::from_string(MANIFEST_SECTION_NAME), SectionType(SectionTypeCode::MANIFEST));
    ASSERT_EQ(SectionType::from_string(ELF_MAIN_SCHEDULE_SECTION_NAME),
              SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE));
    ASSERT_EQ(SectionType::from_string(ELF_INIT_SCHEDULES_SECTION_NAME),
              SectionType(SectionTypeCode::ELF_INIT_SCHEDULES));
    ASSERT_EQ(SectionType::from_string(DYNAMIC_SCHEDULE_SECTION_NAME), SectionType(SectionTypeCode::DYNAMIC_SCHEDULE));
    ASSERT_EQ(SectionType::from_string(IO_LAYOUTS_SECTION_NAME), SectionType(SectionTypeCode::IO_LAYOUTS));
    ASSERT_EQ(SectionType::from_string(BATCH_SIZE_SECTION_NAME), SectionType(SectionTypeCode::BATCH_SIZE));
    ASSERT_EQ(SectionType::from_string(ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME),
              SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG));
    ASSERT_EQ(SectionType::from_string(COMPILER_VERSION_SECTION_NAME), SectionType(SectionTypeCode::COMPILER_VERSION));
}

TEST_F(SectionTypeTest, ValidFromStringLowerCased) {
    ASSERT_EQ(SectionType::from_string(RUNTIME_REQUIREMENTS_SECTION_NAME_LOWER),
              SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS));
    ASSERT_EQ(SectionType::from_string(MANIFEST_SECTION_NAME_LOWER), SectionType(SectionTypeCode::MANIFEST));
    ASSERT_EQ(SectionType::from_string(ELF_MAIN_SCHEDULE_SECTION_NAME_LOWER),
              SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE));
    ASSERT_EQ(SectionType::from_string(ELF_INIT_SCHEDULES_SECTION_NAME_LOWER),
              SectionType(SectionTypeCode::ELF_INIT_SCHEDULES));
    ASSERT_EQ(SectionType::from_string(DYNAMIC_SCHEDULE_SECTION_NAME_LOWER),
              SectionType(SectionTypeCode::DYNAMIC_SCHEDULE));
    ASSERT_EQ(SectionType::from_string(IO_LAYOUTS_SECTION_NAME_LOWER), SectionType(SectionTypeCode::IO_LAYOUTS));
    ASSERT_EQ(SectionType::from_string(BATCH_SIZE_SECTION_NAME_LOWER), SectionType(SectionTypeCode::BATCH_SIZE));
    ASSERT_EQ(SectionType::from_string(ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME_LOWER),
              SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG));
    ASSERT_EQ(SectionType::from_string(COMPILER_VERSION_SECTION_NAME_LOWER),
              SectionType(SectionTypeCode::COMPILER_VERSION));
}

TEST_F(SectionTypeTest, FromStringOnUnknownTypes) {
    ASSERT_EQ(SectionType::from_string("UNKNOWN"), SectionType(SectionTypeCode::UNKNOWN));
    ASSERT_EQ(SectionType::from_string("random_name"), SectionType(SectionTypeCode::UNKNOWN));
    ASSERT_EQ(SectionType::from_string(std::string(BATCH_SIZE_SECTION_NAME) + "_1"),
              SectionType(SectionTypeCode::UNKNOWN));
}

TEST_F(SectionTypeTest, FromInvalidString) {
    OV_EXPECT_THROW(SectionType::from_string(""), ov::Exception, _);
    OV_EXPECT_THROW(SectionType::from_string("0"), ov::Exception, _);
    OV_EXPECT_THROW(SectionType::from_string("102978468432"), ov::Exception, _);
}

TEST_F(SectionTypeTest, IsSectionType) {
    const auto section_type = std::make_shared<SectionType>(SectionTypeCode::MANIFEST);
    const auto section_id = std::make_shared<SectionID>(0);
    const auto cre_special_token = std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND);
    ASSERT_TRUE(is_section_type(section_type));
    ASSERT_FALSE(is_section_type(section_id));
    ASSERT_FALSE(is_section_type(cre_special_token));
}
