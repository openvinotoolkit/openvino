// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/isection.hpp"

using namespace intel_npu;
using testing::_;

using SectionsStringConversionsUnitTests = ::testing::Test;

TEST_F(SectionsStringConversionsUnitTests, TypeAndIDToStringAllValidTypes) {
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS), SectionID(0)),
              "RUNTIME_REQUIREMENTS_0");
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::MANIFEST), SectionID(0)), "MANIFEST_0");
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE), SectionID(0)),
              "ELF_MAIN_SCHEDULE_0");
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::ELF_INIT_SCHEDULES), SectionID(0)),
              "ELF_INIT_SCHEDULES_0");
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::DYNAMIC_SCHEDULE), SectionID(0)),
              "DYNAMIC_SCHEDULE_0");
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::IO_LAYOUTS), SectionID(0)), "IO_LAYOUTS_0");
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::BATCH_SIZE), SectionID(0)), "BATCH_SIZE_0");
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG), SectionID(0)),
              "ENCRYPTED_SCHEDULES_FLAG_0");
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::COMPILER_VERSION), SectionID(0)),
              "COMPILER_VERSION_0");
}

TEST_F(SectionsStringConversionsUnitTests, UnknownTypeToString) {
    OV_EXPECT_THROW(section_type_and_id_to_string(SectionType(SectionTypeCode::UNKNOWN), SectionID(0)),
                    ov::Exception,
                    _);
}

TEST_F(SectionsStringConversionsUnitTests, DifferentIDsToString) {
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::MANIFEST), SectionID(0)), "MANIFEST_0");
    ASSERT_EQ(section_type_and_id_to_string(SectionType(SectionTypeCode::MANIFEST),
                                            SectionID(std::numeric_limits<uint16_t>::max())),
              "MANIFEST_" + std::to_string(std::numeric_limits<uint16_t>::max()));
}

TEST_F(SectionsStringConversionsUnitTests, ValidTypesAndIDsFromString) {
    ASSERT_EQ(section_type_and_id_from_string("RUNTIME_REQUIREMENTS_0"),
              std::make_pair(SectionType(SectionTypeCode::RUNTIME_REQUIREMENTS), std::make_optional(SectionID(0))));
    ASSERT_EQ(section_type_and_id_from_string("MANIFEST_0"),
              std::make_pair(SectionType(SectionTypeCode::MANIFEST), std::make_optional(SectionID(0))));
    ASSERT_EQ(section_type_and_id_from_string("ELF_MAIN_SCHEDULE_0"),
              std::make_pair(SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE), std::make_optional(SectionID(0))));
    ASSERT_EQ(section_type_and_id_from_string("ELF_INIT_SCHEDULES_0"),
              std::make_pair(SectionType(SectionTypeCode::ELF_INIT_SCHEDULES), std::make_optional(SectionID(0))));
    ASSERT_EQ(section_type_and_id_from_string("DYNAMIC_SCHEDULE_0"),
              std::make_pair(SectionType(SectionTypeCode::DYNAMIC_SCHEDULE), std::make_optional(SectionID(0))));
    ASSERT_EQ(section_type_and_id_from_string("IO_LAYOUTS_0"),
              std::make_pair(SectionType(SectionTypeCode::IO_LAYOUTS), std::make_optional(SectionID(0))));
    ASSERT_EQ(section_type_and_id_from_string("BATCH_SIZE_0"),
              std::make_pair(SectionType(SectionTypeCode::BATCH_SIZE), std::make_optional(SectionID(0))));
    ASSERT_EQ(section_type_and_id_from_string("ENCRYPTED_SCHEDULES_FLAG_0"),
              std::make_pair(SectionType(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG), std::make_optional(SectionID(0))));
    ASSERT_EQ(section_type_and_id_from_string("COMPILER_VERSION_0"),
              std::make_pair(SectionType(SectionTypeCode::COMPILER_VERSION), std::make_optional(SectionID(0))));

    ASSERT_EQ(section_type_and_id_from_string("MANIFEST_" + std::to_string(std::numeric_limits<uint16_t>::max())),
              std::make_pair(SectionType(SectionTypeCode::MANIFEST),
                             std::make_optional(SectionID(std::numeric_limits<uint16_t>::max()))));

    ASSERT_EQ(section_type_and_id_from_string("manifest_0"),
              std::make_pair(SectionType(SectionTypeCode::MANIFEST), std::make_optional(SectionID(0))));
}

TEST_F(SectionsStringConversionsUnitTests, ValidTypeAndNoIDFromString) {
    const std::pair<SectionType, std::optional<SectionID>> result =
        std::make_pair(SectionType(SectionTypeCode::MANIFEST), std::nullopt);
    ASSERT_EQ(section_type_and_id_from_string("MANIFEST"), result);
}

TEST_F(SectionsStringConversionsUnitTests, InvalidTypesFromString) {
    const std::pair<SectionType, std::optional<SectionID>> result =
        std::make_pair(SectionType(SectionTypeCode::UNKNOWN), std::nullopt);
    ASSERT_EQ(section_type_and_id_from_string("MANIFEST_"), result);
    ASSERT_EQ(section_type_and_id_from_string("MANIFEST_a"), result);
    ASSERT_EQ(section_type_and_id_from_string("MANIFEST_-1"), result);
    ASSERT_EQ(section_type_and_id_from_string("MANIFEST_1.0"), result);
    ASSERT_EQ(section_type_and_id_from_string("MANIFES"), result);
    ASSERT_EQ(section_type_and_id_from_string("UNKNOWN"), result);
}

TEST_F(SectionsStringConversionsUnitTests, InvalidIDsFromString) {
    OV_EXPECT_THROW(section_type_and_id_from_string("MANIFEST_1000000000000000"), ov::Exception, _);
}

TEST_F(SectionsStringConversionsUnitTests, InvalidTypeValidIDFromString) {
    const std::pair<SectionType, std::optional<SectionID>> result =
        std::make_pair(SectionType(SectionTypeCode::UNKNOWN), std::make_optional(SectionID(0)));
    ASSERT_EQ(section_type_and_id_from_string("MANIFEST_1.0_0"), result);
}

TEST_F(SectionsStringConversionsUnitTests, ChainToStringFromString) {
    ASSERT_EQ(section_type_and_id_from_string(
                  section_type_and_id_to_string(SectionType(SectionTypeCode::MANIFEST), SectionID(0))),
              std::make_pair(SectionType(SectionTypeCode::MANIFEST), std::make_optional(SectionID(0))));
}
