// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>

#include "intel_npu/common/cre_token.hpp"

namespace intel_npu {

/**
 * @brief TODO
 * @note The size needs to be fixed (2 bytes) since this value is written inside the blob manifest
 */
enum class ValidSectionTypeCode : uint16_t {
    UNKNOWN = 100,
    RUNTIME_REQUIREMENTS = 101,
    MANIFEST = 102,
    ELF_MAIN_SCHEDULE = 103,
    ELF_INIT_SCHEDULES = 104,
    DYNAMIC_SCHEDULE = 105,
    IO_LAYOUTS = 106,
    BATCH_SIZE = 107,
    ENCRYPTED_SCHEDULES_FLAG = 108,
    COMPILER_VERSION = 109,
};

// TODO do we need this?
static inline const std::unordered_set<SectionType> ALL_VALID_SECTION_TYPE_CODES{
    ValidSectionTypeCode::RUNTIME_REQUIREMENTS,
    ValidSectionTypeCode::MANIFEST,
    ValidSectionTypeCode::ELF_MAIN_SCHEDULE,
    ValidSectionTypeCode::ELF_INIT_SCHEDULES,
    ValidSectionTypeCode::DYNAMIC_SCHEDULE,
    ValidSectionTypeCode::IO_LAYOUTS,
    ValidSectionTypeCode::BATCH_SIZE,
    ValidSectionTypeCode::ENCRYPTED_SCHEDULES_FLAG,
    ValidSectionTypeCode::COMPILER_VERSION};

/**
 * @brief Identifies the type of the section, along with its corresponding read & write handlers.
 */
class SectionType final : public CREToken {
public:
    SectionType(const ValidSectionTypeCode section_type_code);

    ValidSectionTypeCode get_code() const;

    bool operator==(const SectionType& other) const;

private:
    ValidSectionTypeCode m_code;
};

std::ostream& operator<<(std::ostream& os, const SectionType& type);

std::istream& operator>>(std::istream& is, SectionType& type);

std::string section_type_to_string(const SectionType type);

SectionType section_type_from_string(std::string_view type);

}  // namespace intel_npu
