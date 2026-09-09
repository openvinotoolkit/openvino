// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>

#include "intel_npu/common/cre_token.hpp"

namespace intel_npu {

/**
 * @brief TODO
 * @note The size needs to be fixed (2 bytes) since this value is written inside the blob manifest
 */
enum class SectionTypeCode : uint16_t {
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

/**
 * @brief Identifies the type of the section, along with its corresponding read & write handlers.
 */
class SectionType final : public CREToken {
public:
    SectionType(const SectionTypeCode section_type_code);

    SectionTypeCode get_code() const;

    bool operator==(const SectionType& other) const;

    bool operator!=(const SectionType& other) const;

    bool operator<(const SectionType& other) const;

private:
    SectionTypeCode m_code;
};

std::ostream& operator<<(std::ostream& os, const SectionType& type);

std::istream& operator>>(std::istream& is, SectionType& type);

std::string section_type_to_string(const SectionType type);

SectionType section_type_from_string(std::string_view type);

bool is_section_type(const std::shared_ptr<CREToken>& token);

}  // namespace intel_npu

template <>
struct std::hash<intel_npu::SectionType> {
    std::size_t operator()(const intel_npu::SectionType& type) const noexcept {
        return std::hash<uint16_t>()(static_cast<uint16_t>(type.get_code()));
    }
};
