// Copyright (C) 2018-2026 Intel Corporation
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
 * @brief Identifiers used to distinguish between different section types at runtime and inside the manifest.
 * @note These values need to be unique across plugin versions as well. That is because this code is written inside the
 * manifest, and blob compatibility is involved. There, this list can be expanded, but the current entries can never be
 * modified.
 * @note The size needs to be fixed (2 bytes) since this value is written inside the blob manifest
 */
enum class SectionTypeCode : uint32_t {
    UNKNOWN = 100,
    RUNTIME_REQUIREMENTS = 3,  // Matches the tag ID used by the OV common code
    ELF_MAIN_SCHEDULE = 101,
    ELF_INIT_SCHEDULES = 102,
    DYNAMIC_SCHEDULE = 103,
    IO_LAYOUTS = 104,
    BATCH_SIZE = 105,
    ENCRYPTED_SCHEDULES_FLAG = 106,
    COMPILER_VERSION = 107,
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

    /**
     * @brief Serializes the section type
     * @warning The list of section type codes & strings can be expanded, but already existing entries should be never
     * modified. This is because these values are stored within the blob and act as unique identifiers. Modifying these
     * may break compatibility.
     */
    std::string to_string() const override;

    /**
     * @brief Parses the section type
     * @warning The list of section type codes & strings can be expanded, but already existing entries should be never
     * modified. This is because these values are stored within the blob and act as unique identifiers. Modifying these
     * may break compatibility.
     */
    static SectionType from_string(const std::string_view type);

private:
    SectionTypeCode m_code;
};

std::ostream& operator<<(std::ostream& os, const SectionType& type);

std::istream& operator>>(std::istream& is, SectionType& type);

bool is_section_type(const std::shared_ptr<CREToken>& token);

}  // namespace intel_npu

template <>
struct std::hash<intel_npu::SectionType> {
    std::size_t operator()(const intel_npu::SectionType& type) const noexcept {
        return std::hash<uint32_t>()(static_cast<uint32_t>(type.get_code()));
    }
};
