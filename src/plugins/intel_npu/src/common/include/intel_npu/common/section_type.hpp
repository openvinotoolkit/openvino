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
#include "openvino/runtime/hsm_format.hpp"

namespace intel_npu {

/**
 * @brief Identifiers used to distinguish between different section types at runtime and inside the manifest.
 * @note These values need to be unique across plugin versions as well. That is because this code is written inside the
 * manifest, and blob compatibility is involved. There, this list can be expanded, but the current entries can never be
 * modified.
 * @note The size needs to be fixed (4 bytes) since this value is written inside the blob manifest
 * @note Device specific sections should use a type ID `>= ov::runtime::hsm::core_tag_id_range_end`. The rest of the IDs
 * are reserved for OV common sections. The common sections can be interpreted by the OV common code (and all other
 * plugins). The device specific sections can only be interpreted by the corresponding plugin.
 */
enum class SectionTypeCode : uint32_t {
    UNKNOWN = ov::runtime::hsm::core_tag_id_range_end,
    RUNTIME_REQUIREMENTS = ov::runtime::hsm::core_tag_id_range_end + 1,
    ELF_MAIN_SCHEDULE = ov::runtime::hsm::core_tag_id_range_end + 2,
    ELF_INIT_SCHEDULES = ov::runtime::hsm::core_tag_id_range_end + 3,
    DYNAMIC_SCHEDULE = ov::runtime::hsm::core_tag_id_range_end + 4,
    IO_LAYOUTS = ov::runtime::hsm::core_tag_id_range_end + 5,
    BATCH_SIZE = ov::runtime::hsm::core_tag_id_range_end + 6,
    ENCRYPTED_SCHEDULES_FLAG = ov::runtime::hsm::core_tag_id_range_end + 7,
    COMPILER_VERSION = ov::runtime::hsm::core_tag_id_range_end + 8,
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
