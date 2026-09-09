// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iostream>

#include "intel_npu/common/cre_token.hpp"

namespace intel_npu {

/**
 * @brief Used to distinguish multiple sections of the same type within the same compiled model.
 */
class SectionID final : public CREToken {
public:
    SectionID(const uint16_t section_id);

    uint16_t get_id() const;

    bool operator==(const SectionID& other) const;

    bool operator!=(const SectionID& other) const;

private:
    /**
     * @note The size needs to be fixed (2 bytes) since this value is written inside the blob manifest
     */
    uint16_t m_id;
};

std::ostream& operator<<(std::ostream& os, const SectionID& id);

std::istream& operator>>(std::istream& is, SectionID& id);

std::string section_id_to_string(const SectionID id);

SectionID section_id_from_string(std::string_view id);

bool is_section_id(const std::shared_ptr<CREToken>& token);

}  // namespace intel_npu

template <>
struct std::hash<intel_npu::SectionID> {
    std::size_t operator()(const intel_npu::SectionID& id) const noexcept {
        return std::hash<uint16_t>()(static_cast<uint16_t>(id.get_id()));
    }
};
