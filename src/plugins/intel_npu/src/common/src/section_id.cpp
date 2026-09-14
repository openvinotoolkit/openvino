// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/section_id.hpp"

#include <functional>

#include "intel_npu/utils/utils.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

SectionID::SectionID(const uint16_t section_id) : CREToken(), m_id(section_id) {}

// TODO consider renaming
uint16_t SectionID::get_id() const {
    return m_id;
}

bool SectionID::operator==(const SectionID& other) const {
    return m_id == other.get_id();
}

bool SectionID::operator!=(const SectionID& other) const {
    return !(*this == other);
}

bool SectionID::operator<(const SectionID& other) const {
    return m_id < other.get_id();
}

std::string SectionID::to_string() const {
    return std::to_string(m_id);
}

SectionID SectionID::from_string(const std::string_view id) {
    OPENVINO_ASSERT(utils::has_only_digits(id),
                    "Cannot convert to integer: the id ",
                    id,
                    " is not made exclusively out of digits");

    try {
        return std::stoul(std::string(id));
    } catch (const std::exception&) {
        OPENVINO_THROW("Failed to convert the section id ", id, " to integer");
    }
}

std::ostream& operator<<(std::ostream& os, const SectionID& id) {
    return os << id.to_string();
}

std::istream& operator>>(std::istream& is, SectionID& id) {
    std::string str;
    is >> str;
    id = SectionID::from_string(str);
    return is;
}

bool is_section_id(const std::shared_ptr<CREToken>& token) {
    return std::dynamic_pointer_cast<SectionID>(token) != nullptr;
}

}  // namespace intel_npu
