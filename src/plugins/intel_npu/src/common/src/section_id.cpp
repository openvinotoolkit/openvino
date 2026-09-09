// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/section_id.hpp"

#include <functional>

#include "openvino/core/except.hpp"

namespace {

bool has_only_digits(std::string_view sv) {
    return !sv.empty() && std::all_of(sv.begin(), sv.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
};

}  // namespace

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

std::string section_id_to_string(const SectionID id) {
    return std::to_string(id.get_id());
}

SectionID section_id_from_string(std::string_view id) {
    OPENVINO_ASSERT(has_only_digits(id),
                    "Cannot convert to integer: the id ",
                    id,
                    " is not made exclusively out of digits");

    try {
        return std::stoul(id.data());
    } catch (const std::exception&) {
        OPENVINO_THROW("Failed to convert the section id ", id, " to integer");
    }
}

std::ostream& operator<<(std::ostream& os, const SectionID& id) {
    return os << section_id_to_string(id);
}

std::istream& operator>>(std::istream& is, SectionID& id) {
    std::string str;
    is >> str;
    id = section_id_from_string(str);
    return is;
}

bool is_section_id(const std::shared_ptr<CREToken>& token) {
    return std::dynamic_pointer_cast<SectionID>(token) != nullptr;
}

}  // namespace intel_npu
