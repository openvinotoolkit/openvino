// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/section_type.hpp"

#include "openvino/core/except.hpp"

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

}  // namespace

namespace intel_npu {

SectionType::SectionType(const SectionTypeCode section_type_code) : CREToken(), m_code(section_type_code) {}

SectionTypeCode SectionType::get_code() const {
    return m_code;
}

bool SectionType::operator==(const SectionType& other) const {
    return m_code == other.get_code();
}

// TODO test these
std::string section_type_to_string(const SectionType type) {
    switch (type.get_code()) {
    case SectionTypeCode::RUNTIME_REQUIREMENTS:
        return RUNTIME_REQUIREMENTS_SECTION_NAME.data();
    case SectionTypeCode::MANIFEST:
        return MANIFEST_SECTION_NAME.data();
    case SectionTypeCode::ELF_MAIN_SCHEDULE:
        return ELF_MAIN_SCHEDULE_SECTION_NAME.data();
    case SectionTypeCode::ELF_INIT_SCHEDULES:
        return ELF_INIT_SCHEDULES_SECTION_NAME.data();
    case SectionTypeCode::DYNAMIC_SCHEDULE:
        return DYNAMIC_SCHEDULE_SECTION_NAME.data();
    case SectionTypeCode::IO_LAYOUTS:
        return IO_LAYOUTS_SECTION_NAME.data();
    case SectionTypeCode::BATCH_SIZE:
        return BATCH_SIZE_SECTION_NAME.data();
    case SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG:
        return ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME.data();
    case SectionTypeCode::COMPILER_VERSION:
        return COMPILER_VERSION_SECTION_NAME.data();
    default:
        OPENVINO_THROW("Attempted to convert an unkown section type to string");
    }
}

SectionType section_type_from_string(std::string_view type) {
    if (type == RUNTIME_REQUIREMENTS_SECTION_NAME) {
        return SectionTypeCode::RUNTIME_REQUIREMENTS;
    }
    if (type == MANIFEST_SECTION_NAME) {
        return SectionTypeCode::MANIFEST;
    }
    if (type == ELF_MAIN_SCHEDULE_SECTION_NAME) {
        return SectionTypeCode::ELF_MAIN_SCHEDULE;
    }
    if (type == ELF_INIT_SCHEDULES_SECTION_NAME) {
        return SectionTypeCode::ELF_INIT_SCHEDULES;
    }
    if (type == DYNAMIC_SCHEDULE_SECTION_NAME) {
        return SectionTypeCode::DYNAMIC_SCHEDULE;
    }
    if (type == IO_LAYOUTS_SECTION_NAME) {
        return SectionTypeCode::IO_LAYOUTS;
    }
    if (type == BATCH_SIZE_SECTION_NAME) {
        return SectionTypeCode::BATCH_SIZE;
    }
    if (type == ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME) {
        return SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG;
    }
    if (type == COMPILER_VERSION_SECTION_NAME) {
        return SectionTypeCode::COMPILER_VERSION;
    }

    return SectionTypeCode::UNKNOWN;
}

std::ostream& operator<<(std::ostream& os, const SectionType& type) {
    return os << section_type_to_string(type);
}

std::istream& operator>>(std::istream& is, SectionType& type) {
    std::string str;
    is >> str;
    type = section_type_from_string(str);
    return is;
}

}  // namespace intel_npu
