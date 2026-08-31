// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection.hpp"

namespace {

constexpr std::string_view TYPE_AND_ID_DELIMITER = "_";
constexpr std::string_view CRE_SECTION_NAME = "CRE";
constexpr std::string_view MANIFEST_SECTION_NAME = "MANIFEST";
constexpr std::string_view ELF_MAIN_SCHEDULE_SECTION_NAME = "ELF_MAIN_SCHEDULE";
constexpr std::string_view ELF_INIT_SCHEDULES_SECTION_NAME = "ELF_INIT_SCHEDULES";
constexpr std::string_view DYNAMIC_SCHEDULE_SECTION_NAME = "DYNAMIC_SCHEDULE";
constexpr std::string_view IO_LAYOUTS_SECTION_NAME = "IO_LAYOUTS";
constexpr std::string_view BATCH_SIZE_SECTION_NAME = "BATCH_SIZE";
constexpr std::string_view ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME = "ENCRYPTED_SCHEDULES_FLAG";
constexpr std::string_view COMPILER_VERSION_SECTION_NAME = "COMPILER_VERSION";

bool has_only_digits(std::string_view sv) {
    return !sv.empty() && std::all_of(sv.begin(), sv.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
};

}  // namespace

namespace intel_npu {

ISection::ISection(const SectionType type) : m_type(type) {}

SectionType ISection::get_section_type() const {
    return m_type;
}

std::optional<SectionID> ISection::get_section_id() const {
    return m_id;
}

void ISection::set_id(const SectionID id) const {
    OPENVINO_ASSERT(!m_id.has_value(),
                    "Attempted to set an instance ID to a section that already had one. Section type: ",
                    m_type,
                    ", old instance ID: ",
                    m_id.value());

    m_id = id;
}

std::optional<SectionID> ISection::get_section_id() const {
    return m_id;
}

std::vector<CREToken> ISection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionType, std::unordered_map<SectionID, std::shared_ptr<ISection>>>&
    /*all_registered_sections*/) const {
    // By default, no requirements are added
    return {};
}

bool ISection::evaluate_compatibility_based_on_section_content(BlobReaderInterface& /*reader*/) {
    return true;
}

std::string section_type_to_string(const SectionType type) {
    switch (type) {
    case PredefinedSectionType::CRE:
        return CRE_SECTION_NAME.data();
    case PredefinedSectionType::MANIFEST:
        return MANIFEST_SECTION_NAME.data();
    case PredefinedSectionType::ELF_MAIN_SCHEDULE:
        return ELF_MAIN_SCHEDULE_SECTION_NAME.data();
    case PredefinedSectionType::ELF_INIT_SCHEDULES:
        return ELF_INIT_SCHEDULES_SECTION_NAME.data();
    case PredefinedSectionType::DYNAMIC_SCHEDULE:
        return DYNAMIC_SCHEDULE_SECTION_NAME.data();
    case PredefinedSectionType::IO_LAYOUTS:
        return IO_LAYOUTS_SECTION_NAME.data();
    case PredefinedSectionType::BATCH_SIZE:
        return BATCH_SIZE_SECTION_NAME.data();
    case PredefinedSectionType::ENCRYPTED_SCHEDULES_FLAG:
        return ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME.data();
    case PredefinedSectionType::COMPILER_VERSION:
        return COMPILER_VERSION_SECTION_NAME.data();
    default:
        return std::to_string(type);
    }
}

SectionType section_type_from_string(std::string_view type) {
    if (type == CRE_SECTION_NAME) {
        return PredefinedSectionType::CRE;
    }
    if (type == MANIFEST_SECTION_NAME) {
        return PredefinedSectionType::MANIFEST;
    }
    if (type == ELF_MAIN_SCHEDULE_SECTION_NAME) {
        return PredefinedSectionType::ELF_MAIN_SCHEDULE;
    }
    if (type == ELF_INIT_SCHEDULES_SECTION_NAME) {
        return PredefinedSectionType::ELF_INIT_SCHEDULES;
    }
    if (type == DYNAMIC_SCHEDULE_SECTION_NAME) {
        return PredefinedSectionType::DYNAMIC_SCHEDULE;
    }
    if (type == IO_LAYOUTS_SECTION_NAME) {
        return PredefinedSectionType::IO_LAYOUTS;
    }
    if (type == BATCH_SIZE_SECTION_NAME) {
        return PredefinedSectionType::BATCH_SIZE;
    }
    if (type == ENCRYPTED_SCHEDULES_FLAG_SECTION_NAME) {
        return PredefinedSectionType::ENCRYPTED_SCHEDULES_FLAG;
    }
    if (type == COMPILER_VERSION_SECTION_NAME) {
        return PredefinedSectionType::COMPILER_VERSION;
    }

    OPENVINO_ASSERT(has_only_digits(type),
                    "Attempted to convert unknown section type ",
                    type,
                    " to integer, but it is not made exclusively out of digits");

    try {
        return std::stoul(type.data());
    } catch (const std::exception&) {
        OPENVINO_THROW("Unable to convert section type ", type, " to integer, the type is unknown");
    }
}

// TODO test these
std::string section_type_and_id_to_string(const SectionType type, const SectionID id) {
    return section_type_to_string(type) + TYPE_AND_ID_DELIMITER.data() + std::to_string(id);
}

std::pair<SectionType, SectionID> section_type_and_id_from_string(std::string_view type_and_id) {
    const size_t search_result = type_and_id.rfind(TYPE_AND_ID_DELIMITER);
    OPENVINO_ASSERT(search_result != std::string::npos,
                    "The ",
                    TYPE_AND_ID_DELIMITER,
                    " character that delimits the type and instance IDs is missing from the given section ID string");

    const SectionType type = section_type_from_string(type_and_id.substr(0, search_result));

    const std::string id_string = type_and_id.substr(search_result + 1, std::string::npos).data();
    OPENVINO_ASSERT(has_only_digits(id_string),
                    "Cannot convert to integer: type instance ",
                    id_string,
                    " is not made exclusively out of digits");

    try {
        return std::make_pair<>(type, std::stoul(id_string));
    } catch (const std::exception&) {
        OPENVINO_THROW("Failed to convert the section type instance ", id_string, " to integer");
    }
}

}  // namespace intel_npu
