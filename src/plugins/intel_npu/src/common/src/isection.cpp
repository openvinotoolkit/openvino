// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection.hpp"

namespace {

constexpr std::string_view TYPE_AND_INSTANCE_DELIMITER = "_";

bool has_only_digits(std::string_view sv) {
    return !sv.empty() && std::all_of(sv.begin(), sv.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
};

}  // namespace

namespace intel_npu {

ISection::ISection(const SectionType type) : m_section_type(type) {}

SectionType ISection::get_section_type() const {
    return m_section_type;
}

std::optional<SectionTypeInstance> ISection::get_section_type_instance() const {
    return m_section_type_instance;
}

void ISection::set_section_type_instance(const SectionTypeInstance type_instance) const {
    OPENVINO_ASSERT(!m_section_type_instance.has_value(),
                    "Attempted to set an instance ID to a section that already had one. Section type: ",
                    m_section_type,
                    ", old instance ID: ",
                    m_section_type_instance.value());

    m_section_type_instance = type_instance;
}

std::optional<SectionID> ISection::get_section_id() const {
    if (!m_section_type_instance.has_value()) {
        return std::nullopt;
    }

    return SectionID(m_section_type, m_section_type_instance.value());
}

SectionID::SectionID(SectionType section_type, SectionTypeInstance section_type_instance) {
    type = section_type;
    type_instance = section_type_instance;
}

std::string SectionID::to_string() const {
    std::ostringstream sstream;
    sstream << *this;
    return sstream.str();
}

SectionID SectionID::from_string(std::string_view section_id_string) {
    SectionID section_id;
    std::istringstream sstream(section_id_string.data());
    sstream >> section_id;
    return section_id;
}

bool operator==(const SectionID& sid1, const SectionID& sid2) {
    return sid1.type == sid2.type && sid1.type_instance == sid2.type_instance;
}

std::ostream& operator<<(std::ostream& os, const SectionID& id) {
    switch (id.type) {
    case PredefinedSectionType::CRE:
        os << "CRE";
        break;
    case PredefinedSectionType::OFFSETS_TABLE:
        os << "OFFSETS_TABLE";
        break;
    case PredefinedSectionType::ELF_MAIN_SCHEDULE:
        os << "ELF_MAIN_SCHEDULE";
        break;
    case PredefinedSectionType::ELF_INIT_SCHEDULES:
        os << "ELF_INIT_SCHEDULES";
        break;
    case PredefinedSectionType::IO_LAYOUTS:
        os << "IO_LAYOUTS";
        break;
    case PredefinedSectionType::BATCH_SIZE:
        os << "BATCH_SIZE";
        break;
    default:
        os << id.type;
        break;
    }

    os << id.type_instance;
    return os;
}

// TODO test these
std::istream& operator>>(std::istream& is, SectionID& id) {
    std::string str;
    is >> str;

    const size_t search_result = str.rfind(TYPE_AND_INSTANCE_DELIMITER);
    OPENVINO_ASSERT(
        search_result != std::string::npos,
        "The \"_\" character that delimits the type and instance IDs is missing from the given section ID string");

    const std::string type_string = str.substr(0, search_result);

    if (type_string == "CRE") {
        id.type = PredefinedSectionType::CRE;
    } else if (type_string == "OFFSETS_TABLE") {
        id.type = PredefinedSectionType::OFFSETS_TABLE;
    }
    if (type_string == "ELF_MAIN_SCHEDULE") {
        id.type = PredefinedSectionType::ELF_MAIN_SCHEDULE;
    }
    if (type_string == "ELF_INIT_SCHEDULES") {
        id.type = PredefinedSectionType::ELF_INIT_SCHEDULES;
    }
    if (type_string == "IO_LAYOUTS") {
        id.type = PredefinedSectionType::IO_LAYOUTS;
    }
    if (type_string == "BATCH_SIZE") {
        id.type = PredefinedSectionType::BATCH_SIZE;
    } else {
        OPENVINO_ASSERT(has_only_digits(type_string),
                        "Attempted to convert unknown section type ",
                        type_string,
                        " to integer, but it is not made exclusively out of digits");

        try {
            id.type = std::stoul(type_string);
        } catch (const std::exception&) {
            OPENVINO_THROW("Unable to convert section type ", type_string, " to integer, the type is unknown");
        }
    }

    const std::string type_instance_string = str.substr(search_result + 1, std::string::npos);
    OPENVINO_ASSERT(has_only_digits(type_instance_string),
                    "Cannot convert to integer: type instance ",
                    type_instance_string,
                    " is not made exclusively out of digits");

    try {
        id.type_instance = std::stoul(type_instance_string);
    } catch (const std::exception&) {
        OPENVINO_THROW("Failed to convert the section type instance ", type_instance_string, " to integer");
    }
}

}  // namespace intel_npu

size_t std::hash<intel_npu::SectionID>::operator()(const intel_npu::SectionID& sid) const {
    return std::hash<intel_npu::SectionType>{}(sid.type) ^
           (std::hash<intel_npu::SectionTypeInstance>{}(sid.type_instance) << 1);
}
