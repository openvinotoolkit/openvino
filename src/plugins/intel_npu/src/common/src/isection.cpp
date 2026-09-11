// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection.hpp"

namespace {

constexpr std::string_view TYPE_AND_ID_DELIMITER = "_";

// TODO move to utility
bool has_only_digits(std::string_view sv) {
    return !sv.empty() && std::all_of(sv.begin(), sv.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
};

}  // namespace

namespace intel_npu {

ISection::ISection(const SectionType type) : m_type(type) {}

SectionType ISection::get_type() const {
    return m_type;
}

std::optional<SectionID> ISection::get_id() const {
    return m_id;
}

std::optional<std::string> ISection::get_inidividual_compatibility_requirements() const {
    // No individual requirements by default
    return std::nullopt;
}

void ISection::set_id(const SectionID& id) const {
    OPENVINO_ASSERT(!m_id.has_value(),
                    "Attempted to set an instance ID to a section that already had one. Section type: ",
                    m_type,
                    ", old instance ID: ",
                    m_id.value());

    m_id = id;
}

std::vector<std::shared_ptr<CREToken>> ISection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    // By default, no requirements are added
    return {};
}

bool ISection::evaluate_compatibility_based_on_section_content(BlobReaderInterface& /*reader*/) {
    return true;
}

// TODO test these
std::string section_type_and_id_to_string(const SectionType type, const SectionID id) {
    return section_type_to_string(type) + TYPE_AND_ID_DELIMITER.data() + section_id_to_string(id);
}

// TODO note about optional
std::pair<SectionType, std::optional<SectionID>> section_type_and_id_from_string(std::string_view type_and_id) {
    const size_t search_result = type_and_id.rfind(TYPE_AND_ID_DELIMITER);
    if (search_result == std::string::npos ||
        !has_only_digits(type_and_id.substr(search_result + 1, std::string::npos))) {
        // There's only a section type
        return std::make_pair(section_type_from_string(type_and_id), std::nullopt);
    }

    // Both section type & id are present
    const SectionType type = section_type_from_string(type_and_id.substr(0, search_result));
    const SectionID id = section_id_from_string(type_and_id.substr(search_result + 1, std::string::npos));
    return std::make_pair(type, std::make_optional(id));
}

}  // namespace intel_npu
