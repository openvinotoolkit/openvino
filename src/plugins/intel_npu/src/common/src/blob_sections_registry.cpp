// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_sections_registry.hpp"

#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/major_minor_version.hpp"

namespace {

constexpr uint16_t FIRST_AVAILABLE_SECTION_ID = 0;

}  // namespace

namespace intel_npu {

BlobSectionsRegistry::BlobSectionsRegistry()
    : m_next_section_id(FIRST_AVAILABLE_SECTION_ID),
      m_logger("BlobSectionsRegistry", Logger::global().level()) {}

SectionID BlobSectionsRegistry::register_section(const std::shared_ptr<ISection>& section) {
    OPENVINO_ASSERT(!section->get_id().has_value(),
                    "The given section has already been registered in a sections registry");
    const SectionType section_type = section->get_type();

    // TODO overflow checks
    const SectionID section_id(m_next_section_id++);
    OPENVINO_ASSERT(!m_ids_to_sections.count(section_id),
                    "The same section ID has been attributed to two distinct sections");

    section->set_id(section_id);
    m_write_queue.push(section);
    m_ids_to_sections[section_id] = section;
    m_types_to_sections[section_type].push_back(section);

    m_logger.debug("Registered section %s", section_type_and_id_to_string(section_type, section_id));

    return section_id;
}

std::shared_ptr<ISection> BlobSectionsRegistry::retrieve_section(const SectionID& id) const {
    auto search_result = m_ids_to_sections.find(id);
    if (search_result != m_ids_to_sections.end()) {
        return search_result->second;
    }
    return nullptr;
}

std::shared_ptr<ISection> BlobSectionsRegistry::retrieve_any_section(const SectionType section_type) const {
    if (!m_types_to_sections.count(section_type) || m_types_to_sections.at(section_type).empty()) {
        return nullptr;
    }

    return *m_types_to_sections.at(section_type).begin();
}

std::vector<std::shared_ptr<ISection>> BlobSectionsRegistry::retrieve_sections(const SectionType type) const {
    auto search_result = m_types_to_sections.find(type);
    if (search_result != m_types_to_sections.end()) {
        return search_result->second;
    }
    return {};
}

std::queue<std::shared_ptr<ISection>> BlobSectionsRegistry::get_write_queue() const {
    return m_write_queue;
}

bool BlobSectionsRegistry::has_section_of_type(const SectionType section_type) const {
    return m_types_to_sections.count(section_type) ? m_types_to_sections.at(section_type).size() > 0 : false;
}

size_t BlobSectionsRegistry::count_sections_of_type(const SectionType section_type) const {
    return m_types_to_sections.count(section_type) ? m_types_to_sections.at(section_type).size() : 0;
}

std::unordered_map<SectionType, size_t> BlobSectionsRegistry::get_content_summary() const {
    std::unordered_map<SectionType, size_t> summary;
    for (const auto& [section_type, section_instances] : m_types_to_sections) {
        summary[section_type] = section_instances.size();
    }

    return summary;
}

std::shared_ptr<ISection> BlobSectionsRegistry::build_runtime_requirements_section() const {
    m_logger.debug("Building the runtime requirements");
    std::map<SectionID, std::string> sections_requirements;
    CRE cre(m_logger.level());
    std::unordered_map<SectionID, SectionType> section_id_to_type;

    // Each section can register a compatibility substring, as well as a compatiblity subexpression (between sections)
    for (const auto& [section_id, section] : m_ids_to_sections) {
        const std::optional<std::string> individual_requirements = section->get_individual_compatibility_requirements();
        if (individual_requirements.has_value()) {
            sections_requirements.emplace(section_id, individual_requirements.value());
        }

        cre.append_to_expression(section->get_compatibility_requirements_subexpression(m_ids_to_sections));
        section_id_to_type.emplace(section_id, section->get_type());
    }

    const auto section = std::make_shared<RuntimeRequirementsSection>(
        RuntimeRequirements(sections_requirements, cre, section_id_to_type));
    // TODO move to hpp
    // This section is not registered within this class since the section is dynamically generated based on the current
    // contents of the registry. Future calls to this function may yield different runtime requirements. Despite this, a
    // section ID is set to this section. We are assuming the user of this function may want to write the current
    // section, and a unique section ID may be required for that.
    section->set_id(m_next_section_id);
    return section;
}

}  // namespace intel_npu
