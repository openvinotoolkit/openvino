// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/encrypted_schedules_flag_section.hpp"

#include "intel_npu/common/itt.hpp"

namespace intel_npu {

EncryptedSchedulesFlagSection::EncryptedSchedulesFlagSection(const bool applied_encryption)
    : ISection(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG, {}, true),
      m_flag(applied_encryption),
      m_logger("EncryptedSchedulesFlagSection", Logger::global().level()) {
    m_logger.trace("Section created");
}

std::vector<std::shared_ptr<CREToken>> EncryptedSchedulesFlagSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    // Blob encryption should be supported within the plugin
    m_logger.debug("Added the ENCRYPTED_SCHEDULES_FLAG section type to the CRE");
    return {std::make_shared<SectionType>(get_type())};
}

ov::util::MemoryView EncryptedSchedulesFlagSection::encode() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "EncryptedSchedulesFlagSection::encode");
    m_logger.debug("Encoding the encryption flag %d", m_flag);

    return ov::util::MemoryView(reinterpret_cast<const std::byte*>(&m_flag), sizeof(m_flag));
}

bool EncryptedSchedulesFlagSection::get_flag() const {
    return m_flag;
}

void EncryptedSchedulesFlagSection::set_flag(const bool new_value) {
    m_flag = new_value;
}

ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> EncryptedSchedulesFlagSection::decode =
    [](const ov::runtime::hsm::Section& section) -> std::optional<std::shared_ptr<ISection>> {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "EncryptedSchedulesFlagSection::decode");
    const Logger logger("EncryptedSchedulesFlagSection", Logger::global().level());
    OPENVINO_ASSERT(section.tag().id() == static_cast<uint32_t>(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG),
                    "Logic error: the decoder received a section of the wrong type");
    OPENVINO_ASSERT(section.tag().is_inline() && section.size() >= sizeof(bool),
                    "Logic error: the payload is not inline or the size is not sufficiently large");

    bool flag;
    section.read(0, sizeof(flag), &flag);
    logger.debug("Read the encryption flag %d", flag);

    return std::make_shared<EncryptedSchedulesFlagSection>(flag);
};

}  // namespace intel_npu
