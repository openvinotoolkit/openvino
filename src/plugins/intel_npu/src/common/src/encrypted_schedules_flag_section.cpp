// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "encrypted_schedules_flag_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

EncryptedSchedulesFlagSection::EncryptedSchedulesFlagSection(const bool applied_encryption,
                                                             const ov::log::Level log_level)
    : ISection(PredefinedSectionType::ENCRYPTED_SCHEDULES_FLAG),
      m_flag(applied_encryption),
      m_logger("EncryptedSchedulesFlagSection", log_level) {
    m_logger.trace("Section created");
}

std::vector<CREToken> EncryptedSchedulesFlagSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    m_logger.debug("Added the ENCRYPTED_SCHEDULES_FLAG section type to the CRE");
    return {get_type()};
}

void EncryptedSchedulesFlagSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "EncryptedSchedulesFlagSection::write");
    m_logger.debug("Writting the encryption flag %lu", m_flag);

    writer.write_from(&m_flag, sizeof(m_flag));
}

bool EncryptedSchedulesFlagSection::get_flag() const {
    return m_flag;
}

std::shared_ptr<ISection> EncryptedSchedulesFlagSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "EncryptedSchedulesFlagSection::read");

    const size_t section_length = blob_reader.get_section_length();
    OPENVINO_ASSERT(section_length == sizeof(bool),
                    "EncryptedSchedulesFlagSection: incorrect section length ",
                    section_length,
                    ". Expected: ",
                    sizeof(bool));

    bool flag;
    blob_reader.read_into_buffer(&flag, sizeof(flag));

    Logger("EncryptedSchedulesFlagSection", blob_reader.get_log_level()).debug("Read the encryption flag %lu", flag);

    return std::make_shared<EncryptedSchedulesFlagSection>(flag, blob_reader.get_log_level());
}

}  // namespace intel_npu
