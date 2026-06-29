// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/cre_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

CRESection::CRESection(const CRE& cre, const ov::log::Level log_level)
    : ISection(PredefinedSectionType::CRE),
      m_cre(cre),
      m_logger("CRESection", log_level) {}

void CRESection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CRESection::write");

    writer.write_from(m_cre.get_expression().data(), m_cre.get_expression_length() * sizeof(CREToken));

    m_logger.debug("%lu tokens written", m_cre.get_expression_length());
}

CRE CRESection::get_cre() const {
    return m_cre;
}

std::shared_ptr<ISection> CRESection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CRESection::read");
    Logger logger("CRESection", blob_reader.get_log_level());

    const size_t section_length = blob_reader.get_section_length();
    OPENVINO_ASSERT(section_length % sizeof(CREToken) == 0,
                    "Received a CRE section length that is not divisible by the CRE token size. Section length: ",
                    section_length,
                    ". CRE token size: ",
                    sizeof(CREToken));
    size_t number_of_tokens = section_length / sizeof(CREToken);
    if (number_of_tokens == 0) {
        logger.warning("The parsed CRE is empty. No compatibility checks will be performed");
    }

    logger.debug("Reading %lu tokens", number_of_tokens);

    std::vector<CREToken> tokens(number_of_tokens);
    blob_reader.copy_data_from_source(reinterpret_cast<char*>(tokens.data()), number_of_tokens * sizeof(CREToken));

    return std::make_shared<CRESection>(CRE(tokens, logger.level()), logger.level());
}

}  // namespace intel_npu
