// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_size_section.hpp"

#include "intel_npu/common/itt.hpp"

namespace intel_npu {

// TODO comment why inline
BatchSizeSection::BatchSizeSection(const int64_t batch_size)
    : ISection(SectionTypeCode::BATCH_SIZE, {}, true),
      m_batch_size(batch_size),
      m_logger("BatchSizeSection", Logger::global().level()) {
    m_logger.trace("Section created");
}

std::vector<std::shared_ptr<CREToken>> BatchSizeSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>&
    /*all_registered_sections*/) const {
    m_logger.debug("Added the BATCH_SIZE section type to the CRE");
    return {std::make_shared<SectionType>(get_type())};
}

int64_t BatchSizeSection::get_batch_size() const {
    return m_batch_size;
}

void BatchSizeSection::encode(const ov::runtime::hsm::SectionSink& sink) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "BatchSizeSection::write");
    m_logger.debug("Writting batch size %ld", m_batch_size);

    sink({reinterpret_cast<const std::byte*>(&m_batch_size), sizeof(m_batch_size)});
}

ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> BatchSizeSection::decode =
    [](const ov::runtime::hsm::Section& section) -> std::optional<std::shared_ptr<ISection>> {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "BatchSizeSection::decode");
    const Logger logger("BatchSizeSection", Logger::global().level());
    OPENVINO_ASSERT(section.tag().id() == static_cast<uint32_t>(SectionTypeCode::BATCH_SIZE),
                    "Logic error: the decoder received a section of the wrong type");
    OPENVINO_ASSERT(section.tag().is_inline() && section.size() >= sizeof(int64_t),
                    "Logic error: the payload is not inline or the size is not sufficiently large");

    int64_t batch_size;
    section.read(0, sizeof(batch_size), &batch_size);
    logger.debug("Read batch size %lu", batch_size);

    return std::make_shared<BatchSizeSection>(batch_size);
};

}  // namespace intel_npu
