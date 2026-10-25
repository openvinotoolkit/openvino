// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_version_section.hpp"

#include "intel_npu/common/itt.hpp"

namespace intel_npu {

CompilerVersionSection::CompilerVersionSection(const int32_t version, const ov::log::Level log_level)
    : ISection(SectionTypeCode::COMPILER_VERSION, {}, true),
      m_compiler_version(version),
      m_logger("CompilerVersionSection", log_level) {
    m_logger.trace("Section created");
}

void CompilerVersionSection::encode(const ov::runtime::hsm::SectionSink& sink) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompilerVersionSection::write");
    m_logger.debug("Writting compiler version %lu", m_compiler_version);

    sink({reinterpret_cast<const std::byte*>(&m_compiler_version), sizeof(m_compiler_version)});
}

int32_t CompilerVersionSection::get_compiler_version() const {
    return m_compiler_version;
}

ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> CompilerVersionSection::decode =
    [](const ov::runtime::hsm::Section& section) -> std::optional<std::shared_ptr<ISection>> {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompilerVersionSection::decode");
    const Logger logger("CompilerVersionSection", Logger::global().level());
    OPENVINO_ASSERT(section.tag().id() == static_cast<uint32_t>(SectionTypeCode::COMPILER_VERSION),
                    "Logic error: the decoder received a section of the wrong type");

    const size_t section_length = section.size();
    if (section_length != sizeof(int32_t)) {
        logger.error("Incorrect section length %zu. Expected: %zu", section_length, sizeof(int32_t));
        return std::nullopt;
    }

    int32_t compiler_version;
    section.read(&compiler_version);

    logger.debug("Read compiler version %lu", compiler_version);

    return std::make_shared<CompilerVersionSection>(compiler_version, Logger::global().level());
};

}  // namespace intel_npu
