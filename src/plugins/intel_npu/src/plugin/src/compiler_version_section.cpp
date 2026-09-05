// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_version_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

CompilerVersionSection::CompilerVersionSection(const int32_t version, const ov::log::Level log_level)
    : ISection(PredefinedSectionType::COMPILER_VERSION),
      m_compiler_version(version),
      m_logger("CompilerVersionSection ", log_level) {
    m_logger.trace("Section created");
}

void CompilerVersionSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompilerVersionSection::write");
    m_logger.debug("Writting batch size %lu", m_compiler_version);

    writer.write_from(&m_compiler_version, sizeof(m_compiler_version));
}

int32_t CompilerVersionSection::get_compiler_version() const {
    return m_compiler_version;
}

std::shared_ptr<ISection> CompilerVersionSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompilerVersionSection::read");

    const size_t section_length = blob_reader.get_section_length();
    OPENVINO_ASSERT(section_length == sizeof(int32_t),
                    "CompilerVersionSection: incorrect section length ",
                    section_length,
                    ". Expected: ",
                    sizeof(int32_t));

    int32_t compiler_version;
    blob_reader.read_into_buffer(&compiler_version, sizeof(compiler_version));

    Logger("CompilerVersionSection", blob_reader.get_log_level()).debug("Read batch size %lu", compiler_version);

    return std::make_shared<CompilerVersionSection>(compiler_version, blob_reader.get_log_level());
}

}  // namespace intel_npu
