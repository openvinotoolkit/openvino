// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_writer.hpp"

#include "intel_npu/common/itt.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"

using namespace intel_npu;

namespace {

ov::runtime::hsm::SectionTagReserved build_section_tag(const std::shared_ptr<ISection>& section) {
    OPENVINO_ASSERT(section->get_id().has_value());
    const SectionID id = section->get_id().value();

    const ov::runtime::hsm::SectionTag tag =
        ov::runtime::hsm::SectionTag::make(section->get_type().get_code(), section->is_inline());
    std::array<uint8_t, 4> reserved_bytes{id.get_id() & 0xFF, id.get_id() >> 8 & 0xFF, 0, 0};
    return ov::runtime::hsm::SectionTagReserved(tag, reserved_bytes);
}

}  // namespace

namespace intel_npu::hsm {

// TODO ITT here
Writer::Writer(const std::shared_ptr<BlobSectionsRegistry>& blob_sections_registry)
    : m_blob_sections_registry(blob_sections_registry),
      m_logger("BatchSizeSection", Logger::global().level()) {
    m_logger.trace("NPU HSM writer created");
}

void Writer::write_to(std::ostream& stream) const {
    const auto runtime_requirements_section = m_blob_sections_registry->build_runtime_requirements_section();

    const auto deferred_writer = ov::runtime::hsm::DeferredWriter::open(stream);
    OPENVINO_ASSERT(deferred_writer.has_value(), "A blob writer could not be created. The stream may be invalid");
    // TODO add common sections when available
    deferred_writer->add_section(ov::runtime::hsm::Devices::npu,
                                 build_section_tag(runtime_requirements_section),
                                 runtime_requirements_section,
                                 runtime_requirements_section->get_alignment());

    const std::queue<std::shared_ptr<ISection>> write_queue = m_blob_sections_registry->get_write_queue();
    for (const auto section : write_queue) {
        deferred_writer->add_section(ov::runtime::hsm::Devices::npu,
                                     build_section_tag(section),
                                     section,
                                     section->get_alignment());
    }

    deferred_writer->finalize();
}

}  // namespace intel_npu::hsm
