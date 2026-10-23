// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_reader.hpp"

#include "intel_npu/common/itt.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"

using namespace intel_npu;

namespace {

const ov::runtime::hsm::SectionTag RUNTIME_REQUIREMENTS_TAG =
    ov::runtime::hsm::SectionTag::make(static_cast<uint32_t>(SectionTypeCode::RUNTIME_REQUIREMENTS), false);

// TODO comment mirrors writer conversion
SectionID tag_reserved_bytes_to_section_id(const std::array<uint8_t, 4>& reserved_bytes) {
    return SectionID(reserved_bytes.at(0) + (reserved_bytes.at(1) << 8));
}

}  // namespace

namespace intel_npu::hsm {

// TODO ITT here
Reader::Reader(
    const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>& readers,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators)
    : m_readers(readers),
      m_type_evaluators(type_evaluators),
      m_instance_evaluators(instance_evaluators),
      m_logger("HSMReader", Logger::global().level()) {}

BlobSectionsRegistry Reader::read(ov::runtime::hsm::Reader& base_reader) const {
    // TODO dedicated tags instead of sectin types?
    std::optional<RuntimeRequirements> runtime_requirements;
    const size_t n_runtime_requirements =
        base_reader.count(static_cast<ov::runtime::hsm::DeviceId>(ov::runtime::hsm::Devices::npu),
                          RUNTIME_REQUIREMENTS_TAG);
    OPENVINO_ASSERT(n_runtime_requirements < 2, "Found more than one runtime requirements section");

    if (n_runtime_requirements) {
        OPENVINO_ASSERT(m_readers.count(SectionTypeCode::RUNTIME_REQUIREMENTS), "No reader found for the manifest");
        const auto decode_result =
            base_reader.decode(static_cast<ov::runtime::hsm::DeviceId>(ov::runtime::hsm::Devices::npu),
                               RUNTIME_REQUIREMENTS_TAG,
                               m_readers.at(SectionTypeCode::RUNTIME_REQUIREMENTS));
        OPENVINO_ASSERT(decode_result.has_value());
        const auto runtime_requirements_section =
            std::dynamic_pointer_cast<RuntimeRequirementsSection>(decode_result.value());
        OPENVINO_ASSERT(runtime_requirements_section);

        runtime_requirements = runtime_requirements_section->get_runtime_requirements();
        OPENVINO_ASSERT(runtime_requirements->get_compatibility_check_result(m_type_evaluators, m_instance_evaluators),
                        "The imported model is not compatible");
        m_logger.debug("Runtime requirements evaluation passed");
    } else {
        m_logger.warning(
            "The runtime requirements section was not found within the manifest. Proceeding without performing any "
            "compatibility checks");
    }

    BlobSectionsRegistry registry;

    for (const ov::runtime::hsm::ManifestEntry& manifest_entry : base_reader.entries()) {
        const std::optional<std::shared_ptr<ISection>> parsed_section =
            base_reader.decode(manifest_entry.device,
                               manifest_entry.tag,
                               m_readers.at(static_cast<SectionTypeCode>(manifest_entry.tag.id())));
        if (parsed_section.has_value()) {
            registry.register_section(parsed_section.value(),
                                      tag_reserved_bytes_to_section_id(manifest_entry.tag_reserved));
        }
    }
}

}  // namespace intel_npu::hsm
