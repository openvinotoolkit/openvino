// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_writer.hpp"
#include "intel_npu/common/itt.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"

using namespace intel_npu;

namespace {

const ov::runtime::hsm::SectionTag RUNTIME_REQUIREMENTS_TAG =
    ov::runtime::hsm::SectionTag::make(SectionTypeCode::RUNTIME_REQUIREMENTS, false);

}  // namespace

namespace intel_npu::hsm {

// TODO ITT here
Reader::Reader(
    const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>& readers,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators)
    : m_readers(readers),
      m_type_evaluators(type_evaluators),
      m_instance_evaluators(instance_evaluators) {}

BlobSectionsRegistry Reader::read(ov::runtime::hsm::Reader& base_reader) const {
    // TODO dedicated tags instead of sectin types?
    std::optional<RuntimeRequirements> runtime_requirements;
    const size_t n_runtime_requirements = base_reader.count(ov::runtime::hsm::Devices::npu, );
    OPENVINO_ASSERT(n_runtime_requirements < 2, "Found more than one runtime requirements section");

    if (n_runtime_requirements) {
        OPENVINO_ASSERT(m_readers.count(SectionTypeCode::RUNTIME_REQUIREMENTS), "No reader found for the manifest");
        const auto decode_result = base_reader.decode(ov::runtime::hsm::Devices::npu,
                                                      RUNTIME_REQUIREMENTS_TAG,
                                                      m_readers.at(SectionTypeCode::RUNTIME_REQUIREMENTS));
        OPENVINO_ASSERT(decode_result.has_value());
        const auto runtime_requirements_section =
            std::dynamic_pointer_cast<RuntimeRequirementsSection>(decode_result.value());
        OPENVINO_ASSERT(runtime_requirements_section);

        runtime_requirements = runtime_requirements_section->get_runtime_requirements();
        OPENVINO_ASSERT(runtime_requirements->get_compatibility_check_result(m_section_type_evaluators,
                                                                             m_section_instance_evaluators),
                        "The imported model is not compatible");
        m_logger.debug("Runtime requirements evaluation passed");
    } else {
        m_logger.warning(
            "The runtime requirements section was not found within the manifest. Proceeding without performing any "
            "compatibility checks");
    }

    for (const ov::runtime::hsm::ManifestEntry& manifest_entry : base_reader.entries()) {
        const std::shared_ptr<ISection> parsed_section =
            base_reader.decode(manifest_entry.device,
                               manifest_entry.tag,
                               m_readers.at(SectionTypeCode::RUNTIME_REQUIREMENTS));
    }
}

}  // namespace intel_npu::hsm
