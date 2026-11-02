// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

namespace intel_npu {

ov::runtime::hsm::SectionSink create_generic_section_sink(std::ostream& stream) {
    return [&stream](ov::util::MemoryView data) {
        stream.write(reinterpret_cast<const char*>(data.data()), data.size());
        OPENVINO_ASSERT(stream, "Stream invalidated during writing");
    };
}

ov::runtime::hsm::ManifestEntry create_manifest_entry_of_type(const ::intel_npu::SectionTypeCode type,
                                                              const bool is_inline,
                                                              const bool any_device) {
    ov::runtime::hsm::ManifestEntry manifest_entry;
    manifest_entry.tag = ov::runtime::hsm::SectionTag::make(static_cast<uint32_t>(type), is_inline);
    manifest_entry.device = static_cast<ov::runtime::hsm::DeviceId>(!any_device ? ov::runtime::hsm::Devices::npu
                                                                                : ov::runtime::hsm::Devices::any);

    if (!is_inline) {
        manifest_entry.offset = 0;
    }

    return manifest_entry;
}

MockTypeEvaluator::MockTypeEvaluator(const bool result) : ISectionTypeEvaluator(), m_result(result) {}

bool MockTypeEvaluator::evaluate() const {
    return m_result;
}

ov::CompatibilityCheck MockInstanceEvaluator::evaluate(std::string_view runtime_requirements) const {
    if (runtime_requirements == STRING_THAT_EVALUATES_TO_SUPPORTED) {
        return ov::CompatibilityCheck::SUPPORTED;
    }
    if (runtime_requirements == STRING_THAT_EVALUATES_TO_UNSUPPORTED) {
        return ov::CompatibilityCheck::UNSUPPORTED;
    }
    return ov::CompatibilityCheck::NOT_APPLICABLE;
}

MockSection::MockSection(const std::vector<std::shared_ptr<CREToken>>& cre_tokens,
                         const std::optional<std::string>& individual_requirements,
                         const ov::runtime::hsm::SectionAlignment& alignment)
    : ISection(SectionTypeCode::IO_LAYOUTS, alignment, true),  // TODO comment
      m_cre_tokens(cre_tokens),
      m_individual_requirements(individual_requirements) {}

void MockSection::encode(const ov::runtime::hsm::SectionSink& sink) const {
    return;
}

ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> MockSection::decode =
    [](const ov::runtime::hsm::Section& section) -> std::optional<std::shared_ptr<ISection>> {
    return std::make_shared<MockSection>(std::vector<std::shared_ptr<CREToken>>(), std::nullopt);
};

std::optional<std::string> MockSection::get_individual_compatibility_requirements() const {
    return m_individual_requirements;
}

std::vector<std::shared_ptr<CREToken>> MockSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const {
    return m_cre_tokens;
}

}  // namespace intel_npu
