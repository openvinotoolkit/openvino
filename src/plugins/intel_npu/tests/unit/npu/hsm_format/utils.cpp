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

}  // namespace intel_npu
