// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

namespace ov {
namespace unit_test {
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

    return manifest_entry;
}

}  // namespace intel_npu
}  // namespace unit_test
}  // namespace ov
