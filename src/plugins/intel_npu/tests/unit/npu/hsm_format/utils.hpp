// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/section_type.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov {
namespace unit_test {
namespace intel_npu {

ov::runtime::hsm::SectionSink create_generic_section_sink(std::ostream& stream);

ov::runtime::hsm::ManifestEntry create_manifest_entry_of_type(const ::intel_npu::SectionTypeCode type,
                                                              const bool is_inline = false,
                                                              const bool any_device = false);

}  // namespace intel_npu
}  // namespace unit_test
}  // namespace ov
