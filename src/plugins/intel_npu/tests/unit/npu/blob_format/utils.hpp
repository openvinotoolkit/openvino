// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/hsm_writer.hpp"

namespace ov {
namespace unit_test {
namespace intel_npu {

ov::runtime::hsm::SectionSink create_generic_section_sink(std::ostream& stream);

}  // namespace intel_npu
}  // namespace unit_test
}  // namespace ov
