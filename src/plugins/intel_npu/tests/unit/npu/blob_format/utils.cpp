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

}  // namespace intel_npu
}  // namespace unit_test
}  // namespace ov
