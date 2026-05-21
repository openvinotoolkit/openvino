// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

namespace ov {
namespace unit_test {
namespace intel_npu {

std::unique_ptr<::intel_npu::BlobWriterInterface> create_default_writer_interface(std::ostream& stream) {
    return std::make_unique<::intel_npu::BlobWriterInterface>(
        stream,
        std::queue<std::shared_ptr<::intel_npu::ISection>>(),
        ::intel_npu::CRE(),
        std::unordered_map<::intel_npu::SectionType, ::intel_npu::SectionTypeInstance>());
}

}  // namespace intel_npu
}  // namespace unit_test
}  // namespace ov
