// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"

namespace ov {
namespace unit_test {
namespace intel_npu {

std::unique_ptr<::intel_npu::BlobWriterInterface> create_default_writer_interface(std::ostream& stream);

}  // namespace intel_npu
}  // namespace unit_test
}  // namespace ov
