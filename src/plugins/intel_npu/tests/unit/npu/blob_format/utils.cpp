// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

namespace ov {
namespace unit_test {
namespace intel_npu {

::intel_npu::BlobWriterInterface create_default_writer_interface(std::ostream& stream) {
    return ::intel_npu::BlobWriterInterface(stream, stream.tellp());
}

}  // namespace intel_npu
}  // namespace unit_test
}  // namespace ov
