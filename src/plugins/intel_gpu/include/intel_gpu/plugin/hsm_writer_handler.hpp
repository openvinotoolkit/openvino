// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <ostream>

#include "intel_gpu/plugin/hsm_common.hpp"
#include "openvino/runtime/properties.hpp"

namespace cldnn {
class BinaryOutputBuffer;
}  // namespace cldnn

namespace ov::intel_gpu::hsm {

// Uses the shared writer with bounded transport buffering; never stages a plaintext graph.
void write_model(std::ostream& destination,
                 ov::CacheMode cache_mode,
                 const std::string& requirements,
                 std::function<void(cldnn::BinaryOutputBuffer&)> serialize,
                 Crypt encrypt);

}  // namespace ov::intel_gpu::hsm
