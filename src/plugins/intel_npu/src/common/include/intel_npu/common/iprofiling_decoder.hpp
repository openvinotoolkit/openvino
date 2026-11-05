// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

class IProfilingDecoder {
public:
    virtual std::vector<ov::ProfilingInfo> decode(const std::vector<uint8_t>& profData,
                                                  const ov::Tensor& network) const = 0;
    virtual ~IProfilingDecoder() = default;
};

}  // namespace intel_npu
