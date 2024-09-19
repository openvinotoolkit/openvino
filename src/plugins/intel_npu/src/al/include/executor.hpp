// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/properties.hpp"

namespace intel_npu {

class IExecutor {
public:
    virtual ~IExecutor() = default;

    virtual void setWorkloadType(const ov::WorkloadType workloadType) const = 0;
};

}  // namespace intel_npu
