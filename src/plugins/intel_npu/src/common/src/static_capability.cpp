// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/static_capability.hpp"

#include "openvino/core/except.hpp"

namespace intel_npu {

StaticCapability::StaticCapability(const CRE::Token token) : ICapability(token, true) {}

bool StaticCapability::lazy_check_support() const {
    OPENVINO_NOT_IMPLEMENTED;
}

}  // namespace intel_npu
