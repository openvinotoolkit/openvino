// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/icapability.hpp"

namespace intel_npu {

class StaticCapability final : public ICapability {
public:
    StaticCapability(const CRE::Token token);

    bool lazy_check_support() const override;
};

}  // namespace intel_npu
