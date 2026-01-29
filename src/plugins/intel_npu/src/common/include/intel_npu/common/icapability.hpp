// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_npu/common/cre.hpp"

namespace intel_npu {

class ICapability {
public:
    ICapability(const CRE::Token token);

    ICapability(const CRE::Token token, const bool supported);

    virtual ~ICapability() = default;

    CRE::Token get_token() const;

    bool check_support() const;

    virtual bool lazy_check_support() const = 0;

private:
    CRE::Token m_token;
    mutable std::optional<bool> m_supported;
};

}  // namespace intel_npu
