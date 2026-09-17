// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace intel_npu {

/**
 * @brief Basic unit that composes the compatibility requirements expression
 */
class CREToken {
public:
    virtual ~CREToken() = default;

protected:
    CREToken() = default;
};

}  // namespace intel_npu
