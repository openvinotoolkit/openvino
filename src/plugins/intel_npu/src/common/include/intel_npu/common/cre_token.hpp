// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace intel_npu {

/**
 * @brief Basic unit that composes the compatibility requirements expression
 */
class CREToken {
public:
    virtual ~CREToken() = default;

    virtual std::string to_string() const = 0;

protected:
    CREToken() = default;
};

}  // namespace intel_npu
