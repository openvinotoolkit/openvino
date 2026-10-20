// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/utils/vcl/vcl_api.hpp"

// The holder is shared by every test of the binary, so the compiler library loads at most once.
inline ::intel_npu::VCLFunctionTableProvider testVclFunctions() {
    static const auto holder = std::make_shared<::intel_npu::VCLLoaderHolder>();
    return [] {
        return holder->functions();
    };
}
