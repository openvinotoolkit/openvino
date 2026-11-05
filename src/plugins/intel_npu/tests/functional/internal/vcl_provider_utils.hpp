// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"

namespace vcl_provider {

/// A provider over the real VCL library. All callers in the test process share one loader, like a plugin that stays
/// loaded.
inline intel_npu::VCLFunctionTableProvider processWide() {
    static const auto holder = std::make_shared<const intel_npu::VCLLoaderHolder>();
    return [] {
        return holder->functions();
    };
}

/// A provider whose VCL library cannot be loaded.
inline intel_npu::VCLFunctionTableProvider unavailable() {
    return []() -> std::shared_ptr<const intel_npu::VCLFunctionTable> {
        OPENVINO_THROW("Injected unavailable provider");
    };
}

/// A provider the code under test must never call. A call fails the test.
inline intel_npu::VCLFunctionTableProvider forbidden(std::string reason) {
    return [reason = std::move(reason)]() -> std::shared_ptr<const intel_npu::VCLFunctionTable> {
        ADD_FAILURE() << reason;
        OPENVINO_THROW("Unexpected provider invocation: ", reason);
    };
}

/// Wraps `provider`, counting every call in `calls`. `calls` must outlive the returned provider.
inline intel_npu::VCLFunctionTableProvider counted(std::size_t& calls, intel_npu::VCLFunctionTableProvider provider) {
    return [&calls, provider = std::move(provider)] {
        calls++;
        return provider();
    };
}

}  // namespace vcl_provider
