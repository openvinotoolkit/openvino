// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>

#include "fake_vcl.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "intel_npu/utils/vcl/vcl_function_table_provider.hpp"
#include "openvino/core/except.hpp"

namespace ov::test::npu::vcl_provider {

// A provider over a fake VCL table. The closure keeps the fake instance alive.
inline ::intel_npu::VCLFunctionTableProvider fake(std::initializer_list<std::string> unsupportedOptions = {}) {
    auto fakeVcl = std::make_shared<::fake_vcl::FakeVcl>();
    fakeVcl->unsupportedOptions.insert(unsupportedOptions.begin(), unsupportedOptions.end());
    return [fakeVcl = std::move(fakeVcl)] {
        return fakeVcl->functions();
    };
}

// A provider whose VCL library cannot be loaded.
inline ::intel_npu::VCLFunctionTableProvider unavailable() {
    return []() -> std::shared_ptr<const ::intel_npu::VCLFunctionTable> {
        OPENVINO_THROW("Injected unavailable provider");
    };
}

// A provider the code under test must never call. A call fails the test.
inline ::intel_npu::VCLFunctionTableProvider forbidden(std::string reason) {
    return [reason = std::move(reason)]() -> std::shared_ptr<const ::intel_npu::VCLFunctionTable> {
        ADD_FAILURE() << reason;
        OPENVINO_THROW("Unexpected provider invocation: ", reason);
    };
}

// Wraps `vclFunctionsProvider`, counting every call in `calls`. `calls` must outlive the returned provider.
inline ::intel_npu::VCLFunctionTableProvider counted(std::size_t& calls,
                                                     ::intel_npu::VCLFunctionTableProvider vclFunctionsProvider) {
    return [&calls, vclFunctionsProvider = std::move(vclFunctionsProvider)] {
        calls++;
        return vclFunctionsProvider();
    };
}

}  // namespace ov::test::npu::vcl_provider
