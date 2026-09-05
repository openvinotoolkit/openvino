// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ivcl_compiler.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

namespace fake_vcl {

/// A trivial IVCLCompiler with settable canned returns, for exercising PluginCompilerAdapter's own
/// logic without a driver, a compiler library, or an NPU.
class FakeVCLCompiler : public ::intel_npu::IVCLCompiler {
public:
    //
    // --- canned returns ---
    //

    /// Blob handed back by compile(); a page-sized buffer by default.
    ov::Tensor compileResult = ov::Tensor(ov::element::u8, ov::Shape{4096});
    std::optional<std::string> compatibility = std::string("fake-compat");

    /// Tensors handed back by compileWsOneShot(): init schedules first, main last.
    std::vector<ov::Tensor> wsOneShotResult = {ov::Tensor(ov::element::u8, ov::Shape{4096}),
                                               ov::Tensor(ov::element::u8, ov::Shape{4096})};

    ov::Tensor wsIterativeResult = ov::Tensor(ov::element::u8, ov::Shape{4096});

    ov::SupportedOpsMap queryResult;
    uint32_t version = 0x0007000A;
    std::vector<std::string> supportedOptions = {"OPT_A", "OPT_B"};
    /// Options the fake reports as supported; anything else is unsupported.
    std::vector<std::string> supportedOptionNames = {"OPT_A", "OPT_B"};

    bool throwOnCompile = false;

    //
    // --- recordings ---
    //

    mutable int compileCalls = 0;
    mutable int compileWsOneShotCalls = 0;
    mutable int compileWsIterativeCalls = 0;
    mutable int queryCalls = 0;
    mutable int getSupportedOptionsCalls = 0;
    mutable std::vector<std::pair<std::string, std::optional<std::string>>> optionSupportQueries;
    mutable std::vector<size_t> wsIterativeCallNumbers;

    //
    // --- IVCLCompiler ---
    //

    std::pair<ov::Tensor, std::optional<std::string>> compile(const std::shared_ptr<const ov::Model>&,
                                                              const ::intel_npu::FilteredConfig&) const override {
        ++compileCalls;
        if (throwOnCompile) {
            OPENVINO_THROW("FakeVCLCompiler: compile failed on request");
        }
        return {compileResult, compatibility};
    }

    std::pair<std::vector<ov::Tensor>, std::optional<std::string>> compileWsOneShot(
        const std::shared_ptr<ov::Model>&,
        const ::intel_npu::FilteredConfig&) const override {
        ++compileWsOneShotCalls;
        return {wsOneShotResult, compatibility};
    }

    std::pair<ov::Tensor, std::optional<std::string>> compileWsIterative(const std::shared_ptr<ov::Model>&,
                                                                         const ::intel_npu::FilteredConfig&,
                                                                         size_t callNumber) const override {
        ++compileWsIterativeCalls;
        wsIterativeCallNumbers.push_back(callNumber);
        return {wsIterativeResult, compatibility};
    }

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>&,
                              const ::intel_npu::FilteredConfig&) const override {
        ++queryCalls;
        return queryResult;
    }

    uint32_t get_version() const override {
        return version;
    }

    std::vector<std::string> get_supported_options() const override {
        ++getSupportedOptionsCalls;
        return supportedOptions;
    }

    bool is_option_supported(const std::string& option, const std::optional<std::string>& optValue) const override {
        optionSupportQueries.emplace_back(option, optValue);
        for (const auto& known : supportedOptionNames) {
            if (known == option) {
                return true;
            }
        }
        return false;
    }

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>&,
                                                            const std::vector<uint8_t>&) const override {
        return {};
    }
};

}  // namespace fake_vcl
