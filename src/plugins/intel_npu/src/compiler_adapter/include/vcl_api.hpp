// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/icompiler.hpp"
#include "npu_driver_compiler.h"
#include "openvino/core/except.hpp"

namespace intel_npu {

// clang-format off
#define symbols_list()                                        \
    symbol_statement(vclCompilerCreate)              \
    symbol_statement(vclGetVersion)           \
    symbol_statement(vclCompilerDestroy)           \
    symbol_statement(vclCompilerGetProperties)          \
    symbol_statement(vclAllocatedExecutableCreate)         \
    symbol_statement(vclExecutableDestroy) \
    symbol_statement(vclExecutableGetSerializableBlob)                      \
    symbol_statement(vclQueryNetworkCreate)                     \
    symbol_statement(vclQueryNetwork)                    \
    symbol_statement(vclQueryNetworkDestroy)                      \
    symbol_statement(vclProfilingCreate)                    \
    symbol_statement(vclProfilingDestroy)                   \
    symbol_statement(vclGetDecodedProfilingBuffer)       \
    symbol_statement(vclLogHandleGetString)                         \
    symbol_statement(vclGetCompilerSupportedOptions)                 \
    symbol_statement(vclGetCompilerIsOptionSupported)

// clang-format on

class VCLApi {
public:
    VCLApi();
    VCLApi(const VCLApi& other) = delete;
    VCLApi(VCLApi&& other) = delete;
    void operator=(const VCLApi&) = delete;
    void operator=(VCLApi&&) = delete;

    static const std::shared_ptr<VCLApi>& getInstance();
    std::shared_ptr<void> getLibrary() const {
        return lib;
    }

#define symbol_statement(symbol) decltype(&::symbol) symbol;
    symbols_list();
#undef symbol_statement

private:
    std::shared_ptr<void> lib;
};

#define symbol_statement(symbol)                                                                            \
    template <typename... Args>                                                                             \
    inline typename std::invoke_result<decltype(&::symbol), Args...>::type wrapped_##symbol(Args... args) { \
        const auto& ptr = VCLApi::getInstance();                                                            \
        if (ptr->symbol == nullptr) {                                                                       \
            OPENVINO_THROW("Unsupported symbol " #symbol);                                                  \
        }                                                                                                   \
        return ptr->symbol(std::forward<Args>(args)...);                                                    \
    }
symbols_list();
#undef symbol_statement
#define symbol_statement(symbol) inline decltype(&::symbol) symbol = wrapped_##symbol;
symbols_list();
#undef symbol_statement

class VCLCompilerImpl final : public intel_npu::ICompiler {
public:
    VCLCompilerImpl();
    ~VCLCompilerImpl() override;

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const override;

    uint32_t get_version() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const intel_npu::Config& config) const final override;

private:
    std::shared_ptr<VCLApi> _vclApi;
    vcl_log_handle_t _logHandle = nullptr;
    vcl_compiler_handle_t _compilerHandle = nullptr;
    vcl_compiler_properties_t _compilerProperties;
    Logger _logger;

    // Helper function to serialize the model to a blob
    ov::Tensor serializeModelToBlob(const std::shared_ptr<const ov::Model>& model) const;
};

}  // namespace intel_npu
