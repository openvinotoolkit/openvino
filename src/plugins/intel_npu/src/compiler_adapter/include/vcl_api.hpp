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
#define vcl_symbols_list()                                  \
    vcl_symbol_statement(vclGetVersion)                     \
    vcl_symbol_statement(vclCompilerCreate)                 \
    vcl_symbol_statement(vclCompilerDestroy)                \
    vcl_symbol_statement(vclCompilerGetProperties)          \
    vcl_symbol_statement(vclQueryNetworkCreate)             \
    vcl_symbol_statement(vclQueryNetwork)                   \
    vcl_symbol_statement(vclQueryNetworkDestroy)            \
    vcl_symbol_statement(vclExecutableCreate)               \
    vcl_symbol_statement(vclAllocatedExecutableCreate)      \
    vcl_symbol_statement(vclExecutableDestroy)              \
    vcl_symbol_statement(vclExecutableGetSerializableBlob)  \
    vcl_symbol_statement(vclProfilingCreate)                \
    vcl_symbol_statement(vclGetDecodedProfilingBuffer)      \
    vcl_symbol_statement(vclProfilingDestroy)               \
    vcl_symbol_statement(vclProfilingGetProperties)         \
    vcl_symbol_statement(vclLogHandleGetString)


//unsupported symbols with older ze_loader versions
#define vcl_weak_symbols_list()                             \
    vcl_symbol_statement(vclAllocatedExecutableCreate3)     \
    vcl_symbol_statement(vclAllocatedExecutableCreate2)     \
    vcl_symbol_statement(vclGetCompilerSupportedOptions)    \
    vcl_symbol_statement(vclGetCompilerIsOptionSupported)
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

#define vcl_symbol_statement(vcl_symbol) decltype(&::vcl_symbol) vcl_symbol;
    vcl_symbols_list();
    vcl_weak_symbols_list();
#undef vcl_symbol_statement

private:
    std::shared_ptr<void> lib;
    Logger _logger;
};

#define vcl_symbol_statement(vcl_symbol)                                                                            \
    template <typename... Args>                                                                                     \
    inline typename std::invoke_result<decltype(&::vcl_symbol), Args...>::type wrapped_##vcl_symbol(Args... args) { \
        const auto& ptr = VCLApi::getInstance();                                                                    \
        if (ptr->vcl_symbol == nullptr) {                                                                           \
            OPENVINO_THROW("Unsupported vcl_symbol " #vcl_symbol);                                                  \
        }                                                                                                           \
        return ptr->vcl_symbol(std::forward<Args>(args)...);                                                        \
    }
vcl_symbols_list();
vcl_weak_symbols_list();
#undef vcl_symbol_statement
#define vcl_symbol_statement(vcl_symbol) inline decltype(&::vcl_symbol) vcl_symbol = wrapped_##vcl_symbol;
vcl_symbols_list();
vcl_weak_symbols_list();
#undef vcl_symbol_statement

class VCLCompilerImpl final : public intel_npu::ICompiler {
public:
    VCLCompilerImpl();
    ~VCLCompilerImpl() override;

    static std::shared_ptr<VCLCompilerImpl>& getInstance() {
        static std::shared_ptr<VCLCompilerImpl> compiler = std::make_shared<VCLCompilerImpl>();
        return compiler;
    }

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const override;

    uint32_t get_version() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const intel_npu::Config& config) const final override;

    bool get_supported_options(std::vector<char>& options) const;

    bool is_option_supported(const std::string& option) const;

private:
    std::shared_ptr<VCLApi> _vclApi;
    vcl_log_handle_t _logHandle = nullptr;
    vcl_compiler_handle_t _compilerHandle = nullptr;
    vcl_compiler_properties_t _compilerProperties;
    vcl_version_info_t _vclVersion;
    vcl_version_info_t _vclProfilingVersion;
    Logger _logger;
};

}  // namespace intel_npu
