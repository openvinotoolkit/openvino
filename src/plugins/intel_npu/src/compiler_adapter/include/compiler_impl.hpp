// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>

#include "compiler.h"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/icompiler.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

class VCLApi;

class VCLCompilerImpl final : public intel_npu::ICompiler {
public:
    VCLCompilerImpl();
    ~VCLCompilerImpl() override;
    static const std::shared_ptr<VCLCompilerImpl> getInstance();

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    std::vector<std::shared_ptr<NetworkDescription>> compileWsOneShot(const std::shared_ptr<ov::Model>& model,
                                                                      const Config& config) const override;

    NetworkDescription compileWsIterative(const std::shared_ptr<ov::Model>& model,
                                          const Config& config,
                                          size_t callNumber) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const override;

    uint32_t get_version() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const intel_npu::Config& config) const final override;

    bool get_supported_options(std::vector<char>& options) const;

    bool is_option_supported(const std::string& option, std::optional<std::string> optValue = std::nullopt) const;

    std::shared_ptr<VCLApi> getLinkedLibrary() const;

private:
    vcl_log_handle_t _logHandle = nullptr;
    vcl_compiler_handle_t _compilerHandle = nullptr;
    vcl_compiler_properties_t _compilerProperties;
    vcl_version_info_t _vclVersion;
    vcl_version_info_t _vclProfilingVersion;
    Logger _logger;
};

}  // namespace intel_npu
