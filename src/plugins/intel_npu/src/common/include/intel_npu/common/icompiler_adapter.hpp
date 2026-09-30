// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/config/config.hpp"

namespace intel_npu {

/**
 * @brief Settings consumed by the compiler adapter itself.
 * Each adapter decides which of them are relevant and may ignore the rest.
 */
struct AdapterDescriptor {
    // If set, the adapter skips any internal caching it may have for the compiled graph.
    bool bypassCache = false;
    // If set, the adapter requests secure compilation from its backend, if supported.
    bool secureCompile = false;
};

class ICompilerAdapter {
public:
    /**
     * @param model The model that will be compiled.
     * @param config Will be passed to the compiler.
     * @param adapterDesc Adapter-specific settings, not forwarded to the compiler. See "AdapterDescriptor".
     */
    virtual std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model,
                                            const Config& config,
                                            const AdapterDescriptor& adapterDesc) const = 0;

    /**
     * @brief Compiles the model, weights separation enabled.
     * @details The result of compilation will be a binary object that does not contain a significant portion of
     * weights. The binary object will include two types of schedules: weights initialization and the operations of the
     * main graph. In order to run inference on this weightless blob, the original weights will need to be provided as
     * inputs to the weights initialization schedule. Running this will output the processed weights, that can then be
     * fed to the main schedule and therefore enable it to run predictions.
     *
     * @param model The model that will be compiled.
     * @param config Will be passed to the compiler. Additionally, the "SEPARATE_WEIGHTS_VERSION" option will determine
     * which weights separation implementation will be used. See the weights separation specific methods within
     * "icompiler.hpp".
     * @param adapterDesc Adapter-specific settings, not forwarded to the compiler. See "AdapterDescriptor".
     * @return A "WeightlessGraph" type of object.
     */
    virtual std::shared_ptr<IGraph> compileWS(std::shared_ptr<ov::Model>&& model,
                                              const Config& config,
                                              const AdapterDescriptor& adapterDesc) const = 0;

    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;
    virtual uint32_t get_version() const = 0;
    virtual std::vector<std::string> get_supported_options() const = 0;
    virtual bool is_option_supported(const std::string& optName,
                                     const std::optional<std::string>& optValue = std::nullopt) const = 0;

    /**
     * @brief Resolves the config bundle(s) the platform named in \p config's NPU_PLATFORM needs
     * compiled from it - more than one when that platform ships as several SKU variants (e.g.
     * differing tile counts).
     * @details The default answer is an empty vector, i.e. "resolution is not available" - correct
     * for every adapter that cannot resolve SKU variants itself; the caller then compiles \p config
     * unchanged, as it always has. Only the adapter backed by the VCL compiler library overrides
     * this with a real answer. A single-entry result is the resolved, device-specific bundle (e.g.
     * naming NPU_MAX_TILES) the caller should merge into \p config before compiling.
     */
    virtual std::vector<std::string> resolve_compilation_target_bundles(const Config& config) const {
        static_cast<void>(config);
        return {};
    }

    virtual ~ICompilerAdapter() = default;
};

}  // namespace intel_npu
