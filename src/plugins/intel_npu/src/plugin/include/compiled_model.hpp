// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_npu/al/icompiled_model.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "properties.hpp"

namespace intel_npu {

class CompiledModel final : public ICompiledModel {
public:
    /**
     * @brief The constructor used by the "Plugin::compile_model" method.
     * @note The compilation step has been placed inside this constructor instead of the originating call. This choice
     * was motivated by the possibility of modifying the I/O identifiers via these passes which could potentially lead
     * to bugs.
     * @param model The IR of the model to be compiled
     * @param plugin Pointer towards the NPU plugin instance
     * @param device Backend specific object through which inference requests can be created
     * @param compiler Module used for compiling the IR model.
     * @param profiling Flag indicating if profiling was requested. Setting this to "true" will lead to storing the
     * "compiler" parameter inside the newly created "CompiledModel".
     * @param config Custom configuration object
     */
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const std::shared_ptr<IDevice>& device,
                  const ov::SoPtr<ICompiler>& compiler,
                  const bool profiling,
                  const Config& config);

    /**
     * @brief The constructor used by the "Plugin::import_model" method.
     * @param model The IR of the already compiled model
     * @param plugin Pointer towards the NPU plugin instance
     * @param networkDescription Object holding the compiled model within a buffer along with distinct fields for its
     * metadata
     * @param device Backend specific object through which inference requests can be created
     * @param compiler If set, the module will be stored inside the newly created "CompiledModel"
     * @param config Custom configuration object
     */
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const std::shared_ptr<const NetworkDescription>& networkDescription,
                  const std::shared_ptr<IDevice>& device,
                  const ov::SoPtr<ICompiler>& compiler,
                  const Config& config);

    CompiledModel(const CompiledModel&) = delete;

    CompiledModel& operator=(const CompiledModel&) = delete;

    ~CompiledModel() override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    // For CID path mean it get the blob from compiler through pfnGetNativeBinary using
    // networkDescription->metadata.graphHandle.

    // For CIP path mean it get the blob from
    // networkDescription->compiledNetwork.
    void export_model(std::ostream& stream) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    const std::shared_ptr<const NetworkDescription>& get_network_description() const override;

    const Config& get_config() const override;

    const ov::SoPtr<ICompiler>& get_compiler() const override;

private:
    void configure_stream_executors();

    void create_executor();

    std::shared_ptr<const NetworkDescription> _networkPtr;
    const std::shared_ptr<const ov::Model> _model;
    Config _config;
    Logger _logger;
    const std::shared_ptr<IDevice> _device;
    mutable std::shared_ptr<IExecutor> _executorPtr;
    std::shared_ptr<ov::threading::ITaskExecutor> _resultExecutor;

    std::unique_ptr<Properties> _properties;

    const ov::SoPtr<ICompiler> _compiler;
};

}  //  namespace intel_npu
