// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_npu/common/icompiled_model.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/so_ptr.hpp"

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
     * @param graph Object holding the graph handle along with distinct fields for metadata
     * @param profiling Flag indicating if profiling was requested. Setting this to "true" will lead to storing the
     * "compiler" parameter inside the newly created "CompiledModel".
     * @param config Custom configuration object
     */
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const std::shared_ptr<IDevice>& device,
                  const std::shared_ptr<IGraph>& graph,
                  const Config& config);

    CompiledModel(const CompiledModel&) = delete;

    CompiledModel& operator=(const CompiledModel&) = delete;

    ~CompiledModel() override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    void export_model(std::ostream& stream) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    const std::shared_ptr<IGraph>& get_graph() const override;

    const Config& get_config() const override;

private:
    void initialize_properties();

    void configure_stream_executors();

    Config _config;
    Logger _logger;
    const std::shared_ptr<IDevice> _device;
    std::shared_ptr<ov::threading::ITaskExecutor> _resultExecutor;

    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>> _properties;
    std::vector<ov::PropertyName> _supportedProperties;

    std::shared_ptr<IGraph> _graph;
};

}  //  namespace intel_npu
