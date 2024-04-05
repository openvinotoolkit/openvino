// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_npu/al/icompiled_model.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

class CompiledModel final : public ICompiledModel {
public:
    explicit CompiledModel(const std::shared_ptr<const ov::Model>& model,
                           const std::shared_ptr<const ov::IPlugin>& plugin,
                           const std::shared_ptr<const NetworkDescription>& networkDescription,
                           const std::shared_ptr<IDevice>& device,
                           const std::optional<ov::SoPtr<ICompiler>>& compiler,
                           const Config& config);

    CompiledModel(const CompiledModel&) = delete;

    CompiledModel& operator=(const CompiledModel&) = delete;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    void export_model(std::ostream& stream) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    const std::shared_ptr<const NetworkDescription>& get_network_description() const override;

    const Config& get_config() const override;

    const ov::SoPtr<ICompiler>& get_compiler() const override;

private:
    void initialize_properties();

    void configure_stream_executors();

    std::shared_ptr<const NetworkDescription> _networkPtr;
    const std::shared_ptr<const ov::Model> _model;
    const Config _config;
    Logger _logger;
    const std::shared_ptr<IDevice> _device;
    mutable std::shared_ptr<IExecutor> _executorPtr;
    std::shared_ptr<ov::threading::ITaskExecutor> _resultExecutor;

    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>> _properties;
    std::vector<ov::PropertyName> _supportedProperties;

    std::optional<ov::SoPtr<ICompiler>> _compiler;
};

}  //  namespace intel_npu
