// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "config.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace template_plugin {

class Plugin;
class InferRequest;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
// ! [executable_network:header]
class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                  const Configuration& cfg);

    // Methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    virtual ov::Any get_property(const std::string& name) const override;

    ov::RemoteContext get_context() const override;
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    friend class InferRequest;
    friend class Plugin;

    void compile_model(const std::shared_ptr<ov::Model>& model);
    std::shared_ptr<const Plugin> get_template_plugin() const;

    mutable std::atomic<std::size_t> _requestId = {0};
    Configuration _cfg;
    std::shared_ptr<ov::Model> m_model;
};
// ! [executable_network:header]

}  // namespace template_plugin
}  // namespace ov
