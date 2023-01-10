// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <ngraph/function.hpp>

#include "openvino/runtime/icompiled_model.hpp"
#include "template_async_infer_request.hpp"
#include "template_config.hpp"
#include "template_infer_request.hpp"

namespace TemplatePlugin {

// forward declaration
class Plugin;

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
// ! [executable_network:header]
class ExecutableNetwork : public ov::ICompiledModel {
public:
    ExecutableNetwork(const std::shared_ptr<ov::Model>& model,
                      const Configuration& cfg,
                      const std::shared_ptr<const Plugin>& plugin);

    ExecutableNetwork(std::istream& model, const Configuration& cfg, const std::shared_ptr<const Plugin>& plugin);

    // Methods from a base class ExecutableNetworkThreadSafeDefault

    void export_model(std::ostream& model) const override;
    std::shared_ptr<ov::IInferRequest> create_infer_request_impl() const override;
    std::shared_ptr<ov::IInferRequest> create_infer_request() const override;

    ov::Any get_property(const std::string& name) const override;

private:
    friend class TemplateInferRequest;
    friend class Plugin;

    void InitExecutor();

    std::atomic<std::size_t> _requestId = {0};
    Configuration _cfg;
    std::shared_ptr<const Plugin> get_template_plugin() const;
    std::shared_ptr<ov::Model> m_model;
};
// ! [executable_network:header]

}  // namespace TemplatePlugin
