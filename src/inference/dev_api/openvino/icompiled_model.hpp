// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime CompiledModel interface
 * @file icompiled_model.hpp
 */

#pragma once

#include <memory>
#include <openvino/runtime/common.hpp>
#include <ostream>
#include <vector>

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace InferenceEngine {
class Core;
}

namespace ov {

class IPlugin;

OPENVINO_RUNTIME_API std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> convert_compiled_model_to_legacy(
    const std::shared_ptr<ov::ICompiledModel>& model);

class OPENVINO_RUNTIME_API ICompiledModel : public std::enable_shared_from_this<ICompiledModel> {
public:
    using Ptr = std::shared_ptr<ICompiledModel>;

    ICompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin);
    ICompiledModel(const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& exec_network);

    const std::vector<ov::Output<const ov::Node>>& outputs() const;
    const std::vector<ov::Output<const ov::Node>>& inputs() const;

    virtual std::shared_ptr<InferenceEngine::IInferRequestInternal> create_infer_request() const;

    virtual void export_model(std::ostream& model) const;

    virtual std::shared_ptr<ov::Model> get_runtime_model() const;

    virtual void set_property(const ov::AnyMap& properties);

    virtual ov::Any get_property(const std::string& name) const;

    virtual ov::RemoteContext get_context() const;

protected:
    virtual std::shared_ptr<InferenceEngine::IInferRequestInternal> create_infer_request_impl(
        const std::vector<ov::Output<const ov::Node>>& inputs,
        const std::vector<ov::Output<const ov::Node>>& outputs) const;

private:
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
    std::shared_ptr<const ov::IPlugin> m_plugin;
    std::shared_ptr<void> m_so;
    bool m_loaded_from_cache = false;
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> m_exec_network;

    friend IPlugin;
    friend InferenceEngine::Core;
    friend OPENVINO_RUNTIME_API std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>
    convert_compiled_model_to_legacy(const std::shared_ptr<ov::ICompiledModel>& model);
};

}  // namespace ov
