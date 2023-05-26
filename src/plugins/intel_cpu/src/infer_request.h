// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "graph.h"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace intel_cpu {

class CompiledModel;
class AsyncInferRequest;

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    SyncInferRequest(std::shared_ptr<const CompiledModel> compiled_model);
    virtual ~SyncInferRequest();

    void infer() override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    std::vector<std::shared_ptr<ov::IVariableState>> query_state() const override;

    ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const override;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) override;

    void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors) override;

    /**
     * @brief      Sets the pointer to asynchronous inference request that holds this request
     * @param[in]  asyncRequest Pointer to asynchronous inference request
     */
    void set_async_request(AsyncInferRequest* asyncRequest);

    /**
     * @brief If `_asyncRequest` is initialized throw exception with `ov::Cancelled` status if inference request is
     * canceled
     */
    void throw_if_canceled() const;

protected:
    void create_infer_request();
    InferenceEngine::Precision normToInputSupportedPrec(const std::pair<const std::string, ov::Tensor>& input) const;
    void pushInput(const std::string& inputName, ov::Tensor& inputBlob, InferenceEngine::Precision dataType);

    void prepare_tensor(const std::string& name);
    void PushInputData();

    Graph* graph = nullptr;
    mutable std::unordered_map<std::string, void*> _external_ptr;

private:
    void PushStates();
    void PullStates();
    void redefineMemoryForInputNodes();

    bool find_port_name(const ov::Output<const ov::Node>& port, std::string& name, bool is_input = true) const;

    std::shared_ptr<const CompiledModel> _compiled_model;
    openvino::itt::handle_t _profiling_task;
    std::vector<std::shared_ptr<ov::IVariableState>> _memory_states;
    AsyncInferRequest* _asyncRequest = nullptr;

    std::unordered_map<std::string, ov::Output<const ov::Node>> _input_ports_map;
    std::unordered_map<std::string, ov::Output<const ov::Node>> _output_ports_map;

    std::unordered_map<std::string, ov::Tensor> _outputs;

protected:
    virtual void changeDefaultPtr();
};

}  // namespace intel_cpu
}  // namespace ov
