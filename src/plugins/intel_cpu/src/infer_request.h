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

// CVS_111453: transformation shouldn't change model's input/output's precision
#define WA_PREC_CHANGE_ISSUE

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

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) override;

    void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors) override;

#ifdef WA_PREC_CHANGE_ISSUE
    ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const override;
    std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& _port) const override;
#endif

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

    void init_tensor(const std::string& name);
    void PushInputData();

    Graph* graph = nullptr;
    mutable std::unordered_map<std::string, void*> _external_ptr;

private:
    void PushStates();
    void PullStates();
    void redefineMemoryForInputNodes();

    std::string get_port_name(const ov::Output<const ov::Node>& port) const;
    void check_port(const ov::Output<const ov::Node>& port) const;
    void update_external_inputs();
    bool check_precision_changed(const ov::Output<const ov::Node>& port) const;
    InferenceEngine::TensorDesc create_tensor_desc(const ov::Tensor& tensor);

#ifdef WA_PREC_CHANGE_ISSUE
    const ov::Output<const ov::Node>& get_internal_port(const ov::Output<const ov::Node>& port) const;
    ov::Tensor create_internal_tensor(const ov::Tensor& tensor, const ov::Output<const ov::Node>& port);

    mutable std::unordered_map<std::string, ov::Output<const ov::Node>> _orig_ports_map;
    // Store internal tensor due to precision changes
    std::unordered_map<std::string, ov::Tensor> _internal_tensors;
#endif

    std::shared_ptr<const CompiledModel> _compiled_model;
    openvino::itt::handle_t _profiling_task;
    std::vector<std::shared_ptr<ov::IVariableState>> _memory_states;
    AsyncInferRequest* _asyncRequest = nullptr;

    mutable std::unordered_map<std::string, ov::Output<const ov::Node>> _input_ports_map;
    mutable std::unordered_map<std::string, ov::Output<const ov::Node>> _output_ports_map;
    std::unordered_map<std::string, ov::Tensor> _outputs;

protected:
    virtual void changeDefaultPtr();
};

}  // namespace intel_cpu
}  // namespace ov
