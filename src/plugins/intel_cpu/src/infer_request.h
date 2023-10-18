// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.h"
#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include "cpu_tensor.h"

namespace ov {
namespace intel_cpu {

class ExecNetwork;
class AsyncInferRequest;

class InferRequestBase : public InferenceEngine::IInferRequestInternal {
public:
    virtual ~InferRequestBase();

    void InferImpl() override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> QueryState() override;

    /**
     * @brief      Sets the pointer to asynchronous inference request that holds this request
     * @param[in]  asyncRequest Pointer to asynchronous inference request
     */
    void SetAsyncRequest(AsyncInferRequest* asyncRequest);

    /**
     * @brief If `_asyncRequest` is initialized throw exception with `InferenceEngine::INFER_CANCELLED` status if inference request is canceled
     */
    void ThrowIfCanceled() const;

protected:
    InferRequestBase(InferenceEngine::InputsDataMap networkInputs,
                     InferenceEngine::OutputsDataMap networkOutputs,
                     std::shared_ptr<ExecNetwork> execNetwork_)
    : IInferRequestInternal(networkInputs, networkOutputs), execNetwork(execNetwork_) {}

    InferRequestBase(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                     const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                     std::shared_ptr<ExecNetwork> execNetwork_)
    : IInferRequestInternal(inputs, outputs), execNetwork(execNetwork_) {}

    void CreateInferRequest();
    InferenceEngine::Precision normToInputSupportedPrec(const std::pair<const std::string, InferenceEngine::Blob::Ptr>& input) const;
    void pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob, InferenceEngine::Precision dataType);

protected:
    class OutputControlBlock {
    public:
        using MemMngrPtr = std::shared_ptr<MemoryMngrWithReuse>;

    public:
        OutputControlBlock(const InferenceEngine::Precision& precision, const Shape& shape);

        OutputControlBlock(const OutputControlBlock&) = delete;
        OutputControlBlock& operator=(const OutputControlBlock&) = delete;

        OutputControlBlock(OutputControlBlock&&) = default;
        OutputControlBlock& operator=(OutputControlBlock&&) = default;

        InferenceEngine::Blob::Ptr blob() const {
            return m_blob;
        }

        std::shared_ptr<Tensor> tensor() const {
            return m_tensor;
        }

        const void* rawPtr() const {
            return m_tensor->get_memory()->getData();
        }

        MemMngrPtr currentMemMngr() const {
            return m_buffers[m_buffIndx];
        }

        MemMngrPtr nextMemMngr() {
            m_buffIndx ^= 0x1;
            if (!m_buffers[m_buffIndx]) {
                m_buffers[m_buffIndx] = std::make_shared<MemoryMngrWithReuse>();
            }
            return m_buffers[m_buffIndx];
        }

        void update() {
            m_proxyMemMngr->setMemMngr(currentMemMngr());
            m_blob->allocate(); // WA: update handle
        }

    private:
        std::shared_ptr<Tensor> m_tensor = nullptr;
        InferenceEngine::Blob::Ptr m_blob = nullptr;
        ProxyMemoryMngrPtr m_proxyMemMngr = nullptr;
        std::array<MemMngrPtr, 2> m_buffers;
        int m_buffIndx = 0;
    };

protected:
    virtual void initBlobs() = 0;
    virtual void PushInputData() = 0;

    Graph* graph = nullptr;
    std::unordered_map<std::string, InferenceEngine::Blob::Ptr> externalPtr;

    std::unordered_map<std::string, OutputControlBlock> outputControlBlocks;

private:
    void PushStates();
    void PullStates();
    void redefineMemoryForInputNodes();

    std::shared_ptr<ExecNetwork>        execNetwork;
    openvino::itt::handle_t             profilingTask;
    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> memoryStates;
    AsyncInferRequest*                  _asyncRequest = nullptr;

protected:
    virtual void changeDefaultPtr();
};

class LegacyInferRequest : public InferRequestBase {
public:
    LegacyInferRequest(InferenceEngine::InputsDataMap networkInputs,
                       InferenceEngine::OutputsDataMap networkOutputs,
                       std::shared_ptr<ExecNetwork> execNetwork);

    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) override;
    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;

private:
    void PushInputData() override;
    void initBlobs() override;
    void changeDefaultPtr() override;
};

class InferRequest : public InferRequestBase {
public:
    InferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                 const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                 std::shared_ptr<ExecNetwork> execNetwork);

    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) override;
    void SetBlobsImpl(const std::string& name, const InferenceEngine::BatchedBlob::Ptr& batched_blob) override;
    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;

    void checkBlobs() override;

private:
    void PushInputData() override;
    void initBlobs() override;

    std::unordered_map<std::string, std::shared_ptr<const ov::Node>> modelInputsMap;
    std::unordered_map<std::string, std::shared_ptr<const ov::Node>> modelOutputsMap;
};

}   // namespace intel_cpu
}   // namespace ov
