// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <atomic>
#include <ie_plugin.hpp>
#include <inference_engine.hpp>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include "cldnn_graph.h"

namespace CLDNNPlugin {

struct buf_info {
    size_t buf_offset;
    size_t buf_size;
};

class CLDNNInferRequest : public InferenceEngine::InferRequestInternal {
    static std::atomic<unsigned int> runningCounter;

public:
    // make sure all blobs and cldnn::memory objects
    // are in place and valid
    void checkBlobs() override;
    void InferImpl() override;

    void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override;

    CLDNNInferRequest(InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs);

    CLDNNInferRequest(const CLDNNInferRequest &) = delete;

    virtual ~CLDNNInferRequest() = default;

    void GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) override;
    void SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) override;

    void SetBatch(int batch = -1) override;
    void SetGraph(std::shared_ptr<CLDNNGraph> graph);
    void EnableProfiling() { m_useProfiling = true; }
    void EnableStreams() { m_useStreams = true; }
    static unsigned int GetRunningCounter() { return runningCounter.load(); }

protected:
    std::map<std::string, cldnn::memory> inputsMemory;
    std::map<std::string, cldnn::primitive_id> outputsMap;

    bool m_useProfiling;
    bool m_useStreams;
    std::shared_ptr<CLDNNGraph> m_graph;

    // dynamic batch stuff
    std::map<std::string, std::vector<buf_info>> batchInputs;
    std::map<std::string, std::vector<buf_info>> batchOutputs;

    InferenceEngine::Blob::Ptr createInputBlob(const InferenceEngine::TensorDesc& desc, uint8_t* mem_ptr = nullptr);
    InferenceEngine::Blob::Ptr createOutputBlob(const InferenceEngine::TensorDesc& desc, uint8_t* mem_ptr = nullptr);
    void copyOutputData(const cldnn::memory& outputMemory, InferenceEngine::Blob::Ptr bptr, buf_info* bi = nullptr);
    void copyInputData(std::shared_ptr<cldnn::network> network, const cldnn::primitive_id &inputName,
                       const cldnn::layout& inputLayout, const InferenceEngine::Blob &inputBlob,
                       buf_info* bi = nullptr);

    void input_attach(cldnn::primitive_id name, cldnn::memory& inputMem);
    void input_alloc(cldnn::primitive_id name, const cldnn::layout& layout);
    void AllocateInputs();
    void AllocateOutputs();
    void AllocateInputsDyn();
    void AllocateOutputsDyn();
    void execAndParse();
    void execAndParseDyn();

    void PrepareInput(const cldnn::primitive_id &inputName, const InferenceEngine::Blob &inputBlob);
    void PrepareInputDyn(const cldnn::primitive_id &inputName, const InferenceEngine::Blob &inputBlob);

private:
    static const char fp32_suffix[];
};

};  // namespace CLDNNPlugin
