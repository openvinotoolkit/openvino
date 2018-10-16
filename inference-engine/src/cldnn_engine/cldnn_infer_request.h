// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
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
public:
    void InferImpl() override;

    void
    GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override;

    CLDNNInferRequest(InferenceEnv env, bool useProfiling,
                      InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs);

    CLDNNInferRequest(const CLDNNInferRequest &) = delete;

    virtual ~CLDNNInferRequest() = default;

    void SetBatch(int batch = -1) override;

protected:
    std::map<std::string, cldnn::memory> inputsMemory;
    std::map<std::string, cldnn::primitive_id> outputsMap;
    std::map<cldnn::primitive_id, std::string> implementationsMap;
    bool m_useProfiling;
    InferenceEnv m_env;

    // dynamic batch stuff
    int m_curBatch;
    std::map<std::string, std::vector<buf_info>> batchInputs;
    std::map<std::string, std::vector<buf_info>> batchOutputs;

    InferenceEngine::Blob::Ptr createInputBlob(const InferenceEngine::Precision& p, const InferenceEngine::Layout& l,
                                               const InferenceEngine::SizeVector& sz, uint8_t* mem_ptr = nullptr);
    InferenceEngine::Blob::Ptr createOutputBlob(const InferenceEngine::Precision& p, InferenceEngine::SizeVector& sz,
                                                uint8_t* mem_ptr = nullptr);
    void copyOutputData(const cldnn::memory& outputMemory, InferenceEngine::Blob::Ptr bptr, buf_info* bi = nullptr);
    void copyInputData(std::shared_ptr<cldnn::network> network, const cldnn::primitive_id &inputName,
                                                const cldnn::layout& inputLayout, const InferenceEngine::Blob &inputBlob,
                                                buf_info* bi = nullptr);

    void AllocateInputs();
    void AllocateOutputs();
    void AllocateInputsDyn();
    void AllocateOutputsDyn();
    void execAndParse();
    void execAndParseDyn();

    void PrepareInput(const cldnn::primitive_id &inputName, const InferenceEngine::Blob &inputBlob);
    void PrepareInputDyn(const cldnn::primitive_id &inputName, const InferenceEngine::Blob &inputBlob);

private:
    static const std::string fp32_suffix;
};

};  // namespace CLDNNPlugin
