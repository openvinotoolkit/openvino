// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <atomic>
#include "cldnn_graph.h"
#include <threading/ie_istreams_executor.hpp>

namespace CLDNNPlugin {

struct buf_info {
    size_t buf_offset;
    size_t buf_size;
};

class CLDNNExecNetwork;

class CLDNNInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<CLDNNInferRequest>;
    // make sure all blobs and cldnn::memory objects
    // are in place and valid
    void checkBlobs() override;
    void InferImpl() override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    CLDNNInferRequest(InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs,
                      const std::shared_ptr<CLDNNExecNetwork>& execNetwork);

    CLDNNInferRequest(const CLDNNInferRequest &) = delete;

    virtual ~CLDNNInferRequest() = default;

    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;
    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) override;

    void SetBatch(int batch = -1) override;
    void SetGraph(std::shared_ptr<CLDNNGraph> graph);
    void EnableProfiling() { m_useProfiling = true; }
    void EnableStreams() { m_useStreams = true; }

    void preprocess();
    void enqueue();
    void wait();

    void preprocess_dynamic();
    void enqueue_dynamic();
    void wait_dynamic();

    bool use_external_queue() const { return m_useExternalQueue; }
    void enable_external_queue() { m_useExternalQueue = true; }

private:
    InferenceEngine::BlobMap _deviceOutputs;
    std::map<std::string, cldnn::primitive_id> inputsMap;
    std::map<std::string, cldnn::primitive_id> outputsMap;

    bool m_useProfiling;
    bool m_useStreams;
    bool m_useExternalQueue;
    std::shared_ptr<CLDNNGraph> m_graph;

    // dynamic batch stuff
    std::map<std::string, std::vector<buf_info>> batchInputs;
    std::map<std::string, std::vector<buf_info>> batchOutputs;
    InferenceEngine::IStreamsExecutor* streamExecutor = nullptr;

    void prepare_input(const cldnn::primitive_id &inputName, InferenceEngine::Blob::Ptr &inputBlob,
                       std::vector<cldnn::event::ptr>& dependencies);
    void prepare_output(const cldnn::primitive_id& outputName, InferenceEngine::Blob::Ptr& outputBlob);

    InferenceEngine::Blob::Ptr create_input_host_blob(const InferenceEngine::TensorDesc& desc, uint8_t* mem_ptr = nullptr);
    InferenceEngine::Blob::Ptr create_output_host_blob(const InferenceEngine::TensorDesc& desc, uint8_t* mem_ptr = nullptr);
    InferenceEngine::Blob::Ptr create_device_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout);

    void copy_output_data(cldnn::memory::ptr outputMemory, InferenceEngine::Blob::Ptr bptr, buf_info* bi = nullptr);
    void copy_input_data(std::shared_ptr<cldnn::network> network, const cldnn::primitive_id &inputName,
                         const cldnn::layout& inputLayout, const InferenceEngine::Blob &inputBlob,
                         buf_info* bi = nullptr);

    void allocate_inputs();
    void allocate_outputs();
    void allocate_inputs_dynamic();
    void allocate_outputs_dynamic();

    std::map<cldnn::primitive_id, cldnn::network_output> internal_outputs;
    std::vector<std::map<cldnn::primitive_id, cldnn::network_output>> internal_outputs_dynamic;
};

};  // namespace CLDNNPlugin
