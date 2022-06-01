// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <atomic>
#include "intel_gpu/plugin/graph.hpp"
#include <threading/ie_istreams_executor.hpp>

namespace ov {
namespace runtime {
namespace intel_gpu {

struct buf_info {
    size_t buf_offset;
    size_t buf_size;
};

class CompiledModel;

class InferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<InferRequest>;
    // make sure all blobs and cldnn::memory objects
    // are in place and valid
    void checkBlobs() override;
    void InferImpl() override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    InferRequest(InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs,
                 const std::shared_ptr<CompiledModel>& execNetwork);
    InferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                 const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                 const std::shared_ptr<CompiledModel>& execNetwork);

    InferRequest(const InferRequest &) = delete;

    virtual ~InferRequest() = default;

    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;
    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) override;
    void SetBlobs(const std::string& name, const std::vector<InferenceEngine::Blob::Ptr> &data) override;

    void SetBatch(int batch = -1) override;
    void SetGraph(std::shared_ptr<Graph> graph);
    void EnableProfiling() { m_useProfiling = true; }
    void EnableStreams() { m_useStreams = true; }

    void setup_stream_graph();
    void preprocess_notify();
    void enqueue_notify();
    void wait_notify();

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

    std::map<std::string, std::vector<InferenceEngine::Blob::Ptr>> inputTensorsMap;

    bool m_useProfiling = false;
    bool m_useStreams = false;
    bool m_useExternalQueue = false;
    std::shared_ptr<Graph> m_graph;

    // dynamic batch stuff
    std::map<std::string, std::vector<buf_info>> batchInputs;
    std::map<std::string, std::vector<buf_info>> batchOutputs;
    InferenceEngine::IStreamsExecutor* streamExecutor = nullptr;

    void prepare_input(const cldnn::primitive_id &inputName, InferenceEngine::Blob::Ptr &inputBlob,
                       std::vector<cldnn::event::ptr>& dependencies);
    void prepare_output(const cldnn::primitive_id& outputName, InferenceEngine::Blob::Ptr& outputBlob);

    InferenceEngine::Blob::Ptr create_host_blob(const InferenceEngine::TensorDesc& desc,
                                                std::shared_ptr<InferenceEngine::IAllocator> alloc = nullptr);
    InferenceEngine::Blob::Ptr create_device_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout);

    void copy_output_data(cldnn::memory::ptr outputMemory, InferenceEngine::Blob::Ptr bptr, buf_info* bi = nullptr);
    void copy_input_data(std::shared_ptr<cldnn::network> network, const cldnn::primitive_id &inputName,
                         const cldnn::layout& inputLayout, const InferenceEngine::Blob &inputBlob,
                         buf_info* bi = nullptr);

    InferenceEngine::Blob::Ptr create_shared_device_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout, void* usm_host_mem);
    void allocate_inputs();
    void allocate_outputs();
    void allocate_inputs_dynamic();
    void allocate_outputs_dynamic();

    std::map<cldnn::primitive_id, cldnn::network_output> internal_outputs;
    std::vector<std::map<cldnn::primitive_id, cldnn::network_output>> internal_outputs_dynamic;
    Graph::variable_states_map variables_states_;
};

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
