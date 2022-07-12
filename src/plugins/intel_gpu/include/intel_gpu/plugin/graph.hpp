// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ie_blob.h"
#include "cpp/ie_cnn_network.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/topology.hpp"

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include "intel_gpu/plugin/custom_layer.hpp"
#include "intel_gpu/plugin/device_config.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/program.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

class Graph {
public:
    enum class Stage : uint32_t {
        PREPROC = 1,
        EXECUTE = 2,
        POSTPROC = 4
    };
    typedef std::shared_ptr<Graph> Ptr;
    using variable_states_map = std::map<std::string, std::vector<cldnn::network::VariableState::Ptr>>;

    Graph(InferenceEngine::CNNNetwork& network, InferenceEngine::gpu::ClContext::Ptr context, Config config, uint16_t stream_id = 0);
    explicit Graph(std::shared_ptr<Graph> graph, uint16_t stream_id = 0);
    std::shared_ptr<ngraph::Function> GetExecGraphInfo();

    bool IsLoaded() const;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const;
    void UpdatePerfStatistics();

    const Config& getConfig() const { return m_config; }
    InferenceEngine::gpu::ClContext::Ptr GetContext() { return m_context; }
    std::shared_ptr<cldnn::engine> GetEngine() const { return getContextImpl(m_context)->GetEngine(); }
    int GetMaxDynamicBatchSize() const { return getConfig().max_dynamic_batch; }
    const std::map<std::string, cldnn::layout>& GetInputLayouts() const { return m_program->GetInputLayouts(); }
    const InferenceEngine::InputsDataMap GetNetworkInputs() const { return m_program->GetNetworkInputs(); }
    const InferenceEngine::OutputsDataMap GetNetworkOutputs() const { return m_program->GetNetworkOutputs(); }
    variable_states_map AllocateVariablesMemories();
    std::map<std::string, std::pair<int64_t, int64_t>> GetInputDynBatchDims() { return m_program->m_input_batch_dim; }
    std::map<std::string, int64_t> GetOutputDynBatchDims() { return m_program->m_output_batch_dim; }
    size_t GetNetworksCount() const { return m_networks.size(); }
    std::shared_ptr<cldnn::network> GetNetwork(size_t idx = 0) const;
    InferenceEngine::SizeVector GetOutputSize(std::string outName) const;
    std::string MapOutputName(std::string outName) const;
    std::string getName() const { return m_networkName; }
    void wait(Stage stage_mask) {
        std::unique_lock<std::mutex> lock(m_infer_mutex);
        m_cv.wait(lock, [&] {
            return (m_state & (uint32_t)stage_mask) == 0;
        });
        m_state |= (uint32_t)stage_mask;
    }
    void notify(Stage stage_mask) {
        {
            std::lock_guard<std::mutex> lock(m_infer_mutex);
            m_state &= ~(uint32_t)stage_mask;
        }
        m_cv.notify_one();
    }
    std::mutex& get_mutex() { return m_infer_mutex; }

    bool use_external_queue() const;

protected:
    uint32_t m_state;
    std::condition_variable m_cv;
    std::mutex m_infer_mutex;

    std::string m_networkName;
    Config m_config;

    InferenceEngine::gpu::ClContext::Ptr m_context;
    std::vector<std::shared_ptr<cldnn::network>> m_networks;
    std::map<std::string, cldnn::primitive_id> primitiveIDs;
    std::map<std::string, std::vector<cldnn::primitive_id>> prevPrimitiveIDs;

    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;
    std::vector<cldnn::primitive_id> profilingIDs;

    std::map<std::string, InferenceEngine::SizeVector> outputDims;

    std::shared_ptr<Program> m_program;
    uint16_t m_stream_id;

    std::shared_ptr<cldnn::network> BuildNetwork(std::shared_ptr<cldnn::program> program);
    void Build();
    void UpdateLayersMaps();
    std::shared_ptr<ngraph::Function> GetExecGraphInfoByPrimitivesInfo(std::vector<cldnn::primitive_info>& pi,
                                                                       bool filter_const_primitives = true);
};

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
