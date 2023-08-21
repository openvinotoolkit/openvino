// Copyright (C) 2018-2023 Intel Corporation
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
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_blob.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov {
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

    Graph(InferenceEngine::CNNNetwork& network,
          const RemoteContextImpl::Ptr& context,
          const ExecutionConfig& config,
          uint16_t stream_id = 0,
          InferenceEngine::InputsDataMap* inputs = nullptr,
          InferenceEngine::OutputsDataMap* outputs = nullptr);
    Graph(cldnn::BinaryInputBuffer& ib,
          const RemoteContextImpl::Ptr& context,
          const ExecutionConfig& config,
          uint16_t stream_id = 0,
          InferenceEngine::InputsDataMap* inputs = nullptr,
          InferenceEngine::OutputsDataMap* outputs = nullptr);
    explicit Graph(std::shared_ptr<Graph> graph, uint16_t stream_id = 0);
    void Export(cldnn::BinaryOutputBuffer &ob);
    std::shared_ptr<ngraph::Function> GetExecGraphInfo();

    bool IsLoaded() const;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const;
    void UpdatePerfStatistics();

    cldnn::engine& get_engine() const { return m_context->get_engine(); }
    const ExecutionConfig& get_config() const { return m_config; }

    size_t GetMaxDynamicBatchSize() const { return m_config.get_property(ov::intel_gpu::max_dynamic_batch);}
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
    RemoteContextImpl::Ptr m_context;
    std::shared_ptr<ProgramBuilder> m_program;
    std::string m_networkName;
    ExecutionConfig m_config;
    uint16_t m_stream_id;
    uint32_t m_state;
    std::condition_variable m_cv;
    std::mutex m_infer_mutex;

    std::vector<std::shared_ptr<cldnn::network>> m_networks;
    std::map<std::string, cldnn::primitive_id> primitiveIDs;
    std::map<std::string, std::vector<cldnn::primitive_id>> prevPrimitiveIDs;

    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;
    std::vector<cldnn::primitive_id> profilingIDs;

    std::map<std::string, InferenceEngine::SizeVector> outputDims;

    std::shared_ptr<cldnn::network> BuildNetwork(std::shared_ptr<cldnn::program> program);
    void Build();
    void UpdateLayersMaps();
    std::shared_ptr<ngraph::Function> GetExecGraphInfoByPrimitivesInfo(std::vector<cldnn::primitive_info>& pi,
                                                                       bool filter_const_primitives = true);
};

}  // namespace intel_gpu
}  // namespace ov
