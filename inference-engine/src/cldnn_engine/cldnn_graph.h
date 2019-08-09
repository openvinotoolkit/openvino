// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "ie_blob.h"
#include "ie_plugin.hpp"
#include "cpp/ie_cnn_network.h"
#include "debug_options.h"
#include "inference_engine.hpp"
#include <CPP/network.hpp>
#include <CPP/memory.hpp>
#include <CPP/primitive.hpp>
#include <CPP/topology.hpp>
#include <CPP/pooling.hpp>
#include <CPP/eltwise.hpp>
#include <CPP/concatenation.hpp>
#include <CPP/detection_output.hpp>
#include <CPP/softmax.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <CPP/upsampling.hpp>
#include "cldnn_custom_layer.h"
#include "cldnn_config.h"
#include "cldnn_program.h"

namespace CLDNNPlugin {

class CLDNNGraph {
public:
    typedef std::shared_ptr<CLDNNGraph> Ptr;

    explicit CLDNNGraph(InferenceEngine::ICNNNetwork& network, const Config& config = {}, uint16_t stream_id = 0);
    explicit CLDNNGraph(std::shared_ptr<CLDNNGraph> graph, uint16_t stream_id = 0);
    void GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr& graphPtr);

    bool IsLoaded() const;

    static bool IsLayerSupported(const std::string& type) {
        return Program::LayerTypeFromStr(type) != Program::NO_TYPE;
    }

    void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const;
    void UpdatePerfStatistics();

    int GetMaxDynamicBatchSize() const { return m_config.max_dynamic_batch; }
    const std::map<std::string, cldnn::layout>& GetInputLayouts() const { return m_program->getInputLayouts(); }
    std::shared_ptr<const cldnn::engine> GetEngine() const { return m_engine; }
    size_t GetNetworksCount() const { return m_networks.size(); }
    std::shared_ptr<cldnn::network> GetNetwork(size_t idx = 0) const;
    InferenceEngine::SizeVector GetOutputSize(std::string outName) const;
    std::string MapOutputName(std::string outName) const;
    std::string getName() const { return m_networkName; }
    const Config& getConfig() const { return m_config; }

protected:
    std::string m_networkName;

    std::shared_ptr<const cldnn::engine> m_engine;
    std::vector<std::shared_ptr<cldnn::network>> m_networks;
    std::map<std::string, cldnn::primitive_id> primitiveIDs;
    std::map<cldnn::primitive_id, std::vector<std::string>> primitivesToIRLayersMap;
    std::map<std::string, std::vector<cldnn::primitive_id>> prevPrimitiveIDs;

    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;
    std::map<cldnn::primitive_id, std::string> implementationsMap;
    std::vector<cldnn::primitive_id> profilingIDs;

    std::map<std::string, InferenceEngine::SizeVector> outputDims;

    std::shared_ptr<Program> m_program;
    Config m_config;
    uint16_t m_stream_id;

    std::shared_ptr<cldnn::network> BuildNetwork(std::shared_ptr<cldnn::program> program);
    void Build();
    void UpdateLayersMaps();
    void UpdateImplementationsMap();
    InferenceEngine::ICNNNetwork::Ptr GetExecGraphInfoByPrimitivesInfo(std::vector<cldnn::primitive_info>& pi,
                                                                       bool filter_const_primitives = true);
};

}  // namespace CLDNNPlugin
