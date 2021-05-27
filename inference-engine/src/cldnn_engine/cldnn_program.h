// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <cstdint>
#include <mutex>

#include <cpp/ie_cnn_network.h>

#include "cldnn_config.h"

#include <api/engine.hpp>
#include <api/topology.hpp>

// Forward declarations for cldnn part
namespace cldnn {
enum class activation_func;
struct activation_additional_params;
enum class reduce_mode : uint16_t;
enum class eltwise_mode : int32_t;
}  // namespace cldnn

// Forward declarations for ngraph part
namespace ngraph {
class Node;
class DiscreteTypeInfo;
}  // namespace ngraph

#define REGISTER_FACTORY_IMPL(op_version, op_name)                                                \
void __register ## _ ## op_name ## _ ## op_version() {                                            \
    Program::RegisterFactory<ngraph::op::op_version::op_name>(                                    \
    [](Program& p, const std::shared_ptr<ngraph::Node>& op) {                                     \
        auto op_casted = std::dynamic_pointer_cast<ngraph::op::op_version::op_name>(op);          \
        if (!op_casted)                                                                           \
            IE_THROW() << "Invalid ngraph Node type passed into " << __PRETTY_FUNCTION__; \
        Create##op_name##Op(p, op_casted);                                                        \
       });                                                                                        \
}

namespace CLDNNPlugin {

std::string layer_type_lower(const ngraph::Node* op);
std::string layer_type_name_ID(const ngraph::Node* op);
std::string layer_type_lower(const std::shared_ptr<ngraph::Node>& op);
std::string layer_type_name_ID(const std::shared_ptr<ngraph::Node>& op);

struct PerfCounter {
    InferenceEngine::InferenceEngineProfileInfo::LayerStatus status;
    bool isCPU;
    uint64_t realTime_uSec;
    uint64_t cpu_uSec;
    uint32_t num;
    std::string layerType;
    std::string parentPrimitive;

public:
    PerfCounter() : realTime_uSec(0), cpu_uSec(0), num(0),
                    status(InferenceEngine::InferenceEngineProfileInfo::NOT_RUN), isCPU(false) {}

    long long realTime_avg() const { return (num == 0) ? 0 : realTime_uSec / num; }
    long long cpu_avg() const { return (num == 0) ? 0 : cpu_uSec / num; }
};

class Program {
public:
    Program(InferenceEngine::CNNNetwork& network, std::shared_ptr<const cldnn::engine> engine, const Config& config, bool createTopologyOnly = false);
    Program(std::shared_ptr<const cldnn::engine> engine, const Config& config) : m_config(config), m_engine(engine),
            m_curBatch(-1), queryMode(false), m_max_batch(1) {}
    Program() : m_config({}), m_engine(nullptr), m_curBatch(-1), queryMode(false), m_max_batch(1) {}

    static const cldnn::primitive_id m_preProcessTag;
    static const cldnn::primitive_id m_meanValuesTag;
    static const cldnn::primitive_id m_workaroundTag;
    static const cldnn::primitive_id m_preCustomLayerTag;
    static const cldnn::primitive_id m_postCustomLayerTag;

    std::map<std::string, cldnn::primitive_id> primitiveIDs;
    std::map<cldnn::primitive_id, std::vector<std::string>> primitivesToIRLayersMap;
    std::map<cldnn::primitive_id, std::string> IRToNgraphLayersMap;
    std::map<std::string, std::vector<cldnn::primitive_id>> prevPrimitiveIDs;
    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;

    std::vector<cldnn::primitive_id> profilingIDs;

    std::map<std::string, InferenceEngine::SizeVector> outputDims;
    std::map<std::string, cldnn::layout> inputLayouts;
    using BlobCacheKey = std::pair<const char*, std::vector<size_t>>;
    std::map<BlobCacheKey, cldnn::primitive_id> blobMemCache;

    int m_max_batch;
    int m_curBatch;

    std::shared_ptr<cldnn::program> GetCompiledProgram(int program_id = 0);
    const std::map<std::string, cldnn::layout>& GetInputLayouts() const { return inputLayouts; }
    InferenceEngine::InputsDataMap GetNetworkInputs() const { return m_networkInputs; }
    InferenceEngine::OutputsDataMap GetNetworkOutputs() const { return m_networkOutputs; }
    const cldnn::engine& GetEngine() const { return *m_engine; }
    std::shared_ptr<const cldnn::engine> GetEnginePtr() const { return m_engine; }
    const Config& GetConfig() const { return m_config; }
    int GetMaxBatchSizeForSingleProgram();

    bool IsOpSupported(const InferenceEngine::CNNNetwork& network, const std::shared_ptr<ngraph::Node>& op);

    // Profiling utils
    void InitProfileInfo(const std::string& layerName,
                         const std::string& layerType,
                         bool isCPU = false,
                         InferenceEngine::InferenceEngineProfileInfo::LayerStatus status
                         = InferenceEngine::InferenceEngineProfileInfo::EXECUTED,
                         std::string parentId = "");
    void AddPrimitiveToProfiler(cldnn::primitive_id id, const std::shared_ptr<ngraph::Node>& op,
                                cldnn::primitive_id customOutputId = "");
    void AddPrimitiveToProfiler(const std::shared_ptr<ngraph::Node>& op,
                                cldnn::primitive_id customOutputId = "");
    void AddInnerPrimitiveToProfiler(cldnn::primitive_id id, cldnn::primitive_id parentId,
                                     const std::shared_ptr<ngraph::Node>& op);

    // Graph construction helpers
    void ValidateInputs(const std::shared_ptr<ngraph::Node>& op, std::vector<size_t> validInputsCount);
    std::vector<cldnn::primitive_id> GetInputPrimitiveIDs(const std::shared_ptr<ngraph::Node>& op) const;

    using factory_t = std::function<void(Program&, const std::shared_ptr<ngraph::Node>&)>;
    using factories_map_t = std::map<ngraph::DiscreteTypeInfo, factory_t>;

    template<typename OpType, typename std::enable_if<std::is_base_of<ngraph::Node, OpType>::value, int>::type = 0>
    static void RegisterFactory(factory_t func) {
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);
        if (Program::factories_map.find(OpType::type_info) == Program::factories_map.end())
            Program::factories_map.insert({OpType::type_info, func});
    }

    template<typename PType>
    void AddPrimitive(PType prim) {
        if (m_topology == nullptr) {
            IE_THROW() << "m_topology object was not created in clDNNPlugin::Program";
        }

        m_topology->add(prim);
    }

    std::shared_ptr<cldnn::topology> GetTopology() const { return m_topology; }

private:
    static factories_map_t factories_map;
    std::vector<std::shared_ptr<cldnn::program>> m_programs;
    std::shared_ptr<const cldnn::engine> m_engine;
    Config m_config;

    std::shared_ptr<cldnn::topology> m_topology;
    InferenceEngine::InputsDataMap m_networkInputs;
    InferenceEngine::OutputsDataMap m_networkOutputs;

    bool queryMode;

    void EnableQueryMode() { queryMode = true; }
    void DisableQueryMode() { queryMode = false; }

    void PrepareBuild(InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs);
    void CleanupBuild();

    // TODO(eunsoo): remove createTopolpgyOnly argument and add another method to create topology from ngraph function
    std::shared_ptr<cldnn::program> BuildProgram(const std::vector<std::shared_ptr<ngraph::Node>>& ops,
                                                 InferenceEngine::InputsDataMap networkInputs,
                                                 InferenceEngine::OutputsDataMap networkOutputs,
                                                 bool createTopologyOnly = false);

    void CreateSingleLayerPrimitive(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op);
    bool CanProcessDynBatch(std::vector<std::shared_ptr<ngraph::Node>> ops, InferenceEngine::InputsDataMap networkInputs) const;
    void ChangeInputBatch(int batch);
};

void CreateCustomOp(Program& p, const std::shared_ptr<ngraph::Node>& node, CLDNNCustomLayerPtr customLayer);
void CreateUnaryEltwiseOp(Program& p, const std::shared_ptr<ngraph::Node>& node,
                          cldnn::activation_func func, cldnn::activation_additional_params params);
void CreateElementwiseOp(Program& p, const std::shared_ptr<ngraph::Node>& node, cldnn::eltwise_mode mode);

bool IsNodeOnConstPath(const std::shared_ptr<ngraph::Node>& node);

}  // namespace CLDNNPlugin
