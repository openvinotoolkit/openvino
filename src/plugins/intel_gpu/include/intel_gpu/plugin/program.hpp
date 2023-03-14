// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <cstdint>
#include <mutex>
#include <set>

#include <cpp/ie_cnn_network.h>
#include <ngraph/ngraph.hpp>
#include "gpu/gpu_config.hpp"


#include "intel_gpu/plugin/custom_layer.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"

// Forward declarations for cldnn part
namespace cldnn {
enum class activation_func;
struct activation_additional_params;
enum class reduce_mode : uint16_t;
enum class eltwise_mode : int32_t;
}  // namespace cldnn

#define REGISTER_FACTORY_IMPL(op_version, op_name)                                                \
void __register ## _ ## op_name ## _ ## op_version();                                             \
void __register ## _ ## op_name ## _ ## op_version() {                                            \
    Program::RegisterFactory<ov::op::op_version::op_name>(                                        \
    [](Program& p, const std::shared_ptr<ov::Node>& op) {                                         \
        auto op_casted = std::dynamic_pointer_cast<ov::op::op_version::op_name>(op);              \
        if (!op_casted)                                                                           \
            IE_THROW() << "Invalid ov Node type passed into " << __PRETTY_FUNCTION__;             \
        Create##op_name##Op(p, op_casted);                                                        \
       });                                                                                        \
}

namespace ov {
namespace intel_gpu {

template<class T>
struct is_smart_pointer : std::false_type {};
template<class T>
struct is_smart_pointer<std::shared_ptr<T>> : std::true_type {};
template<class T>
struct is_smart_pointer<std::shared_ptr<const T>> : std::true_type {};

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
    PerfCounter()
    : status(InferenceEngine::InferenceEngineProfileInfo::NOT_RUN)
    , isCPU(false)
    , realTime_uSec(0)
    , cpu_uSec(0)
    , num(0) {}

    long long realTime_avg() const { return (num == 0) ? 0 : realTime_uSec / num; }
    long long cpu_avg() const { return (num == 0) ? 0 : cpu_uSec / num; }
};

class Program {
public:
    Program(InferenceEngine::CNNNetwork& network, cldnn::engine& engine, const ExecutionConfig& config,
            bool createTopologyOnly = false, bool partialBuild = false);
    Program(cldnn::engine& engine, const ExecutionConfig& config)
        : m_max_batch(1)
        , m_curBatch(-1)
        , m_config(config)
        , m_engine(engine)
        , queryMode(false) {}

    static const cldnn::primitive_id m_preProcessTag;
    static const cldnn::primitive_id m_meanValuesTag;
    static const cldnn::primitive_id m_workaroundTag;
    static const cldnn::primitive_id m_preCustomLayerTag;
    static const cldnn::primitive_id m_postCustomLayerTag;

    std::map<std::string, cldnn::primitive_id> primitive_ids;
    std::map<std::string, std::vector<cldnn::primitive_id>> prevPrimitiveIDs;
    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;

    std::vector<cldnn::primitive_id> profiling_ids;

    std::map<std::string, InferenceEngine::SizeVector> outputDims;
    std::map<std::string, cldnn::layout> inputLayouts;
    using BlobCacheKey = std::pair<const char*, std::vector<size_t>>;
    std::map<BlobCacheKey, cldnn::primitive_id> blobMemCache;
    CustomLayerMap m_custom_layers;

    int m_max_batch;
    int m_curBatch;
    std::map<std::string, std::pair<int64_t, int64_t>> m_input_batch_dim;
    std::map<std::string, int64_t> m_output_batch_dim;

    std::shared_ptr<cldnn::program> GetCompiledProgram(int program_id = 0);
    const std::map<std::string, cldnn::layout>& GetInputLayouts() const { return inputLayouts; }
    InferenceEngine::InputsDataMap GetNetworkInputs() const { return m_networkInputs; }
    InferenceEngine::OutputsDataMap GetNetworkOutputs() const { return m_networkOutputs; }
    cldnn::engine& get_engine() const { return m_engine; }
    const ExecutionConfig& get_config() const { return m_config; }
    int GetMaxBatchSizeForSingleProgram();

    bool IsOpSupported(const InferenceEngine::CNNNetwork& network, const std::shared_ptr<ngraph::Node>& op);
    bool IsDynBatchModel(const std::shared_ptr<ov::Model>& model,
                         std::map<std::string, ov::PartialShape>& shapes,
                         std::map<std::string, std::pair<int64_t, int64_t>>& batch_dim);

    // Profiling utils
    void init_profile_info(const cldnn::primitive& prim);

    // Graph construction helpers
    std::vector<cldnn::input_info> GetInputInfo(const std::shared_ptr<ngraph::Node>& op) const;

    using factory_t = std::function<void(Program&, const std::shared_ptr<ngraph::Node>&)>;
    using factories_map_t = std::map<ngraph::DiscreteTypeInfo, factory_t>;

    template<typename OpType>
    static void RegisterFactory(factory_t func) {
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);
        if (Program::factories_map.find(OpType::get_type_info_static()) == Program::factories_map.end())
            Program::factories_map.insert({OpType::get_type_info_static(), func});
    }

    template<typename PType, typename = typename std::enable_if<!is_smart_pointer<PType>::value>::type>
    void add_primitive(const ngraph::Node& op, PType prim, std::vector<std::string> aliases = {}) {
        add_primitive(op, std::static_pointer_cast<cldnn::primitive>(std::make_shared<PType>(prim)), aliases);
    }

    void add_primitive(const ngraph::Node& op, std::shared_ptr<cldnn::primitive> prim, std::vector<std::string> aliases = {});

    std::shared_ptr<cldnn::topology> GetTopology() const { return m_topology; }

    using variables_state_info_map = std::map<std::string, std::set<cldnn::layout>>;

    void AddVariableStateInfo(const std::string& variable_id, const cldnn::layout& layout);

    const variables_state_info_map& GetVariablesStatesInfo() const { return m_variablesStateInfo; }

    bool use_new_shape_infer() const { return allow_new_shape_infer; }

private:
    static factories_map_t factories_map;
    std::vector<std::shared_ptr<cldnn::program>> m_programs;
    ExecutionConfig m_config;
    cldnn::engine& m_engine;

    std::shared_ptr<cldnn::topology> m_topology;
    InferenceEngine::InputsDataMap m_networkInputs;
    InferenceEngine::OutputsDataMap m_networkOutputs;
    variables_state_info_map m_variablesStateInfo;

    bool allow_new_shape_infer = false;

    bool queryMode;

    void EnableQueryMode() { queryMode = true; }
    void DisableQueryMode() { queryMode = false; }

    void PrepareBuild(InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs);
    void CleanupBuild();

    // TODO(eunsoo): remove createTopolpgyOnly argument and add another method to create topology from ngraph function
    std::shared_ptr<cldnn::program> BuildProgram(const std::vector<std::shared_ptr<ngraph::Node>>& ops,
                                                 InferenceEngine::InputsDataMap networkInputs,
                                                 InferenceEngine::OutputsDataMap networkOutputs,
                                                 bool createTopologyOnly = false, bool partialBuild = false);

    void CreateSingleLayerPrimitive(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op);
    void ChangeInputBatch(int batch);
};

void CreateCustomOp(Program& p, const std::shared_ptr<ngraph::Node>& node, CustomLayerPtr customLayer);
void CreateUnaryEltwiseOp(Program& p, const std::shared_ptr<ngraph::Node>& node,
                          cldnn::activation_func func, cldnn::activation_additional_params params);
void CreateElementwiseOp(Program& p,
                         const std::shared_ptr<ngraph::Node>& node,
                         cldnn::eltwise_mode mode,
                         std::vector<float> coefficients = {});

bool IsNodeOnConstPath(const std::shared_ptr<ngraph::Node>& node);

void validate_inputs_count(const std::shared_ptr<ngraph::Node>& op, std::vector<size_t> possible_inputs_count);

inline bool ends_with(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size())
        return false;
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

}  // namespace intel_gpu
}  // namespace ov
