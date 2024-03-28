// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/op/parameter.hpp"

#include "intel_gpu/plugin/custom_layer.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <cstdint>
#include <mutex>
#include <set>

#if defined(_WIN32) && !defined(__GNUC__)
#    define __PRETTY_FUNCTION__ __FUNCSIG__
#else
#    define __PRETTY_FUNCTION__ __PRETTY_FUNCTION__
#endif

// Forward declarations for cldnn part
namespace cldnn {
enum class activation_func;
struct activation_additional_params;
enum class reduce_mode : uint16_t;
enum class eltwise_mode : int32_t;
}  // namespace cldnn

#define REGISTER_FACTORY_IMPL(op_version, op_name)                                                  \
void __register ## _ ## op_name ## _ ## op_version();                                               \
void __register ## _ ## op_name ## _ ## op_version() {                                              \
    ProgramBuilder::RegisterFactory<ov::op::op_version::op_name>(                                   \
    [](ProgramBuilder& p, const std::shared_ptr<ov::Node>& op) {                                    \
        auto op_casted = std::dynamic_pointer_cast<ov::op::op_version::op_name>(op);                \
        OPENVINO_ASSERT(op_casted, "[GPU] Invalid ov Node type passed into ", __PRETTY_FUNCTION__); \
        Create##op_name##Op(p, op_casted);                                                          \
       });                                                                                          \
}

namespace ov {
namespace intel_gpu {

template<class T>
struct is_smart_pointer : std::false_type {};
template<class T>
struct is_smart_pointer<std::shared_ptr<T>> : std::true_type {};
template<class T>
struct is_smart_pointer<std::shared_ptr<const T>> : std::true_type {};

std::string layer_type_lower(const ov::Node* op);
std::string layer_type_name_ID(const ov::Node* op);
std::string layer_type_lower(const std::shared_ptr<ov::Node>& op);
std::string layer_type_name_ID(const std::shared_ptr<ov::Node>& op);

struct PerfCounter {
    ov::ProfilingInfo::Status status = ov::ProfilingInfo::Status::NOT_RUN;
    bool isCPU = false;
    uint64_t realTime_uSec = 0;
    uint64_t cpu_uSec = 0;
    uint32_t num = 0;
    std::string layerType = "";
    std::string parentPrimitive = "";

    PerfCounter() = default;

    long long realTime_avg() const { return (num == 0) ? 0 : realTime_uSec / num; }
    long long cpu_avg() const { return (num == 0) ? 0 : cpu_uSec / num; }
};

class ProgramBuilder final {
public:
    ProgramBuilder(std::shared_ptr<ov::Model> model, cldnn::engine& engine, const ExecutionConfig& config,
            bool createTopologyOnly = false, bool partialBuild = false,
            std::shared_ptr<ov::threading::IStreamsExecutor> task_executor = nullptr,
            std::shared_ptr<cldnn::ICompilationContext> compilation_context = nullptr,
            bool innerProgram = false);
    ProgramBuilder(cldnn::engine& engine, const ExecutionConfig& config);

    static const cldnn::primitive_id m_preProcessTag;
    static const cldnn::primitive_id m_preCustomLayerTag;
    static const cldnn::primitive_id m_postCustomLayerTag;

    std::map<std::string, cldnn::primitive_id> primitive_ids;
    std::map<size_t, std::vector<cldnn::primitive_id>> inputPrimitiveIDs;
    std::map<size_t, cldnn::primitive_id> prevPrimitiveIDs;
    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;

    std::vector<cldnn::primitive_id> profiling_ids;

    std::map<size_t, cldnn::layout> inputLayouts;
    using BlobCacheKey = std::tuple<const char*, ov::Shape, ov::element::Type>;
    std::map<BlobCacheKey, cldnn::primitive_id> blobMemCache;

    std::shared_ptr<cldnn::program> get_compiled_program() const;
    std::shared_ptr<cldnn::topology> get_topology() const { return m_topology; }

    const std::map<size_t, cldnn::layout>& get_input_layouts() const { return inputLayouts; }
    cldnn::engine& get_engine() const { return m_engine; }
    const ExecutionConfig& get_config() const { return m_config; }

    int64_t get_parameter_index(const std::shared_ptr<ov::op::v0::Parameter>& parameter) const;
    int64_t get_result_index(const ov::Output<ov::Node>& value) const;
    int64_t get_result_index(const ov::Output<const ov::Node>& value) const;

    bool is_op_supported(const std::shared_ptr<ov::Node>& op);

    // Profiling utils
    void init_profile_info(const cldnn::primitive& prim);

    // Graph construction helpers
    std::vector<cldnn::input_info> GetInputInfo(const std::shared_ptr<ov::Node>& op) const;

    using factory_t = std::function<void(ProgramBuilder&, const std::shared_ptr<ov::Node>&)>;
    using factories_map_t = std::map<ov::DiscreteTypeInfo, factory_t>;

    template<typename OpType>
    static void RegisterFactory(factory_t func) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (ProgramBuilder::factories_map.find(OpType::get_type_info_static()) == ProgramBuilder::factories_map.end()) {
            ProgramBuilder::factories_map.insert({OpType::get_type_info_static(), func});
        }
    }

    template<typename PType, typename = typename std::enable_if<!is_smart_pointer<PType>::value>::type>
    void add_primitive(const ov::Node& op, PType prim, std::vector<std::string> aliases = {}) {
        add_primitive(op, std::static_pointer_cast<cldnn::primitive>(std::make_shared<PType>(prim)), std::move(aliases));
    }

    void add_primitive(const ov::Node& op, std::shared_ptr<cldnn::primitive> prim, std::vector<std::string> aliases = {});

    bool use_new_shape_infer() const { return allow_new_shape_infer; }
    bool requires_new_shape_infer(const std::shared_ptr<ov::Node>& op) const;
    bool is_inner_program() const { return m_is_inner_program; }
    bool is_query_mode() { return queryMode; }

    std::shared_ptr<ov::threading::IStreamsExecutor> get_task_executor() const { return m_task_executor; }
    std::shared_ptr<cldnn::ICompilationContext> get_compilation_context() const { return m_compilation_context; }

private:
    static factories_map_t factories_map;
    std::shared_ptr<cldnn::program> m_program;
    std::shared_ptr<ov::Model> m_model;
    ExecutionConfig m_config;
    cldnn::engine& m_engine;
    static std::mutex m_mutex;

    std::shared_ptr<cldnn::topology> m_topology;
    CustomLayerMap m_custom_layers;

    bool allow_new_shape_infer = false;

    bool queryMode;

    std::shared_ptr<ov::threading::IStreamsExecutor> m_task_executor;
    std::shared_ptr<cldnn::ICompilationContext> m_compilation_context;

    bool m_is_inner_program = false;

    void EnableQueryMode() { queryMode = true; }
    void DisableQueryMode() { queryMode = false; }

    void prepare_build();
    void cleanup_build();

    // TODO(eunsoo): remove createTopolpgyOnly argument and add another method to create topology from ngraph function
    std::shared_ptr<cldnn::program> build(const std::vector<std::shared_ptr<ov::Node>>& ops,
                                          bool createTopologyOnly = false, bool partialBuild = false, bool innerProgram = false);

    void CreateSingleLayerPrimitive(cldnn::topology& topology, const std::shared_ptr<ov::Node>& op);
};

void CreatePagedAttention(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op);
void CreateCustomOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& node, CustomLayerPtr customLayer);
void CreateUnaryEltwiseOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& node,
                          cldnn::activation_func func, cldnn::activation_additional_params params);
void CreateElementwiseOp(ProgramBuilder& p,
                         const std::shared_ptr<ov::Node>& node,
                         cldnn::eltwise_mode mode,
                         std::vector<float> coefficients = {},
                         bool pythondiv = true);

bool IsNodeOnConstPath(const std::shared_ptr<ov::Node>& node);

void validate_inputs_count(const std::shared_ptr<ov::Node>& op, std::vector<size_t> possible_inputs_count);

inline bool ends_with(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size())
        return false;
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

}  // namespace intel_gpu
}  // namespace ov
