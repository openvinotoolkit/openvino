// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ie_metric_helpers.hpp"  // must be included first

#include "compiled_model.h"
#include "async_infer_request.h"
#include "infer_request.h"
#include "itt.h"
#include "low_precision/low_precision.hpp"
#include "memory_state.h"
#include "nodes/memory.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "serialize.h"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/transformation_pipeline.h"
#define FIX_62820 0
#if FIX_62820 && ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
#    include <openvino/runtime/threading/tbb_streams_executor.hpp>
#endif

#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "transformations/utils/utils.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstring>
#include <utility>

using namespace ov::threading;

namespace ov {
namespace intel_cpu {

struct ImmediateSerialExecutor : public ov::threading::ITaskExecutor {
    void run(ov::threading::Task task) override {
        std::lock_guard<std::mutex> l{_mutex};
        task();
    }
    std::mutex _mutex;
};

CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const Config& cfg,
                             const ExtensionManager::Ptr& extMgr,
                             const bool loaded_from_cache)
    : ov::ICompiledModel::ICompiledModel(model, plugin),
      m_model(model),
      m_plugin(plugin),
      m_cfg{cfg},
      extensionManager(extMgr),
      m_name{model->get_name()},
      m_loaded_from_cache(loaded_from_cache) {
    bool isFloatModel = !ov::op::util::has_op_with_type<ov::op::v0::FakeQuantize>(m_model);
    m_mutex = std::make_shared<std::mutex>();
    const auto& core = m_plugin->get_core();
    if (!core)
        OPENVINO_THROW("Unable to get API version. Core is unavailable");
    m_cfg.isLegacyApi = !core->is_new_api();

    if (cfg.exclusiveAsyncRequests) {
        // special case when all InferRequests are muxed into a single queue
        m_task_executor = m_plugin->get_executor_manager()->get_executor("CPU");
    } else {
        auto streamsExecutorConfig =
            is_cpu_map_available()
                ? m_cfg.streamExecutorConfig
                : IStreamsExecutor::Config::make_default_multi_threaded(m_cfg.streamExecutorConfig, isFloatModel);
        streamsExecutorConfig._name = "CPUStreamsExecutor";
        m_cfg.streamExecutorConfig._threads = streamsExecutorConfig._threads;
#if FIX_62820 && (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
        m_task_executor = std::make_shared<TBBStreamsExecutor>(streamsExecutorConfig);
#else
        m_task_executor = m_plugin->get_executor_manager()->get_idle_cpu_streams_executor(streamsExecutorConfig);
#endif
    }
    if (0 != cfg.streamExecutorConfig._streams) {
#if FIX_62820 && (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
        // There is no additional threads but we still need serialize callback execution to preserve legacy behaviour
        m_callback_executor = std::make_shared<ImmediateSerialExecutor>();
#else
        m_callback_executor = m_plugin->get_executor_manager()->get_idle_cpu_streams_executor(
            IStreamsExecutor::Config{"CPUCallbackExecutor", 1, 0, IStreamsExecutor::ThreadBindingType::NONE});
#endif
    } else {
        m_callback_executor = m_task_executor;
    }

    if (m_task_executor)
        set_task_executor(m_task_executor);
    if (m_callback_executor)
        set_callback_executor(m_callback_executor);

    int streams = std::max(1, m_cfg.streamExecutorConfig._streams);
    std::vector<Task> tasks;
    tasks.resize(streams);
    m_graphs.resize(streams);
    if (m_cfg.streamExecutorConfig._streams != 0) {
        auto all_graphs_ready = [&] {
            return std::all_of(m_graphs.begin(), m_graphs.end(), [&](Graph& graph) {
                return graph.IsReady();
            });
        };
        do {
            for (auto&& task : tasks) {
                task = [this] {
                    CompiledModel::get_graph();
                };
            }
            m_task_executor->run_and_wait(tasks);
        } while (!all_graphs_ready());
    } else {
        CompiledModel::get_graph();
    }
}

CompiledModel::GraphGuard::Lock CompiledModel::get_graph() const {
    int streamId = 0;
    int socketId = 0;
    auto streamsExecutor = std::dynamic_pointer_cast<IStreamsExecutor>(m_task_executor);
    if (nullptr != streamsExecutor) {
        streamId = streamsExecutor->get_stream_id();
        socketId = streamsExecutor->get_socket_id();
    }
    auto graphLock = GraphGuard::Lock(m_graphs[streamId % m_graphs.size()]);
    if (!graphLock._graph.IsReady()) {
        std::exception_ptr exception;
        auto makeGraph = [&] {
            try {
                GraphContext::Ptr ctx;
                {
                    std::lock_guard<std::mutex> lock{*m_mutex.get()};
                    // disable weights caching if graph was created only once
                    auto weightsCache = m_cfg.streamExecutorConfig._streams != 1 ? m_socketWeights[socketId] : nullptr;
                    auto isQuantizedFlag =
                        (m_cfg.lpTransformsMode == Config::On) &&
                        ov::pass::low_precision::LowPrecision::isFunctionQuantized(m_model);

                    ctx = std::make_shared<GraphContext>(m_cfg, extensionManager, weightsCache, isQuantizedFlag);
                }
                const std::shared_ptr<const ov::Model> model = m_model;
                graphLock._graph.CreateGraph(model, ctx);
            } catch (...) {
                exception = std::current_exception();
            }
        };
        if (nullptr != streamsExecutor) {
            streamsExecutor->execute(makeGraph);
        } else {
            makeGraph();
        }
        if (exception) {
            std::rethrow_exception(exception);
        }
    }
    return graphLock;
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    m_numRequests++;
    return std::make_shared<SyncInferRequest>(std::static_pointer_cast<const CompiledModel>(shared_from_this()));
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request =
        std::make_shared<AsyncInferRequest>(std::static_pointer_cast<SyncInferRequest>(internal_request),
                                            get_task_executor(),
                                            get_callback_executor());
    return async_infer_request;
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    if (m_graphs.empty())
        OPENVINO_THROW("No graph was found");

    return get_graph()._graph.dump();
}

ov::Any CompiledModel::get_metric_legacy(const std::string& name, const GraphGuard& graph) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, graph.dump()->get_friendly_name());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto&& key : graph.getConfig()._config) {
            configKeys.push_back(key.first);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        Config engConfig = graph.getConfig();
        auto option = engConfig._config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
        OPENVINO_ASSERT(option != engConfig._config.end());
        auto streams = std::stoi(option->second);
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(streams ? streams : 1));
    } else {
        OPENVINO_THROW("Unsupported property: ", name);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    if (m_graphs.empty())
        OPENVINO_THROW("No graph was found");

    if (name == ov::loaded_from_cache) {
        return m_loaded_from_cache;
    }

    Config engConfig = get_graph()._graph.getConfig();
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        return option->second;
    }

    // @todo Can't we just use local copy (_cfg) instead?
    auto graphLock = get_graph();
    const auto& graph = graphLock._graph;
    const auto& config = graph.getConfig();

    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };

    if (name == ov::supported_properties) {
        return std::vector<ov::PropertyName>{
            RO_property(ov::supported_properties.name()),
            RO_property(ov::model_name.name()),
            RO_property(ov::optimal_number_of_infer_requests.name()),
            RO_property(ov::num_streams.name()),
            RO_property(ov::affinity.name()),
            RO_property(ov::inference_num_threads.name()),
            RO_property(ov::enable_profiling.name()),
            RO_property(ov::hint::inference_precision.name()),
            RO_property(ov::hint::performance_mode.name()),
            RO_property(ov::hint::execution_mode.name()),
            RO_property(ov::hint::num_requests.name()),
            RO_property(ov::hint::enable_cpu_pinning.name()),
            RO_property(ov::hint::scheduling_core_type.name()),
            RO_property(ov::hint::enable_hyper_threading.name()),
            RO_property(ov::execution_devices.name()),
            RO_property(ov::intel_cpu::denormals_optimization.name()),
            RO_property(ov::intel_cpu::sparse_weights_decompression_rate.name()),
        };
    }

    if (name == ov::model_name) {
        // @todo Does not seem ok to 'dump()' the whole graph everytime in order to get a name
        const std::string modelName = graph.dump()->get_friendly_name();
        return decltype(ov::model_name)::value_type(modelName);
    } else if (name == ov::optimal_number_of_infer_requests) {
        const auto streams = config.streamExecutorConfig._streams;
        return decltype(ov::optimal_number_of_infer_requests)::value_type(
            streams);  // ov::optimal_number_of_infer_requests has no negative values
    } else if (name == ov::num_streams) {
        const auto streams = config.streamExecutorConfig._streams;
        return decltype(ov::num_streams)::value_type(
            streams);  // ov::num_streams has special negative values (AUTO = -1, NUMA = -2)
    } else if (name == ov::affinity) {
        const auto affinity = config.streamExecutorConfig._threadBindingType;
        switch (affinity) {
        case IStreamsExecutor::ThreadBindingType::NONE:
            return ov::Affinity::NONE;
        case IStreamsExecutor::ThreadBindingType::CORES:
            return ov::Affinity::CORE;
        case IStreamsExecutor::ThreadBindingType::NUMA:
            return ov::Affinity::NUMA;
        case IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
            return ov::Affinity::HYBRID_AWARE;
        }
        return ov::Affinity::NONE;
    } else if (name == ov::inference_num_threads) {
        const auto num_threads = config.streamExecutorConfig._threads;
        return decltype(ov::inference_num_threads)::value_type(num_threads);
    } else if (name == ov::enable_profiling.name()) {
        const bool perfCount = config.collectPerfCounters;
        return decltype(ov::enable_profiling)::value_type(perfCount);
    } else if (name == ov::hint::inference_precision) {
        return decltype(ov::hint::inference_precision)::value_type(config.inferencePrecision);
    } else if (name == ov::hint::performance_mode) {
        const auto perfHint = ov::util::from_string(config.perfHintsConfig.ovPerfHint, ov::hint::performance_mode);
        return perfHint;
    } else if (name == ov::hint::enable_cpu_pinning.name()) {
        const bool use_pin = config.enableCpuPinning;
        return decltype(ov::hint::enable_cpu_pinning)::value_type(use_pin);
    } else if (name == ov::hint::scheduling_core_type) {
        const auto core_type = config.schedulingCoreType;
        return core_type;
    } else if (name == ov::hint::enable_hyper_threading.name()) {
        const bool use_ht = config.enableHyperThreading;
        return decltype(ov::hint::enable_hyper_threading)::value_type(use_ht);
    } else if (name == ov::hint::execution_mode) {
        return config.executionMode;
    } else if (name == ov::hint::num_requests) {
        const auto perfHintNumRequests = config.perfHintsConfig.ovPerfHintNumRequests;
        return decltype(ov::hint::num_requests)::value_type(perfHintNumRequests);
    } else if (name == ov::execution_devices) {
        return decltype(ov::execution_devices)::value_type{m_plugin->get_device_name()};
    } else if (name == ov::intel_cpu::denormals_optimization) {
        return decltype(ov::intel_cpu::denormals_optimization)::value_type(config.denormalsOptMode ==
                                                                           Config::DenormalsOptMode::DO_On);
    } else if (name == ov::intel_cpu::sparse_weights_decompression_rate) {
        return decltype(ov::intel_cpu::sparse_weights_decompression_rate)::value_type(
            config.fcSparseWeiDecompressionRate);
    }
    /* Internally legacy parameters are used with new API as part of migration procedure.
     * This fallback can be removed as soon as migration completed */
    return get_metric_legacy(name, graph);
}

void CompiledModel::export_model(std::ostream& modelStream) const {
    ModelSerializer serializer(modelStream, extensionManager);
    serializer << m_model;
}

}  // namespace intel_cpu
}  // namespace ov
