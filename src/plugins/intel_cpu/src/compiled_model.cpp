// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.h"

#include <cstring>
#include <utility>

#include "async_infer_request.h"
#include "config.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "graph.h"
#include "infer_request.h"
#include "itt.h"
#include "low_precision/low_precision.hpp"
#include "memory_control.hpp"
#include "memory_state.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/debug_capabilities.h"
#include "utils/memory_stats_dump.hpp"
#include "utils/serialize.hpp"
#include "utils/denormals.hpp"

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_ie_scheduler.hpp"
#endif

using namespace ov::threading;

namespace ov::intel_cpu {

struct ImmediateSerialExecutor : public ov::threading::ITaskExecutor {
    void run(ov::threading::Task task) override {
        std::lock_guard<std::mutex> l{_mutex};
        task();
    }
    std::mutex _mutex;
};

CompiledModel::~CompiledModel() {
    if (m_has_sub_compiled_models) {
        m_sub_compiled_models.clear();
        m_sub_memory_manager->_memorys_table.clear();
    }
    CPU_DEBUG_CAP_ENABLE(dumpMemoryStats(m_cfg.debugCaps, m_name, m_graphs, m_socketWeights));
}

CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             Config cfg,
                             const bool loaded_from_cache,
                             std::shared_ptr<SubMemoryManager> sub_memory_manager)
    : ov::ICompiledModel::ICompiledModel(model, plugin),
      m_model(model),
      m_plugin(plugin),
      m_cfg{std::move(cfg)},
      m_name{model->get_name()},
      m_loaded_from_cache(loaded_from_cache),
      m_sub_memory_manager(std::move(sub_memory_manager)),
      m_model_name(model->get_friendly_name()) {
    m_mutex = std::make_shared<std::mutex>();
    const auto& core = m_plugin->get_core();
    if (!core) {
        OPENVINO_THROW("Unable to get API version. Core is unavailable");
    }


    IStreamsExecutor::Config executor_config;
    if (m_cfg.get_exclusive_async_requests()) {
        // special case when all InferRequests are muxed into a single queue
        m_task_executor = m_plugin->get_executor_manager()->get_executor("CPU");
    } else {
        executor_config = m_cfg.get_num_sub_streams() > 0 ? IStreamsExecutor::Config{"CPUMainStreamExecutor",
                                                                             1,
                                                                             1,
                                                                             ov::hint::SchedulingCoreType::ANY_CORE,
                                                                             false,
                                                                             true}
                                                          : m_cfg.get_stream_executor_config();
        m_task_executor = m_plugin->get_executor_manager()->get_idle_cpu_streams_executor(executor_config);
    }
    if (0 != m_cfg.get_stream_executor_config().get_streams()) {
        m_callback_executor = m_plugin->get_executor_manager()->get_idle_cpu_streams_executor(
            IStreamsExecutor::Config{"CPUCallbackExecutor", 1, 0});
    } else {
        m_callback_executor = m_task_executor;
    }

    if (m_task_executor) {
        set_task_executor(m_task_executor);
    }
    if (m_callback_executor) {
        set_callback_executor(m_callback_executor);
    }

    int streams = std::max(1, executor_config.get_streams());
    std::vector<Task> tasks;
    tasks.resize(streams);
    m_graphs.resize(streams);
    if (executor_config.get_streams() != 0) {
        auto all_graphs_ready = [&] {
            return std::all_of(m_graphs.begin(), m_graphs.end(), [&](Graph& graph) {
                return graph.IsReady();
            });
        };
        do {
            for (auto&& task : tasks) {
                task = [this] {
#if defined(OV_CPU_WITH_ACL)
                    static std::once_flag flag_once;
                    std::call_once(flag_once, [&]() {
                        std::shared_ptr<arm_compute::IScheduler> acl_scheduler = std::make_shared<ACLScheduler>();
                        arm_compute::Scheduler::set(std::static_pointer_cast<arm_compute::IScheduler>(acl_scheduler));
                    });
#endif
                    CompiledModel::get_graph();
                };
            }
            m_task_executor->run_and_wait(tasks);
        } while (!all_graphs_ready());
    } else {
        CompiledModel::get_graph();
    }

    if (m_cfg.get_num_sub_streams() > 0) {
        m_has_sub_compiled_models = true;
        auto message = message_manager();
        m_sub_memory_manager = std::make_shared<SubMemoryManager>(m_cfg.get_num_sub_streams());
        message->set_num_sub_streams(m_cfg.get_num_sub_streams());
        for (int i = 0; i < m_cfg.get_num_sub_streams(); i++) {
            auto sub_cfg = m_cfg.clone(i, true);
            m_sub_compiled_models.push_back(
                std::make_shared<CompiledModel>(model, plugin, sub_cfg, loaded_from_cache, m_sub_memory_manager));
        }
    }
}

static bool set_denormals_optimization(const ov::intel_cpu::DenormalsOptimization& value){
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        if (value.m_mode == DenormalsOptimization::Mode::ON) {
            flush_to_zero(true);
            return denormals_as_zero(true);
        } else if (value.m_mode == DenormalsOptimization::Mode::OFF) {
            flush_to_zero(false);
            denormals_as_zero(false);
        }
    }
    return false;
}

CompiledModel::GraphGuard::Lock CompiledModel::get_graph() const {
    int streamId = 0;
    int socketId = 0;
    auto streamsExecutor = std::dynamic_pointer_cast<IStreamsExecutor>(m_task_executor);
    if (nullptr != streamsExecutor) {
        streamId = streamsExecutor->get_stream_id();
        socketId = std::max(0, streamsExecutor->get_socket_id());
    }
    auto graphLock = GraphGuard::Lock(m_graphs[streamId % m_graphs.size()]);
    if (!graphLock._graph.IsReady()) {
        std::exception_ptr exception;
        auto makeGraph = [&] {
            try {
                GraphContext::Ptr ctx;
                {
                    std::lock_guard<std::mutex> lock{*m_mutex.get()};
                    auto isQuantizedFlag = (m_cfg.get_enable_lp_transformations()) &&
                                           ov::pass::low_precision::LowPrecision::isFunctionQuantized(m_model);
                                               // SSE runtime check is needed for some ATOM machine, which is x86-64 but w/o SSE

                    bool denormalsAsZero = set_denormals_optimization(m_cfg.get_denormals_optimization());
                    ctx = std::make_shared<GraphContext>(m_cfg,
                                                         m_socketWeights[socketId],
                                                         isQuantizedFlag,
                                                         denormalsAsZero,
                                                         streamsExecutor,
                                                         m_sub_memory_manager);
                }

                const std::shared_ptr<const ov::Model> model = m_model;
                graphLock._graph.Init(model, ctx);
                graphLock._graph.Activate();
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
    return std::make_shared<SyncInferRequest>(std::static_pointer_cast<const CompiledModel>(shared_from_this()));
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request =
        std::make_shared<AsyncInferRequest>(std::static_pointer_cast<SyncInferRequest>(internal_request),
                                            get_task_executor(),
                                            get_callback_executor());
    if (m_has_sub_compiled_models) {
        std::vector<std::shared_ptr<IAsyncInferRequest>> requests;
        requests.reserve(m_sub_compiled_models.size());
        for (const auto& model : m_sub_compiled_models) {
            requests.push_back(model->create_infer_request());
        }
        async_infer_request->setSubInferRequest(requests);
        async_infer_request->setSubInfer(true);
    }
    return async_infer_request;
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    if (m_graphs.empty()) {
        OPENVINO_THROW("No graph was found");
    }

    return get_graph()._graph.dump();
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };

    if (name == ov::supported_properties) {
        std::vector<ov::PropertyName> ro_properties{
            RO_property(ov::supported_properties.name()),
            RO_property(ov::model_name.name()),
            RO_property(ov::optimal_number_of_infer_requests.name()),
            RO_property(ov::num_streams.name()),
            RO_property(ov::inference_num_threads.name()),
            RO_property(ov::enable_profiling.name()),
            RO_property(ov::hint::inference_precision.name()),
            RO_property(ov::hint::performance_mode.name()),
            RO_property(ov::hint::execution_mode.name()),
            RO_property(ov::hint::num_requests.name()),
            RO_property(ov::hint::enable_cpu_pinning.name()),
            RO_property(ov::hint::enable_cpu_reservation.name()),
            RO_property(ov::hint::scheduling_core_type.name()),
            RO_property(ov::hint::model_distribution_policy.name()),
            RO_property(ov::hint::enable_hyper_threading.name()),
            RO_property(ov::execution_devices.name()),
            RO_property(ov::intel_cpu::denormals_optimization.name()),
            RO_property(ov::log::level.name()),
            RO_property(ov::intel_cpu::sparse_weights_decompression_rate.name()),
            RO_property(ov::hint::dynamic_quantization_group_size.name()),
            RO_property(ov::hint::kv_cache_precision.name()),
            RO_property(ov::key_cache_precision.name()),
            RO_property(ov::value_cache_precision.name()),
            RO_property(ov::key_cache_group_size.name()),
            RO_property(ov::value_cache_group_size.name()),
        };

        return ro_properties;
    }

    if (name == ov::model_name) {
        return decltype(ov::model_name)::value_type {m_model_name};
    }
    if (name == ov::loaded_from_cache) {
        return decltype(ov::loaded_from_cache)::value_type {m_loaded_from_cache};
    }
    if (name == ov::optimal_number_of_infer_requests) {
        const auto streams = m_cfg.get_stream_executor_config().get_streams();
        return decltype(ov::optimal_number_of_infer_requests)::value_type(
            streams > 0 ? streams : 1);  // ov::optimal_number_of_infer_requests has no negative values
    }
    if (name == ov::execution_devices) {
        return decltype(ov::execution_devices)::value_type{m_plugin->get_device_name()};
    }

    return m_cfg.get_property(name, OptionVisibility::RELEASE);
}

void CompiledModel::export_model(std::ostream& modelStream) const {
    ModelSerializer serializer(modelStream, m_cfg.get_cache_encryption_callbacks().encrypt);
    serializer << m_model;
}

void CompiledModel::release_memory() {
    for (auto&& graph : m_graphs) {
        // try to lock mutex, since it may be already locked (e.g by an infer request)
        std::unique_lock<std::mutex> lock(graph._mutex, std::try_to_lock);
        OPENVINO_ASSERT(lock.owns_lock(),
                        "Attempt to call release_memory() on a compiled model in a busy state. Please ensure that all "
                        "infer requests are completed before releasing memory.");
        auto ctx = graph.getGraphContext();
        ctx->releaseMemory();
    }
}

}  // namespace ov::intel_cpu
