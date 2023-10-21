// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/legacy_api_helper.hpp"
#include "intel_gpu/plugin/legacy_remote_context.hpp"

#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/util/common_util.hpp"

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/async_infer_request.hpp"

#include <fstream>
#include <utility>
#include <sys/types.h>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace ov {
namespace intel_gpu {

namespace {
std::shared_ptr<ov::threading::ITaskExecutor> create_task_executor(const std::shared_ptr<const ov::IPlugin>& plugin, const ExecutionConfig& config) {
    if (config.get_property(ov::internal::exclusive_async_requests)) {
        //exclusive_async_requests essentially disables the streams (and hence should be checked first) => aligned with the CPU behavior
        return plugin->get_executor_manager()->get_executor("GPU");
    } else if (config.get_property(ov::hint::enable_cpu_pinning)) {
        auto executor_config =
            ov::threading::IStreamsExecutor::Config{"Intel GPU plugin executor",
                                                    0,
                                                    0,
                                                    ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
                                                    1,
                                                    0,
                                                    0,
                                                    ov::threading::IStreamsExecutor::Config::PreferredCoreType::BIG,
                                                    {{config.get_property(ov::num_streams), MAIN_CORE_PROC, 1, 0, 0}},
                                                    true};
        auto post_config = ov::threading::IStreamsExecutor::Config::reserve_cpu_threads(executor_config);
        return std::make_shared<ov::threading::CPUStreamsExecutor>(post_config);
    } else {
        return std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"Intel GPU plugin executor", config.get_property(ov::num_streams)});
    }
}
}  // namespace

CompiledModel::CompiledModel(std::shared_ptr<ov::Model> model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             RemoteContextImpl::Ptr context,
                             const ExecutionConfig& config)
    : ov::ICompiledModel(model,
                         plugin,
                         wrap_if_old_api(context, plugin->is_new_api()),
                         create_task_executor(plugin, config))
    , m_context(context)
    , m_config(config)
    , m_wait_executor(std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{"Intel GPU plugin wait executor"}))
    , m_model(model->is_dynamic() ? model : nullptr)
    , m_model_name(model->get_friendly_name())
    , m_inputs(ov::ICompiledModel::inputs())
    , m_outputs(ov::ICompiledModel::outputs())
    , m_loaded_from_cache(false) {
    auto graph_base = std::make_shared<Graph>(model, m_context, m_config, 0);
    for (uint16_t n = 0; n < m_config.get_property(ov::num_streams); n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<Graph>(graph_base, n);
        m_graphs.push_back(graph);
    }
}

CompiledModel::CompiledModel(cldnn::BinaryInputBuffer ib,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             RemoteContextImpl::Ptr context,
                             const ExecutionConfig& config)
    : ov::ICompiledModel(nullptr,
                         plugin,
                         wrap_if_old_api(context, plugin->is_new_api()),
                         create_task_executor(plugin, config),
                         nullptr)
    , m_context(context)
    , m_config(config)
    , m_wait_executor(std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{"Intel GPU plugin wait executor"}))
    , m_model(nullptr)
    , m_model_name("")
    , m_loaded_from_cache(true) {
    {
        size_t num_params;
        ib >> num_params;

        for (size_t idx = 0; idx < num_params; ++idx) {
            std::string param_name;
            ib >> param_name;
            ov::element::Type param_element_type;
            std::string str_element_type;
            ib >> str_element_type;
            std::stringstream oss(str_element_type);
            oss >> param_element_type;
            ov::PartialShape param_shape;
            ib >> param_shape;
            std::unordered_set<std::string> param_names;
            size_t num_names;
            ib >> num_names;
            for (size_t i = 0; i < num_names; ++i) {
                std::string name;
                ib >> name;
                param_names.emplace(name);
            }

            auto new_param = std::make_shared<ov::op::v0::Parameter>(param_element_type, param_shape);
            new_param->set_friendly_name(param_name);
            new_param->set_element_type(param_element_type);
            new_param->output(0).get_tensor().set_names(param_names);
            new_param->validate_and_infer_types();
            m_inputs.push_back(new_param->output(0));
        }
    }

    {
        size_t num_results;
        ib >> num_results;

        for (size_t idx = 0; idx < num_results; ++idx) {
            ov::element::Type fake_element_type;
            std::string str_element_type;
            ib >> str_element_type;
            std::stringstream oss(str_element_type);
            oss >> fake_element_type;

            ov::PartialShape fake_shape;
            ib >> fake_shape;

            std::string fake_name;
            ib >> fake_name;

            std::string param_name;
            ib >> param_name;

            std::unordered_set<std::string> param_names;
            size_t num_names;
            ib >> num_names;
            for (size_t i = 0; i < num_names; ++i) {
                std::string name;
                ib >> name;
                param_names.emplace(name);
            }

            auto fake_param = std::make_shared<ov::op::v0::Parameter>(fake_element_type, fake_shape);
            fake_param->set_friendly_name(fake_name);
            fake_param->validate_and_infer_types();

            auto new_result = std::make_shared<ov::op::v0::Result>(fake_param);
            new_result->set_friendly_name(param_name);
            new_result->output(0).get_tensor().set_names(param_names);
            new_result->validate_and_infer_types();
            m_outputs.push_back(new_result->output(0));
        }
    }

    auto pos = ib.tellg();
    for (uint16_t n = 0; n < m_config.get_property(ov::num_streams); n++) {
        ib.seekg(pos);
        auto graph = std::make_shared<Graph>(ib, context, m_config, 0);
        m_graphs.push_back(graph);
    }
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    auto sync_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<AsyncInferRequest>(std::static_pointer_cast<SyncInferRequest>(sync_request),
                                                                   get_task_executor(),
                                                                   m_wait_executor,
                                                                   get_callback_executor());
    return async_infer_request;
}

// Cache blob format:
//     [ is_dynamic flag ]
//     [ ov::Node::Input/ ov::Node::Output ]
//     [ ov::intel_gpu::Graph ]
void CompiledModel::export_model(std::ostream& model) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::export_model");
    OPENVINO_ASSERT(!m_graphs.empty(), "[GPU] Model not loaded");

    cldnn::BinaryOutputBuffer ob(model);

    bool is_dynamic = get_graph(0)->get_network()->is_dynamic();

    ob << is_dynamic;
    // Inputs
    {
        const auto& params = inputs();
        ob << params.size();

        for (const auto& param : params) {
            std::stringstream ss;
            ss << param.get_element_type();

            ob << param.get_node()->get_friendly_name();
            ob << ss.str();
            ob << param.get_partial_shape();
            ob << param.get_names().size();
            for (const auto& name : param.get_names()) {
                ob << name;
            }
        }
    }

    // Outputs
    {
        const auto& results = outputs();
        ob << results.size();

        for (const auto& param : results) {
            std::stringstream ss;
            ss << param.get_element_type();

            ob << ss.str();
            ob << param.get_partial_shape();
            ob << param.get_node()->get_input_node_ptr(0)->get_friendly_name();
            ob << param.get_node()->get_friendly_name();
            ob << param.get_names().size();
            for (const auto& name : param.get_names()) {
                ob << name;
            }
        }
    }

    if (is_dynamic) {
        ov::pass::StreamSerialize serializer(model, {}, ov::pass::Serialize::Version::UNSPECIFIED);
        serializer.run_on_model(m_model);
    } else {
        get_graph(0)->export_model(ob);
    }
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    return get_graph(0)->get_runtime_model();
}
const std::vector<std::shared_ptr<Graph>>& CompiledModel::get_graphs() const {
    return m_graphs;
}
std::shared_ptr<Graph> CompiledModel::get_graph(size_t n) const {
    OPENVINO_ASSERT(m_graphs.size() >= n, "[GPU] Invalid graph idx: ", n, ". Only ", m_graphs.size(), " were created");
    return m_graphs[n];
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {
            // Metrics
            ov::PropertyName{ov::supported_properties.name(), PropertyMutability::RO},
            ov::PropertyName{ov::model_name.name(), PropertyMutability::RO},
            ov::PropertyName{ov::optimal_number_of_infer_requests.name(), PropertyMutability::RO},

            // Configs
            ov::PropertyName{ov::enable_profiling.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::enable_cpu_pinning.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::model_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::host_task_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::queue_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::queue_throttle.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::enable_loop_unrolling.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::disable_winograd_convolution.name(), PropertyMutability::RO},
            ov::PropertyName{ov::cache_dir.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RO},
            ov::PropertyName{ov::num_streams.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RO},
            ov::PropertyName{ov::device::id.name(), PropertyMutability::RO},
            ov::PropertyName{ov::execution_devices.name(), PropertyMutability::RO}
        };
    } else if (name == ov::model_name) {
        return decltype(ov::model_name)::value_type {m_model_name};
    } else if (name == ov::loaded_from_cache) {
        return decltype(ov::loaded_from_cache)::value_type {m_loaded_from_cache};
    OPENVINO_SUPPRESS_DEPRECATED_START
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        static const std::vector<std::string> metrics {
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
        };
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        static const std::vector<std::string> config_keys {
            CONFIG_KEY(MODEL_PRIORITY),
            CONFIG_KEY(PERFORMANCE_HINT),
            CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS),
            CONFIG_KEY(PERF_COUNT),
            CONFIG_KEY(CONFIG_FILE),
            CONFIG_KEY(DEVICE_ID),
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),
            CONFIG_KEY(CACHE_DIR),
            CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
            GPU_CONFIG_KEY(PLUGIN_PRIORITY),
            GPU_CONFIG_KEY(PLUGIN_THROTTLE),
            GPU_CONFIG_KEY(HOST_TASK_PRIORITY),
            GPU_CONFIG_KEY(NV12_TWO_INPUTS),
            GPU_CONFIG_KEY(MAX_NUM_THREADS),
            GPU_CONFIG_KEY(ENABLE_LOOP_UNROLLING),
        };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, config_keys);
    OPENVINO_SUPPRESS_DEPRECATED_END
    } else if (name == ov::optimal_number_of_infer_requests) {
        unsigned int nr = m_config.get_property(ov::num_streams);
        if (m_config.get_property(ov::hint::performance_mode) != ov::hint::PerformanceMode::LATENCY)
            nr *= 2;
        return decltype(ov::optimal_number_of_infer_requests)::value_type {nr};
    } else if (name == ov::execution_devices) {
        return decltype(ov::execution_devices)::value_type{m_context->get_device_name()};
    }

    auto actual_name = name;
    if (LegacyAPIHelper::is_legacy_property({name, nullptr}, is_new_api())) {
        actual_name = LegacyAPIHelper::convert_legacy_property({name, nullptr}).first;
    }

    auto val = m_config.get_property(actual_name);
    if (LegacyAPIHelper::is_legacy_property({name, nullptr}, is_new_api())) {
        val = LegacyAPIHelper::convert_to_legacy_property({actual_name, val}).second;
    }

    return val;
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::create_sync_infer_request");
    OPENVINO_ASSERT(!m_graphs.empty(), "[GPU] Model not loaded");

    for (auto& graph : m_graphs) {
        OPENVINO_ASSERT(graph != nullptr, "[GPU] Model not loaded: graph is nullptr");
        OPENVINO_ASSERT(graph->is_loaded(), "[GPU] Model not loaded: invalid graph");
    }

    return std::make_shared<SyncInferRequest>(std::static_pointer_cast<const CompiledModel>(shared_from_this()));
}

}  // namespace intel_gpu
}  // namespace ov
