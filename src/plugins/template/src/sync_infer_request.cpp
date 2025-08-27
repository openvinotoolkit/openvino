
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "sync_infer_request.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "compiled_model.hpp"
#include "itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/variable_context.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "perf_counter.hpp"
#include "plugin.hpp"
#include "remote_tensor.hpp"
#include "template/remote_tensor.hpp"
#include "variable_state.hpp"

using Time = std::chrono::high_resolution_clock;

namespace {

void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor,
                          const ov::element::Type& element_type,
                          const ov::Shape& shape) {
    if (!tensor || tensor->get_element_type() != element_type) {
        tensor = ov::make_tensor(element_type, shape);
    } else {
        tensor->set_shape(shape);
    }
}

void collect_variables(const std::shared_ptr<ov::Model>& ov_model,
                       ov::op::util::VariableContext& variable_context,
                       std::vector<ov::SoPtr<ov::IVariableState>>& list_of_variables) {
    for (const auto& op : ov_model->get_ordered_ops()) {
        if (auto multi_subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_graph : multi_subgraph_op->get_functions()) {
                collect_variables(sub_graph, variable_context, list_of_variables);
            }
        }
    }
    for (const auto& variable : ov_model->get_variables()) {
        if (!variable_context.get_variable_value(variable)) {
            const auto& shape = variable->get_info().data_shape.is_dynamic()
                                    ? ov::Shape{0}
                                    : variable->get_info().data_shape.to_shape();
            ov::Tensor tensor = ov::Tensor(variable->get_info().data_type, shape);
            variable_context.set_variable_value(variable, std::make_shared<ov::op::util::VariableValue>(tensor));
            auto state =
                std::make_shared<ov::template_plugin::VariableState>(variable->get_info(),
                                                                     variable_context.get_variable_value(variable));
            list_of_variables.emplace_back(state);
        }
    }
}

struct PortsDiscovery {
    std::vector<ov::template_plugin::ScoresPort> scores;
    std::vector<ov::template_plugin::LayerPorts> layers;
};

static PortsDiscovery discover_ports(const std::shared_ptr<const ov::template_plugin::CompiledModel>& cm) {
    PortsDiscovery out;
    auto runtime = cm->get_runtime_model();
    const auto& results = runtime->get_results();
    size_t layer_id = 0;

    for (size_t i = 0; i < results.size(); ++i) {
        auto res = results[i];
        if (res->get_output_size() != 1)
            continue;
        auto src = res->input_value(0);
        auto pat = std::dynamic_pointer_cast<ov::op::PagedAttentionExtension>(src.get_node_shared_ptr());
        if (!pat)
            continue;
        // by contract: second output (index 1) is scores
        if (src.get_index() != 1)
            continue;

        ov::template_plugin::LayerPorts lp;
        lp.k_param = pat->input_value(3);
        lp.v_param = pat->input_value(4);
        out.layers.push_back(lp);

        ov::template_plugin::ScoresPort sp{};
        sp.result_index = i;
        sp.layer_id = layer_id++;
        out.scores.push_back(sp);
    }
    return out;
}

}  // namespace

namespace ov {
namespace template_plugin {

// ! [infer_request:ctor]
InferRequest::InferRequest(const std::shared_ptr<const ov::template_plugin::CompiledModel>& model)
    : ov::ISyncInferRequest(model) {
    auto requestID = std::to_string(model->m_request_id.fetch_add(1));

    std::string name = model->m_model->get_friendly_name() + "_Req" + requestID;
    m_profiling_task = {
        openvino::itt::handle("Template" + std::to_string(model->m_cfg.device_id) + "_" + name + "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(model->m_cfg.device_id) + "_" + name + "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(model->m_cfg.device_id) + "_" + name + "_StartPipeline"),
        openvino::itt::handle("Template" + std::to_string(model->m_cfg.device_id) + "_" + name + "_WaitPipline"),
    };
    m_durations = {};
    m_executable = model->get_template_plugin()->m_backend->compile(model->m_model);

    // Allocate plugin backend specific memory handles
    m_backend_input_tensors.resize(get_inputs().size());
    m_backend_output_tensors.resize(get_outputs().size());

    // Allocate input/output tensors
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 input.get_element_type(),
                                 input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 output.get_element_type(),
                                 output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }

    // Save variable states
    ov::op::util::VariableContext variable_context;
    const auto& ov_model = m_executable->get_model();
    collect_variables(ov_model, variable_context, m_variable_states);
    m_eval_context.emplace("VariableContext", variable_context);

    // ---- NEW: Cache manager and eviction configuration ----
    m_cache_mgr = model->get_or_create_cache_manager_locked();  // method added in CompiledModel
    auto ports = discover_ports(model);
    m_scores_ports = std::move(ports.scores);
    m_layer_ports = std::move(ports.layers);

    const size_t block_size = m_cache_mgr ? m_cache_mgr->get_block_size() : 32;
    const size_t num_layers = m_cache_mgr ? m_cache_mgr->get_num_decoder_layers() : m_scores_ports.size();
    const auto& ecfg = model->m_eviction_cfg;  // assumed member in CompiledModel
    m_eviction =
        std::make_unique<ov::cache::CacheEvictionAlgorithm>(ecfg, block_size, num_layers, ecfg.snapkv_window_size);
    m_cfg_max_kv_blocks = 0;  // keep 0 unless you pass a property to set it
}
// ! [infer_request:ctor]

// ! [infer_request:dtor]
InferRequest::~InferRequest() = default;
// ! [infer_request:dtor]

// ! [infer_request:set_tensors_impl]
void InferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                    const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    for (const auto& input : get_inputs()) {
        if (input == port) {
            m_batched_tensors[input.get_tensor_ptr()] = tensors;
            return;
        }
    }
    OPENVINO_THROW("Cannot find input tensors for port ", port);
}
// ! [infer_request:set_tensors_impl]

// ! [infer_request:query_state]
std::vector<ov::SoPtr<ov::IVariableState>> InferRequest::query_state() const {
    return m_variable_states;
}
// ! [infer_request:query_state]

std::shared_ptr<const ov::template_plugin::CompiledModel> InferRequest::get_template_model() const {
    auto& compiled_model = get_compiled_model();
    auto template_model = std::dynamic_pointer_cast<const ov::template_plugin::CompiledModel>(compiled_model);
    OPENVINO_ASSERT(template_model);
    return template_model;
}

// ! [infer_request:infer]
void InferRequest::infer() {
    // TODO: fill with actual list of pipeline stages, which are executed synchronously for sync infer requests
    infer_preprocess();
    start_pipeline();
    wait_pipeline();  // does nothing in current implementation
    infer_postprocess();
}
// ! [infer_request:infer]

// ! [infer_request:infer_preprocess]
void InferRequest::infer_preprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[Preprocess]);
    auto start = Time::now();

    // NEW: bind KV cache before tensor export
    ensure_kv_cache_bound();

    convert_batched_tensors();
    check_tensors();

    // Allocate backend tensors
    OPENVINO_ASSERT(get_inputs().size() == m_backend_input_tensors.size());
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto tensor = get_tensor(get_inputs()[i]);
        if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
            auto vector_tensor = std::dynamic_pointer_cast<ov::template_plugin::VectorImpl>(tensor._ptr);
            OPENVINO_ASSERT(vector_tensor, "Template plugin supports only VectorTensor with remote context.");
            auto element_type = vector_tensor->get_element_type();
            void* data = vector_tensor->get_data();
            OPENVINO_ASSERT(data != nullptr);
            // Create backend tenor
            m_backend_input_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(element_type,
                                                                                      vector_tensor->get_shape(),
                                                                                      data);
        } else if (tensor->is_continuous()) {
            // No ROI extraction is needed
            m_backend_input_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor->get_element_type(),
                                                                                      tensor->get_shape(),
                                                                                      tensor->data());
        } else {
            OPENVINO_ASSERT(tensor->get_element_type().bitwidth() % 8 == 0,
                            "Template plugin: Unsupported ROI tensor with element type having ",
                            std::to_string(tensor->get_element_type().bitwidth()),
                            " bits size");
            ov::Shape shape = tensor->get_shape();
            // Perform manual extraction of ROI tensor
            // Basic implementation doesn't take axis order into account `desc.getBlockingDesc().getOrder()`
            // Performance of manual extraction is not optimal, but it is ok for template implementation
            m_backend_input_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor->get_element_type(),
                                                                                      tensor->get_shape());
            tensor->copy_to(ov::get_tensor_impl(m_backend_input_tensors[i])._ptr);
        }
    }
    // Tensors can be dynamic, so in this case we need to allocate tensors with right shape
    OPENVINO_ASSERT(get_outputs().size() == m_backend_output_tensors.size());
    for (size_t i = 0; i < get_outputs().size(); i++) {
        const auto& result = get_template_model()->m_model->get_results()[i];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            m_backend_output_tensors[i] = get_template_model()->get_template_plugin()->m_backend->create_tensor();
            continue;
        }
        auto tensor = make_tensor(get_tensor(get_outputs()[i]));
        if (tensor.is_continuous() && !tensor.is<ov::RemoteTensor>())
            m_backend_output_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor.get_element_type(),
                                                                                      tensor.get_shape(),
                                                                                      tensor.data());
        else
            m_backend_output_tensors[i] =
                get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor.get_element_type(),
                                                                                      tensor.get_shape());
    }
    m_durations[Preprocess] = Time::now() - start;
}
// ! [infer_request:infer_preprocess]

// ! [infer_request:start_pipeline]
void InferRequest::start_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[StartPipeline])
    auto start = Time::now();
    m_executable->call(m_backend_output_tensors,
                       m_backend_input_tensors,
                       m_eval_context,
                       get_template_model()->m_cfg.perf_count);
    m_durations[StartPipeline] = Time::now() - start;
}
// ! [infer_request:start_pipeline]

// ! [infer_request:wait_pipeline]
void InferRequest::wait_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[WaitPipeline])
    auto start = Time::now();
    // TODO: Wait pipeline using driver API or other synchronizations methods
    // NOTE: not used in current implementation since `startPipeline` executes pipiline synchronously
    m_durations[WaitPipeline] = Time::now() - start;
}
// ! [infer_request:wait_pipeline]

// ! [infer_request:infer_postprocess]
void InferRequest::infer_postprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[Postprocess]);
    auto start = Time::now();
    OPENVINO_ASSERT(get_outputs().size() == m_backend_output_tensors.size());
    for (size_t i = 0; i < get_outputs().size(); i++) {
        const auto& result = get_template_model()->m_model->get_results()[i];
        const auto& host_tensor = m_backend_output_tensors[i];
        auto tensor = get_tensor(get_outputs()[i]);
        if (result->get_output_partial_shape(0).is_dynamic()) {
            ov::Output<const ov::Node> output{result->output(0).get_node(), result->output(0).get_index()};
            allocate_tensor(output, [&host_tensor](ov::SoPtr<ov::ITensor>& tensor) {
                allocate_tensor_impl(tensor, host_tensor.get_element_type(), host_tensor.get_shape());
                host_tensor.copy_to(ov::make_tensor(tensor));
            });
        } else if (!tensor->is_continuous()) {
            host_tensor.copy_to(ov::make_tensor(tensor));
        } else if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
            auto vector_tensor = std::dynamic_pointer_cast<ov::template_plugin::VectorImpl>(tensor._ptr);
            OPENVINO_ASSERT(vector_tensor, "Template plugin supports only VectorTensor with remote context.");
            void* data = vector_tensor->get_data();
            // Copy to vector
            std::memcpy(data, host_tensor.data(), tensor->get_byte_size());
        }
    }

    // NEW: harvest scores & run eviction policy
    register_scores_and_evict();

    m_durations[Postprocess] = Time::now() - start;
}
// ! [infer_request:infer_postprocess]

// ! [infer_request:get_profiling_info]
std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    const auto fill_profiling_info = [](const std::string& name,
                                        const std::chrono::duration<float, std::micro>& time) -> ov::ProfilingInfo {
        ov::ProfilingInfo p_info;
        p_info.status = ov::ProfilingInfo::Status::EXECUTED;
        p_info.node_name = name;
        p_info.cpu_time = p_info.real_time = std::chrono::duration_cast<std::chrono::milliseconds>(time);
        return p_info;
    };

    info.emplace_back(fill_profiling_info("input preprocessing", m_durations[Preprocess]));
    info.emplace_back(fill_profiling_info("execution time", m_durations[StartPipeline]));
    auto template_model = get_template_model();
    for (const auto& op : template_model->get_runtime_model()->get_ops()) {
        auto rt_info = op->get_rt_info();
        const auto& it = rt_info.find(ov::runtime::interpreter::PERF_COUNTER_NAME);
        OPENVINO_ASSERT(it != rt_info.end(), "Operation ", op, " doesn't contain performance counter");
        auto counter = it->second.as<std::shared_ptr<ov::runtime::interpreter::PerfCounter>>();
        info.emplace_back(fill_profiling_info(op->get_friendly_name(), counter->duration()));
    }
    info.emplace_back(fill_profiling_info("output postprocessing", m_durations[Postprocess]));

    return info;
}
// ! [infer_request:get_profiling_info]

// ! [infer_request:cancel]
void InferRequest::cancel() {
    m_executable->cancel();
}
// ! [infer_request:cancel]

// ---- NEW HELPERS ----
void InferRequest::ensure_kv_cache_bound() {
    if (!m_cache_mgr)
        return;
    // If you want to cap total pages externally, set m_cfg_max_kv_blocks via property elsewhere
    if (m_cfg_max_kv_blocks > 0)
        m_cache_mgr->allocate_cache_if_needed(m_cfg_max_kv_blocks);

    const size_t L = std::min(m_layer_ports.size(), m_cache_mgr->get_num_decoder_layers());
    for (size_t l = 0; l < L; ++l) {
        const auto& lp = m_layer_ports[l];
        auto kc = m_cache_mgr->get_key_cache(l);
        auto vc = m_cache_mgr->get_value_cache(l);
        if (kc)
            set_tensor(lp.k_param, ov::get_tensor_impl(kc));
        if (vc)
            set_tensor(lp.v_param, ov::get_tensor_impl(vc));
    }
}

void InferRequest::register_scores_and_evict() {
    if (!m_eviction)
        return;

    ov::cache::AttentionScoresForEachDecoderLayer per_layer_scores;
    per_layer_scores.resize(m_scores_ports.size());
    for (const auto& sp : m_scores_ports) {
        // pick the host-side tensor that corresponds to this Result
        per_layer_scores[sp.layer_id] = m_backend_output_tensors.at(sp.result_index);
    }

    std::set<size_t> skipped{};
    m_eviction->register_new_token_scores(per_layer_scores, skipped);

    // In this skeleton we just query and drop; your next step would be to
    // translate logical evictions into physical page ops via CacheManager.
    auto evicted = m_eviction->evict_logical_blocks();
    (void)evicted;
}

}  // namespace template_plugin
}  // namespace ov
