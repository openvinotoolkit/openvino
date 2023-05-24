// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_infer_request.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

// #include "itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/util/variable_context.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "plugin.hpp"
#include "compiled_model.hpp"
// #include "template/remote_tensor.hpp"
// #include "variable_state.hpp"

using Time = std::chrono::high_resolution_clock;

namespace {

void allocate_tensor_impl(ov::Tensor& tensor, const ov::element::Type& element_type, const ov::Shape& shape) {
    if (!tensor || tensor.get_element_type() != element_type) {
        tensor = ov::Tensor(element_type, shape);
    } else {
        tensor.set_shape(shape);
    }
}

}  // namespace

ov::hetero::InferRequest::InferRequest(const std::shared_ptr<const ov::hetero::CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    
    int index = 0;
    for (auto&& subnetwork : compiled_model->m_networks) {
        InferRequest::SubRequestDesc desc;
        desc._network = subnetwork._network;
        // desc._profilingTask = openvino::itt::handle("Infer" + std::to_string(index++));
        desc._request = {desc._network->create_infer_request(), desc._network._so};

        
        // Allocate input/output tensors
        for (const auto& input : subnetwork._network->inputs()) {
            allocate_tensor(input, [input](ov::Tensor& tensor) {
                // Can add a check to avoid double work in case of shared tensors
                allocate_tensor_impl(tensor,
                                    input.get_element_type(),
                                    input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
            });
        }
        for (const auto& output : subnetwork._network->outputs()) {
            allocate_tensor(output, [output](ov::Tensor& tensor) {
                // Can add a check to avoid double work in case of shared tensors
                allocate_tensor_impl(tensor,
                                    output.get_element_type(),
                                    output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
            });
        }    
        
        m_infer_requests.push_back(desc);
    }
}

// ! [infer_request:dtor]
ov::hetero::InferRequest::~InferRequest() = default;
// ! [infer_request:dtor]

// ! [infer_request:set_tensors_impl]
void ov::hetero::InferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                                         const std::vector<ov::Tensor>& tensors) {
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
std::vector<std::shared_ptr<ov::IVariableState>> ov::hetero::InferRequest::query_state() const {
    return m_variable_states;
}
// ! [infer_request:query_state]

std::shared_ptr<const ov::hetero::CompiledModel> ov::hetero::InferRequest::get_hetero_model()
    const {
    auto& compiled_model = get_compiled_model();
    auto hetero_model = std::dynamic_pointer_cast<const ov::hetero::CompiledModel>(compiled_model);
    OPENVINO_ASSERT(hetero_model);
    return hetero_model;
}

// ! [infer_request:infer]
void ov::hetero::InferRequest::infer() {
    // TODO: fill with actual list of pipeline stages, which are executed synchronously for sync infer requests
    infer_preprocess();
    start_pipeline();
    wait_pipeline();  // does nothing in current implementation
    infer_postprocess();
}
// ! [infer_request:infer]

// ! [infer_request:infer_preprocess]
void ov::hetero::InferRequest::infer_preprocess() {
    // OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[Preprocess]);
    // auto start = Time::now();
    // convert_batched_tensors();
    // check_tensors();

    // // Allocate backend tensors
    // OPENVINO_ASSERT(get_inputs().size() == m_backend_input_tensors.size());
    // for (size_t i = 0; i < get_inputs().size(); i++) {
    //     auto tensor = get_tensor(get_inputs()[i]);
    //     if (tensor.is<ov::RemoteTensor>()) {
    //         OPENVINO_ASSERT(tensor.is<ov::hetero::VectorTensor>(),
    //                         "Template plugin supports only VectorTensor with remote context.");
    //         auto vector_tensor = tensor.as<ov::hetero::VectorTensor>();
    //         auto element_type = vector_tensor.get_element_type();
    //         void* data = vector_tensor.get_data();
    //         OPENVINO_ASSERT(data != nullptr);
    //         // Create backend tenor
    //         m_backend_input_tensors[i] =
    //             get_template_model()->get_template_plugin()->m_backend->create_tensor(element_type,
    //                                                                                   vector_tensor.get_shape(),
    //                                                                                   data);
    //     } else if (tensor.is_continuous()) {
    //         // No ROI extraction is needed
    //         m_backend_input_tensors[i] =
    //             get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor.get_element_type(),
    //                                                                                   tensor.get_shape(),
    //                                                                                   tensor.data());
    //     } else {
    //         OPENVINO_ASSERT(tensor.get_element_type().bitwidth() % 8 == 0,
    //                         "Template plugin: Unsupported ROI tensor with element type having ",
    //                         std::to_string(tensor.get_element_type().bitwidth()),
    //                         " bits size");
    //         ov::Shape shape = tensor.get_shape();
    //         // Perform manual extraction of ROI tensor
    //         // Basic implementation doesn't take axis order into account `desc.getBlockingDesc().getOrder()`
    //         // Performance of manual extraction is not optimal, but it is ok for template implementation
    //         m_backend_input_tensors[i] =
    //             get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor.get_element_type(),
    //                                                                                   tensor.get_shape());
    //         tensor.copy_to(m_backend_input_tensors[i]);
    //     }
    // }
    // // Tensors can be dynamic, so in this case we need to allocate tensors with right shape
    // OPENVINO_ASSERT(get_outputs().size() == m_backend_output_tensors.size());
    // for (size_t i = 0; i < get_outputs().size(); i++) {
    //     const auto& result = get_template_model()->m_model->get_results()[i];
    //     if (result->get_output_partial_shape(0).is_dynamic()) {
    //         m_backend_output_tensors[i] = get_template_model()->get_template_plugin()->m_backend->create_tensor();
    //         continue;
    //     }
    //     auto tensor = get_tensor(get_outputs()[i]);
    //     if (tensor.is_continuous() && !tensor.is<ov::RemoteTensor>())
    //         m_backend_output_tensors[i] =
    //             get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor.get_element_type(),
    //                                                                                   tensor.get_shape(),
    //                                                                                   tensor.data());
    //     else
    //         m_backend_output_tensors[i] =
    //             get_template_model()->get_template_plugin()->m_backend->create_tensor(tensor.get_element_type(),
    //                                                                                   tensor.get_shape());
    // }
    // m_durations[Preprocess] = Time::now() - start;
}
// ! [infer_request:infer_preprocess]

// ! [infer_request:start_pipeline]
void ov::hetero::InferRequest::start_pipeline() {
    // OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[StartPipeline])
    // auto start = Time::now();
    // m_executable->call(m_backend_output_tensors,
    //                    m_backend_input_tensors,
    //                    m_eval_context,
    //                    get_template_model()->m_cfg.perf_count);
    // m_durations[StartPipeline] = Time::now() - start;
}
// ! [infer_request:start_pipeline]

// ! [infer_request:wait_pipeline]
void ov::hetero::InferRequest::wait_pipeline() {
    // OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[WaitPipeline])
    // auto start = Time::now();
    // // TODO: Wait pipeline using driver API or other synchronizations methods
    // // NOTE: not used in current implementation since `startPipeline` executes pipiline synchronously
    // m_durations[WaitPipeline] = Time::now() - start;
}
// ! [infer_request:wait_pipeline]

// ! [infer_request:infer_postprocess]
void ov::hetero::InferRequest::infer_postprocess() {
    // OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[Postprocess]);
    // auto start = Time::now();
    // OPENVINO_ASSERT(get_outputs().size() == m_backend_output_tensors.size());
    // for (size_t i = 0; i < get_outputs().size(); i++) {
    //     const auto& result = get_template_model()->m_model->get_results()[i];
    //     auto host_tensor = m_backend_output_tensors[i];
    //     auto tensor = get_tensor(get_outputs()[i]);
    //     if (result->get_output_partial_shape(0).is_dynamic()) {
    //         ov::Output<const ov::Node> output{result->output(0).get_node(), result->output(0).get_index()};
    //         allocate_tensor(output, [host_tensor](ov::Tensor& tensor) {
    //             allocate_tensor_impl(tensor, host_tensor.get_element_type(), host_tensor.get_shape());
    //             host_tensor.copy_to(tensor);
    //         });
    //     } else if (!tensor.is_continuous()) {
    //         host_tensor.copy_to(tensor);
    //     } else if (tensor.is<ov::RemoteTensor>()) {
    //         OPENVINO_ASSERT(tensor.is<ov::hetero::VectorTensor>(),
    //                         "Template plugin supports only VectorTensor with remote context.");
    //         auto vector_tensor = tensor.as<ov::hetero::VectorTensor>();
    //         void* data = vector_tensor.get_data();
    //         // Copy to vector
    //         std::memcpy(data, host_tensor.data(), tensor.get_byte_size());
    //     }
    // }
    // m_durations[Postprocess] = Time::now() - start;
}
// ! [infer_request:infer_postprocess]

// ! [infer_request:get_profiling_info]
std::vector<ov::ProfilingInfo> ov::hetero::InferRequest::get_profiling_info() const {
    // std::vector<ov::ProfilingInfo> info;
    // const auto fill_profiling_info = [](const std::string& name,
    //                                     const std::chrono::duration<float, std::micro>& time) -> ov::ProfilingInfo {
    //     ov::ProfilingInfo p_info;
    //     p_info.status = ov::ProfilingInfo::Status::EXECUTED;
    //     p_info.node_name = name;
    //     p_info.cpu_time = p_info.real_time = std::chrono::duration_cast<std::chrono::milliseconds>(time);
    //     return p_info;
    // };
    // info.emplace_back(fill_profiling_info("input preprocessing", m_durations[Preprocess]));
    // info.emplace_back(fill_profiling_info("execution time", m_durations[StartPipeline]));
    // info.emplace_back(fill_profiling_info("output postprocessing", m_durations[Postprocess]));
    // return info;
}
// ! [infer_request:get_profiling_info]

// ! [infer_request:cancel]
void ov::hetero::InferRequest::cancel() {
    // m_executable->cancel();
}
// ! [infer_request:cancel]
