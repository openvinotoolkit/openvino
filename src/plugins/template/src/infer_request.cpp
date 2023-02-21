// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "plugin.hpp"
#include "template_itt.hpp"

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

// ! [infer_request:ctor]
TemplatePlugin::InferRequest::InferRequest(const std::shared_ptr<const TemplatePlugin::CompiledModel>& model)
    : ov::ISyncInferRequest(model) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto requestID = std::to_string(get_template_model()->_requestId.fetch_add(1));

    std::string name = get_template_model()->m_model->get_friendly_name() + "_Req" + requestID;
    m_profiling_task = {
        openvino::itt::handle("Template" + std::to_string(get_template_model()->_cfg.deviceId) + "_" + name +
                              "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(get_template_model()->_cfg.deviceId) + "_" + name +
                              "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(get_template_model()->_cfg.deviceId) + "_" + name +
                              "_StartPipeline"),
        openvino::itt::handle("Template" + std::to_string(get_template_model()->_cfg.deviceId) + "_" + name +
                              "_WaitPipline"),
    };

    m_executable = get_template_model()->get_template_plugin()->_backend->compile(get_template_model()->m_model);

    // Allocate plugin backend specific memory handles
    m_backend_input_tensors.resize(get_inputs().size());
    m_backend_output_tensors.resize(get_outputs().size());

    // Allocate input/output tensors
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [input](ov::Tensor& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 input.get_element_type(),
                                 input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::Tensor& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 output.get_element_type(),
                                 output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }
}
// ! [infer_request:ctor]

std::vector<ov::VariableState> TemplatePlugin::InferRequest::query_state() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const TemplatePlugin::CompiledModel> TemplatePlugin::InferRequest::get_template_model() const {
    auto& compiled_model = get_compiled_model();
    auto template_model = std::dynamic_pointer_cast<const TemplatePlugin::CompiledModel>(compiled_model);
    OPENVINO_ASSERT(template_model);
    return template_model;
}

// ! [infer_request:dtor]
TemplatePlugin::InferRequest::~InferRequest() = default;
// ! [infer_request:dtor]

// ! [infer_request:infer_impl]
void TemplatePlugin::InferRequest::infer() {
    // TODO: fill with actual list of pipeline stages, which are executed synchronously for sync infer requests
    infer_preprocess();
    start_pipeline();
    wait_pipeline();  // does nothing in current implementation
    infer_postprocess();
}
// ! [infer_request:infer_impl]

// ! [infer_request:infer_preprocess]
void TemplatePlugin::InferRequest::infer_preprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[Preprocess]);
    auto start = Time::now();
    convert_batched_tensors();
    check_tensors();

    // Allocate backend tensors
    OPENVINO_ASSERT(get_inputs().size() == m_backend_input_tensors.size());
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto tensor = get_tensor(get_inputs()[i]);
        if (tensor.is_continuous()) {
            // No ROI extraction is needed
            m_backend_input_tensors[i] =
                get_template_model()->get_template_plugin()->_backend->create_tensor(tensor.get_element_type(),
                                                                                     tensor.get_shape(),
                                                                                     tensor.data());
        } else {
            OPENVINO_ASSERT(tensor.get_element_type().bitwidth() % 8 == 0,
                            "Template plugin: Unsupported ROI tensor with element type having ",
                            std::to_string(tensor.get_element_type().bitwidth()),
                            " bits size");
            ov::Shape shape = tensor.get_shape();
            // Perform manual extraction of ROI tensor
            // Basic implementation doesn't take axis order into account `desc.getBlockingDesc().getOrder()`
            // Performance of manual extraction is not optimal, but it is ok for template implementation
            m_backend_input_tensors[i] =
                get_template_model()->get_template_plugin()->_backend->create_tensor(tensor.get_element_type(),
                                                                                     tensor.get_shape());
            auto* src_data = static_cast<uint8_t*>(tensor.data());
            auto dst_tensor = std::dynamic_pointer_cast<ngraph::runtime::HostTensor>(m_backend_input_tensors[i]);
            OPENVINO_ASSERT(dst_tensor, "Template plugin error: Can't cast created tensor to HostTensor");
            auto* dst_data = dst_tensor->get_data_ptr<uint8_t>();
            std::vector<size_t> indexes(shape.size());
            for (size_t dst_idx = 0; dst_idx < ov::shape_size(shape); dst_idx++) {
                size_t val = dst_idx;
                size_t src_idx = 0;
                for (size_t j1 = 0; j1 < indexes.size(); j1++) {
                    size_t j = indexes.size() - j1 - 1;
                    indexes[j] = val % shape[j];
                    val /= shape[j];
                    src_idx += indexes[j] * tensor.get_strides()[j];
                }
                memcpy(dst_data + dst_idx * tensor.get_element_type().size(),
                       src_data + src_idx,
                       tensor.get_element_type().size());
            }
        }
    }
    // Tensors can be dynamic, so in this case we need to allocate tensors with right shape
    OPENVINO_ASSERT(get_outputs().size() == m_backend_output_tensors.size());
    for (size_t i = 0; i < get_outputs().size(); i++) {
        const auto& result = get_template_model()->m_model->get_results()[i];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            m_backend_output_tensors[i] = get_template_model()->get_template_plugin()->_backend->create_tensor();
            continue;
        }
        auto tensor = get_tensor(get_outputs()[i]);
        m_backend_output_tensors[i] =
            get_template_model()->get_template_plugin()->_backend->create_tensor(tensor.get_element_type(),
                                                                                 tensor.get_shape(),
                                                                                 tensor.data());
    }
    m_durations[Preprocess] = Time::now() - start;
}
// ! [infer_request:infer_preprocess]

// ! [infer_request:start_pipeline]
void TemplatePlugin::InferRequest::start_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[StartPipeline])
    auto start = Time::now();
    m_executable->call(m_backend_output_tensors, m_backend_input_tensors);
    m_durations[StartPipeline] = Time::now() - start;
}
// ! [infer_request:start_pipeline]

void TemplatePlugin::InferRequest::wait_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[WaitPipeline])
    auto start = Time::now();
    // TODO: Wait pipeline using driver API or other synchronizations methods
    // NOTE: not used in current implementation since `startPipeline` executes pipiline synchronously
    m_durations[WaitPipeline] = Time::now() - start;
}

// ! [infer_request:infer_postprocess]
void TemplatePlugin::InferRequest::infer_postprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[Postprocess]);
    auto start = Time::now();
    OPENVINO_ASSERT(get_outputs().size() == m_backend_output_tensors.size());
    for (size_t i = 0; i < get_outputs().size(); i++) {
        const auto& result = get_template_model()->m_model->get_results()[i];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            auto host_tensor = m_backend_output_tensors[i];
            ov::Output<const ov::Node> output{result->output(0).get_node(), result->output(0).get_index()};
            allocate_tensor(output, [host_tensor](ov::Tensor& tensor) {
                allocate_tensor_impl(tensor, host_tensor->get_element_type(), host_tensor->get_shape());
                // tensor.set_shape(host_tensor->get_shape());
                host_tensor->read(static_cast<char*>(tensor.data()), host_tensor->get_size_in_bytes());
            });
        }
    }
    m_durations[Postprocess] = Time::now() - start;
}
// ! [infer_request:infer_postprocess]

// ! [infer_request:set_blobs_impl]
void TemplatePlugin::InferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                                    const std::vector<ov::Tensor>& tensors) {
    for (const auto& input : get_inputs()) {
        if (input == port) {
            m_batched_tensors[input.get_tensor_ptr()] = tensors;
            return;
        }
    }
    OPENVINO_UNREACHABLE("Cannot find input tensors for port ", port);
}
// ! [infer_request:set_blobs_impl]

// ! [infer_request:get_performance_counts]
std::vector<ov::ProfilingInfo> TemplatePlugin::InferRequest::get_profiling_info() const {
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
    info.emplace_back(fill_profiling_info("output postprocessing", m_durations[Postprocess]));
    return info;
}
// ! [infer_request:get_performance_counts]
