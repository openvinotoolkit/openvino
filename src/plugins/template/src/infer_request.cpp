// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "openvino/core/except.hpp"
#include "plugin.hpp"
#include "template_itt.hpp"

using Time = std::chrono::high_resolution_clock;

namespace {

static void AllocateImplSingle(std::vector<ov::Tensor>& tensors,
                               size_t idx,
                               const ov::element::Type& element_type,
                               const ov::Shape& shape) {
    OPENVINO_ASSERT(idx < tensors.size());
    if (!tensors.at(idx) || tensors.at(idx).get_element_type() != element_type) {
        tensors.at(idx) = ov::Tensor(element_type, shape);
    } else {
        tensors.at(idx).set_shape(shape);
    }
}

static void AllocateImpl(std::vector<ov::Tensor>& tensors, const std::vector<ov::Output<const ov::Node>>& ports) {
    OPENVINO_ASSERT(tensors.size() == ports.size());

    for (size_t i = 0; i < tensors.size(); i++) {
        AllocateImplSingle(tensors,
                           i,
                           ports.at(i).get_element_type(),
                           ports.at(i).get_partial_shape().is_dynamic() ? ov::Shape{0} : ports.at(i).get_shape());
    }
}

}  // namespace

// ! [infer_request:ctor]
TemplatePlugin::InferRequest::InferRequest(const std::shared_ptr<const TemplatePlugin::CompiledModel>& model)
    : ov::ISyncInferRequest(model) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto compiled_model = std::const_pointer_cast<TemplatePlugin::CompiledModel>(get_template_model());
    auto requestID = std::to_string(compiled_model->_requestId.fetch_add(1));

    std::string name = compiled_model->m_model->get_friendly_name() + "_Req" + requestID;
    m_profiling_task = {
        openvino::itt::handle("Template" + std::to_string(compiled_model->_cfg.deviceId) + "_" + name + "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(compiled_model->_cfg.deviceId) + "_" + name + "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(compiled_model->_cfg.deviceId) + "_" + name +
                              "_StartPipline"),
        openvino::itt::handle("Template" + std::to_string(compiled_model->_cfg.deviceId) + "_" + name + "_WaitPipline"),
    };

    m_executable = compiled_model->get_template_plugin()->_backend->compile(compiled_model->m_model);

    // Allocate plugin backend specific memory handles
    m_plugin_input_tensors.resize(get_inputs().size());
    m_plugin_output_tensors.resize(get_outputs().size());
    m_input_tensors.resize(get_inputs().size());
    m_output_tensors.resize(get_outputs().size());

    AllocateImpl(m_input_tensors, get_inputs());
    AllocateImpl(m_output_tensors, get_outputs());
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
TemplatePlugin::InferRequest::~InferRequest() {
    auto compiled_model = std::const_pointer_cast<TemplatePlugin::CompiledModel>(get_template_model());
    compiled_model->_requestId--;
}
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

    OPENVINO_ASSERT(m_input_tensors.size() == m_plugin_input_tensors.size());
    for (size_t i = 0; i < m_input_tensors.size(); i++) {
        // No ROI extraction is needed
        m_plugin_input_tensors[i] = get_template_model()->get_template_plugin()->_backend->create_tensor(
            m_input_tensors.at(i).get_element_type(),
            m_input_tensors.at(i).get_shape(),
            m_input_tensors.at(i).data());
    }
    OPENVINO_ASSERT(m_output_tensors.size() == m_plugin_output_tensors.size());
    for (size_t i = 0; i < m_output_tensors.size(); i++) {
        m_plugin_output_tensors[i] = get_template_model()->get_template_plugin()->_backend->create_tensor(
            m_output_tensors.at(i).get_element_type(),
            m_output_tensors.at(i).get_shape(),
            m_output_tensors.at(i).data());
    }
    m_durations[Preprocess] = Time::now() - start;
}
// ! [infer_request:infer_preprocess]

// ! [infer_request:start_pipeline]
void TemplatePlugin::InferRequest::start_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, m_profiling_task[StartPipeline])
    auto start = Time::now();
    m_executable->call(m_plugin_output_tensors, m_plugin_input_tensors);
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
    // We don't need post-processing in template plugin
    m_durations[Postprocess] = Time::now() - start;
}
// ! [infer_request:infer_postprocess]

// ! [infer_request:get_blob]
ov::Tensor TemplatePlugin::InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "get_tensor");
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "Failed to get tensor, port ", port, " cannot be found.");

    ov::Tensor data;
    if (found_port.is_input()) {
        data = m_input_tensors[found_port.idx];
        ov::Shape shape;
        if (!data.get_size()) {
            auto&& parameters = get_template_model()->m_model->get_parameters();
            const auto& pshape = parameters.at(found_port.idx)->get_partial_shape();
            shape = pshape.is_dynamic() ? ov::Shape{0} : pshape.get_shape();
            auto this_non_const = const_cast<InferRequest*>(this);
            AllocateImplSingle(this_non_const->m_input_tensors, found_port.idx, port.get_element_type(), shape);
            data = m_input_tensors[found_port.idx];
        } else {
            shape = data.get_shape();
        }
        check_tensor(port, data);
    } else {
        data = m_output_tensors[found_port.idx];
        ov::Shape shape;
        auto has_zeros = [](const ov::Shape& vec) {
            return std::any_of(vec.cbegin(), vec.cend(), [](size_t e) {
                return e == 0;
            });
        };
        if (port.get_partial_shape().is_static() && !has_zeros(port.get_shape())) {
            shape = port.get_shape();
        } else if (!has_zeros(data.get_shape())) {
            shape = data.get_shape();
        } else {
            shape = ov::Shape(
                port.get_partial_shape().rank().is_dynamic() ? 1 : port.get_partial_shape().rank().get_length(),
                0);
        }
        if (data.get_shape() != shape) {
            auto this_non_const = const_cast<InferRequest*>(this);
            AllocateImplSingle(this_non_const->m_output_tensors, found_port.idx, port.get_element_type(), shape);
            data = m_output_tensors[found_port.idx];
        }
        check_tensor(port, data);
    }
    return data;
}
// ! [infer_request:get_blob]

// ! [infer_request:set_blob]
void TemplatePlugin::InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "set_tensor");
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "Failed to set tensor, port ", port, " cannot be found.");

    OPENVINO_ASSERT(tensor.get_element_type() == port.get_element_type(),
                    "Failed to set tensor with element type: ",
                    tensor.get_element_type(),
                    ", this element type is not corresponding to model element type: ",
                    port.get_element_type());
    OPENVINO_ASSERT(port.get_partial_shape().is_dynamic() || port.get_shape() == tensor.get_shape(),
                    "Tensor shape (",
                    tensor.get_shape(),
                    ") doesn't match with model shape (",
                    port.get_partial_shape(),
                    ").");

    if (found_port.is_input()) {
        m_input_tensors.at(found_port.idx) = tensor;
        m_batched_tensors.erase(found_port.idx);
    } else {
        m_output_tensors.at(found_port.idx) = tensor;
    }
}
// ! [infer_request:set_blob]

// ! [infer_request:set_blobs_impl]
void TemplatePlugin::InferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                                    const std::vector<ov::Tensor>& tensors) {
    m_batched_tensors[find_port(port).idx] = tensors;
}
// ! [infer_request:set_blobs_impl]

// ! [infer_request:get_performance_counts]
std::vector<ov::ProfilingInfo> TemplatePlugin::InferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    // const auto fill_profiling_info = [](const std::string& name, const std::chrono::microseconds& time) {
    //     ov::ProfilingInfo p_info;
    //     p_info.status = ov::ProfilingInfo::Status::EXECUTED;
    //     p_info.node_name = name;
    //     p_info.cpu_time = p_info.real_time = time;
    //     return p_info;
    // };
    // info.emplace_back(fill_profiling_info("input preprocessing", m_durations[Preprocess].count()));
    // info.emplace_back(fill_profiling_info("execution time", m_durations[StartPipeline]));
    // info.emplace_back(fill_profiling_info("output postprocessing", m_durations[Postprocess]));
    return info;
}
// ! [infer_request:get_performance_counts]
