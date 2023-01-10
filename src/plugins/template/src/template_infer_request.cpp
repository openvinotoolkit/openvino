// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_infer_request.hpp"

#include <debug.h>
#include <ie_compound_blob.h>

#include <algorithm>
#include <map>
#include <memory>
#include <ngraph/runtime/host_tensor.hpp>
#include <ngraph/runtime/reference/convert.hpp>
#include <string>
#include <utility>

#include "blob_factory.hpp"
#include "ie_api.h"
#include "ie_common.h"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "template_executable_network.hpp"
#include "template_itt.hpp"
#include "template_plugin.hpp"

using namespace TemplatePlugin;

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
}  // namespace

}  // namespace

// ! [infer_request:ctor]
TemplateInferRequest::TemplateInferRequest(const std::shared_ptr<TemplatePlugin::ExecutableNetwork>& executableNetwork)
    : ov::IInferRequest(executableNetwork) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks
    auto requestID = std::to_string(get_template_model()->_requestId.fetch_add(1));

    std::string name = get_template_model()->m_model->get_friendly_name() + "_Req" + requestID;
    _profilingTask = {
        openvino::itt::handle("Template" + std::to_string(get_template_model()->_cfg.deviceId) + "_" + name +
                              "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(get_template_model()->_cfg.deviceId) + "_" + name +
                              "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(get_template_model()->_cfg.deviceId) + "_" + name +
                              "_StartPipline"),
        openvino::itt::handle("Template" + std::to_string(get_template_model()->_cfg.deviceId) + "_" + name +
                              "_WaitPipline"),
    };

    _executable = get_template_model()->get_template_plugin()->_backend->compile(get_template_model()->m_model);

    // Allocate plugin backend specific memory handles
    _inputTensors.resize(get_inputs().size());
    _outputTensors.resize(get_outputs().size());
    m_input_tensors.resize(get_inputs().size());
    m_output_tensors.resize(get_outputs().size());

    AllocateImpl(m_input_tensors, get_inputs());
    AllocateImpl(m_output_tensors, get_outputs());
}

// ! [infer_request:ctor]

std::shared_ptr<TemplatePlugin::ExecutableNetwork> TemplateInferRequest::get_template_model() const {
    auto& compiled_model = get_compiled_model();

    auto template_model = std::dynamic_pointer_cast<TemplatePlugin::ExecutableNetwork>(compiled_model);
    OPENVINO_ASSERT(template_model);

    return template_model;
}

// ! [infer_request:dtor]
TemplateInferRequest::~TemplateInferRequest() {
    get_template_model()->_requestId--;
}
// ! [infer_request:dtor]

// ! [infer_request:infer_impl]
void TemplateInferRequest::infer() {
    // TODO: fill with actual list of pipeline stages, which are executed synchronously for sync infer requests
    infer_preprocess();
    start_pipeline();
    wait_pipeline();  // does nothing in current implementation
    infer_postprocess();
}
// ! [infer_request:infer_impl]

// ! [infer_request:infer_preprocess]
void TemplateInferRequest::infer_preprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[Preprocess]);
    auto start = Time::now();
    convert_batched_tensors();
    // NOTE: After IInferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in constructor.
    // IInferRequestInternal::execDataPreprocessing(_deviceInputs);
    OPENVINO_ASSERT(m_input_tensors.size() == _inputTensors.size());
    for (size_t i = 0; i < m_input_tensors.size(); i++) {
        // No ROI extraction is needed
        _inputTensors[i] = get_template_model()->get_template_plugin()->_backend->create_tensor(
            m_input_tensors.at(i).get_element_type(),
            m_input_tensors.at(i).get_shape(),
            m_input_tensors.at(i).data());
    }
    OPENVINO_ASSERT(m_output_tensors.size() == _outputTensors.size());
    for (size_t i = 0; i < m_output_tensors.size(); i++) {
        _outputTensors[i] = get_template_model()->get_template_plugin()->_backend->create_tensor(
            m_output_tensors.at(i).get_element_type(),
            m_output_tensors.at(i).get_shape(),
            m_output_tensors.at(i).data());
    }
    _durations[Preprocess] = Time::now() - start;
}
// ! [infer_request:infer_preprocess]

// ! [infer_request:start_pipeline]
void TemplateInferRequest::start_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[StartPipeline])
    auto start = Time::now();
    _executable->call(_outputTensors, _inputTensors);
    _durations[StartPipeline] = Time::now() - start;
}
// ! [infer_request:start_pipeline]

void TemplateInferRequest::wait_pipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[WaitPipeline])
    auto start = Time::now();
    // TODO: Wait pipeline using driver API or other synchronizations methods
    // NOTE: not used in current implementation since `startPipeline` executes pipiline synchronously
    _durations[WaitPipeline] = Time::now() - start;
}

// ! [infer_request:infer_postprocess]
void TemplateInferRequest::infer_postprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[Postprocess]);
    auto start = Time::now();
    // FIXME: We don't need postprocessing anymore
    // for (auto&& output : _networkOutputs) {
    //     auto index = get_template_model()->_outputIndex[output.first];
    //     const auto& result = get_template_model()->m_model->get_results()[index];
    //     if (result->get_output_partial_shape(0).is_dynamic()) {
    //         // Touch blob to allocate it
    //         GetBlob(output.first);
    //     }
    //     auto outputBlob = _outputs.at(output.first);
    //     auto networkOutput = _networkOutputBlobs[output.first];
    //     if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision()) {
    //         blobCopy(networkOutput, outputBlob);
    //     } else if (result->get_output_partial_shape(0).is_dynamic()) {
    //         auto tensor = _outputTensors[get_template_model()->_outputIndex.at(output.first)];
    //         tensor->read(InferenceEngine::as<InferenceEngine::MemoryBlob>(outputBlob)->wmap().as<char*>(),
    //                      tensor->get_size_in_bytes());
    //     }
    // }
    _durations[Postprocess] = Time::now() - start;
}
// ! [infer_request:infer_postprocess]

// ! [infer_request:get_blob]
ov::Tensor TemplateInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
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
            auto this_non_const = const_cast<TemplateInferRequest*>(this);
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
            auto this_non_const = const_cast<TemplateInferRequest*>(this);
            AllocateImplSingle(this_non_const->m_output_tensors, found_port.idx, port.get_element_type(), shape);
            data = m_output_tensors[found_port.idx];
        }
        check_tensor(port, data);
    }
    return data;
}
// ! [infer_request:get_blob]

// ! [infer_request:set_blob]
void TemplateInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
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
void TemplateInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                            const std::vector<ov::Tensor>& tensors) {
    m_batched_tensors[find_port(port).idx] = tensors;
}
// ! [infer_request:set_blobs_impl]

// ! [infer_request:get_performance_counts]
std::vector<ov::ProfilingInfo> TemplateInferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    // const auto fill_profiling_info = [](const std::string& name, const std::chrono::microseconds& time) {
    //     ov::ProfilingInfo p_info;
    //     p_info.status = ov::ProfilingInfo::Status::EXECUTED;
    //     p_info.node_name = name;
    //     p_info.cpu_time = p_info.real_time = time;
    //     return p_info;
    // };
    // info.emplace_back(fill_profiling_info("input preprocessing", _durations[Preprocess].count()));
    // info.emplace_back(fill_profiling_info("execution time", _durations[StartPipeline]));
    // info.emplace_back(fill_profiling_info("output postprocessing", _durations[Postprocess]));
    return info;
}
// ! [infer_request:get_performance_counts]
