// Copyright (C) 2018-2023 Intel Corporation
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
#include "ie_remote_blob.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "template_compiled_model.hpp"
#include "template_itt.hpp"
#include "template_plugin.hpp"

using namespace TemplatePlugin;
using namespace InferenceEngine;

using Time = std::chrono::high_resolution_clock;

// ! [infer_request:ctor]
TemplateInferRequest::TemplateInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                           const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                           const std::shared_ptr<const TemplatePlugin::CompiledModel>& compiled_model)
    : IInferRequestInternal(inputs, outputs),
      m_compiled_model(compiled_model) {
    createInferRequest();
}

void TemplateInferRequest::createInferRequest() {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto compiled_model = std::const_pointer_cast<TemplatePlugin::CompiledModel>(m_compiled_model);
    auto requestID = std::to_string(compiled_model->_requestId.fetch_add(1));

    std::string name = m_compiled_model->m_model->get_friendly_name() + "_Req" + requestID;
    _profilingTask = {
        openvino::itt::handle("Template" + std::to_string(m_compiled_model->_cfg.deviceId) + "_" + name +
                              "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(m_compiled_model->_cfg.deviceId) + "_" + name +
                              "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(m_compiled_model->_cfg.deviceId) + "_" + name +
                              "_StartPipline"),
        openvino::itt::handle("Template" + std::to_string(m_compiled_model->_cfg.deviceId) + "_" + name +
                              "_WaitPipline"),
    };

    _executable = m_compiled_model->get_template_plugin()->_backend->compile(m_compiled_model->m_model);

    allocateDeviceBuffers();
    allocateBlobs();
}
// ! [infer_request:ctor]

// ! [infer_request:dtor]
TemplateInferRequest::~TemplateInferRequest() {
    auto compiled_model = std::const_pointer_cast<TemplatePlugin::CompiledModel>(m_compiled_model);
    compiled_model->_requestId--;
}
// ! [infer_request:dtor]

void TemplateInferRequest::allocateDeviceBuffers() {
    // Allocate plugin backend specific memory handles
    _inputTensors.resize(_networkInputs.size());
    _outputTensors.resize(_networkOutputs.size());
}

template <typename BlobData>
static void AllocateImplSingle(const BlobData& blobData, BlobMap& blobMap, const SizeVector& dims) {
    const auto& precision = blobData.second->getTensorDesc().getPrecision();
    auto layout = blobData.second->getTensorDesc().getLayout();
    Blob::Ptr& blob = blobMap[blobData.first];
    if (!blob) {
        blob = make_blob_with_precision({precision, dims, layout});
        blob->allocate();
    } else {
        blob->setShape(dims);
    }
}

template <typename BlobDataMap>
static void AllocateImpl(const BlobDataMap& userDataMap, BlobMap& userBlobMap, bool isInputBlob = true) {
    for (const auto& userData : userDataMap) {
        auto tensorDesc = userData.second->getTensorDesc();
        AllocateImplSingle(userData, userBlobMap, tensorDesc.getDims());
    }
}

void TemplateInferRequest::allocateBlobs() {
    AllocateImpl(_networkInputs, _inputs);
    AllocateImpl(_networkOutputs, _outputs, false);
}

// ! [infer_request:infer_impl]
void TemplateInferRequest::InferImpl() {
    // TODO: fill with actual list of pipeline stages, which are executed synchronously for sync infer requests
    inferPreprocess();
    startPipeline();
    waitPipeline();  // does nothing in current implementation
    inferPostprocess();
}
// ! [infer_request:infer_impl]

// ! [infer_request:infer_preprocess]
void TemplateInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[Preprocess]);
    auto start = Time::now();
    convertBatchedInputBlobs();
    // NOTE: After IInferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in constructor.
    for (auto&& input : _inputs) {
        auto index = m_compiled_model->_inputIndex.at(input.first);
        const auto& parameter_type = m_compiled_model->m_model->get_parameters()[index]->get_element_type();
        auto mem_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(input.second);
        // No ROI extraction is needed
        _inputTensors[index] = m_compiled_model->get_template_plugin()->_backend->create_tensor(
            parameter_type,
            mem_blob->getTensorDesc().getBlockingDesc().getBlockDims(),
            mem_blob->rmap().as<void*>());
    }
    for (auto&& output : _outputs) {
        auto mem_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(output.second);
        auto index = m_compiled_model->_outputIndex.at(output.first);
        const auto& result = m_compiled_model->m_model->get_results()[index];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            _outputTensors[index] = m_compiled_model->get_template_plugin()->_backend->create_tensor();
            continue;
        }
        const auto& resultShape = result->get_shape();
        const auto& resultType = result->get_element_type();
        _outputTensors[index] =
            m_compiled_model->get_template_plugin()->_backend->create_tensor(resultType,
                                                                             resultShape,
                                                                             mem_blob->wmap().as<void*>());
    }
    _durations[Preprocess] = Time::now() - start;
}
// ! [infer_request:infer_preprocess]

// ! [infer_request:start_pipeline]
void TemplateInferRequest::startPipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[StartPipeline])
    auto start = Time::now();
    _executable->call(_outputTensors, _inputTensors);
    _durations[StartPipeline] = Time::now() - start;
}
// ! [infer_request:start_pipeline]

void TemplateInferRequest::waitPipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[WaitPipeline])
    auto start = Time::now();
    // TODO: Wait pipeline using driver API or other synchronizations methods
    // NOTE: not used in current implementation since `startPipeline` executes pipiline synchronously
    _durations[WaitPipeline] = Time::now() - start;
}

// ! [infer_request:infer_postprocess]
void TemplateInferRequest::inferPostprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[Postprocess]);
    auto start = Time::now();
    for (auto&& output : _outputs) {
        auto index = m_compiled_model->_outputIndex.at(output.first);
        const auto& result = m_compiled_model->m_model->get_results()[index];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            // Touch blob to allocate it
            GetBlob(output.first);
        }
        auto mem_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(output.second);
        if (result->get_output_partial_shape(0).is_dynamic()) {
            auto tensor = _outputTensors[m_compiled_model->_outputIndex.at(output.first)];
            tensor->read(mem_blob->wmap().as<char*>(), tensor->get_size_in_bytes());
        }
    }
    _durations[Postprocess] = Time::now() - start;
}
// ! [infer_request:infer_postprocess]

// ! [infer_request:get_blob]
InferenceEngine::Blob::Ptr TemplateInferRequest::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "GetBlob");
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    Blob::Ptr data;
    const SizeVector oneVector = {1};
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        data = _inputs[name];
        SizeVector dims;
        if (!data) {
            auto&& parameters = m_compiled_model->m_model->get_parameters();
            const auto& pshape = parameters.at(m_compiled_model->_inputIndex.at(name))->get_partial_shape();
            dims = pshape.is_dynamic() ? SizeVector({0}) : pshape.get_shape();
            AllocateImplSingle(*_networkInputs.find(name), _inputs, dims);
            data = _inputs[name];
        } else {
            dims = data->getTensorDesc().getDims();
        }
        checkBlob(data, name, true, foundInput->getTensorDesc().getLayout() != SCALAR ? dims : oneVector);
    } else {
        data = _outputs[name];
        SizeVector dims;
        auto has_zeros = [](const SizeVector& vec) {
            return std::any_of(vec.cbegin(), vec.cend(), [](size_t e) {
                return e == 0;
            });
        };
        if (!has_zeros(foundOutput->getTensorDesc().getDims())) {
            dims = foundOutput->getTensorDesc().getDims();
        } else if (_outputTensors[m_compiled_model->_outputIndex.at(name)] &&
                   _outputTensors[m_compiled_model->_outputIndex.at(name)]->get_partial_shape().is_static()) {
            dims = _outputTensors[m_compiled_model->_outputIndex.at(name)]->get_shape();
        } else {
            auto rank = foundOutput->getTensorDesc().getDims().size();
            dims = SizeVector(rank == 0 ? 1 : rank, 0);
        }

        if (data->getTensorDesc().getDims() != dims) {
            AllocateImplSingle(*_networkOutputs.find(name), _outputs, dims);
            data = _outputs[name];
        }
        checkBlob(data, name, false, foundOutput->getTensorDesc().getLayout() != SCALAR ? dims : oneVector);
    }
    return data;
}
// ! [infer_request:get_blob]

// ! [infer_request:set_blob]
void TemplateInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& userBlob) {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "SetBlob");
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }
    if (!userBlob)
        IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    auto has_zeros = [](const SizeVector& vec) {
        return std::any_of(vec.cbegin(), vec.cend(), [](size_t e) {
            return e == 0;
        });
    };
    const bool isInput = findInputAndOutputBlobByName(name, foundInput, foundOutput);
    const bool compoundBlobPassed = userBlob->is<CompoundBlob>();
    const bool remoteBlobPassed = userBlob->is<InferenceEngine::RemoteBlob>();
    if (!compoundBlobPassed && !remoteBlobPassed && userBlob->buffer() == nullptr)
        IE_THROW(NotAllocated) << "Input data was not allocated. Input name: \'" << name << "\'";
    bool input_dynamic = foundInput && has_zeros(foundInput->getInputData()->getDims());
    bool output_dynamic = foundOutput && has_zeros(foundOutput->getDims());
    if (userBlob->size() == 0 && !(input_dynamic || output_dynamic)) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }

    size_t dataSize = userBlob->size();
    if (isInput) {
        // ilavreno: the condition below is obsolete, but we need an exact list of precisions
        // which are supports by G-API preprocessing
        if (foundInput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user input precision";
        }
        if (foundInput->getLayout() != userBlob->getTensorDesc().getLayout()) {
            IE_THROW(ParameterMismatch) << "Failed to set Blob with layout not corresponding to user input layout";
        }

        size_t inputSize = userBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                               ? InferenceEngine::details::product(userBlob->getTensorDesc().getDims())
                               : 1;
        if (dataSize != inputSize) {
            IE_THROW() << "Input blob size is not equal network input size (" << dataSize << "!=" << inputSize << ").";
        }
        _inputs[name] = userBlob;
        _batched_inputs.erase(name);
    } else {
        if (compoundBlobPassed) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }
        size_t outputSize = userBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                                ? details::product(userBlob->getTensorDesc().getDims())
                                : 1;
        if (dataSize != outputSize) {
            IE_THROW() << "Output blob size is not equal network output size (" << dataSize << "!=" << outputSize
                       << ").";
        }
        if (foundOutput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user output precision";
        }
        if (foundOutput->getLayout() != userBlob->getTensorDesc().getLayout()) {
            IE_THROW(ParameterMismatch) << "Failed to set Blob with layout not corresponding to user input layout";
        }
        _outputs[name] = userBlob;
    }
}
// ! [infer_request:set_blob]

// ! [infer_request:set_blobs_impl]
void TemplateInferRequest::SetBlobsImpl(const std::string& name, const InferenceEngine::BatchedBlob::Ptr& batchedBlob) {
    _batched_inputs[name] = batchedBlob;
}
// ! [infer_request:set_blobs_impl]

// ! [infer_request:get_performance_counts]
std::map<std::string, InferenceEngineProfileInfo> TemplateInferRequest::GetPerformanceCounts() const {
    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    InferenceEngineProfileInfo info;
    info.execution_index = 0;
    info.status = InferenceEngineProfileInfo::EXECUTED;

    info.cpu_uSec = info.realTime_uSec = static_cast<long long>(_durations[Preprocess].count());
    perfMap["1. input preprocessing"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["2. input transfer to a device"] = info;
    info.cpu_uSec = info.realTime_uSec = static_cast<long long>(_durations[StartPipeline].count());
    perfMap["3. execution time"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["4. output transfer from a device"] = info;
    info.cpu_uSec = info.realTime_uSec = static_cast<long long>(_durations[Postprocess].count());
    perfMap["5. output postprocessing"] = info;
    return perfMap;
}
// ! [infer_request:get_performance_counts]
