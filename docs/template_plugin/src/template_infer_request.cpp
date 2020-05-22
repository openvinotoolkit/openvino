// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <utility>
#include <algorithm>
#include <memory>
#include <string>
#include <map>

#include <ie_blob.h>
#include <ie_plugin.hpp>
#include <description_buffer.hpp>
#include <debug.h>
#include <ie_layouts.h>
#include <threading/ie_executor_manager.hpp>
#include <blob_transform.hpp>
#include <ie_parallel.hpp>
#include <ie_memcpy.h>
#include <precision_utils.h>
#include <template/template_config.hpp>

#include "template_infer_request.hpp"
#include "template_executable_network.hpp"
#include "template_plugin.hpp"

using namespace TemplatePlugin;

using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;
using fsec = std::chrono::duration<float>;

// ! [infer_request:ctor]
TemplateInferRequest::TemplateInferRequest(const InferenceEngine::InputsDataMap&                     networkInputs,
                                           const InferenceEngine::OutputsDataMap&                    networkOutputs,
                                           const std::shared_ptr<TemplatePlugin::ExecutableNetwork>& executableNetwork) :
    InferRequestInternal(networkInputs, networkOutputs),
    _executableNetwork(executableNetwork) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto requestID = std::to_string(_executableNetwork->_requestId);
    _executableNetwork->_requestId++;

    std::string name = _executableNetwork->_name + "_Req" + requestID;
    _profilingTask = { {
        { ProfilingTask("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_Preprocess") },
        { ProfilingTask("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_Postprocess") },
        { ProfilingTask("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_StartPipline") },
        { ProfilingTask("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_WaitPipline") },
    } };

    allocateDeviceBuffers();
    allocateInputBlobs();
    allocateOutputBlobs();
}
// ! [infer_request:ctor]

// ! [infer_request:dtor]
TemplateInferRequest::~TemplateInferRequest() {
    _executableNetwork->_requestId--;
}
// ! [infer_request:dtor]

void TemplateInferRequest::allocateDeviceBuffers() {
    // TODO: allocate device buffers if Template device is a remote one
}

void TemplateInferRequest::allocateInputBlobs() {
    for (auto &networkInput : _networkInputs) {
        SizeVector dims = networkInput.second->getTensorDesc().getDims();
        Precision precision = networkInput.second->getTensorDesc().getPrecision();
        Layout input_layout = networkInput.second->getInputData()->getLayout();
        Blob::Ptr inputBlob;
        Blob::Ptr inputBlobNCHW;
        switch (precision) {
        case Precision::FP32 :
            inputBlobNCHW = inputBlob = InferenceEngine::make_shared_blob<float>({ precision, dims, input_layout });
            if (input_layout == Layout::NHWC) {
                inputBlobNCHW = InferenceEngine::make_shared_blob<float>({ precision, dims, Layout::NCHW });
            }
            break;
        case Precision::FP16 :
        case Precision::I16 :
            inputBlobNCHW = inputBlob = InferenceEngine::make_shared_blob<int16_t>({ precision, dims, input_layout });
            if (input_layout == Layout::NHWC) {
                inputBlobNCHW = InferenceEngine::make_shared_blob<int16_t>({ precision, dims, Layout::NCHW });
            }
            break;
        case Precision::U8 :
            inputBlobNCHW = inputBlob = InferenceEngine::make_shared_blob<uint8_t>({ precision, dims, input_layout });
            if (input_layout == Layout::NHWC) {
                inputBlobNCHW = InferenceEngine::make_shared_blob<uint8_t>({ precision, dims, Layout::NCHW });
            }
            break;
        default:
            THROW_IE_EXCEPTION << "Unsupported network precision: " << precision
                << precision << "! Supported precisions are: FP32, FP16, I16, U8";
        }
        // allocate the input blob
        inputBlob->allocate();
        _inputs[networkInput.first] = inputBlob;
        if (inputBlobNCHW != inputBlob) {
            inputBlobNCHW->allocate();
        }
        _inputsNCHW[networkInput.first] = inputBlobNCHW;
    }
}

void TemplateInferRequest::allocateOutputBlobs() {
    for (auto &networkOutput : _networkOutputs) {
        SizeVector dims = networkOutput.second->getTensorDesc().getDims();
        Precision precision = networkOutput.second->getPrecision();
        Blob::Ptr outputBlob;

        // allocate the output blob
        Blob::Ptr outputBlobNCHW;
        switch (precision) {
        case Precision::FP32 :
            outputBlobNCHW = outputBlob = InferenceEngine::make_shared_blob<float>({ precision, dims, networkOutput.second->getLayout() });
            if (networkOutput.second->getLayout() == Layout::NHWC) {
                outputBlobNCHW = InferenceEngine::make_shared_blob<float>({ precision, dims,  Layout::NCHW });
            }
            break;
        case Precision::FP16 :
            outputBlobNCHW = outputBlob = InferenceEngine::make_shared_blob<int16_t>({ precision, dims, networkOutput.second->getLayout() });
            if (networkOutput.second->getLayout() == Layout::NHWC) {
                outputBlobNCHW = InferenceEngine::make_shared_blob<int16_t>({ precision, dims, Layout::NCHW });
            }
            break;
        default:
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: "
                << precision << "! Supported precisions are: FP32, FP16";
        }
        // allocate the output blob
        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
        if (outputBlobNCHW != outputBlob) {
            outputBlobNCHW->allocate();
        }
        _outputsNCHW[networkOutput.first] = outputBlobNCHW;
    }

    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }
}

// ! [infer_request:infer_impl]
void TemplateInferRequest::InferImpl() {
    // TODO: fill with actual list of pipeline stages, which are executed syncronously for sync infer requests
    inferPreprocess();
    startPipeline();
    waitPipeline();
    inferPostprocess();
}
// ! [infer_request:infer_impl]

// ! [infer_request:infer_preprocess]
void TemplateInferRequest::inferPreprocess() {
    auto prev = Time::now();

    // execute input pre-processing.
    InferRequestInternal::execDataPreprocessing(_inputs);

    for (auto &input : InferRequestInternal::_inputs) {
        auto& src = input.second;
        auto& dst = _inputsNCHW[input.first];
        if (src != dst) {
            if (src->getTensorDesc().getPrecision() == dst->getTensorDesc().getPrecision()
                && src->getTensorDesc().getDims() == dst->getTensorDesc().getDims()
                && src->getTensorDesc().getLayout() == dst->getTensorDesc().getLayout()) {
                _inputsNCHW[input.first] = input.second;
            } else {  // Convert Layout to NCHW
                InferenceEngine::blob_copy(src, dst);
            }
        }
    }

    // TODO: Preprocessing on inputs if needed: work _inputsNCHW

    _inputPreprocessTime = static_cast<double>(std::chrono::duration_cast<ns>(Time::now() - prev).count());
}
// ! [infer_request:infer_preprocess]

void TemplateInferRequest::startPipeline() {
    IE_PROFILING_AUTO_SCOPE_TASK(_profilingTask[StartPipeline])
    // TODO: Start pipeline and fill _inputTransferTime, _executeTime, _outputTransferTime
}

void TemplateInferRequest::waitPipeline() {
    IE_PROFILING_AUTO_SCOPE_TASK(_profilingTask[WaitPipeline])
    auto prev = Time::now();
    // TODO: Wait pipeline using driver API or other synronizations methods
    _inputPreprocessTime = static_cast<double>(std::chrono::duration_cast<ns>(Time::now() - prev).count());
}

void TemplateInferRequest::inferPostprocess() {
    IE_PROFILING_AUTO_SCOPE_TASK(_profilingTask[Postprocess])
    auto prev = Time::now();
    // TODO: perform post-processing and convert to NHWC layout
    _outputPostProcessTime = static_cast<double>(std::chrono::duration_cast<ns>(Time::now() - prev).count());
}

// ! [infer_request:get_performance_counts]
void TemplateInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    InferenceEngineProfileInfo info;
    info.execution_index = 0;
    info.status = InferenceEngineProfileInfo::EXECUTED;
    info.cpu_uSec = info.realTime_uSec = _inputPreprocessTime / 1000;
    perfMap["1. input preprocessing"] = info;
    info.cpu_uSec = 0;
    info.realTime_uSec = _inputTransferTime / 1000;
    perfMap["2. input transfer to a device"] = info;
    info.cpu_uSec = 0;
    info.realTime_uSec = _executeTime / 1000;
    perfMap["3. execution time"] = info;
    info.cpu_uSec = 0;
    info.realTime_uSec = _outputTransferTime / 1000;
    perfMap["4. output transfer from a device"] = info;
    info.cpu_uSec = info.realTime_uSec = _outputPostProcessTime / 1000;
    perfMap["5. output postprocessing"] = info;
}
// ! [infer_request:get_performance_counts]
