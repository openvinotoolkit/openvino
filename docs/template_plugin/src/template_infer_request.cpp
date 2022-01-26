// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_infer_request.hpp"

#include <debug.h>
#include <ie_compound_blob.h>

#include <algorithm>
#include <map>
#include <memory>
#include <ngraph/runtime/reference/convert.hpp>
#include <string>
#include <utility>

#include "blob_factory.hpp"
#include "ie_api.h"
#include "ie_common.h"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "template_executable_network.hpp"
#include "template_itt.hpp"
#include "template_plugin.hpp"

using namespace TemplatePlugin;
using namespace InferenceEngine;

using Time = std::chrono::high_resolution_clock;

// ! [infer_request:ctor]
TemplateInferRequest::TemplateInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                           const InferenceEngine::OutputsDataMap& networkOutputs,
                                           const std::shared_ptr<TemplatePlugin::ExecutableNetwork>& executableNetwork)
    : IInferRequestInternal(networkInputs, networkOutputs),
      _executableNetwork(executableNetwork) {
    createInferRequest();
}

TemplateInferRequest::TemplateInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                           const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                           const std::shared_ptr<TemplatePlugin::ExecutableNetwork>& executableNetwork)
    : IInferRequestInternal(inputs, outputs),
      _executableNetwork(executableNetwork) {
    createInferRequest();
}

void TemplateInferRequest::createInferRequest() {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto requestID = std::to_string(_executableNetwork->_requestId.fetch_add(1));

    std::string name = _executableNetwork->_function->get_friendly_name() + "_Req" + requestID;
    _profilingTask = {
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name +
                              "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name +
                              "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name +
                              "_StartPipline"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name +
                              "_WaitPipline"),
    };

    _executable = _executableNetwork->_plugin->_backend->compile(_executableNetwork->_function);

    allocateDeviceBuffers();
    allocateBlobs();
}
// ! [infer_request:ctor]

// ! [infer_request:dtor]
TemplateInferRequest::~TemplateInferRequest() {
    _executableNetwork->_requestId--;
}
// ! [infer_request:dtor]

void TemplateInferRequest::allocateDeviceBuffers() {
    // Allocate plugin backend specific memory handles
    _inputTensors.resize(_networkInputs.size());
    _outputTensors.resize(_networkOutputs.size());
}

template <typename BlobData, typename GetNetworkPrecisionF>
static void AllocateImplSingle(BlobMap& blobMap,
                               BlobMap& networkBlobMap,
                               const BlobData& blobData,
                               GetNetworkPrecisionF&& GetNetworkPrecision,
                               const SizeVector& dims) {
    const auto& precision = blobData.second->getTensorDesc().getPrecision();
    auto layout = blobData.second->getTensorDesc().getLayout();
    const auto deviceLayout = TensorDesc::getLayoutByDims(dims);
    Blob::Ptr& blob = blobMap[blobData.first];
    if (!blob) {
        blob = make_blob_with_precision({precision, dims, layout});
        blob->allocate();
    } else {
        blob->setShape(dims);
    }

    auto networkPrecision = InferenceEngine::details::convertPrecision(GetNetworkPrecision(blobData.first));
    Blob::Ptr networkBlob;
    if (precision == networkPrecision && layout == deviceLayout) {
        networkBlob = blob;
    } else {
        networkBlob = make_blob_with_precision({networkPrecision, dims, deviceLayout});
        networkBlob->allocate();
    }
    networkBlobMap[blobData.first] = networkBlob;
}

template <typename BlobDataMap, typename GetNetworkPrecisionF>
static void AllocateImpl(const BlobDataMap& userDataMap,
                         BlobMap& userBlobMap,
                         BlobMap& deviceBlobMap,
                         GetNetworkPrecisionF&& GetNetworkPrecision,
                         bool isInputBlob = true) {
    for (const auto& userData : userDataMap) {
        IE_SUPPRESS_DEPRECATED_START
        auto partialShape = userData.second->getPartialShape();
        IE_SUPPRESS_DEPRECATED_END
        SizeVector dims;
        if (partialShape.is_static()) {
            dims = userData.second->getTensorDesc().getDims();
        } else if (partialShape.rank().is_static()) {
            dims = SizeVector(partialShape.rank().get_length(), 0);
        } else {
            dims = SizeVector{0};
        }
        AllocateImplSingle(userBlobMap, deviceBlobMap, userData, GetNetworkPrecision, dims);
    }
}

void TemplateInferRequest::allocateBlobs() {
    auto&& parameters = _executableNetwork->_function->get_parameters();
    AllocateImpl(_networkInputs, _inputs, _deviceInputs, [&](const std::string& blobName) {
        return parameters.at(_executableNetwork->_inputIndex.at(blobName))->get_element_type();
    });
    auto&& results = _executableNetwork->_function->get_results();
    AllocateImpl(
        _networkOutputs,
        _outputs,
        _networkOutputBlobs,
        [&](const std::string& blobName) {
            return results.at(_executableNetwork->_outputIndex.at(blobName))->get_element_type();
        },
        false);
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

template <typename SrcT, typename DstT>
static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    ngraph::runtime::reference::convert<SrcT, DstT>(
        InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const SrcT*>(),
        InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<DstT*>(),
        src->size());
}

static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    switch (src->getTensorDesc().getPrecision()) {
    case Precision::U8: {
        switch (dst->getTensorDesc().getPrecision()) {
        case Precision::U8:
            break;
        case Precision::FP32: {
            blobCopy<std::uint8_t, float>(src, dst);
        } break;
        default: {
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision()
                                     << " to " << dst->getTensorDesc().getPrecision();
        }
        }
    } break;
    case Precision::FP32: {
        switch (dst->getTensorDesc().getPrecision()) {
        case Precision::FP32:
            break;
        case Precision::U8: {
            blobCopy<float, std::uint8_t>(src, dst);
        } break;
        default: {
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision()
                                     << " to " << dst->getTensorDesc().getPrecision();
        }
        }
    } break;
    case Precision::I64: {
        switch (dst->getTensorDesc().getPrecision()) {
        case Precision::I64:
            break;
        case Precision::I32: {
            blobCopy<int64_t, int32_t>(src, dst);
        } break;
        default: {
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision()
                                     << " to " << dst->getTensorDesc().getPrecision();
        }
        }
    } break;
    case Precision::I16: {
        switch (dst->getTensorDesc().getPrecision()) {
        case Precision::I16:
            break;
        case Precision::FP32: {
            blobCopy<int16_t, float>(src, dst);
        } break;
        default: {
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision()
                                     << " to " << dst->getTensorDesc().getPrecision();
        }
        }
    } break;
    case Precision::I8: {
        switch (dst->getTensorDesc().getPrecision()) {
        case Precision::I8:
            break;
        case Precision::FP32: {
            blobCopy<int8_t, float>(src, dst);
        } break;
        default: {
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision()
                                     << " to " << dst->getTensorDesc().getPrecision();
        }
        }
    } break;
    case Precision::BOOL: {
        switch (dst->getTensorDesc().getPrecision()) {
        case Precision::BOOL:
            break;
        case Precision::FP32: {
            blobCopy<bool, float>(src, dst);
        } break;
        default: {
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision()
                                     << " to " << dst->getTensorDesc().getPrecision();
        }
        }
    } break;
    case Precision::U16: {
        switch (dst->getTensorDesc().getPrecision()) {
        case Precision::U16:
            break;
        case Precision::FP32: {
            blobCopy<uint16_t, float>(src, dst);
        } break;
        default: {
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision()
                                     << " to " << dst->getTensorDesc().getPrecision();
        }
        }
    } break;
    default: {
        IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision();
    }
    }
}

// ! [infer_request:infer_preprocess]
void TemplateInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[Preprocess]);
    auto start = Time::now();
    convertBatchedInputBlobs();
    // NOTE: After IInferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in constructor.
    IInferRequestInternal::execDataPreprocessing(_deviceInputs);
    for (auto&& networkInput : _deviceInputs) {
        auto index = _executableNetwork->_inputIndex[networkInput.first];
        const auto& parameter = _executableNetwork->_function->get_parameters()[index];
        auto parameterShape = networkInput.second->getTensorDesc().getDims();
        const auto& parameterType = parameter->get_element_type();
        _inputTensors[index] = _executableNetwork->_plugin->_backend->create_tensor(
            parameterType,
            parameterShape,
            InferenceEngine::as<InferenceEngine::MemoryBlob>(networkInput.second)->rmap().as<void*>());
    }
    for (auto&& output : _outputs) {
        auto outputBlob = output.second;
        auto networkOutput = _networkOutputBlobs[output.first];
        auto index = _executableNetwork->_outputIndex[output.first];
        if (outputBlob->getTensorDesc().getPrecision() == networkOutput->getTensorDesc().getPrecision()) {
            networkOutput = outputBlob;
        }
        const auto& result = _executableNetwork->_function->get_results()[index];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            _outputTensors[index] = _executableNetwork->_plugin->_backend->create_tensor();
            continue;
        }
        const auto& resultShape = result->get_shape();
        const auto& resultType = result->get_element_type();
        _outputTensors[index] = _executableNetwork->_plugin->_backend->create_tensor(
            resultType,
            resultShape,
            InferenceEngine::as<InferenceEngine::MemoryBlob>(networkOutput)->wmap().as<void*>());
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
    for (auto&& output : _networkOutputs) {
        auto index = _executableNetwork->_outputIndex[output.first];
        const auto& result = _executableNetwork->_function->get_results()[index];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            // Touch blob to allocate it
            GetBlob(output.first);
        }
        auto outputBlob = _outputs.at(output.first);
        auto networkOutput = _networkOutputBlobs[output.first];
        if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision()) {
            blobCopy(networkOutput, outputBlob);
        } else if (result->get_output_partial_shape(0).is_dynamic()) {
            auto tensor = _outputTensors[_executableNetwork->_outputIndex.at(output.first)];
            tensor->read(InferenceEngine::as<InferenceEngine::MemoryBlob>(outputBlob)->wmap().as<char*>(),
                         tensor->get_size_in_bytes());
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
        // ROI blob is returned only if it was set previously. Otherwise default blob is returned.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
        } else {
            data = _inputs[name];
            SizeVector dims;
            if (!data) {
                auto&& parameters = _executableNetwork->_function->get_parameters();
                const auto& pshape = parameters.at(_executableNetwork->_inputIndex.at(name))->get_partial_shape();
                dims = pshape.is_dynamic() ? SizeVector({0}) : pshape.get_shape();
                AllocateImplSingle(
                    _inputs,
                    _deviceInputs,
                    *_networkInputs.find(name),
                    [&](const std::string& blobName) {
                        return parameters.at(_executableNetwork->_inputIndex.at(blobName))->get_element_type();
                    },
                    dims);
                data = _inputs[name];
            } else {
                dims = data->getTensorDesc().getDims();
            }
            checkBlob(data, name, true, foundInput->getTensorDesc().getLayout() != SCALAR ? dims : oneVector);
            auto& devBlob = _deviceInputs[name];
            if (preProcessingRequired(foundInput, data, devBlob)) {
                // if no devBlob, performs inplace
                addInputPreProcessingFor(name, data, devBlob ? devBlob : _inputs[name]);
            }
        }
    } else {
        data = _outputs[name];
        SizeVector dims;
        IE_SUPPRESS_DEPRECATED_START
        if (!foundOutput->isDynamic()) {
            dims = foundOutput->getTensorDesc().getDims();
        } else if (_outputTensors[_executableNetwork->_outputIndex.at(name)] &&
                   _outputTensors[_executableNetwork->_outputIndex.at(name)]->get_partial_shape().is_static()) {
            dims = _outputTensors[_executableNetwork->_outputIndex.at(name)]->get_shape();
        } else {
            auto rank = foundOutput->getPartialShape().rank();
            dims = SizeVector(rank.is_dynamic() ? 1 : rank.get_length(), 0);
        }
        IE_SUPPRESS_DEPRECATED_END

        if (data->getTensorDesc().getDims() != dims) {
            auto&& results = _executableNetwork->_function->get_results();
            AllocateImplSingle(
                _outputs,
                _networkOutputBlobs,
                *_networkOutputs.find(name),
                [&](const std::string& blobName) {
                    return results.at(_executableNetwork->_outputIndex.at(blobName))->get_element_type();
                },
                dims);
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
    const bool isInput = findInputAndOutputBlobByName(name, foundInput, foundOutput);
    const bool compoundBlobPassed = userBlob->is<CompoundBlob>();
    const bool remoteBlobPassed = userBlob->is<RemoteBlob>();
    if (!compoundBlobPassed && !remoteBlobPassed && userBlob->buffer() == nullptr)
        IE_THROW(NotAllocated) << "Input data was not allocated. Input name: \'" << name << "\'";
    IE_SUPPRESS_DEPRECATED_START
    if (userBlob->size() == 0 &&
        !((foundInput && foundInput->getInputData()->isDynamic()) || (foundOutput && foundOutput->isDynamic()))) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }
    IE_SUPPRESS_DEPRECATED_END

    size_t dataSize = userBlob->size();
    if (isInput) {
        // ilavreno: the condition below is obsolete, but we need an exact list of precisions
        // which are supports by G-API preprocessing
        if (foundInput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user input precision";
        }

        auto& devBlob = _deviceInputs[name];
        auto usrDims = userBlob->getTensorDesc().getDims();
        auto usrLayout = userBlob->getTensorDesc().getLayout();
        auto devDims = devBlob->getTensorDesc().getDims();
        auto devLayout = devBlob->getTensorDesc().getLayout();
        auto devPrecision = devBlob->getTensorDesc().getPrecision();
        IE_SUPPRESS_DEPRECATED_START
        if (foundInput->getInputData()->isDynamic() && (devDims != usrDims || devLayout != usrLayout)) {
            devBlob = make_blob_with_precision({devPrecision, usrDims, TensorDesc::getLayoutByDims(usrDims)});
            devBlob->allocate();
            _deviceInputs[name] = devBlob;
        }
        IE_SUPPRESS_DEPRECATED_END
        const bool preProcRequired = preProcessingRequired(foundInput, userBlob, devBlob);
        if (compoundBlobPassed && !preProcRequired) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            addInputPreProcessingFor(name, userBlob, devBlob ? devBlob : _inputs[name]);
        } else {
            size_t inputSize = devBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                                   ? InferenceEngine::details::product(devBlob->getTensorDesc().getDims())
                                   : 1;
            if (dataSize != inputSize) {
                IE_THROW() << "Input blob size is not equal network input size (" << dataSize << "!=" << inputSize
                           << ").";
            }
            _inputs[name] = userBlob;
            devBlob = userBlob;
        }
        _batched_inputs.erase(name);
    } else {
        if (compoundBlobPassed) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }
        auto& devBlob = _networkOutputBlobs[name];
        auto usrDims = userBlob->getTensorDesc().getDims();
        auto usrLayout = userBlob->getTensorDesc().getLayout();
        auto devDims = devBlob->getTensorDesc().getDims();
        auto devLayout = devBlob->getTensorDesc().getLayout();
        auto devPrecision = devBlob->getTensorDesc().getPrecision();
        IE_SUPPRESS_DEPRECATED_START
        if (foundOutput->isDynamic() && (devDims != usrDims || devLayout != usrLayout)) {
            devBlob = make_blob_with_precision({devPrecision, usrDims, TensorDesc::getLayoutByDims(usrDims)});
            devBlob->allocate();
            _networkOutputBlobs[name] = devBlob;
        }
        IE_SUPPRESS_DEPRECATED_END
        size_t outputSize = devBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                                ? details::product(devBlob->getTensorDesc().getDims())
                                : 1;
        if (dataSize != outputSize) {
            IE_THROW() << "Output blob size is not equal network output size (" << dataSize << "!=" << outputSize
                       << ").";
        }
        if (foundOutput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user output precision";
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

    info.cpu_uSec = info.realTime_uSec = _durations[Preprocess].count();
    perfMap["1. input preprocessing"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["2. input transfer to a device"] = info;
    info.cpu_uSec = info.realTime_uSec = _durations[StartPipeline].count();
    perfMap["3. execution time"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["4. output transfer from a device"] = info;
    info.cpu_uSec = info.realTime_uSec = _durations[Postprocess].count();
    perfMap["5. output postprocessing"] = info;
    return perfMap;
}
// ! [infer_request:get_performance_counts]
