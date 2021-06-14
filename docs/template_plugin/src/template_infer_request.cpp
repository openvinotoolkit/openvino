// Copyright (C) 2018-2021 Intel Corporation
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
#include "ie_ngraph_utils.hpp"
#include "template_executable_network.hpp"
#include "template_itt.hpp"
#include "template_plugin.hpp"

using namespace TemplatePlugin;
using namespace InferenceEngine;

using Time = std::chrono::high_resolution_clock;

// ! [infer_request:ctor]
TemplateInferRequest::TemplateInferRequest(const InferenceEngine::InputsDataMap& networkInputs, const InferenceEngine::OutputsDataMap& networkOutputs,
                                           const std::shared_ptr<TemplatePlugin::ExecutableNetwork>& executableNetwork)
    : IInferRequestInternal(networkInputs, networkOutputs), _executableNetwork(executableNetwork) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto requestID = std::to_string(_executableNetwork->_requestId.fetch_add(1));

    std::string name = _executableNetwork->_function->get_friendly_name() + "_Req" + requestID;
    _profilingTask = {
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_StartPipline"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_WaitPipline"),
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
static void AllocateImplSingle(BlobMap& blobMap, BlobMap& networkBlobMap, BlobData& blobData, GetNetworkPrecisionF&& GetNetworkPrecision,
                               const SizeVector& dims) {
    auto& precision = blobData.second->getTensorDesc().getPrecision();
    auto layout = blobData.second->getTensorDesc().getLayout();
    if (dims.size() > 0 && layout == InferenceEngine::Layout::SCALAR) {
        layout = InferenceEngine::Layout::ANY;
    }
    const auto deviceLayout = TensorDesc::getLayoutByDims(dims);
    Blob::Ptr blob;
    blob = make_blob_with_precision({precision, dims, layout});
    blob->allocate();
    blobMap[blobData.first] = blob;

    auto networkPresion = InferenceEngine::details::convertPrecision(GetNetworkPrecision(blobData.first));
    Blob::Ptr networkBlob;
    if (precision == networkPresion && layout == deviceLayout) {
        networkBlob = blob;
    } else {
        networkBlob = make_blob_with_precision({networkPresion, dims, deviceLayout});
        networkBlob->allocate();
    }
    networkBlobMap[blobData.first] = networkBlob;
}

template <typename BlobDataMap, typename GetNetworkPrecisionF>
static void AllocateImpl(const BlobDataMap& userDataMap, BlobMap& userBlobMap, BlobMap& deviceBlobMap, GetNetworkPrecisionF&& GetNetworkPrecision,
                         bool isInputBlob = true) {
    for (auto&& userData : userDataMap) {
        auto& dims = userData.second->getTensorDesc().getDims();
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
        _networkOutputs, _outputs, _networkOutputBlobs,
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
    ngraph::runtime::reference::convert<SrcT, DstT>(InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const SrcT*>(),
                                                    InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<DstT*>(), src->size());
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
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision() << " to "
                                     << dst->getTensorDesc().getPrecision();
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
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision() << " to "
                                     << dst->getTensorDesc().getPrecision();
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
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision() << " to "
                                     << dst->getTensorDesc().getPrecision();
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
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision() << " to "
                                     << dst->getTensorDesc().getPrecision();
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
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision() << " to "
                                     << dst->getTensorDesc().getPrecision();
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
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision() << " to "
                                     << dst->getTensorDesc().getPrecision();
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
            IE_THROW(NotImplemented) << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision() << " to "
                                     << dst->getTensorDesc().getPrecision();
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
    // NOTE: After IInferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in constructor.
    IInferRequestInternal::execDataPreprocessing(_deviceInputs);
    for (auto&& networkInput : _deviceInputs) {
        auto index = _executableNetwork->_inputIndex[networkInput.first];
        const auto& parameter = _executableNetwork->_function->get_parameters()[index];
        auto parameterShape =
            m_realShapes.find(networkInput.first) != m_realShapes.end() ? ngraph::Shape(m_realShapes.at(networkInput.first)) : parameter->get_shape();
        const auto& parameterType = parameter->get_element_type();
        _inputTensors[index] = _executableNetwork->_plugin->_backend->create_tensor(
            parameterType, parameterShape, InferenceEngine::as<InferenceEngine::MemoryBlob>(networkInput.second)->rmap().as<void*>());
    }
    // Go over all outputs in the model, not over all allocated blobs because for a part of the outputs
    // blobs may not be yet allocated due to unknown dimensions
    // TODO: should we really go over all results in the network, or it is better to go over _networkOutputs?
    for (size_t index = 0; index < _executableNetwork->_function->get_results().size(); ++index) {
        _outputTensors[index] = _executableNetwork->_plugin->_backend->create_tensor();
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
        auto outputBlob = output.second;
        auto networkOutput = _networkOutputBlobs[output.first];
        // perform precision conversion of network output's precision and computational
        // graph output's precision are different
        if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision()) {
            blobCopy(networkOutput, outputBlob);
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
            if (m_realShapes.find(name) == m_realShapes.end() && foundInput->getInputData()->isDynamic())
                IE_THROW() << "Cannot get blob " << name << " which contains dynamic shapes";
            const auto& dims = m_realShapes.find(name) != m_realShapes.end() ? m_realShapes[name] : foundInput->getTensorDesc().getDims();
            if (data) {
                if (data->getTensorDesc().getDims() != dims) {
                    // TODO: implement something smart here instead of raw re-allocation
                    data.reset();
                }
            }
            if (!data) {
                auto&& parameters = _executableNetwork->_function->get_parameters();
                AllocateImplSingle(
                    _inputs, _deviceInputs, *_networkInputs.find(name),
                    [&](const std::string& blobName) {
                        return parameters.at(_executableNetwork->_inputIndex.at(blobName))->get_element_type();
                    },
                    dims);
                data = _inputs[name];
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
        if (!foundOutput->isDynamic()) {
            dims = foundOutput->getTensorDesc().getDims();
        } else if (_outputTensors[_executableNetwork->_outputIndex.at(name)]->get_partial_shape().is_static()) {
            dims = _outputTensors[_executableNetwork->_outputIndex.at(name)]->get_shape();
        } else {
            IE_THROW() << "Output blob dimensions are not all known for output name " << name << " with partial shape: " << foundOutput->getPartialShape();
        }

        if (data) {
            if (data->getTensorDesc().getDims() != dims) {
                // TODO: implement something smart here instead of raw re-allocation
                data.reset();
            }
        }

        if (!data) {
            auto&& results = _executableNetwork->_function->get_results();
            AllocateImplSingle(
                _outputs, _networkOutputBlobs, *_networkOutputs.find(name),
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
    const bool compoundBlobPassed = userBlob->is<CompoundBlob>();
    const bool remoteBlobPassed = userBlob->is<RemoteBlob>();
    if (!compoundBlobPassed && !remoteBlobPassed && userBlob->buffer() == nullptr)
        IE_THROW(NotAllocated) << "Input data was not allocated. Input name: \'" << name << "\'";
    if (userBlob->size() == 0) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }

    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    size_t dataSize = userBlob->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        // ilavreno: the condition below is obsolete, but we need an exact list of precisions
        // which are supports by G-API preprocessing
        if (foundInput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set Blob with precision not corresponding to user input precision";
        }

        auto& devBlob = _deviceInputs[name];
        auto usrDims = userBlob->getTensorDesc().getDims();
        auto usrLayout = userBlob->getTensorDesc().getLayout();
        auto devDims = devBlob->getTensorDesc().getDims();
        auto devLayout = devBlob->getTensorDesc().getLayout();
        auto devPrecision = devBlob->getTensorDesc().getPrecision();
        if (foundInput->getInputData()->isDynamic() && (devDims != usrDims || devLayout != usrLayout)) {
            m_realShapes[name] = usrDims;
            devBlob = make_blob_with_precision({devPrecision, usrDims, TensorDesc::getLayoutByDims(usrDims)});
            devBlob->allocate();
            _deviceInputs[name] = devBlob;
        }
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
                IE_THROW() << "Input blob size is not equal network input size (" << dataSize << "!=" << inputSize << ").";
            }
            _inputs[name] = userBlob;
            devBlob = userBlob;
        }
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
        if (foundOutput->isDynamic() && (devDims != usrDims || devLayout != usrLayout)) {
            m_realShapes[name] = usrDims;
            devBlob = make_blob_with_precision({devPrecision, usrDims, TensorDesc::getLayoutByDims(usrDims)});
            devBlob->allocate();
            _networkOutputBlobs[name] = devBlob;
        }
        size_t outputSize = devBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR ? details::product(devBlob->getTensorDesc().getDims()) : 1;
        if (dataSize != outputSize) {
            IE_THROW() << "Output blob size is not equal network output size (" << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set Blob with precision not corresponding to user output precision";
        }
        _outputs[name] = userBlob;
    }
}
// ! [infer_request:set_blob]

// ! [infer_request:set_shape]
void TemplateInferRequest::SetShape(const std::string& name, const InferenceEngine::SizeVector& dims) {
    // Check partial shape compatibility
    ngraph::PartialShape newShape(dims);
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (!foundInput->getInputData()->getPartialShape().compatible(newShape))
            IE_THROW() << "New shape " << newShape << " for " << name << " is incompatible with original shape "
                       << foundInput->getInputData()->getPartialShape();
    } else {
        if (!foundOutput->getPartialShape().compatible(newShape))
            IE_THROW() << "New shape " << newShape << " for " << name << " is incompatible with original shape " << foundOutput->getPartialShape();
    }

    m_realShapes[name] = dims;
}
// ! [infer_request:set_shape]

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
