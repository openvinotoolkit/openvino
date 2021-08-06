// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <memory>
#include <string>

#include <ie_blob.h>
#include <ie_common.h>
#include <ie_preprocess.hpp>
#include <ie_compound_blob.h>
#include <ie_algorithm.hpp>
#include <ie_remote_context.hpp>
#include <debug.h>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/plugin_itt.hpp>


namespace InferenceEngine {

IInferRequestInternal::~IInferRequestInternal() {}

IInferRequestInternal::IInferRequestInternal(const InputsDataMap& networkInputs, const OutputsDataMap& networkOutputs) :
    // We should copy maps since they can be overriden in SetBlob with preprocess
    _networkInputs{copyInfo(networkInputs)},
    _networkOutputs{copyInfo(networkOutputs)} {
}

void IInferRequestInternal::Infer() {
    checkBlobs();
    InferImpl();
}

void IInferRequestInternal::InferImpl() {
    IE_THROW(NotImplemented);
}

void IInferRequestInternal::Cancel() {
    IE_THROW(NotImplemented);
}

std::map<std::string, InferenceEngineProfileInfo> IInferRequestInternal::GetPerformanceCounts() const {
    IE_THROW(NotImplemented);
}

void IInferRequestInternal::SetBlob(const std::string& name, const Blob::Ptr& userBlob) {
    OV_ITT_SCOPED_TASK(itt::domains::Plugin, "SetBlob");
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }
    if (!userBlob) IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";
    const bool compoundBlobPassed = userBlob->is<CompoundBlob>();
    const bool remoteBlobPassed   = userBlob->is<RemoteBlob>();
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
        const bool preProcRequired = preProcessingRequired(foundInput, userBlob, devBlob);
        if (compoundBlobPassed && !preProcRequired) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            addInputPreProcessingFor(name, userBlob, devBlob ? devBlob : _inputs[name]);
        } else {
            size_t inputSize = foundInput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                ? InferenceEngine::details::product(foundInput->getTensorDesc().getDims())
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
        size_t outputSize = foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
            ? details::product(foundOutput->getTensorDesc().getDims()) :
            1;
        if (dataSize != outputSize) {
            IE_THROW() << "Output blob size is not equal network output size (" << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch) << "Failed to set Blob with precision not corresponding to user output precision";
        }
        // ilavreno: this condition is valid for most plugins except MYRIAD
        // it is able to perform layout conversion for output blob dynamically
        // if (foundOutput->getLayout() != userBlob->getTensorDesc().getLayout()) {
        //     IE_THROW(ParameterMismatch) << "Failed to set Blob with layout not corresponding to user output layout";
        // }
        _outputs[name] = userBlob;
    }
}

Blob::Ptr IInferRequestInternal::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::Plugin, "GetBlob");
    Blob::Ptr data;
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    const SizeVector oneVector = { 1 };
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        // ROI blob is returned only if it was set previously. Otherwise default blob is returned.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
        } else {
            data = _inputs[name];
            checkBlob(data, name, true,
                foundInput->getTensorDesc().getLayout() != SCALAR
                ? foundInput->getTensorDesc().getDims()
                : oneVector);

            auto& devBlob = _deviceInputs[name];
            if (preProcessingRequired(foundInput, data, devBlob)) {
                // if no devBlob, performs inplace
                addInputPreProcessingFor(name, data, devBlob ? devBlob : _inputs[name]);
            }
        }
    } else {
        data = _outputs[name];
        checkBlob(data, name, false,
            foundOutput->getTensorDesc().getLayout() != SCALAR
            ? foundOutput->getTensorDesc().getDims()
            : oneVector);
    }
    return data;
}

void IInferRequestInternal::SetBlob(const std::string& name, const Blob::Ptr& data, const PreProcessInfo& info) {
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
       foundInput->getPreProcess() = copyPreProcess(info);
    } else {
        IE_THROW() << "Pre-process can't be set to output blob";
    }

    SetBlob(name, data);
}

const PreProcessInfo& IInferRequestInternal::GetPreProcess(const std::string& name) const {
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        return foundInput->getPreProcess();
    } else {
        IE_THROW() << "Output blob can't have pre-processing";
    }
}

void IInferRequestInternal::SetBatch(int batch) {
    IE_THROW(NotImplemented);
}

std::vector<std::shared_ptr<IVariableStateInternal>> IInferRequestInternal::QueryState() {
    IE_THROW(NotImplemented);
}

void IInferRequestInternal::StartAsync() {
    checkBlobs();
    StartAsyncImpl();
}

void IInferRequestInternal::StartAsyncImpl() {
    IE_THROW(NotImplemented);
}

StatusCode IInferRequestInternal::Wait(int64_t millis_timeout) {
    IE_THROW(NotImplemented);
}

void IInferRequestInternal::SetCallback(Callback callback) {
    _callback = std::move(callback);
}

void IInferRequestInternal::execDataPreprocessing(InferenceEngine::BlobMap& preprocessedBlobs, bool serial) {
    for (auto& input : preprocessedBlobs) {
        // If there is a pre-process entry for an input then it must be pre-processed
        // using preconfigured resize algorithm.
        auto it = _preProcData.find(input.first);
        if (it != _preProcData.end()) {
            it->second->execute(input.second, _networkInputs[input.first]->getPreProcess(), serial, m_curBatch);
        }
    }
}

bool IInferRequestInternal::findInputAndOutputBlobByName(const std::string& name, InputInfo::Ptr& foundInput, DataPtr& foundOutput) const {
    foundInput = nullptr;
    foundOutput = nullptr;
    if (_networkOutputs.empty()) {
        IE_THROW() << "Internal error: network outputs is not set";
    }
    auto foundInputPair = std::find_if(std::begin(_networkInputs), std::end(_networkInputs),
                                        [&](const std::pair<std::string, InputInfo::Ptr>& pair) {
                                            return pair.first == name;
                                        });
    auto foundOutputPair = std::find_if(std::begin(_networkOutputs), std::end(_networkOutputs),
                                        [&](const std::pair<std::string, DataPtr>& pair) {
                                            return pair.first == name;
                                        });
    bool retVal;

    if (foundInputPair != std::end(_networkInputs)) {
        foundInput = foundInputPair->second;
        retVal = true;
    } else if (foundOutputPair != std::end(_networkOutputs)) {
        foundOutput = foundOutputPair->second;
        retVal = false;
    } else {
        IE_THROW(NotFound) << "Failed to find input or output with name: \'" << name << "\'";
    }
    return retVal;
}

void IInferRequestInternal::checkBlob(const Blob::Ptr& blob, const std::string& name, bool isInput, const SizeVector& refDims) const {
    std::string bType = isInput ? "Input" : "Output";
    std::string sType = isInput ? "input" : "output";
    std::string strNotAllocated(bType + " data was not allocated.");
    std::string strNotMatched("The " + sType + " blob size is not equal to the network " + sType + " size");

    if (!blob) {
        IE_THROW(NotAllocated) << strNotAllocated;
    }
    size_t refSize;
    if (refDims.empty()) {
        SizeVector dims;
        if (isInput) {
            auto foundInputPair = std::find_if(std::begin(_networkInputs), std::end(_networkInputs),
                                                [&](const std::pair<std::string, InputInfo::Ptr>& pair) {
                                                    return pair.first == name;
                                                });
            if (foundInputPair == std::end(_networkInputs)) {
                IE_THROW(NotFound) << "Failed to find input with name: \'" << name << "\'";
            }
            dims = foundInputPair->second->getTensorDesc().getDims();
            refSize = foundInputPair->second->getTensorDesc().getLayout() != SCALAR
                ? details::product(dims)
                : 1;
        } else {
            auto foundOutputPair = std::find_if(std::begin(_networkOutputs), std::end(_networkOutputs),
                                                [&](const std::pair<std::string, DataPtr>& pair) {
                                                    return pair.first == name;
                                                });
            if (foundOutputPair == std::end(_networkOutputs)) {
                IE_THROW(NotFound) << "Failed to find output with name: \'" << name << "\'";
            }
            dims = foundOutputPair->second->getTensorDesc().getDims();
            refSize = foundOutputPair->second->getTensorDesc().getLayout() != SCALAR
                ? details::product(dims)
                : 1;
        }
    } else {
        refSize = details::product(refDims);
    }

    if (refSize != blob->size()) {
        IE_THROW() << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
    }
    const bool remoteBlobPassed = blob->is<RemoteBlob>();
    if (!remoteBlobPassed && blob->buffer() == nullptr) IE_THROW() << strNotAllocated;
}

void IInferRequestInternal::checkBlobs() {
    for (auto const& input : _inputs) {
        checkBlob(input.second, input.first, true);
    }
    for (auto const& output : _outputs) {
        checkBlob(output.second, output.first, false);
    }
}

void IInferRequestInternal::setPointerToExecutableNetworkInternal(const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork) {
    _exeNetwork = exeNetwork;
}

bool IInferRequestInternal::preProcessingRequired(const InputInfo::Ptr& info, const Blob::Ptr& userBlob, const Blob::Ptr& deviceBlob) {
    // pre-processing is required if:
    // 1. resize algorithm is specified (resize required)
    // 2. color format specified:
    // 2.a. color format is not equal to network's expected (color conversion required)
    // 2.b. network's layout != blob's layout (reorder required)
    // 3. precision conversion is required

    const auto& preProcessInfo = info->getPreProcess();
    const auto inputColorFormat = preProcessInfo.getColorFormat();
    // FIXME: support other network's input formats once the API is ready. Assuming input is in
    // the BGR format by default
    const auto networkColorFormat = ColorFormat::BGR;
    const bool colorFormatSpecified = inputColorFormat != ColorFormat::RAW;

    auto blob_layout = [](const Blob::Ptr& b) { return b->getTensorDesc().getLayout();   };
    auto blob_prec   = [](const Blob::Ptr& b) { return b->getTensorDesc().getPrecision();};

    auto dst_layout = deviceBlob ? blob_layout(deviceBlob) : info->getLayout();
    auto dst_prec   = deviceBlob ? blob_prec(deviceBlob)   : info->getPrecision();

    //FIXME: remove the first part to allow any needed conversion?
    const bool need_layout_conv = (colorFormatSpecified || deviceBlob) &&
                                    (blob_layout(userBlob) != dst_layout);

    return preProcessInfo.getResizeAlgorithm() != ResizeAlgorithm::NO_RESIZE ||
            (colorFormatSpecified && inputColorFormat != networkColorFormat) ||
            need_layout_conv ||
            (blob_prec(userBlob) != dst_prec);
}

void IInferRequestInternal::addInputPreProcessingFor(const std::string& name, Blob::Ptr const& from, const Blob::Ptr& to) {
    auto ppDataIt = _preProcData.find(name);
    if (ppDataIt == _preProcData.end()) {
        ppDataIt = (_preProcData.emplace(name, CreatePreprocDataHelper())).first;
    }

    auto& preproc_ptr = ppDataIt->second;
    preproc_ptr->isApplicable(from,  to);
    // Stores the given blob as ROI blob. It will be used to fill in network input
    // during pre-processing
    preproc_ptr->setRoiBlob(from);
}

void* IInferRequestInternal::GetUserData() noexcept {
    return _userData;
}

void IInferRequestInternal::SetUserData(void* userData) noexcept {
    _userData = userData;
}
}  // namespace InferenceEngine
