// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"

#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "cpp_interfaces/plugin_itt.hpp"
#include "debug.h"
#include "ie_algorithm.hpp"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_ngraph_utils.hpp"
#include "openvino/core/partial_shape.hpp"
#include "transformations/utils/utils.hpp"

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START

IInferRequestInternal::~IInferRequestInternal() {}

IInferRequestInternal::IInferRequestInternal(const InputsDataMap& networkInputs, const OutputsDataMap& networkOutputs)
    :  // We should copy maps since they can be overriden in SetBlob with preprocess
      _networkInputs{copyInfo(networkInputs)},
      _networkOutputs{copyInfo(networkOutputs)} {}

IInferRequestInternal::IInferRequestInternal(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                             const std::vector<std::shared_ptr<const ov::Node>>& outputs)
    : _parameters(inputs),
      _results(outputs) {
    const auto& create_old_data = [](const ov::Output<const ov::Node>& output) -> InferenceEngine::DataPtr {
        auto name = ov::op::util::get_ie_output_name(output);
        auto shape = output.get_partial_shape();
        auto rank = shape.rank().is_static() ? shape.rank().get_length() : -1;
        SizeVector dims(1, 0);
        if (shape.is_static()) {
            dims = output.get_shape();
        } else if (rank >= 0) {
            dims = SizeVector(rank, 0);
        }
        for (const auto& dim : shape) {
            if (dim.is_static() && dim.get_length() == 0)
                IE_THROW() << name << " has zero dimension which is not allowed";
        }
        const Layout rankLayout = rank < 0 ? Layout::BLOCKED : TensorDesc::getLayoutByRank(rank);
        const auto precision = InferenceEngine::details::convertPrecision(output.get_element_type());
        return std::make_shared<Data>(name, TensorDesc{precision, dims, rankLayout});
    };
    const auto& create_old_input_data =
        [create_old_data](const ov::Output<const ov::Node>& output) -> InferenceEngine::InputInfo::Ptr {
        auto info = std::make_shared<InferenceEngine::InputInfo>();
        info->setInputData(create_old_data(output));
        return info;
    };

    for (const auto& param : _parameters) {
        const auto& input = create_old_input_data(param->output(0));
        input->setName(param->get_friendly_name());
        _networkInputs[input->name()] = input;
    }

    for (const auto& result : _results) {
        auto input = result->input_value(0);
        const auto& output = create_old_data(ov::Output<const ov::Node>(input.get_node(), input.get_index()));
        _networkOutputs[output->getName()] = output;
    }
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

std::shared_ptr<const ov::Node> IInferRequestInternal::findInputByNodeName(const std::string& name) const {
    for (const auto& input : GetInputs()) {
        if (input->get_friendly_name() == name)
            return input;
    }
    return nullptr;
}

std::shared_ptr<const ov::Node> IInferRequestInternal::findOutputByNodeName(const std::string& name) const {
    for (const auto& output : GetOutputs()) {
        if (output->input_value(0).get_node()->get_friendly_name() == name)
            return output;
    }
    return nullptr;
}

void IInferRequestInternal::SetBlob(const std::string& name, const Blob::Ptr& userBlob) {
    OV_ITT_SCOPED_TASK(itt::domains::Plugin, "SetBlob");
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }
    if (!userBlob)
        IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    const bool isInput = findInputAndOutputBlobByName(name, foundInput, foundOutput);
    const auto input = findInputByNodeName(name);
    const auto output = findOutputByNodeName(name);

    if (userBlob->buffer() == nullptr)
        IE_THROW(NotAllocated) << "Input data was not allocated. Input name: \'" << name << "\'";
    if (userBlob->size() == 0 && !((input && input->get_output_partial_shape(0).is_dynamic()) ||
                                   (output && output->get_output_partial_shape(0).is_dynamic()))) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }
    const bool isInputDynamic = input && input->get_output_partial_shape(0).is_dynamic();
    const bool isOutputDynamic = output && output->get_input_partial_shape(0).is_dynamic();

    size_t dataSize = userBlob->size();
    if (isInput) {
        // ilavreno: the condition below is obsolete, but we need an exact list of precisions
        // which are supports by G-API preprocessing
        if (foundInput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user input precision";
        }

        auto& devBlob = _deviceInputs[name];

        size_t inputSize = foundInput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                               ? InferenceEngine::details::product(foundInput->getTensorDesc().getDims())
                               : 1;
        if (!isInputDynamic && dataSize != inputSize) {
            IE_THROW() << "Input tensor size is not equal network input size (" << dataSize << "!=" << inputSize
                       << ").";
        }
        _inputs[name] = userBlob;
        devBlob = userBlob;
    } else {
        size_t outputSize = foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                                ? details::product(foundOutput->getTensorDesc().getDims())
                                : 1;
        if (!isOutputDynamic && dataSize != outputSize) {
            IE_THROW() << "Output blob size is not equal network output size (" << dataSize << "!=" << outputSize
                       << ").";
        }
        if (foundOutput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user output precision";
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
    const SizeVector oneVector = {1};
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        const auto input = findInputByNodeName(name);
        const bool isInputDynamic = input && input->get_output_partial_shape(0).is_dynamic();
        data = _inputs[name];
        const auto& dims = foundInput->getTensorDesc().getDims();
        if (isInputDynamic)
            checkBlob(data, name, true);
        else
            checkBlob(data, name, true, foundInput->getTensorDesc().getLayout() != SCALAR ? dims : oneVector);
    } else {
        const auto output = findOutputByNodeName(name);
        const bool isOutputDynamic = output && output->get_output_partial_shape(0).is_dynamic();
        data = _outputs[name];
        const auto& dims = foundOutput->getTensorDesc().getDims();
        if (isOutputDynamic)
            checkBlob(data, name, false);
        else
            checkBlob(data, name, false, foundOutput->getTensorDesc().getLayout() != SCALAR ? dims : oneVector);
    }
    return data;
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

void IInferRequestInternal::execDataPreprocessing(InferenceEngine::BlobMap& preprocessedBlobs, bool serial) {}

bool IInferRequestInternal::findInputAndOutputBlobByName(const std::string& name,
                                                         InputInfo::Ptr& foundInput,
                                                         DataPtr& foundOutput) const {
    foundInput = nullptr;
    foundOutput = nullptr;
    if (_networkOutputs.empty()) {
        IE_THROW() << "Internal error: network outputs is not set";
    }
    auto foundInputPair = std::find_if(std::begin(_networkInputs),
                                       std::end(_networkInputs),
                                       [&](const std::pair<std::string, InputInfo::Ptr>& pair) {
                                           return pair.first == name;
                                       });
    auto foundOutputPair = std::find_if(std::begin(_networkOutputs),
                                        std::end(_networkOutputs),
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

void IInferRequestInternal::checkBlob(const Blob::Ptr& blob,
                                      const std::string& name,
                                      bool isInput,
                                      const SizeVector& refDims) const {
    std::string bType = isInput ? "Input" : "Output";
    std::string sType = isInput ? "input" : "output";
    std::string strNotAllocated(bType + " data was not allocated.");
    std::string strNotMatched("The " + sType + " blob size is not equal to the network " + sType + " size");

    if (!blob) {
        IE_THROW(NotAllocated) << strNotAllocated;
    }
    size_t refSize;
    bool isDynamic = false;
    if (refDims.empty()) {
        SizeVector dims;
        if (isInput) {
            auto foundInputPair = std::find_if(std::begin(_networkInputs),
                                               std::end(_networkInputs),
                                               [&](const std::pair<std::string, InputInfo::Ptr>& pair) {
                                                   return pair.first == name;
                                               });
            if (foundInputPair == std::end(_networkInputs)) {
                IE_THROW(NotFound) << "Failed to find input with name: \'" << name << "\'";
            }
            const auto input = findInputByNodeName(name);
            isDynamic = input && input->get_output_partial_shape(0).is_dynamic();
            dims = foundInputPair->second->getTensorDesc().getDims();
            refSize = foundInputPair->second->getTensorDesc().getLayout() != SCALAR ? details::product(dims) : 1;
        } else {
            auto foundOutputPair = std::find_if(std::begin(_networkOutputs),
                                                std::end(_networkOutputs),
                                                [&](const std::pair<std::string, DataPtr>& pair) {
                                                    return pair.first == name;
                                                });
            if (foundOutputPair == std::end(_networkOutputs)) {
                IE_THROW(NotFound) << "Failed to find output with name: \'" << name << "\'";
            }
            const auto output = findOutputByNodeName(name);
            isDynamic = output && output->get_output_partial_shape(0).is_dynamic();
            ngraph::PartialShape blobPartialShape(blob->getTensorDesc().getDims());
            if (output && output->get_output_partial_shape(0).compatible(blobPartialShape)) {
                dims = blob->getTensorDesc().getDims();
            } else {
                // TODO: it is strange to request tensor desc from data when the shapes are not compatible, probably we
                // need to immediately throw here
                dims = foundOutputPair->second->getTensorDesc().getDims();
            }
            refSize = foundOutputPair->second->getTensorDesc().getLayout() != SCALAR ? details::product(dims) : 1;
        }
    } else {
        refSize = details::product(refDims);
    }

    if (!isDynamic && refSize != blob->size()) {
        IE_THROW() << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
    }
    if (blob->buffer() == nullptr)
        IE_THROW() << strNotAllocated;
}

void IInferRequestInternal::checkBlobs() {
    for (auto const& input : _inputs) {
        checkBlob(input.second, input.first, true);
    }
    for (auto const& output : _outputs) {
        checkBlob(output.second, output.first, false);
    }
}

void IInferRequestInternal::setPointerToExecutableNetworkInternal(
    const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork) {
    _exeNetwork = exeNetwork;
}

std::shared_ptr<IExecutableNetworkInternal> IInferRequestInternal::getPointerToExecutableNetworkInternal() const {
    return _exeNetwork;
}

void IInferRequestInternal::setPointerToSo(const std::shared_ptr<void>& so) {
    _so = so;
}

std::shared_ptr<void> IInferRequestInternal::getPointerToSo() const {
    return _so;
}

void* IInferRequestInternal::GetUserData() noexcept {
    return _userData;
}

void IInferRequestInternal::SetUserData(void* userData) noexcept {
    _userData = userData;
}

void IInferRequestInternal::setModelInputsOutputs(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                  const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    _parameters = inputs;
    _results = outputs;
}

const std::vector<std::shared_ptr<const ov::Node>>& IInferRequestInternal::GetInputs() const {
    return _parameters;
}

const std::vector<std::shared_ptr<const ov::Node>>& IInferRequestInternal::GetOutputs() const {
    return _results;
}
}  // namespace InferenceEngine
