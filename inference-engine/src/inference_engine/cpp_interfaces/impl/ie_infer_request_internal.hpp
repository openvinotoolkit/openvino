// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <blob_factory.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "debug.h"
#include "ie_compound_blob.h"
#include "ie_memcpy.h"
#include "ie_preprocess_data.hpp"

namespace InferenceEngine {

class ExecutableNetworkInternal;

typedef std::shared_ptr<ExecutableNetworkInternal> ExecutableNetworkInternalPtr;

/**
 * @brief optional implementation of IInferRequestInternal to avoid duplication in all plugins
 */
class InferRequestInternal : virtual public IInferRequestInternal {
public:
    typedef std::shared_ptr<InferRequestInternal> Ptr;

    InferRequestInternal(const InputsDataMap& networkInputs, const OutputsDataMap& networkOutputs): m_curBatch(-1) {
        // We should copy maps in order to avoid modifications in the future.
        for (const auto& it : networkInputs) {
            InputInfo::Ptr newPtr;
            if (it.second) {
                newPtr.reset(new InputInfo());
                DataPtr newData(new Data(*it.second->getInputData()));
                copyPreProcess(it.second->getPreProcess(), newPtr->getPreProcess());

                newData->getInputTo().clear();
                newPtr->setInputData(newData);
            }
            _networkInputs[it.first] = newPtr;
        }

        for (const auto& it : networkOutputs) {
            DataPtr newData;
            if (it.second) {
                newData.reset(new Data(*it.second));
                newData->getInputTo().clear();
            }
            _networkOutputs[it.first] = newData;
        }
    }

    /**
     * @brief The minimal infer function to be implemented by plugins. It infers specified input(s) in synchronous mode
     * @note blocks all method of IInferRequest while request is ongoing (running or waiting in queue)
     */
    virtual void InferImpl() = 0;

    /**
     * @brief Default common implementation for all plugins with checking input and output blobs before inference
     */
    void Infer() override {
        checkBlobs();
        InferImpl();
    }

    /**
     * @brief Given optional implementation of setting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input
     * precision and size.
     */
    void SetBlob(const char* name, const Blob::Ptr& data) override {
        IE_PROFILING_AUTO_SCOPE(SetBlob)
        if (name == nullptr) {
            THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
        }
        if (!data) THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
        const bool compoundBlobPassed = data->is<CompoundBlob>();
        if (!compoundBlobPassed && data->buffer() == nullptr)
            THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
        if (data->size() == 0) {
            THROW_IE_EXCEPTION << "Input data is empty. Input name: \'" << name << "\'";
        }

        InputInfo::Ptr foundInput;
        DataPtr foundOutput;
        size_t dataSize = data->size();
        if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
            if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                                   << "Failed to set Blob with precision not corresponding to user input precision";
            }

            const bool preProcRequired = preProcessingRequired(foundInput, data);
            if (compoundBlobPassed && !preProcRequired) {
                THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                                   << "cannot set compound blob: supported only for input pre-processing";
            }

            if (preProcRequired) {
                if (_preProcData.find(name) == _preProcData.end()) {
                    _preProcData.emplace(name, CreatePreprocDataHelper());
                }
                _preProcData[name]->isApplicable(data, _inputs[name]);
                // Stores the given blob as ROI blob. It will be used to fill in network input
                // during pre-processing
                _preProcData[name]->setRoiBlob(data);
            } else {
                size_t inputSize = details::product(foundInput->getTensorDesc().getDims());
                if (dataSize != inputSize) {
                    THROW_IE_EXCEPTION << "Input blob size is not equal network input size (" << dataSize
                                       << "!=" << inputSize << ").";
                }
                _inputs[name] = data;
            }
        } else {
            if (compoundBlobPassed) {
                THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                                   << "cannot set compound blob: supported only for input pre-processing";
            }
            size_t outputSize = details::product(foundOutput->getDims());
            if (dataSize != outputSize) {
                THROW_IE_EXCEPTION << "Output blob size is not equal network output size (" << dataSize
                                   << "!=" << outputSize << ").";
            }
            if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                                   << "Failed to set Blob with precision not corresponding to user output precision";
            }
            _outputs[name] = data;
        }
    }

    /**
     * @brief Given optional implementation of getting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input
     * precision and size.
     * @note if ROI blob was previously set it is returned (without dimensions checks) instead of default blob.
     */
    void GetBlob(const char* name, Blob::Ptr& data) override {
        IE_PROFILING_AUTO_SCOPE(GetBlob)
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
            }
        } else {
            data = _outputs[name];
            checkBlob(data, name, false,
                foundOutput->getTensorDesc().getLayout() != SCALAR
                ? foundOutput->getTensorDesc().getDims()
                : oneVector);
        }
    }

    /**
     * @brief Sets pre-process for input data
     * @param name Name of input blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     * @param info Preprocess info for blob.
     */
    void SetBlob(const char* name, const Blob::Ptr& data, const PreProcessInfo& info) override {
        InputInfo::Ptr foundInput;
        DataPtr foundOutput;
        if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
            copyPreProcess(info, foundInput->getPreProcess());
        } else {
            THROW_IE_EXCEPTION << "Pre-process can't be set to output blob";
        }

        SetBlob(name, data);
    }

    /**
     * @brief Gets pre-process for input data
     * @param name Name of input blob.
     * @param info pointer to a pointer to PreProcessInfo structure
     */
    void GetPreProcess(const char* name, const PreProcessInfo** info) const override {
        InputInfo::Ptr foundInput;
        DataPtr foundOutput;
        if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
            *info = &foundInput->getPreProcess();
        } else {
            THROW_IE_EXCEPTION << "Output blob can't have pre-processing";
        }
    }

    void setPointerToExecutableNetworkInternal(ExecutableNetworkInternalPtr exeNetwork) {
        _exeNetwork = exeNetwork;
    }

    virtual void checkBlobs() {
        for (auto const& input : _inputs) {
            checkBlob(input.second, input.first, true);
        }
        for (auto const& output : _outputs) {
            checkBlob(output.second, output.first, false);
        }
    }

    void SetBatch(int /* batch */) override {
        THROW_IE_EXCEPTION << "Dynamic batch is not supported";
    };

    /**
     * @brief Checks and executes input data pre-processing if needed.
     */
    void execDataPreprocessing(InferenceEngine::BlobMap& inputs, bool serial = false) {
        for (auto& input : inputs) {
            // If there is a pre-process entry for an input then it must be pre-processed
            // using preconfigured resize algorithm.
            auto it = _preProcData.find(input.first);
            if (it != _preProcData.end()) {
                _preProcData[input.first]->execute(input.second, _networkInputs[input.first]->getPreProcess(), serial,
                                                   m_curBatch);
            }
        }
    }

protected:
    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;
    InferenceEngine::BlobMap _inputs;
    InferenceEngine::BlobMap _outputs;
    ExecutableNetworkInternalPtr _exeNetwork;
    std::map<std::string, PreProcessDataPtr> _preProcData;  // pre-process data per input
    int m_curBatch;                                         // current batch value used in dynamic batching

protected:
    /**
     * @brief helper to find input or output blob by name
     * @param name - a name of input or output blob.
     * @return true - if loaded network has input with provided name,
     *         false - if loaded network has output with provided name
     * @throws [parameter_mismatch] exception if input and output has the same name
     * @throws [not_found] exception if there is no input and output layers with given name
     */
    bool findInputAndOutputBlobByName(const char* name, InputInfo::Ptr& foundInput, DataPtr& foundOutput) const {
        foundInput = nullptr;
        foundOutput = nullptr;
        if (_networkInputs.empty() || _networkOutputs.empty()) {
            THROW_IE_EXCEPTION << "Internal error: network inputs and outputs is not set";
        }
        auto foundInputPair = std::find_if(std::begin(_networkInputs), std::end(_networkInputs),
                                           [&](const std::pair<std::string, InputInfo::Ptr>& pair) {
                                               return pair.first == name;
                                           });
        auto foundOutputPair = std::find_if(std::begin(_networkOutputs), std::end(_networkOutputs),
                                            [&](const std::pair<std::string, DataPtr>& pair) {
                                                return pair.first == name;
                                            });
        if (foundOutputPair == std::end(_networkOutputs) && (foundInputPair == std::end(_networkInputs))) {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to find input or output with name: \'" << name << "\'";
        }
        if (foundInputPair != std::end(_networkInputs)) {
            foundInput = foundInputPair->second;
            return true;
        } else {
            foundOutput = foundOutputPair->second;
            return false;
        }
    }

    void checkBlob(const Blob::Ptr& blob, const std::string& name, bool isInput, const SizeVector& refDims = {}) const {
        std::string bType = isInput ? "Input" : "Output";
        std::string sType = isInput ? "input" : "output";
        std::string strNotAllocated(bType + " data was not allocated.");
        std::string strNotMatched("The " + sType + " blob size is not equal to the network " + sType + " size");

        if (!blob) {
            THROW_IE_EXCEPTION << strNotAllocated;
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
                    THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to find input with name: \'" << name << "\'";
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
                    THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to find output with name: \'" << name << "\'";
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
            THROW_IE_EXCEPTION << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
        }
        if (blob->buffer() == nullptr) THROW_IE_EXCEPTION << strNotAllocated;
    }

    void copyPreProcess(const PreProcessInfo& from, PreProcessInfo& to) {
        to = from;
        if (from.getMeanVariant() == MEAN_IMAGE) {
            for (size_t i = 0; i < from.getNumberOfChannels(); i++) {
                auto& from_blob = from[i]->meanData;
                auto to_blob = make_blob_with_precision(from[i]->meanData->getTensorDesc());
                to_blob->allocate();
                ie_memcpy(to_blob->buffer(), to_blob->byteSize(), from_blob->cbuffer(), from_blob->byteSize());

                to.setMeanImageForChannel(to_blob, i);
            }
        }
    }

    /**
     * @brief helper to decide whether pre-processing is required
     * @param info InputInfo corresponding to input blob
     * @param blob input Blob object corresponding to input info
     * @return true if pre-processing is required, false otherwise
     */
    bool preProcessingRequired(const InputInfo::Ptr& info, const Blob::Ptr& blob) {
        // pre-processing is required if:
        // 1. resize algorithm is specified (resize required)
        // 2. color format specified:
        // 2.a. color format is not equal to network's expected (color conversion required)
        // 2.b. network's layout != blob's layout (reorder required)
        const auto& preProcessInfo = info->getPreProcess();
        const auto inputColorFormat = preProcessInfo.getColorFormat();
        // FIXME: support other network's input formats once the API is ready. Assuming input is in
        // the BGR format by default
        const auto networkColorFormat = ColorFormat::BGR;

        const bool colorFormatSpecified = inputColorFormat != ColorFormat::RAW;
        return preProcessInfo.getResizeAlgorithm() != ResizeAlgorithm::NO_RESIZE ||
               (colorFormatSpecified && inputColorFormat != networkColorFormat) ||
               (colorFormatSpecified && info->getLayout() != blob->getTensorDesc().getLayout());
    }
};

}  // namespace InferenceEngine
