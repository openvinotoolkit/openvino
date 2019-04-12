// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <blob_factory.hpp>
#include <ie_input_info.hpp>
#include <ie_icnn_network.hpp>
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "debug.h"
#include "cpp_interfaces/exception2status.hpp"
#include "ie_preprocess_data.hpp"
#include "ie_memcpy.h"

namespace InferenceEngine {

class ExecutableNetworkInternal;

typedef std::shared_ptr<ExecutableNetworkInternal> ExecutableNetworkInternalPtr;

/**
 * @brief optional implementation of IInferRequestInternal to avoid duplication in all plugins
 */
class InferRequestInternal : virtual public IInferRequestInternal {
public:
    typedef std::shared_ptr<InferRequestInternal> Ptr;

    InferRequestInternal(InputsDataMap networkInputs, OutputsDataMap networkOutputs)
            : m_curBatch(-1) {
        // We should copy maps in order to avoid modifications in the future.
        for (const auto &it : networkInputs) {
            InputInfo::Ptr newPtr;
            if (it.second) {
                newPtr.reset(new InputInfo());
                DataPtr newData(new Data(*it.second->getInputData()));
                newPtr->getPreProcess() = it.second->getPreProcess();
                if (newPtr->getPreProcess().getMeanVariant() == MEAN_IMAGE) {
                    for (size_t i = 0; i < newPtr->getPreProcess().getNumberOfChannels(); i++) {
                        auto blob = newPtr->getPreProcess()[i]->meanData;
                        newPtr->getPreProcess()[i]->meanData =
                                make_blob_with_precision(newPtr->getPreProcess()[i]->meanData->getTensorDesc());
                        newPtr->getPreProcess()[i]->meanData->allocate();
                        ie_memcpy(newPtr->getPreProcess()[i]->meanData->buffer(), newPtr->getPreProcess()[i]->meanData->byteSize(),
                                  blob->cbuffer(), blob->byteSize());
                    }
                }
                newData->inputTo.clear();
                newPtr->setInputData(newData);
            }
            _networkInputs[it.first] = newPtr;
        }

        for (const auto &it : networkOutputs) {
            DataPtr newData;
            if (it.second) {
                newData.reset(new Data(*it.second));
                newData->inputTo.clear();
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
    };

    /**
     * @brief Given optional implementation of setting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    void SetBlob(const char *name, const Blob::Ptr &data) override {
        if (!data)
            THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
        if (data->buffer() == nullptr)
            THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
        if (name == nullptr) {
            THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
        }
        InputInfo::Ptr foundInput;
        DataPtr foundOutput;
        size_t dataSize = data->size();
        if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
            if (foundInput->getInputPrecision() != data->precision()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                                   << "Failed to set Blob with precision not corresponding to user input precision";
            }

            if (foundInput->getPreProcess().getResizeAlgorithm() != ResizeAlgorithm::NO_RESIZE) {
                PreProcessData::isApplicable(data, _inputs[name]);
                // Stores the given blob as ROI blob. It will be used to fill in network input during pre-processing.
                _preProcData[name].setRoiBlob(data);
            } else {
                size_t inputSize = details::product(foundInput->getDims());
                if (dataSize != inputSize) {
                    THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                                       << dataSize << "!=" << inputSize << ").";
                }
                _inputs[name] = data;
            }
        } else {
            size_t outputSize = details::product(foundOutput->getDims());
            if (dataSize != outputSize) {
                THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                                   << dataSize << "!=" << outputSize << ").";
            }
            if (foundOutput->getPrecision() != data->precision()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                                   << "Failed to set Blob with precision not corresponding to user output precision";
            }
            _outputs[name] = data;
        }
    }

    /**
     * @brief Given optional implementation of getting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     * @note if ROI blob was previously set it is returned (without dimensions checks) instead of default blob.
     */
    void GetBlob(const char *name, Blob::Ptr &data) override {
        InputInfo::Ptr foundInput;
        DataPtr foundOutput;
        if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
            // ROI blob is returned only if it was set previously. Otherwise default blob is returned.
            auto it = _preProcData.find(name);
            if (it != _preProcData.end()) {
                data = it->second.getRoiBlob();
            } else {
                data = _inputs[name];
                checkBlob(data, name, true, foundInput->getDims());
            }
        } else {
            data = _outputs[name];
            checkBlob(data, name, false, foundOutput->getDims());
        }
    }

    void setPointerToExecutableNetworkInternal(ExecutableNetworkInternalPtr exeNetwork) {
        _exeNetwork = exeNetwork;
    }

    void checkBlobs() const {
        for (auto const &input : _inputs) {
            checkBlob(input.second, input.first, true);
        }
        for (auto const &output : _outputs) {
            checkBlob(output.second, output.first, false);
        }
    }

    void SetBatch(int batch) override {
        THROW_IE_EXCEPTION << "Dynamic batch is not supported";
    };

    /**
     * @brief Checks and executes input data pre-processing if needed.
     */
    void execDataPreprocessing(InferenceEngine::BlobMap& inputs, bool serial = false) {
        for (auto &input : inputs) {
            // If there is a pre-process entry for an input then it must be pre-processed
            // using preconfigured resize algorithm.
            auto it = _preProcData.find(input.first);
            if (it != _preProcData.end()) {
                _preProcData[input.first].execute(input.second,
                                                  _networkInputs[input.first]->getPreProcess().getResizeAlgorithm(),
                                                  serial,
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
    std::map<std::string, PreProcessData> _preProcData;  // pre-process data per input
    int m_curBatch;  // current batch value used in dynamic batching

protected:
    /**
     * @brief helper to find input or output blob by name
     * @param name - a name of input or output blob.
     * @return true - if loaded network has input with provided name,
     *         false - if loaded network has output with provided name
     * @throws [parameter_mismatch] exception if input and output has the same name
     * @throws [not_found] exception if there is no input and output layers with given name
     */
    bool findInputAndOutputBlobByName(const char *name, InputInfo::Ptr &foundInput, DataPtr &foundOutput) const {
        foundInput = nullptr;
        foundOutput = nullptr;
        if (_networkInputs.empty() || _networkOutputs.empty()) {
            THROW_IE_EXCEPTION << "Internal error: network inputs and outputs is not set";
        }
        auto foundInputPair = std::find_if(std::begin(_networkInputs),
                                           std::end(_networkInputs),
                                           [&](const std::pair<std::string, InputInfo::Ptr> &pair) {
                                               return pair.first == name;
                                           });
        auto foundOutputPair = std::find_if(std::begin(_networkOutputs),
                                            std::end(_networkOutputs),
                                            [&](const std::pair<std::string, DataPtr> &pair) {
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

    void checkBlob(const Blob::Ptr &blob, const std::string &name, bool isInput, const SizeVector& refDims = {}) const {
        std::string bType = isInput ? "Input" : "Output";
        std::string sType = isInput ? "input" : "output";
        std::string strNotAllocated(bType + " data was not allocated.");
        std::string strNotMatched("The " + sType + " blob size is not equal to the network " + sType + " size");

        if (!blob) THROW_IE_EXCEPTION << strNotAllocated;
        size_t refSize;
        if (refDims.empty()) {
            SizeVector dims;
            if (isInput) {
                auto foundInputPair = std::find_if(std::begin(_networkInputs),
                                                   std::end(_networkInputs),
                                                   [&](const std::pair<std::string, InputInfo::Ptr>& pair) {
                                                       return pair.first == name;
                                                   });
                if (foundInputPair == std::end(_networkInputs)) {
                    THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to find input with name: \'" << name << "\'";
                }
                dims = foundInputPair->second->getDims();
            } else {
                auto foundOutputPair = std::find_if(std::begin(_networkOutputs),
                                                    std::end(_networkOutputs),
                                                    [&](const std::pair<std::string, DataPtr>& pair) {
                                                        return pair.first == name;
                                                    });
                if (foundOutputPair == std::end(_networkOutputs)) {
                    THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to find output with name: \'" << name << "\'";
                }
                dims = foundOutputPair->second->getDims();
            }
            refSize = details::product(dims);
        } else {
            refSize = details::product(refDims);
        }

        if (refSize != blob->size()) {
            THROW_IE_EXCEPTION << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
        }
        if (blob->buffer() == nullptr) THROW_IE_EXCEPTION << strNotAllocated;
    }
};

}  // namespace InferenceEngine
