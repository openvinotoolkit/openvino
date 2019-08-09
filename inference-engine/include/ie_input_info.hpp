// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for InputInfo class
 * @file ie_input_info.hpp
 */
#pragma once

#include <string>
#include <memory>
#include <map>
#include "ie_common.h"
#include "ie_data.h"
#include "ie_preprocess.hpp"
#include "ie_blob.h"
#include "ie_precision.hpp"

namespace InferenceEngine {

/**
 * @brief This class contains information about each input of the network
 */
class InputInfo {
public:
    /** @brief A smart pointer to the InputInfo instance */
    using Ptr = std::shared_ptr<InputInfo>;
    /** @brief A smart pointer to the constant InputInfo instance */
    using CPtr = std::shared_ptr<const InputInfo>;

    /**
     * @deprecated Use InputInfo::getPrecision
     * @brief Gets a precision of the input data provided by user
     *
     * By default it matches the layers precision, but there are exceptions of this rule
     * For Q78 precision networks the input is expected in I16 by default
     * For FP16 precision networks the input is expected in FP32 by default
     *
     * @details By default it matches the layers precision, but there are exceptions of this rule.
     * For Q78 precision networks the input is expected in I16 by default.
     * For FP16 precision networks the input is expected in FP32 by default.
     * The default input precision might be changed preferred one using InputInfo::setPrecision
     * function.
     * For example, for a Q78 precision network you can pass FP32 input data
     * @return The precision used for input blob creation
     */
    INFERENCE_ENGINE_DEPRECATED
    Precision getInputPrecision() const {
        return getPrecision();
    }

    /**
     * @deprecated Use InputInfo::setPrecision
     * @brief Changes the precision of the input data provided by the user.
     * This function should be called before loading the network to the plugin
     * @param p A new precision of the input data to set
     */
    INFERENCE_ENGINE_DEPRECATED
    void setInputPrecision(Precision p) {
        setPrecision(p);
    }

    /**
     * @brief Gets a precision of the input data provided by user
     *
     * By default it matches the layers precision, but there are exceptions of this rule
     * For Q78 precision networks the input is expected in I16 by default
     * For FP16 precision networks the input is expected in FP32 by default
     *
     * @details By default it matches the layers precision, but there are exceptions of this rule.
     * For Q78 precision networks the input is expected in I16 by default.
     * For FP16 precision networks the input is expected in FP32 by default.
     * The default input precision might be changed preferred one using InputInfo::setPrecision()
     * function.
     * For example, for a Q78 precision network you can pass FP32 input data
     * @return The precision used for input blob creation
     */
    Precision getPrecision() const {
        if (!_inputData) {
            THROW_IE_EXCEPTION << "Data is empty!";
        }
        return _inputData->getPrecision();
    }

    /**
     * @brief Changes the precision of the input data provided by the user.
     * This function should be called before loading the network to the plugin
     * @param p A new precision of the input data to set
     */
    void setPrecision(Precision p) {
        if (!_inputData) {
            THROW_IE_EXCEPTION << "Data is empty!";
        }
        _inputData->setPrecision(p);
    }

    /**
     * @brief Gets a layout of the input data provided by user
     * @details By default it matches the layers precision and depends on number of its dimensions:
     * C - for 1-dimensional,
     * NC - for 2-dimensional,
     * CHW - for 3-dimensional,
     * NCHW - for 4-dimensional
     * The default input layout might be changed preferred one using setLayout() function.
     * @return The precision used for input blob creation
     */
    Layout getLayout() {
        if (!_inputData) {
            THROW_IE_EXCEPTION << "Data is empty!";
        }
        return _inputData->getLayout();
    }

    /**
     * @brief Changes the layout of the input data provided by the user.
     * This function should be called before loading the network to the plugin
     * @param p A new layout of the input data to set
     */
    void setLayout(Layout l) {
        if (!_inputData) {
            THROW_IE_EXCEPTION << "Data is empty!";
        }
        _inputData->setLayout(l);
    }

    /**
     * @brief Gets the name of the input
     * @return A string - the name of the input
     */
    const std::string &name() const { return _inputData->getName(); }

    /**
     * @brief Gets the input data
     * @return A smart pointer to the input data
     */
    DataPtr getInputData() {
        return _inputData;
    }

    /**
     * @brief Initializes the pointer to the input data that stores the main input parameters like dims, etc.
     * This method initializes the precision with the information from the inputPtr if it was not set
     * explicitly through InputInfo::setPrecision. If InputInfo::setPrecision is called, this method does not overwrite the precision.
     * @param inputPtr Pointer to the input data to set
     */
    void setInputData(DataPtr inputPtr) {
        _inputData = inputPtr;
    }

    /**
     * @deprecated Please use InputInfo::getTensorDesc for working with layouts and dimensions
     * @brief Gets dimensions/shape of the input data with reversed order
     * @return A SizeVector object that contains dimensions of the input data. If the data is not set, the method returns an empty SizeVector object.
     */
    INFERENCE_ENGINE_DEPRECATED
    SizeVector getDims() const {
        if (_inputData) {
            auto dims = _inputData->getTensorDesc().getDims();
            std::reverse(dims.begin(), dims.end());
            return dims;
        } else {
            return SizeVector();
        }
    }

    /**
     * @brief Returns the tensor descriptor
     */
    const TensorDesc &getTensorDesc() const {
        if (!_inputData) {
            THROW_IE_EXCEPTION << "Data is empty!";
        }
        return _inputData->getTensorDesc();
    }

    /**
     * @brief Gets pre-process info for the input
     * @return A reference to the PreProcessInfo instance that contains pre-process info for this input
     */
    PreProcessInfo &getPreProcess() { return _preProcessInfo; }

protected:
    /**
     * @brief Pre-process info for the input
     */
    PreProcessInfo _preProcessInfo;

    /**
     * @brief A smart pointer to the input data
     */
    DataPtr _inputData;
};

/**
 * @brief A collection that contains string as key, and InputInfo smart pointer as value
 */
using InputsDataMap = std::map<std::string, InputInfo::Ptr>;

/**
 * @brief A collection that contains string as key, and const InputInfo smart pointer as value
 */
using ConstInputsDataMap = std::map<std::string, InputInfo::CPtr>;

}  // namespace InferenceEngine
