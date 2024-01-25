// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file containing ngraph implementation of public CNNNetwork wrapper
 * @file cnn_network_ngraph_impl.hpp
 */

#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpp/ie_cnn_network.h"
#include "description_buffer.hpp"
#include "ie_api.h"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"
#include "ie_input_info.hpp"

namespace InferenceEngine {
namespace details {

ov::element::Type toLegacyType(const ov::element::Type& ngraph_type, bool input);

IE_SUPPRESS_DEPRECATED_START

/**
 * @brief Ngraph-based implementation of the CNNNetwork.
 */
class INFERENCE_ENGINE_API_CLASS(CNNNetworkNGraphImpl) final : public ICNNNetwork {
public:
    CNNNetworkNGraphImpl(const std::shared_ptr<::ov::Model>& nGraph, bool newAPI = false);

    CNNNetworkNGraphImpl(const CNNNetwork& nGraph);

    void getOutputsInfo(std::map<std::string, DataPtr>& out) const noexcept override;

    void getInputsInfo(InputsDataMap& inputs) const noexcept override;

    InputInfo::Ptr getInput(const std::string& inputName) const noexcept override;
    const std::string& getName() const noexcept override;

    size_t layerCount() const override;

    void setInputInfo(InputInfo::Ptr data);

    // public version
    StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept override;

    size_t getBatchSize() const override;

    StatusCode addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept override;

    void addOutput(const ::ov::Output<::ov::Node>& dataName);

    std::shared_ptr<const ::ov::Model> getFunction() const noexcept override {
        return _ngraph_function;
    }
    std::shared_ptr<::ov::Model> getFunction() noexcept override {
        return _ngraph_function;
    }

    virtual void validate(int = 10);

    StatusCode reshape(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp) noexcept override;
    StatusCode reshape(const std::map<std::string, ov::PartialShape>& inputShapes,
                       ResponseDesc* resp) noexcept override;

    StatusCode serialize(const std::string& xmlPath, const std::string& binPath, ResponseDesc* resp) const
        noexcept override;

    StatusCode serialize(std::ostream& xmlBuf, std::ostream& binBuf, ResponseDesc* resp) const noexcept override;

    StatusCode serialize(std::ostream& xmlBuf, Blob::Ptr& binBlob, ResponseDesc* resp) const noexcept override;

    StatusCode getOVNameForTensor(std::string& ov_name, const std::string& orig_name, ResponseDesc* resp) const
        noexcept override;

    // used by convertFunctionToICNNNetwork from legacy library
    std::map<std::string, DataPtr> _data;

protected:
    std::shared_ptr<::ov::Model> _ngraph_function;

private:
    InferenceEngine::InputsDataMap _inputData;
    std::map<std::string, DataPtr> _outputData;
    std::unordered_map<std::string, std::string> _tensorNames;
    bool _new_api = false;

    /**
     * @brief Create DataPtr for nGraph operation
     *
     * @param output output port from nGraph op
     * @param outName name for DataPtr
     * @param ptr reference to new DataPtr
     */
    void createDataForResult(const ::ov::Output<::ov::Node>& output, const std::string& outName, DataPtr& ptr);

    /**
     * @brief Reshape on the same shape
     */
    void reshape();
    void reshape(const std::map<std::string, ov::PartialShape>& inputShapes);
    void validateFunctionNames() const;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
