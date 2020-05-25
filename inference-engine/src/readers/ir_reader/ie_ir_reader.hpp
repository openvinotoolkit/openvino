// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <ie_blob.h>
#include <ie_common.h>
#include <ie_iextension.h>

#include <ie_icnn_network.hpp>
#include <ie_reader.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pugi {
class xml_node;
class xml_document;
}  // namespace pugi

namespace ngraph {
class Function;
}  // namespace ngraph

namespace InferenceEngine {

/**
 * @brief This class is the main interface to build and parse a network from a given IR
 *
 * All methods here do not throw exceptions and return a StatusCode and ResponseDesc object.
 * Alternatively, to use methods that throw exceptions, refer to the CNNNetReader wrapper class.
 */
class INFERENCE_ENGINE_API_CLASS(IRReader): public IReader {
public:
    IRReader() = default;

    void Release() noexcept override {
        delete this;
    }
    /**
     * @brief Checks that reader supports format of the model
     * @param model stream with model
     * @return true if format is supported
     */
    bool supportModel(std::istream& model) const override;
    /**
     * @brief Reads the model to CNNNetwork
     * @param model stream with model
     * @param exts vector with extensions
     *
     * @return CNNNetwork
     */
    CNNNetwork read(std::istream& model, const std::vector<IExtensionPtr>& exts) const override;
    /**
     * @brief Reads the model to CNNNetwork
     * @param model stream with model
     * @param weights stream with binary data
     * @param exts vector with extensions
     *
     * @return CNNNetwork
     */
    CNNNetwork read(std::istream& model, std::istream& weights, const std::vector<IExtensionPtr>& exts) const override;

    std::vector<std::string> getDataFileExtensions() const override {
        return {"bin"};
    }
};

}  // namespace InferenceEngine
