// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides interface for network reader that is used to build networks from a given IR
 * @file ie_icnn_net_reader.h
 */
#pragma once

#include <map>
#include <string>
#include <memory>

#include <ie_blob.h>
#include <ie_common.h>
#include <ie_api.h>
#include <ie_icnn_network.hpp>

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
#ifdef ENABLE_IR_READER
class INFERENCE_ENGINE_API_CLASS(IRReader) {
#else
class IRReader {
#endif
public:
    /**
     * @brief Reads IR xml and bin (with the same name) files
     * @param modelPath path to IR file
     * @return shared pointer to nGraph function
     */
    std::shared_ptr<ngraph::Function> read(const std::string& modelPath);
    /**
     * @brief Reads IR xml and bin files
     * @param modelPath path to IR file
     * @param binPath path to bin file
     * @return shared pointer to nGraph function
     */
    std::shared_ptr<ngraph::Function> read(const std::string& modelPath, const std::string& binPath);
    /**
     * @brief Reads IR xml and bin (with the same name) files
     * @param model string with IR
     * @param weights shared pointer to constant blob with weights
     * @return shared pointer to nGraph function
     */
    std::shared_ptr<ngraph::Function> read(const std::string& model, const Blob::CPtr& weights);

private:
    std::shared_ptr<ngraph::Function> readXml(const pugi::xml_document& xmlDoc, const Blob::CPtr& weights);
};

/**
 * @brief Converts nGraph function to ICNNNetwork
 * @param nGraph shared pointer to nGraph function
 * @return pointer to ICNNNetwork
 */
INFERENCE_ENGINE_API_CPP(ICNNNetwork::Ptr) convertFunctionToICNNNetwork(const std::shared_ptr<ngraph::Function>& nGraph);

}  // namespace InferenceEngine

