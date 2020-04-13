// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_icnn_network.hpp>
#include <string>
#include <vector>

namespace pugi {
class xml_document;
}

namespace InferenceEngine {
namespace Serialization {
/**
    * @brief Serialize network into IE IR XML file and binary weights file
    * @param xmlPath   Path to XML file
    * @param binPath   Path to BIN file
    * @param network   network to be serialized
    */
INFERENCE_ENGINE_API_CPP(void) Serialize(const std::string& xmlPath, const std::string& binPath,
                                         const InferenceEngine::ICNNNetwork& network);

/**
    * @brief Fill XML representation using network
    * @param network   Loaded network
    * @param doc       XML object
    * @param execGraphInfoSerialization    If true scip some info serialization
    * @param dumpWeights                   If false does not serialize waights info
    * @return Size of all weights in network
    */
INFERENCE_ENGINE_API_CPP(std::size_t) FillXmlDoc(const InferenceEngine::ICNNNetwork& network, pugi::xml_document& doc,
                                                 const bool execGraphInfoSerialization = false, const bool dumpWeights = true);

/**
    * @brief Write all weights in network into output stream
    * @param stream    Output stream
    * @param network   Loaded network
    */
INFERENCE_ENGINE_API_CPP(void) SerializeBlobs(std::ostream& stream,
                                              const InferenceEngine::ICNNNetwork& network);

/**
    * @brief Returns set of topologically sorted layers
    * @param network network to be sorted
    * @return `std::vector` of topologically sorted CNN layers
    */
INFERENCE_ENGINE_API_CPP(std::vector<CNNLayerPtr>) TopologicalSort(const InferenceEngine::ICNNNetwork& network);
}  // namespace Serialization
}  // namespace InferenceEngine
