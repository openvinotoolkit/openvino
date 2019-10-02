// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace pugi {
class xml_document;
}


namespace InferenceEngine {
namespace details {

/**
* Class for serialization of model been presented as ICNNNetwork to the disk
*/
class NetworkSerializer {
public:
    static void serialize(
        const std::string &xmlPath,
        const std::string &binPath,
        const InferenceEngine::ICNNNetwork& network);

/**
 * @brief Fill XML representation using network
 * @param network   Loaded network
 * @param doc       XML object
 * @param execGraphInfoSerialization    If true scip some info serialization
 * @param dumpWeights                   If false does not serialize waights info
 * @return Size of all weights in network
 */
    static INFERENCE_ENGINE_API_CPP(std::size_t) fillXmlDoc(const InferenceEngine::ICNNNetwork&  network,
                                                              pugi::xml_document&                  doc,
                                                              const bool                           execGraphInfoSerialization = false,
                                                              const bool                           dumpWeights = true);

/**
 * @brief Write all weights in network into output stream
 * @param stream    Output stream
 * @param network   Loaded network
 */
    static INFERENCE_ENGINE_API_CPP(void) serializeBlobs(
        std::ostream&                       stream,
        const InferenceEngine::ICNNNetwork& network);

    static INFERENCE_ENGINE_API_CPP(void) updateStdLayerParams(const InferenceEngine::CNNLayer::Ptr &layer);
};

}  // namespace details
}  // namespace InferenceEngine
