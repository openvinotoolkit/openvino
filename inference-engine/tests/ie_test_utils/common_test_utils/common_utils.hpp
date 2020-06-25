// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <string>
#include <sstream>
#include <iterator>
#include <vector>

#include <cpp/ie_cnn_network.h>
#include <details/ie_cnn_network_iterator.hpp>

namespace CommonTestUtils {

template<typename vecElementType>
inline std::string vec2str(const std::vector<vecElementType> &vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<vecElementType>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return std::string("()");
}

template<typename vecElementType>
inline std::string vec2str(const std::vector<std::vector<vecElementType>> &vec) {
    std::ostringstream result;
    for (const auto &v : vec) {
        result << vec2str<vecElementType>(v);
    }
    return result.str();
}

inline InferenceEngine::CNNLayerPtr getLayerByName(const InferenceEngine::ICNNNetwork * icnnnetwork,
                                                   const std::string & layerName) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::details::CNNNetworkIterator i(icnnnetwork), end;
    while (i != end) {
        auto layer = *i;
        if (layer->name == layerName)
            return layer;
        ++i;
    }

    std::stringstream stream;
    stream << "Layer " << layerName << " not found in network";
    throw InferenceEngine::NotFound(stream.str());
    IE_SUPPRESS_DEPRECATED_END
}

inline InferenceEngine::CNNLayerPtr getLayerByName(const InferenceEngine::CNNNetwork & network,
                                                   const std::string & layerName) {
    const InferenceEngine::ICNNNetwork & icnnnetwork = static_cast<const InferenceEngine::ICNNNetwork&>(network);
    return getLayerByName(&icnnnetwork, layerName);
}

}  // namespace CommonTestUtils
