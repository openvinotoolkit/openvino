// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <legacy/details/ie_cnn_network_iterator.hpp>

namespace CommonTestUtils {

IE_SUPPRESS_DEPRECATED_START

inline std::shared_ptr<InferenceEngine::CNNLayer> getLayerByName(const InferenceEngine::CNNNetwork& network,
                                                          const std::string& layerName) {
    InferenceEngine::details::CNNNetworkIterator i(network), end;
    while (i != end) {
        auto layer = *i;
        if (layer->name == layerName)
            return layer;
        ++i;
    }
    IE_THROW(NotFound) << "Layer " << layerName << " not found in network";
}

IE_SUPPRESS_DEPRECATED_END

}  // namespace CommonTestUtils
