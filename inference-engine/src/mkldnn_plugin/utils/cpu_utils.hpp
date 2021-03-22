// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


namespace MKLDNNPlugin {

/**
* @brief Returns normalized by size dims where missing dimensions are filled with units from the beginning
* Example: dims = {2, 3, 5}; ndims = 5; result = {1, 1, 2, 3, 5}
* @param dims
* shape to normalize
* @param ndims
* rank of resulting shape
* @return normalized vector
*/
inline std::vector<size_t> getNormalizedDimsBySize(const InferenceEngine::SizeVector &dims, size_t ndims) {
    if (dims.size() >= ndims)
        return dims;

    std::vector<size_t> normalizedDims = dims;
    for (size_t i = 0; i < (ndims - dims.size()); i++) {
        normalizedDims.insert(normalizedDims.begin(), 1);
    }
    return normalizedDims;
}

/**
* @brief Checked that secondInputDims unidirectional broadcastable per tensor or per channel to firstInputDims
* @param firstInputDims
* shape on which should be broadcastable
* @param secondInputDims
* shape which should be broadcastable
* @return true if broadcastable, false otherwise.
*/
inline bool isPerTensorOrPerChannelBroadcastable(const InferenceEngine::SizeVector &firstInputDims, const InferenceEngine::SizeVector& secondInputDims) {
    if (secondInputDims.size() > firstInputDims.size())
        return false;
    if (std::accumulate(secondInputDims.begin(), secondInputDims.end(), 1, std::multiplies<size_t>()) == 1)
        return true;

    std::vector<size_t> normalizedSecondInputDims = getNormalizedDimsBySize(secondInputDims, firstInputDims.size());
    for (size_t i = 0; i < normalizedSecondInputDims.size(); i++) {
        if ((i == 1 && normalizedSecondInputDims[i] != firstInputDims[1]) || (i != 1 && normalizedSecondInputDims[i] != 1))
            return false;
    }
    return true;
}

}  // namespace MKLDNNPlugin