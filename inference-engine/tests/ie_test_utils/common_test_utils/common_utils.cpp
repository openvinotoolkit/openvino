// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include <legacy/details/ie_cnn_network_iterator.hpp>

::std::ostream& ngraph::operator << (::std::ostream & os, const Function&) {
    throw std::runtime_error("should not be called");
    return os;
}

namespace CommonTestUtils {

IE_SUPPRESS_DEPRECATED_START

std::shared_ptr<InferenceEngine::CNNLayer>
getLayerByName(const InferenceEngine::CNNNetwork & network, const std::string & layerName) {
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

std::ostream& operator<<(std::ostream & os, OpType type) {
    switch (type) {
        case OpType::SCALAR:
            os << "SCALAR";
            break;
        case OpType::VECTOR:
            os << "VECTOR";
            break;
        default:
            IE_THROW() << "NOT_SUPPORTED_OP_TYPE";
    }
    return os;
}

}  // namespace CommonTestUtils
