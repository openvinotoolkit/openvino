// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_ngraph_utils.hpp>
#include "cnn_network_ngraph_impl.hpp"
#include "ie_itt.hpp"

namespace InferenceEngine {
namespace details {

CNNNetwork cloneNetwork(const CNNNetwork& network) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "cloneNetwork");

    if (network.getFunction()) {
        IE_SUPPRESS_DEPRECATED_START
        return CNNNetwork(std::make_shared<details::CNNNetworkNGraphImpl>(network));
        IE_SUPPRESS_DEPRECATED_END
    }

    IE_THROW() << "InferenceEngine::details::cloneNetwork requires ngraph-based `network` object to clone";
}

}  // namespace details
}  // namespace InferenceEngine
