// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_common.h"
#include "cpp/ie_cnn_network.h"
#include <ngraph/ngraph.hpp>
#include <ie_icore.hpp>

InferenceEngine::QueryNetworkResult getQueryNetwork(const InferenceEngine::CNNNetwork& network, const InferenceEngine::ICore* core,
                                                    const std::string& pluginName);
