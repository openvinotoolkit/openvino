// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/concat_quantization.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConcatQuantization, CompareWithRefImpl) {
    InferenceEngine::Core* core = PluginCache::get().ie(targetDevice).get();
    if (!configuration.empty()) {
        core->SetConfig(configuration, targetDevice);
    }

    try {
        InferenceEngine::CNNNetwork cnnNetwork = InferenceEngine::CNNNetwork{ function };
        executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice);
    }
    catch (InferenceEngine::details::InferenceEngineException ex) {
        FAIL() << ex.what();
    }
};


}  // namespace SubgraphTestsDefinitions
