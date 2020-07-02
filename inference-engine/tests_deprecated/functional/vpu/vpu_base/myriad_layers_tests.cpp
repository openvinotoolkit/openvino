// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

#include <thread>
#include <chrono>
#include <iostream>

#include "functional_test_utils/plugin_cache.hpp"

using namespace InferenceEngine;

void myriadLayersTests_nightly::makeSingleLayerNetwork(const LayerParams& layerParams,
                                                       const NetworkParams& networkParams,
                                                       const WeightsBlob::Ptr& weights) {
    // Disable reorder in per-layer tests to make sure intended layout is used
    _config[VPU_CONFIG_KEY(DISABLE_REORDER)] = CONFIG_VALUE(YES);

    // White list of per-layer tests that allowed to reorder
    if (layerParams._layerType == "Flatten") {
        _config[VPU_CONFIG_KEY(DISABLE_REORDER)] = CONFIG_VALUE(NO);
    }

    ASSERT_NO_FATAL_FAILURE(
        makeSingleLayerNetworkImpl(layerParams, networkParams, weights);
    );
}

const std::vector<InferenceEngine::SizeVector> g_poolingInput = {
    {{1,  1,  16,  16},
     {1,  8, 228, 128},
     {1, 16,  32,  64}}
};

const std::vector<InferenceEngine::SizeVector> g_poolingInput_postOp = {
    {{1, 32,  86, 100}, // postOp issue MX
     {1, 32,  62, 104}} // postOp issue M2
};

const std::vector<pooling_layer_params> g_poolingLayerParamsFull = {
    /* kernel stride  pad */
    {{2, 2}, {2, 2}, {0, 0}},
    {{2, 2}, {2, 2}, {1, 1}},
    {{2, 2}, {2, 2}, {2, 2}},
    {{2, 2}, {1, 1}, {0, 0}},
    {{2, 2}, {1, 1}, {1, 1}},
    {{2, 2}, {1, 1}, {2, 2}},
    {{4, 2}, {2, 2}, {0, 0}},
    {{4, 2}, {2, 2}, {1, 1}},
    {{4, 2}, {2, 2}, {2, 2}},
    {{4, 2}, {1, 1}, {0, 0}},
    {{4, 2}, {1, 1}, {1, 1}},
    {{4, 2}, {1, 1}, {2, 2}},
    {{2, 4}, {2, 2}, {0, 0}},
    {{2, 4}, {2, 2}, {1, 1}},
    {{2, 4}, {2, 2}, {2, 2}},
    {{2, 4}, {1, 1}, {0, 0}},
    {{2, 4}, {1, 1}, {1, 1}},
    {{2, 4}, {1, 1}, {2, 2}},
};

const std::vector<pooling_layer_params> g_poolingLayerParamsLite = {
    /* kernel stride  pad */
    {{2, 2}, {1, 1}, {0, 0}},
    {{4, 2}, {2, 2}, {1, 1}},
    {{2, 4}, {1, 1}, {2, 2}},
};

const std::vector<vpu::LayoutPreference> g_poolingLayout = {
    vpu::LayoutPreference::ChannelMajor,
    vpu::LayoutPreference::ChannelMinor
};

const std::vector<InferenceEngine::SizeVector> g_convolutionTensors = {
    {1, 8, 4, 16},

    // FIXME: the test is written for [N]HWC layout, but InferenceEngine doesn't have 3D HWC layout.
//    {16, 8, 16},
};

const std::vector<InferenceEngine::SizeVector> g_convolutionTensors_postOp = {
    {{1, 32, 112, 96}}  /* postOp issue */
};

const std::vector<fcon_test_params> g_fcTestParamsSubset = {
    {{1, 4, 8, 16}, 4, 0.065f},
    {{1, 16, 8, 8}, 8, 0.065f}
};

/* tests subset to check 2 layers operation invocation */
/* additional tests for 2D and 3D tensors added        */
const std::vector<int32_t> g_dimensionsFC = {
    4, 3
};

const std::vector<int32_t> g_addBiasFC = {
    1, 0
};
