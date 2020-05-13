// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

static const float ERROR_BOUND = 0.0f;

struct SliceParams {
    SizeVector inputDims;
    IN_OUT_desc outputs;
    int axis;
};

PRETTY_PARAM(SliceTestParams, SliceParams);

typedef myriadLayerTestBaseWithParam<SliceTestParams> myriadLayersTestsSlice_smoke;

TEST_P(myriadLayersTestsSlice_smoke, Slice) {
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    const SliceParams testParams = GetParam();

    const auto inputDims = testParams.inputDims;
    const auto outputs = testParams.outputs;
    const auto axis = testParams.axis;
    std::map<std::string, std::string> layerParams;
    layerParams["axis"] = std::to_string(axis);

    SetInputTensors({inputDims});

    SetOutputTensors(outputs);
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Slice").params(layerParams)));

    BlobMap refBlobMap;
    const auto inputBlob = _inputMap.begin()->second;

    const auto layout = vpu::deviceLayout(
            inputBlob->getTensorDesc().getLayout(),
            vpu::LayoutPreference::ChannelMajor);
    inputBlob->getTensorDesc().setLayout(layout);

    for (const auto& item : _outputMap) {
        item.second->getTensorDesc().setLayout(layout);
        refBlobMap[item.first] = make_shared_blob<ie_fp16>({
           Precision::FP16,
           item.second->getTensorDesc().getDims(),
           item.second->getTensorDesc().getLayout()
        });
        refBlobMap[item.first]->allocate();
    }

    ASSERT_TRUE(Infer());
    ref_Split(inputBlob, refBlobMap, axis);

    for (const auto& item : _outputMap)
        CompareCommonAbsolute(item.second, refBlobMap[item.first], ERROR_BOUND);
}
