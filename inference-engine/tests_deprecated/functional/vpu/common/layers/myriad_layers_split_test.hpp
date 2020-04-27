// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

static const float ERROR_BOUND = 0.0f;

struct SplitParams {
    SizeVector dims;
    int axis;
    int numSplit;
};

PRETTY_PARAM(SplitTestParams, SplitParams);

typedef myriadLayerTestBaseWithParam<SplitTestParams> myriadLayersTestsSplit_nightly;

TEST_P(myriadLayersTestsSplit_nightly, Split) {
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    const SplitParams testParams = GetParam();

    const auto inputDims = testParams.dims;
    const auto axis = testParams.axis;
    const auto numSplit = testParams.numSplit;
    std::map<std::string, std::string> layerParams;
    layerParams["axis"] = std::to_string(axis);
    layerParams["num_split"] = std::to_string(numSplit);

    SetInputTensors({inputDims});

    IN_OUT_desc output;
    auto dims = inputDims;
    for (size_t i = 0; i < numSplit; ++i) {
        const int begin = (i + 0) * inputDims[axis] / numSplit;
        const int end   = (i + 1) * inputDims[axis] / numSplit;
        const int dimSize = end - begin;
        dims[axis] = dimSize;
        output.push_back(dims);
    }
    SetOutputTensors(output);
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Split").params(layerParams)));

    BlobMap refBlobMap;

    const auto inputBlob = _inputMap.begin()->second;
    const auto layout = vpu::deviceLayout(
            inputBlob->getTensorDesc().getLayout(),
            vpu::LayoutPreference::ChannelMajor);
    inputBlob->getTensorDesc().setLayout(layout);

    for (const auto& item : _outputMap) {
        item.second->getTensorDesc().setLayout(layout);
        refBlobMap[item.first] = make_shared_blob<ie_fp16>(
                {
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
