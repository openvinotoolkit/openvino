// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <algorithm>
#include "ie_memcpy.h"

using std::tuple;
using std::get;

using namespace InferenceEngine;

PRETTY_PARAM(NDims, nd_tensor_test_params);

typedef myriadLayerTestBaseWithParam<tuple<NDims, int>> myriadLayerCopy_smoke;

TEST_P(myriadLayerCopy_smoke, Copy) {

    nd_tensor_test_params input_dims = get<0>(GetParam());
    int ndims = get<1>(GetParam());

    IN_OUT_desc inputTensors;
    IN_OUT_desc outputTensors;
    outputTensors.resize(1);
    inputTensors.resize(1);
    inputTensors[0].resize(ndims);
    outputTensors[0].resize(ndims);

    for (int i = 0; i < ndims; i++)
    {
        inputTensors[0][i] = input_dims.dims[i];
        outputTensors[0][i] = input_dims.dims[i];
    }

    SetInputTensors(inputTensors);
    SetOutputTensors(outputTensors);

    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Copy")));
    SetFirstInputToRange(1.0f, 100.0f);

    ASSERT_TRUE(Infer());
    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    CompareCommonAbsolute(outputBlob, inputBlob, 0);
}
