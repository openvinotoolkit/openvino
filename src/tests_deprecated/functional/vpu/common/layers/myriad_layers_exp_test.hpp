// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <cmath>

#define BOUND (5.0f)
#define REL_ERROR_BOUND (0.003f)

using namespace InferenceEngine;

class myriadLayersTestsExp_smoke: public myriadLayersTests_nightly,
                                    public testing::WithParamInterface<Dims> {};

TEST_P(myriadLayersTestsExp_smoke, TestsExp)
{
    auto p = ::testing::WithParamInterface<Dims>::GetParam();
    SetInputTensor(p);
    SetOutputTensor(p);
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Exp")));
    SetFirstInputToRange(-BOUND, BOUND);
    ASSERT_TRUE(Infer());

    /* output check */
    ref_exp(_inputMap.begin()->second, _refBlob);
    CompareCommonRelative(_outputMap.begin()->second, _refBlob, REL_ERROR_BOUND);
}

static std::vector<Dims> s_expParams = {
    {{1, 1, 16, 16}},
    {{1, 2, 16, 16}},
    {{1, 3, 16, 16}},
    {{1, 1, 53, 16}},
    {{1, 2, 53, 16}},
    {{1, 3, 53, 16}},
    {{1, 1, 224, 224}},
    {{1, 2, 224, 224}},
    {{1, 3, 224, 224}},
    {{1, 1, 224, 235}},
    {{1, 2, 224, 235}},
    {{1, 3, 224, 235}},
    // TODO: rewrite to ngraph to have reshape functionality
    // {{10, 17191, 1, 1}},
    {{1, 1, 10, 17191}}
};
