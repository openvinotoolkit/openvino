// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <cmath>

#define BOUND (10.0f)
#define ERROR_BOUND (1.e-2f)
#define ERROR_BOUND_WITH_LOG (1.e-2f)

using namespace InferenceEngine;

class myriadLayersTestsLog_nightly: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<Dims> {
public:
};

TEST_P(myriadLayersTestsLog_nightly, TestsLog)
{
    auto p = ::testing::WithParamInterface<Dims>::GetParam();
    SetInputTensor(p);
    SetOutputTensor(p);
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Log")));
    SetFirstInputToRange(-BOUND, BOUND);
    ASSERT_TRUE(Infer());

    /* output check */
    ref_log(_inputMap.begin()->second, _refBlob);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_logParams = {
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
    {{10, 17191, 1, 1}},
    {{1, 1, 10, 17191}}
};
