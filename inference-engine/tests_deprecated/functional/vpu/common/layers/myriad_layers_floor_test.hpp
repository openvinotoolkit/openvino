// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <cmath>

#define BOUND (1.e+7f)
#define ERROR_BOUND (0.f)

using namespace InferenceEngine;

void ref_floor(const InferenceEngine::Blob::Ptr src,
             InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->getTensorDesc().getDims().size(), dst->getTensorDesc().getDims().size());
    ie_fp16 *srcData = src->buffer();
    ie_fp16 *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    for (size_t indx = 0; indx < src->size(); indx++) {
        dstData[indx] =
                PrecisionUtils::f32tof16(floorf(PrecisionUtils::f16tof32(srcData[indx])));
    }
}

class myriadLayersTestsFloor_nightly: public myriadLayersTests_nightly,
                                    public testing::WithParamInterface<Dims> {
public:
};

TEST_P(myriadLayersTestsFloor_nightly, TestsFloor)
{
    auto p = ::testing::WithParamInterface<Dims>::GetParam();
    SetInputTensor(p);
    SetOutputTensor(p);
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Floor")));
    SetFirstInputToRange(-BOUND, BOUND);
    ASSERT_TRUE(Infer());

    /* output check */
    ref_floor(_inputMap.begin()->second, _refBlob);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_FloorParams = {
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
