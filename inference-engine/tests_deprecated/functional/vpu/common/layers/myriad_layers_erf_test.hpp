// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <cmath>

// 5. may be also too wide, consider lowering it to 3. - 4. outside +-3.0 it is +-1 with precision of 3 digits.
#define BOUND (5.0f)
#define ERROR_BOUND (1.2e-3f)

using namespace InferenceEngine;

void ref_erf(const InferenceEngine::Blob::Ptr src,
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
                PrecisionUtils::f32tof16(erff(PrecisionUtils::f16tof32(srcData[indx])));
    }
}

class myriadLayersTestsErf_smoke: public myriadLayersTests_nightly,
                                    public testing::WithParamInterface<SizeVector> {
public:
};

TEST_P(myriadLayersTestsErf_smoke, TestsErf)
{
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    auto p = ::testing::WithParamInterface<SizeVector>::GetParam();
    SetInputTensors({p});
    SetOutputTensors({p});
    makeSingleLayerNetwork(LayerInitParams("Erf"));
    SetFirstInputToRange(-BOUND, BOUND);
    ASSERT_TRUE(Infer());

    /* output check */
    ref_erf(_inputMap.begin()->second, _refBlob);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static const std::vector<SizeVector> s_ErfDims = {
    {4, 1, 16, 16},
    {4, 2, 16, 16},
    {4, 3, 16, 16},
    {4, 4, 1, 53, 16},
    {4, 4, 2, 53, 16},
    {4, 4, 3, 53, 16},
    {4, 4, 1, 224, 224},
    {4, 4, 4, 2, 224, 224},
    {4, 4, 4, 3, 224, 224},
    {4, 4, 4, 1, 224, 235},
    {4, 4, 4, 2, 224, 235},
    {4, 4, 4, 3, 224, 235},
    {1, 1, 277, 230},
    {1, 2, 277, 230},
    {1, 3, 277, 230},
    {32, 8, 16}
};
