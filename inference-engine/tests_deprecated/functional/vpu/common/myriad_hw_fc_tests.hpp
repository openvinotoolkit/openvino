// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_hw_tests_base.hpp"

class MyriadX_HW_FullyConnected_Tests_nightly
        : public MyriadX_HW_Tests_nightly,
          public testing::WithParamInterface<fcon_test_params> {
public:
    fcon_test_params p;

    size_t numWeights;
    size_t numBias;

    IN_OUT_desc in_tensor, out_tensor;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(MyriadX_HW_Tests_nightly::SetUp());

        p = GetParam();

        numWeights = p.in.c * p.in.h * p.in.w * p.out_c;
        numBias = p.out_c;

        in_tensor.push_back({p.in.n, p.in.c, p.in.h, p.in.w});
        out_tensor.push_back({p.in.n, p.out_c});

        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
    }

    void AddFCLayer() {
        std::map<std::string, std::string> fcParams;
        fcParams["out-size"] = std::to_string(p.out_c);
        _testNet.addLayer(LayerInitParams("FullyConnected")
                 .params(fcParams)
                 .weights(numWeights).fillWeights(defaultWeightsRange)
                 .biases(numBias).fillBiases(defaultWeightsRange)
                 .in(in_tensor)
                 .out(out_tensor),
                 ref_innerproduct_wrap);
    }

    void AddReLULayer(float negativeSlope = 0.0) {
        ParamsStruct reluParams = {
            {"negative_slope", std::to_string(negativeSlope)}
        };
        _testNet.addLayer(LayerInitParams("ReLU")
                 .params(reluParams)
                 .in(out_tensor)
                 .out(out_tensor),
                 ref_ReLU_wrap);
    }
};

TEST_P(MyriadX_HW_FullyConnected_Tests_nightly, Single) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddFCLayer();

    CompareWithSW(p.error_bound);
}

TEST_P(MyriadX_HW_FullyConnected_Tests_nightly, Single_NC) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    if (p.in.h != 1 || p.in.w != 1) {
        GTEST_SKIP() << "Non NC case";
    }

    in_tensor.clear();
    in_tensor.push_back({p.in.n, p.in.c});

    AddFCLayer();

    CompareWithSW(p.error_bound);
}

TEST_P(MyriadX_HW_FullyConnected_Tests_nightly, WithReLU) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddFCLayer();
    AddReLULayer(0.0f);

    CompareWithSW(p.error_bound);
}

TEST_P(MyriadX_HW_FullyConnected_Tests_nightly, MultipleInfer) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddFCLayer();

    CompareWithItself(100);
}

INSTANTIATE_TEST_SUITE_P(fc_1024to1000, MyriadX_HW_FullyConnected_Tests_nightly,
                        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 1024, 1, 1}, 1000, 0.25f))
);

INSTANTIATE_TEST_SUITE_P(fc_4096to1000, MyriadX_HW_FullyConnected_Tests_nightly,
                        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 4096, 1, 1}, 1000, 0.82f))
);

INSTANTIATE_TEST_SUITE_P(fc_4096to4096, MyriadX_HW_FullyConnected_Tests_nightly,
                        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 4096, 1, 1}, 4096, 0.9f))
);

INSTANTIATE_TEST_SUITE_P(fc_16x16x16to16, MyriadX_HW_FullyConnected_Tests_nightly,
                        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 16, 16, 16}, 16, 0.71f))
);

INSTANTIATE_TEST_SUITE_P(fc_512x7x7to4096, MyriadX_HW_FullyConnected_Tests_nightly,
                        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 512, 7, 7}, 4096, 4.38f))
);

INSTANTIATE_TEST_SUITE_P(fc_256x7x7to1470, MyriadX_HW_FullyConnected_Tests_nightly,
                        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 256, 7, 7}, 1470, 2.375f))
);

INSTANTIATE_TEST_SUITE_P(fc_576to128, MyriadX_HW_FullyConnected_Tests_nightly,
                        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 576, 1, 1}, 128, 0.76f))
);

INSTANTIATE_TEST_SUITE_P(fc_1152to128, MyriadX_HW_FullyConnected_Tests_nightly,
                        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 1152, 1, 1}, 128, 0.76f))
);

INSTANTIATE_TEST_SUITE_P(fc_batch, MyriadX_HW_FullyConnected_Tests_nightly,
                        ::testing::Values(MAKE_STRUCT(fcon_test_params, {100, 256, 1, 1}, 1024, 0.1f))
);
