// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_hw_tests_base.hpp"

using HWPoolingParams = std::tuple<DimsInput, kernel, stride, pad>;

class MyriadX_HW_Pooling_Tests_nightly
        : public MyriadX_HW_Tests_nightly,
          public testing::WithParamInterface<HWPoolingParams> {
public:
    tensor_test_params in_dims;
    param_size kernel;
    param_size stride;
    param_size pad;

    tensor_test_params out_dims;

    IN_OUT_desc in_tensor, out_tensor;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(MyriadX_HW_Tests_nightly::SetUp());

        in_dims = std::get<0>(GetParam());
        kernel = std::get<1>(GetParam());
        stride = std::get<2>(GetParam());
        pad = std::get<3>(GetParam());

        size_t out_w = std::ceil((in_dims.w + 2.0 * pad.x - kernel.x) / stride.x + 1);
        size_t out_h = std::ceil((in_dims.h + 2.0 * pad.y - kernel.y) / stride.y + 1);

        out_dims = {in_dims.n, in_dims.c, out_h, out_w};

        in_tensor.push_back({in_dims.n, in_dims.c, in_dims.h, in_dims.w});
        out_tensor.push_back({out_dims.n, out_dims.c, out_dims.h, out_dims.w});

        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
    }

    void AddPoolingLayer(const std::string& poolMethod) {
        std::map<std::string, std::string> poolParams = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"pool-method", poolMethod}
        };
        _testNet.addLayer(LayerInitParams("Pooling")
                 .params(poolParams)
                 .in(in_tensor)
                 .out(out_tensor),
                 ref_pooling_wrap);
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

    void RunSingleTest(const std::string& poolMethod, float tolerance) {
        if (!CheckMyriadX()) {
            GTEST_SKIP() << "Non-MyriadX device";
        }

        AddPoolingLayer(poolMethod);

        CompareWithSW(tolerance);
    }

    void RunWithReLUTest(const std::string& poolMethod, float tolerance) {
        if (!CheckMyriadX()) {
            GTEST_SKIP() << "Non-MyriadX device";
        }

        AddPoolingLayer(poolMethod);
        AddReLULayer(0.0f);

        CompareWithSW(tolerance);
    }

    void RunMultipleInferTest(const std::string& poolMethod) {
        if (!CheckMyriadX()) {
            GTEST_SKIP() << "Non-MyriadX device";
        }

        AddPoolingLayer(poolMethod);

        CompareWithItself(100);
    }
};

TEST_P(MyriadX_HW_Pooling_Tests_nightly, Max_Single) {
    RunSingleTest("max", 0.0f);
}

TEST_P(MyriadX_HW_Pooling_Tests_nightly, Avg_Single) {
    // this case is not supported by HW
    if (kernel.x == 3 && kernel.y == 3 &&
        stride.x == 2 && stride.y == 2) {
        GTEST_SKIP() << "Unsupported case";
    }
    if ((kernel.x % 2 == 0 || kernel.y % 2 == 0) &&
        (in_dims.w % 2 == 1 || in_dims.h % 2 == 1)) {
        GTEST_SKIP() << "Unsupported case";
    }

    RunSingleTest("avg", 0.0015f);
}

TEST_P(MyriadX_HW_Pooling_Tests_nightly, Max_WithReLU) {
    RunWithReLUTest("max", 0.0f);
}

TEST_P(MyriadX_HW_Pooling_Tests_nightly, Avg_WithReLU) {
    // this case is not supported by HW
    if (kernel.x == 3 && kernel.y == 3 &&
        stride.x == 2 && stride.y == 2) {
        GTEST_SKIP() << "Unsupported case";
    }
    if ((kernel.x % 2 == 0 || kernel.y % 2 == 0) &&
        (in_dims.w % 2 == 1 || in_dims.h % 2 == 1)) {
        GTEST_SKIP() << "Unsupported case";
    }

    RunWithReLUTest("avg", 0.0015f);
}

TEST_P(MyriadX_HW_Pooling_Tests_nightly, Max_MultipleInfer) {
    RunMultipleInferTest("max");
}

TEST_P(MyriadX_HW_Pooling_Tests_nightly, Avg_MultipleInfer) {
    // this case is not supported by HW
    if (kernel.x == 3 && kernel.y == 3 &&
        stride.x == 2 && stride.y == 2) {
        GTEST_SKIP() << "Unsupported case";
    }
    if ((kernel.x % 2 == 0 || kernel.y % 2 == 0) &&
        (in_dims.w % 2 == 1 || in_dims.h % 2 == 1)) {
        GTEST_SKIP() << "Unsupported case";
    }

    RunMultipleInferTest("avg");
}

INSTANTIATE_TEST_SUITE_P(pool_2x2s1p0, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 112, 112))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_2x2s2p0, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 224, 224),
                                                             MAKE_STRUCT(tensor_test_params, 1, 128, 112, 112),
                                                             MAKE_STRUCT(tensor_test_params, 1, 256, 56, 56),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 28, 28),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 14, 14))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_2x2s2p0_yolo_tiny_v1, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 448, 448))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_3x3s1p0, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 192, 28, 28),
                                                             MAKE_STRUCT(tensor_test_params, 1, 100, 28, 28))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_3x3s1p1, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 192, 28, 28))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

// TODO : 3x3s2p0 HW seems to work only for Max Pooling
INSTANTIATE_TEST_SUITE_P(pool_3x3s2p0, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 112, 112))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_3x3s2p1, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 576, 7, 7),
                                                             MAKE_STRUCT(tensor_test_params, 1, 16, 35, 35),
                                                             MAKE_STRUCT(tensor_test_params, 1, 16, 75, 75),
                                                             MAKE_STRUCT(tensor_test_params, 1, 16, 35, 2045))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_7x7s1p0, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 1024, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_14x14s1p0, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 1024, 14, 14),
                                                             MAKE_STRUCT(tensor_test_params, 1, 1000, 14, 14))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 14, 14))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_15x15s1p0, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 1024, 15, 15),
                                                             MAKE_STRUCT(tensor_test_params, 1, 1000, 15, 15))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 15, 15))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_2x2s1p1_odd, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 13, 13))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_2x2s2p0_odd, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 75, 75),
                                                             MAKE_STRUCT(tensor_test_params, 2, 64, 75, 75),
                                                             MAKE_STRUCT(tensor_test_params, 2, 64, 76, 75),
                                                             MAKE_STRUCT(tensor_test_params, 2, 64, 75, 76))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_3x3s1p0_odd, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 192, 37, 37),
                                                             MAKE_STRUCT(tensor_test_params, 1, 832, 9, 9),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 19, 19))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_3x3s2p0_odd, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 96, 93, 93),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 23, 23),
                                                             MAKE_STRUCT(tensor_test_params, 1, 192, 75, 75),
                                                             MAKE_STRUCT(tensor_test_params, 1, 480, 37, 37))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_3x3s2p0_extra, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 96, 32, 52))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(pool_7x7s7p0_rfcn_batch, MyriadX_HW_Pooling_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 300, 5, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

using PoolTFParams = std::tuple<DimsInput, kernel, stride, tfPad>;

class MyriadX_HW_PoolTF_Tests_nightly :
        public MyriadX_HW_Tests_nightly,
        public testing::WithParamInterface<PoolTFParams>{
public:
    tensor_test_params inDims;
    tensor_test_params outDims;
    param_size kernel;
    param_size stride;
    paddings4 pad;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(MyriadX_HW_Tests_nightly::SetUp());

        inDims = std::get<0>(GetParam());
        kernel = std::get<1>(GetParam());
        stride = std::get<2>(GetParam());
        pad = std::get<3>(GetParam());

        size_t out_w = std::ceil((inDims.w + pad.left + pad.right - kernel.x) / stride.x + 1);
        size_t out_h = std::ceil((inDims.h + pad.top + pad.bottom - kernel.y) / stride.y + 1);

        outDims = {inDims.n, inDims.c, out_h, out_w};
    }

    void AddPoolingLayer() {
        std::map<std::string, std::string> poolParams = {
            {"pool-method", "max"},
            {"kernel-x", std::to_string(kernel.x)},
            {"kernel-y", std::to_string(kernel.y)},
            {"stride-x", std::to_string(stride.x)},
            {"stride-y", std::to_string(stride.y)},
            {"exclude-pad", "true"},
            {"rounding-type", "floor"},
            {"pad-x", std::to_string(pad.left)},
            {"pad-r", std::to_string(pad.right)},
            {"pad-y", std::to_string(pad.top)},
            {"pad-b", std::to_string(pad.bottom)}
        };

        _testNet.addLayer(LayerInitParams("Pooling")
                 .params(poolParams)
                 .in({{inDims.n, inDims.c, inDims.h, inDims.w}})
                 .out({{outDims.n, outDims.c, outDims.h, outDims.w}}),
                 ref_pooling_wrap);
    }
};

TEST_P(MyriadX_HW_PoolTF_Tests_nightly, Single) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddPoolingLayer();

    CompareWithSW(0.0f);
}

INSTANTIATE_TEST_SUITE_P(pool_2x2_3x3, MyriadX_HW_PoolTF_Tests_nightly,
    ::testing::Combine(
        ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 128, 128)),
        ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2),
                                  MAKE_STRUCT(param_size, 3, 3)),
        ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1),
                                  MAKE_STRUCT(param_size, 2, 2)),
        ::testing::Values<tfPad>(MAKE_STRUCT(paddings4, 0, 0, 0, 0),
                                 MAKE_STRUCT(paddings4, 0, 0, 1, 1),
                                 MAKE_STRUCT(paddings4, 1, 0, 0, 0),
                                 MAKE_STRUCT(paddings4, 1, 1, 0, 0),
                                 MAKE_STRUCT(paddings4, 1, 1, 1, 1))
    )
);

INSTANTIATE_TEST_SUITE_P(pool4x4, MyriadX_HW_PoolTF_Tests_nightly,
    ::testing::Combine(
        ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 128, 128)),
        ::testing::Values<kernel>(MAKE_STRUCT(param_size, 4, 4)),
        ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1),
                                  MAKE_STRUCT(param_size, 2, 2)),
        ::testing::Values<tfPad>(MAKE_STRUCT(paddings4, 0, 0, 0, 0),
                                 MAKE_STRUCT(paddings4, 0, 0, 2, 2),
                                 MAKE_STRUCT(paddings4, 1, 1, 2, 2),
                                 MAKE_STRUCT(paddings4, 2, 0, 0, 0),
                                 MAKE_STRUCT(paddings4, 2, 0, 0, 1),
                                 MAKE_STRUCT(paddings4, 2, 0, 1, 0),
                                 MAKE_STRUCT(paddings4, 2, 2, 0, 0),
                                 MAKE_STRUCT(paddings4, 2, 2, 1, 1),
                                 MAKE_STRUCT(paddings4, 2, 2, 2, 2))
    )
);

INSTANTIATE_TEST_SUITE_P(pool_with_large_width, MyriadX_HW_PoolTF_Tests_nightly,
    ::testing::Combine(
        ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 8, 640, 960),
                                     MAKE_STRUCT(tensor_test_params, 1, 64, 6, 1000)),
        ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2),
                                  MAKE_STRUCT(param_size, 3, 3),
                                  MAKE_STRUCT(param_size, 4, 4)),
        ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1),
                                  MAKE_STRUCT(param_size, 2, 2)),
        ::testing::Values<tfPad>(MAKE_STRUCT(paddings4, 0, 0, 0, 0))
    )
);

INSTANTIATE_TEST_SUITE_P(tf, MyriadX_HW_PoolTF_Tests_nightly,
    ::testing::Values(
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 64, 112, 112),   // input
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 2, 2),                      // stride
            MAKE_STRUCT(paddings4, 0, 0, 1, 1)                  // pad
        )
    )
);
