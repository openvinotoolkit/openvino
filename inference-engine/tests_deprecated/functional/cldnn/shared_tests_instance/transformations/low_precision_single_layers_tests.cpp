// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include <gtest/gtest.h>
#include <string>
#include <memory>

using namespace ::testing;
using namespace InferenceEngine;


TEST_P(SingleLayerTransformationsTest, LPT) {
}

INSTANTIATE_TEST_CASE_P(
        SingleLayerTransformationsTestFP32,
        SingleLayerTransformationsTest,
        ::testing::Values(
                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new FullyConnectedAndScaleShiftsOnActivationsTestModel()),
                //    { { 1, 2048 } },
                //    { { 1, 1000 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnSignedActivationsAndWeightsPositiveTestModel()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnSignedActivationsAndWeightsNegativeTestModel()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnUnsignedActivationsAndWeightsTestModel()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnSignedActivationsAndInvertedWeightsTestModel()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeReshapePoolingTestModelWithConstants()),
                        { { 1, 1280, 7 } },
                        { { 1, 1280, 7 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeReshapePoolingTestModelWithoutConstants()),
                        { { 1, 1280, 7 } },
                        { { 1, 1280, 7 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FullyConnectedAndQuantizeTestModel()),
                        { { 1, 32, 1, 1 } },
                        { { 1, 32, 1, 1 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FullyConnectedAndScaleShiftsOnActivationsTestModel()),
                        { { 1, 2048 } },
                        { { 1, 1000 } }),

//                SingleLayerTransformationsTestParams(
//                        "GPU",
//                        SingleLayerTestModel::Ptr(new GemmAndQuantizeTestModel()),
//                        { { 1, 32, 149, 149 } },
//                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new PoolingTestModel()),
                        { { 149, 149, 32, 1 } },
                        { { 149, 149, 32, 1 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel()),
                        { { 1, 32, 147, 147 } },
                        { { 1, 64, 147, 147 } }),

                // Const transformation is disabled
                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnWeightsWithoutConstTransformationTestModel()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndPoolingAndQuantizeOnActivationsTestModel()),
                        { { 1, 64, 147, 147 } },
                        { { 1, 80, 73,  73  } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnActivationsTestModel()),
                        { { 1, 3,  299, 299 } },
                        { { 1, 32, 149, 149 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel()),
                        { { 1, 3,  299, 299 } },
                        { { 1, 32, 149, 149 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel()),
                        { { 1, 3, 299, 299 } },
                        { { 1, 32, 149, 149 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionDepthwiseTestModel()),
                        { { 1, 32, 112, 112 } },
                        { { 1, 32, 112, 112 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionGroupedTestModel()),
                        { { 1, 32, 112, 112 } },
                        { { 1, 32, 112, 112 } }),

//                SingleLayerTransformationsTestParams(
//                        "GPU",
//                        SingleLayerTestModel::Ptr(new EltwiseTestModel()),
//                        { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
//                        { { 1, 3, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new EltwiseCpuTestModel()),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConcatTestModel(true, true, true)),
                        { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
                        { { 1, 6, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConcatTestModel(true, true, false)),
                        { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
                        { { 1, 6, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConcatTestModel(false)),
                        { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
                        { { 1, 6, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConcatMultiChannelTestModel()),
                        { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
                        { { 1, 6, 299, 299 } }),

                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new ConcatMultiBranchTestModel()),
                //    { { 299, 299, 3, 1 }, { 299, 299, 3, 1 } },
                //    { { 299, 299, 12, 1 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new QuantizationOnWeightsTestModel()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new QuantizationOnInvertedWeightsTestModel()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeAsOutputTest()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeWithMultiOutputsTest()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeAndScaleShiftTestModel()),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationTestModel({ {-10.25, 10.1641} })),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationTestModel({ {-0.00174255, 0.00174255} })),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationTestModel({ {-329.688, 327.188} })),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationWithNegativeScalesTestModel()),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationWithNegativeSlopeTestModel()),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ScaleShiftAndFakeQuantizeTestModel()),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } })

        ),
        SingleLayerTransformationsTestParams::getLowPrecisionTransformerSingleLayerTestName);


INSTANTIATE_TEST_CASE_P(
        SingleLayerTransformationsTestFP16,
        SingleLayerTransformationsTest,
        ::testing::Values(
                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FullyConnectedAndScaleShiftsOnActivationsTestModel()),
                        { { 1, 2048 } },
                        { { 1, 1000 } },
                        "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FullyConnectedAndQuantizeTestModel()),
                        { { 1, 32, 1, 1 } },
                        { { 1, 32, 1, 1 } },
                        "FP16"),

                // TODO: uncomment after fix
                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnSignedActivationsAndWeightsTestModel()),
                //    { { 1, 32, 149, 149 } },
                //    { { 1, 32, 147, 147 } },
                //    "FP16"),

                // TODO: uncomment after fix
                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnUnsignedActivationsAndWeightsTestModel()),
                //    { { 1, 32, 149, 149 } },
                //    { { 1, 32, 147, 147 } },
                //    "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeReshapePoolingTestModelWithConstants()),
                        { { 1, 1280, 7 } },
                        { { 1, 1280, 7 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeReshapePoolingTestModelWithoutConstants()),
                        { { 1, 1280, 7 } },
                        { { 1, 1280, 7 } }),


                //Not parametrized yet. Executed on FP32

                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new FullyConnectedAndQuantizeTestModel()),
                //    { { 1, 32, 149, 149 } },
                //    { { 1, 32, 147, 147 } },
                //    "FP16"),

                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new GemmAndQuantizeTestModel()),
                //    { { 1, 32, 149, 149 } },
                //    { { 1, 32, 147, 147 } },
                //    "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new PoolingTestModel()),
                        { { 149, 149, 32, 1 } },
                        { { 149, 149, 32, 1 } },
                        "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel()),
                        { { 1, 32, 147, 147 } },
                        { { 1, 64, 147, 147 } },
                        "FP16"),

                // TODO: uncomment after fix
                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnWeightsWithoutConstTransformationTestModel()),
                //    { { 1, 32, 149, 149 } },
                //    { { 1, 32, 147, 147 } },
                //    "FP16"),

                // TODO: uncomment after fix
                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new ConvolutionAndPoolingAndQuantizeOnActivationsTestModel()),
                //    { { 1, 64, 147, 147 } },
                //    { { 1, 80, 73,  73  } },
                //    "FP16"),

                // TODO: uncomment after fix
                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnActivationsTestModel()),
                //    { { 1, 3,  299, 299 } },
                //    { { 1, 32, 149, 149 } },
                //    "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel()),
                        { { 1, 3,  299, 299 } },
                        { { 1, 32, 149, 149 } },
                        "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel()),
                        { { 1, 3, 299, 299 } },
                        { { 1, 32, 149, 149 } },
                        "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionDepthwiseTestModel()),
                        { { 1, 32, 112, 112 } },
                        { { 1, 32, 112, 112 } },
                        "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConvolutionGroupedTestModel()),
                        { { 1, 32, 112, 112 } },
                        { { 1, 32, 112, 112 } }),

//                SingleLayerTransformationsTestParams(
//                        "GPU",
//                        SingleLayerTestModel::Ptr(new EltwiseTestModel()),
//                        { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
//                        { { 1, 3, 299, 299 } },
//                        "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new EltwiseCpuTestModel()),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } }),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConcatTestModel(true)),
                        { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
                        { { 1, 6, 299, 299 } },
                        "FP16"),

                SingleLayerTransformationsTestParams(
                    "GPU",
                    SingleLayerTestModel::Ptr(new ConcatTestModel(false)),
                    { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
                    { { 1, 6, 299, 299 } },
                    "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new ConcatMultiChannelTestModel()),
                        { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
                        { { 1, 6, 299, 299 } }),

                //SingleLayerTransformationsTestParams(
                //    "GPU",
                //    SingleLayerTestModel::Ptr(new ConcatMultiBranchTestModel()),
                //    { { 299, 299, 3, 1 }, { 299, 299, 3, 1 } },
                //    { { 299, 299, 12, 1 } },
                //    "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new QuantizationOnWeightsTestModel()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } },
                        "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new QuantizationOnInvertedWeightsTestModel()),
                        { { 1, 32, 149, 149 } },
                        { { 1, 32, 147, 147 } },
                        "FP16"),

                SingleLayerTransformationsTestParams(
                        "GPU",
                        SingleLayerTestModel::Ptr(new FakeQuantizeAndScaleShiftTestModel()),
                        { { 1, 3, 299, 299 } },
                        { { 1, 3, 299, 299 } },
                        "FP16")
        ),
        SingleLayerTransformationsTestParams::getLowPrecisionTransformerSingleLayerTestName);
