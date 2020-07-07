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
    smoke_SingleLayerTransformationsTestFP32,
    SingleLayerTransformationsTest,
    ::testing::Values(
        SingleLayerTransformationsTestParams(
            "CPU",
            PowerTestModel::Ptr(new PowerTestModel(1.f, 1.f, 0)),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            PowerTestModel::Ptr(new PowerTestModel(1.f, 2.89f, 64)),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            PowerTestModel::Ptr(new PowerTestModel(1.f, -32.f, 0)),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            PowerTestModel::Ptr(new PowerTestModel(1.f, 1.f, -64.f)),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            PowerTestModel::Ptr(new PowerTestModel(3.5f, 1.f, 0)),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ResampleTestModel()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FullyConnectedAndScaleShiftsOnActivationsTestModel()),
            { { 1, 2048 } },
            { { 1, 1000 } }),

        // TODO: uncomment later
        //SingleLayerTransformationsTestParams(
        //    "MKLDNNPlugin",
        //    SingleLayerTestModel::Ptr(new FullyConnectedTestModel({ 1, 128, 12, 64 }, { 128, 768 })),
        //    { { 1, 128, 12, 64 } },
        //    { { 128, 768 } }),

        // SingleLayerTransformationsTestParams(
        //     "CPU",
        //     SingleLayerTestModel::Ptr(new FullyConnectedTestModel({ 1, 128, 12, 64 }, { 1, 128, 768 })),
        //     { { 1, 128, 12, 64 } },
        //     { { 1, 128, 768 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnSignedActivationsAndWeightsPositiveTestModel()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnSignedActivationsAndWeightsNegativeTestModel()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnUnsignedActivationsAndWeightsTestModel()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnSignedActivationsAndInvertedWeightsTestModel()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeReshapePoolingTestModelWithConstants()),
            { { 1, 1280, 7 } },
            { { 1, 1280, 7 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeReshapePoolingTestModelWithoutConstants()),
            { { 1, 1280, 7 } },
            { { 1, 1280, 7 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeReshapeTestModelWithConstants()),
            { { 1, 256, 6, 6 } },
            { { 1, 9216 } }),

        // TODO: fix asymmetric patern creation issue for NC layout and uncomment
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new FullyConnectedAndQuantizeTestModel()),
        //    { { 1, 32, 1, 1 } },
        //    { { 1, 32, 1, 1 } }),

        // TODO: uncomment when biases correction with absent biases will be fixed
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new GemmAndQuantizeTestModel()),
        //    { { 1, 32, 149, 149 } },
        //    { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new PoolingTestModel()),
            { { 149, 149, 32, 1 } },
            { { 149, 149, 32, 1 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel()),
            { { 1, 32, 147, 147 } },
            { { 1, 64, 147, 147 } }),

        // Const transformation is disabled
        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnWeightsWithoutConstTransformationTestModel()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndPoolingAndQuantizeOnActivationsTestModel()),
            { { 1, 64, 147, 147 } },
            { { 1, 80, 73,  73  } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnActivationsTestModel()),
            { { 1, 3,  299, 299 } },
            { { 1, 32, 149, 149 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel()),
            { { 1, 3,  299, 299 } },
            { { 1, 32, 149, 149 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel()),
            { { 1, 3, 299, 299 } },
            { { 1, 32, 149, 149 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionDepthwiseTestModel()),
            { { 1, 32, 112, 112 } },
            { { 1, 32, 112, 112 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionGroupedTestModel()),
            { { 1, 32, 112, 112 } },
            { { 1, 32, 112, 112 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatMultiChannelTestModel()),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConcatMultiBranchTestModel()),
        //    { { 299, 299, 3, 1 }, { 299, 299, 3, 1 } },
        //    { { 299, 299, 12, 1 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new QuantizationOnWeightsTestModel()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } }),

        // TODO: fix later
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new QuantizationOnInvertedWeightsTestModel()),
        //    { { 1, 32, 149, 149 } },
        //    { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeAsOutputTest()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeWithMultiOutputsTest()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeAndScaleShiftTestModel()),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationTestModel({ {-10.25, 10.1641} })),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationTestModel({ {-0.00174255, 0.00174255} })),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationTestModel({ {-329.688, 327.188} })),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationWithNegativeScalesTestModel()),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeAndActivationWithNegativeSlopeTestModel()),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ScaleShiftAndFakeQuantizeTestModel()),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeWithTwoScaleShiftsAsOutput()),
            { { 1, 32, 28, 28 }, { } },
            { { } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new MvnTestModel(0ul, 0ul)),
            { { 1, 4, 128, 128, 128 } },
            { { 1, 4, 128, 128, 128 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new MvnTestModel(1ul, 0ul)),
            { { 1, 4, 128, 128, 128 } },
            { { 1, 4, 128, 128, 128 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new MvnTestModel(0ul, 1ul)),
            { { 1, 4, 128, 128, 128 } },
            { { 1, 4, 128, 128, 128 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new MvnTestModel(1ul, 1ul)),
            { { 1, 4, 128, 128, 128 } },
            { { 1, 4, 128, 128, 128 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new PrecisionSelectionMultibranchPreservedTestModel(true)),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 149, 149 }, { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new PrecisionSelectionMultibranchPreservedTestModel(false)),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 149, 149 }, { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new PrecisionSelectionMultibranchNotPreservedTestModel(true)),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 149, 149 }, { 1, 32, 147, 147 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new PrecisionSelectionMultibranchNotPreservedTestModel(false)),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 149, 149 }, { 1, 32, 147, 147 } })
    ),
    SingleLayerTransformationsTestParams::getLowPrecisionTransformerSingleLayerTestName);

INSTANTIATE_TEST_CASE_P(
    smoke_EltwiseTestFP32,
    SingleLayerTransformationsTest,
    ::testing::Values(
        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseTestModel(true, "sum", true)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseTestModel(true, "sum", false)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseTestModel(true, "mul", true)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseTestModel(true, "mul", false)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseFqWithChildrenTestModel(true, "sum", true)),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseFqWithChildrenTestModel(true, "sum", false)),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseFqWithChildrenTestModel(true, "mul", true)),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseFqWithChildrenTestModel(true, "mul", false)),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } })
    ),
    SingleLayerTransformationsTestParams::getLowPrecisionTransformerSingleLayerTestName);

INSTANTIATE_TEST_CASE_P(
    smoke_ConcatTestFP32,
    SingleLayerTransformationsTest,
    ::testing::Values(
        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatTestModel(true, true, true)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatTestModel(true, true, false)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatTestModel(true, false)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatTestModel(false, true)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConcatTestModel(false, false)),
        //    { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
        //    { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatTestModel(true, true, true, { 100, 1 })),
            { { 100, 1 }, { 100, 1 } },
            { { 100, 2 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatTestModel(true, true, false, { 100, 1 })),
            { { 100, 1 }, { 100, 1 } },
            { { 100, 2 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatTestModel(false, true, true, { 100, 1 })),
            { { 100, 1 }, { 100, 1 } },
            { { 100, 2 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatTestModel(false, true, false, { 100, 1 })),
            { { 100, 1 }, { 100, 1 } },
            { { 100, 2 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatWithPoolingTestModel(false, false, false, 2.0)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatWithPoolingTestModel(false, true, false, 2.0)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatWithPoolingTestModel(true, false, false, 2.0)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatWithPoolingTestModel(true, true, false, 2.0)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatWithPoolingTestModel(false, false, true, 2.0)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatWithPoolingTestModel(false, true, true, 2.0)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatWithPoolingTestModel(true, false, true, 2.0)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatWithPoolingTestModel(true, true, true, 2.0)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } })
    ),
    SingleLayerTransformationsTestParams::getLowPrecisionTransformerSingleLayerTestName);

INSTANTIATE_TEST_CASE_P(
    smoke_ScaleShiftToConvolutionFP32,
    SingleLayerTransformationsTest,
    ::testing::Values(
        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ScaleShiftToConvolutionAfterNotConcatIgnoreTestModel()),
            { { 1, 64, 112, 112 } },
            { { 1, 64, 112, 112 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ScaleShiftToConvolutionAfterFakeQuantizeIgnoreTestModel()),
            { { 1, 64, 112, 112 } },
            { { 1, 64, 112, 112 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ScaleShiftToConvolutionAfterConcatTestModel(true)),
            { { 1, 32, 299, 299 }, { 1, 32, 299, 299 } },
            { { 1, 64, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ScaleShiftToConvolutionAfterConcatTestModel(false)),
            { { 1, 32, 299, 299 }, { 1, 32, 299, 299 } },
            { { 1, 64, 299, 299 } })
    ),
    SingleLayerTransformationsTestParams::getLowPrecisionTransformerSingleLayerTestName);

INSTANTIATE_TEST_CASE_P(
    smoke_UpdateBiases,
    SingleLayerTransformationsTest,
    ::testing::Values(
        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new UpdateBiasesConvolutionTestModel(false)),
            { { 1, 32, 112, 112 } },
            { { 1, 32, 112, 112 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new UpdateBiasesConvolutionTestModel(true)),
            { { 1, 32, 112, 112 } },
            { { 1, 32, 112, 112 } })

        // TODO: uncomment later
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new UpdateBiasesFullyConnectedTestModel(false)),
        //    { { 1, 32, 112, 112 } },
        //    { { 1, 100 } }),

        // TODO: uncomment later
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new UpdateBiasesFullyConnectedTestModel(true)),
        //    { { 1, 32, 112, 112 } },
        //    { { 1, 100 } })
    ),
    SingleLayerTransformationsTestParams::getLowPrecisionTransformerSingleLayerTestName);

INSTANTIATE_TEST_CASE_P(
    smoke_EltwiseCpuWithPooling,
    SingleLayerTransformationsTest,
    ::testing::Values(
        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseWithPoolingTestModel(true, "mul", false)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseWithPoolingTestModel(true, "mul", true)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseWithPoolingTestModel(true, "sum", false)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseWithPoolingTestModel(true, "sum", true)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } })
    ),
    SingleLayerTransformationsTestParams::getLowPrecisionTransformerSingleLayerTestName);

INSTANTIATE_TEST_CASE_P(
    smoke_Eltwise,
    SingleLayerTransformationsTest,
    ::testing::Values(
        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseWithPoolingTestModel(true, "sum", false)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseWithPoolingTestModel(true, "sum", true)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseCpuTestModel()),
            { { 1, 3, 299, 299 } },
            { {} }),

//        SingleLayerTransformationsTestParams(
//            "CPU",
//            SingleLayerTestModel::Ptr(new EltwiseTestModel()),
//            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
//            { {} },
//            "FP16"),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseCpuTestModel()),
            { { 1, 3, 299, 299 } },
            { {} },
            "FP16"),

//        SingleLayerTransformationsTestParams(
//            "CPU",
//            SingleLayerTestModel::Ptr(new EltwiseBroadcastTestModel()),
//            { { 1, 128, 128 }, { 1, 128, 128 } },
//            { { 1, 128, 128 } }),

        SingleLayerTransformationsTestParams( // 5
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseBroadcastTestModel()),
            { { 1, 1,   128 }, { 1, 128, 128 } },
            { { 1, 128, 128 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseBroadcastTestModel()),
            { { 1, 128, 128 }, { 1, 128, 1 } },
            { { 1, 128, 128 } }),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new EltwiseBroadcastTestModel()),
            { { 1, 1,   128 }, { 1, 128, 1 } },
            { { 1, 128, 128 } })));

INSTANTIATE_TEST_CASE_P(
    smoke_SingleLayerTransformationsTestFP16,
    SingleLayerTransformationsTest,
    ::testing::Values(
        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FullyConnectedAndScaleShiftsOnActivationsTestModel()),
            { { 1, 2048 } },
            { { 1, 1000 } },
            "FP16"),

        // TODO: uncomment after fix
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnSignedActivationsAndWeightsTestModel()),
        //    { { 1, 32, 149, 149 } },
        //    { { 1, 32, 147, 147 } },
        //    "FP16"),

        // TODO: uncomment after fix
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnUnsignedActivationsAndWeightsTestModel()),
        //    { { 1, 32, 149, 149 } },
        //    { { 1, 32, 147, 147 } },
        //    "FP16"),

        // TODO: uncomment after fix
//        SingleLayerTransformationsTestParams(
//            "CPU",
//            SingleLayerTestModel::Ptr(new FakeQuantizeReshapePoolingTestModelWithConstants()),
//            { { 1, 1280, 7 } },
//            { { 1, 1280, 7 } },
//            "FP16"),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeReshapePoolingTestModelWithoutConstants()),
            { { 1, 1280, 7 } },
            { { 1, 1280, 7 } },
            "FP16"),

        // TODO: uncomment after fix
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new FakeQuantizeReshapeTestModelWithConstants()),
        //    { { 1, 256, 6, 6 } },
        //    { { 1, 9216 } },
        //    "FP16"),

        //Not parametrized yet. Executed on FP32

        // TODO: fix asymmetric patern creation issue for NC layout and uncomment
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new FullyConnectedAndQuantizeTestModel()),
        //    { { 1, 32, 149, 149 } },
        //    { { 1, 32, 147, 147 } },
        //    "FP16"),

        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new GemmAndQuantizeTestModel()),
        //    { { 1, 32, 149, 149 } },
        //    { { 1, 32, 147, 147 } },
        //    "FP16"),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new PoolingTestModel()),
            { { 149, 149, 32, 1 } },
            { { 149, 149, 32, 1 } },
            "FP16"),

        // TODO: failed on I8 on activations - uncomment after fix
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel()),
        //    { { 1, 32, 147, 147 } },
        //    { { 1, 64, 147, 147 } },
        //    "FP16"),

        // TODO: uncomment after fix
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnWeightsWithoutConstTransformationTestModel()),
        //    { { 1, 32, 149, 149 } },
        //    { { 1, 32, 147, 147 } },
        //    "FP16"),

        // TODO: uncomment after fix
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConvolutionAndPoolingAndQuantizeOnActivationsTestModel()),
        //    { { 1, 64, 147, 147 } },
        //    { { 1, 80, 73,  73  } },
        //    "FP16"),

        // TODO: uncomment after fix
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConvolutionAndQuantizeOnActivationsTestModel()),
        //    { { 1, 3,  299, 299 } },
        //    { { 1, 32, 149, 149 } },
        //    "FP16"),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel()),
            { { 1, 3,  299, 299 } },
            { { 1, 32, 149, 149 } },
            "FP16"),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel()),
            { { 1, 3, 299, 299 } },
            { { 1, 32, 149, 149 } },
            "FP16"),

        // TODO: fix and uncomment
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConvolutionDepthwiseTestModel()),
        //    { { 1, 32, 112, 112 } },
        //    { { 1, 32, 112, 112 } },
        //    "FP16"),

        // TODO: fix and uncomment
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConvolutionGroupedTestModel()),
        //    { { 1, 32, 112, 112 } },
        //    { { 1, 32, 112, 112 } },
        //    "FP16"),

        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConcatTestModel(true)),
        //    { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
        //    { { 1, 6, 299, 299 } },
        //    "FP16"),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatTestModel(false, true)),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 } },
            "FP16"),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new ConcatMultiChannelTestModel()),
            { { 1, 3, 299, 299 }, { 1, 3, 299, 299 } },
            { { 1, 6, 299, 299 }, },
            "FP16"),

        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ConcatMultiBranchTestModel()),
        //    { { 299, 299, 3, 1 }, { 299, 299, 3, 1 } },
        //    { { 299, 299, 12, 1 } },
        //    "FP16"),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new QuantizationOnWeightsTestModel()),
            { { 1, 32, 149, 149 } },
            { { 1, 32, 147, 147 } },
            "FP16"),

        // TODO: fix later
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new QuantizationOnInvertedWeightsTestModel()),
        //    { { 1, 32, 149, 149 } },
        //    { { 1, 32, 147, 147 } },
        //    "FP16"),

        SingleLayerTransformationsTestParams(
            "CPU",
            SingleLayerTestModel::Ptr(new FakeQuantizeAndScaleShiftTestModel()),
            { { 1, 3, 299, 299 } },
            { { 1, 3, 299, 299 } },
            "FP16")

        // TODO: fix and uncomment
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ScaleShiftToConvolutionAfterNotConcatIgnoreTestModel()),
        //    { { 1, 64, 112, 112 } },
        //    { { 1, 64, 112, 112 } },
        //    "FP16")

        // TODO: fix and uncomment
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ScaleShiftToConvolutionAfterFakeQuantizeIgnoreTestModel()),
        //    { { 1, 64, 112, 112 } },
        //    { { 1, 64, 112, 112 } },
        //    "FP16")

        // TODO: fix and uncomment
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new ScaleShiftToConvolutionAfterConcatTestModel()),
        //    { { 1, 32, 299, 299 }, { 1, 32, 299, 299 } },
        //    { { 1, 64, 299, 299 } },
        //    "FP16")

        // TODO: fix and uncomment
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new UpdateBiasesConvolutionTestModel(false)),
        //    { { 1, 32, 112, 112 } },
        //    { { 1, 32, 112, 112 } },
        //    "FP16"),

        // TODO: fix and uncomment
        //SingleLayerTransformationsTestParams(
        //    "CPU",
        //    SingleLayerTestModel::Ptr(new UpdateBiasesConvolutionTestModel(true)),
        //    { { 1, 32, 112, 112 } },
        //    { { 1, 32, 112, 112 } },
        //    "FP16")
        ),
    SingleLayerTransformationsTestParams::getLowPrecisionTransformerSingleLayerTestName);
