// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bfloat16_helpers.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/interpolate.hpp"

#include "ngraph/opsets/opset3.hpp"

using namespace InferenceEngine;
using namespace ngraph;

namespace LayerTestsDefinitions {

class InterpolateLayerTestBF16 : public testing::WithParamInterface<InterpolateLayerTestParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerTestParams> obj) {
        return LayerTestsDefinitions::InterpolateLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::InterpolateLayerTestParams>(
                obj.param, 0));
    }

protected:
    std::shared_ptr<ngraph::Function> createGraph(LayerTestsDefinitions::InterpolateLayerTestParams &params) {
        //    Power (FP32)
        //        |
        //       Interpolation (BF16)

        InterpolateSpecificParams interpolateParams;
        std::vector<size_t> inputShape, targetShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;

        std::tie(interpolateParams, netPrecision, inputShape, targetShape, targetDevice) = params;

        // STAGE1: construction of the GRAPH
        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;

        // multiply
        auto input1 = std::make_shared<opset3::Parameter>(ntype, ngraph::Shape{inputShape});
        input1->set_friendly_name("Input_1");
        std::shared_ptr<ngraph::opset1::Constant> const1 = nullptr;
        if (netPrecision == Precision::FP32) {
            const1 = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            const1 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto mulNode = std::make_shared<opset1::Multiply>(input1, const1);

        // add
        std::shared_ptr<ngraph::opset1::Constant> addConst = nullptr;
        if (netPrecision == Precision::FP32) {
            addConst = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            addConst = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto addNode = std::make_shared<opset1::Add>(mulNode, addConst);
        addNode->set_friendly_name("Add_1");

        // interpolation
        std::vector<size_t> padBegin, padEnd;
        bool antialias;
        ngraph::op::v4::Interpolate::InterpolateMode mode;
        ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode;
        ngraph::op::v4::Interpolate::NearestMode nearestMode;

        using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;
        ShapeCalcMode shape_calc_node = ShapeCalcMode::sizes;
        double cubeCoef;
        tie(mode, coordinateTransformMode, nearestMode, antialias, padBegin, padEnd, cubeCoef) = interpolateParams;

        auto constant = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {targetShape.size()}, targetShape);
        auto targetShapeConst = std::make_shared<ngraph::opset3::Constant>(constant);

        std::vector<float> scales(targetShape.size(), 1.0f);
        auto scales_const = ngraph::opset3::Constant(ngraph::element::Type_t::f32, {scales.size()}, scales);

        auto scalesInput = std::make_shared<ngraph::opset3::Constant>(scales_const);

        ngraph::op::v4::Interpolate::InterpolateAttrs interpolateAttributes{mode, shape_calc_node, padBegin,
                                                                            padEnd, coordinateTransformMode, nearestMode, antialias, cubeCoef};
        auto interpolNode = std::make_shared<ngraph::op::v4::Interpolate>(addNode,
                                                                         targetShapeConst,
                                                                         scalesInput,
                                                                         interpolateAttributes);

        interpolNode->set_friendly_name("Interp");

        return std::make_shared<ngraph::Function>(interpolNode, ngraph::ParameterVector{input1});
    }

    void Run() override {
        if (!InferenceEngine::with_cpu_x86_bfloat16()) {
            // on platforms which do not support bfloat16, we are disabling bf16 tests since there are no bf16 primitives,
            // tests are useless on such platforms
            GTEST_SKIP();
        }

        LayerTestsCommon::Run();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        auto blobPtr = make_blob_with_precision(info.getTensorDesc());
        blobPtr->allocate();
        if (0 == FuncTestUtils::fillInputsBySinValues(blobPtr)) {
            return blobPtr;
        } else {
            return nullptr;
        }
    }

    void ValidateWithFP32Inference() {
        auto outBlobBF16 = GetOutputs();
        InferenceEngine::MemoryBlob::CPtr mout1 = InferenceEngine::as<InferenceEngine::MemoryBlob>(outBlobBF16.front());
        ASSERT_NE(mout1, nullptr);
        auto lm1 = mout1->rmap();

        auto testParams = GetParam();

        // FP32 inference
        // if netPrecision is not eq to the FP32 - change network precision and recreate network
        std::get<1>(testParams) = Precision::FP32;
        InferenceEngine::CNNNetwork cnnNetFP32(createGraph(testParams));
        std::string inputNameFP32 = cnnNetFP32.getInputsInfo().begin()->first;
        std::string outputNameFP32 = cnnNetFP32.getOutputsInfo().begin()->first;
        BasicBF16Test::setNetInOutPrecision(cnnNetFP32, inPrc);
        auto exec_net2 = core->LoadNetwork(cnnNetFP32, targetDevice,
                                        { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO } });
        auto req2 = exec_net2.CreateInferRequest();

        req2.SetBlob(inputNameFP32, inputs.front());

        req2.Infer();
        auto outBlobFP32 = req2.GetBlob(outputNameFP32);
        InferenceEngine::MemoryBlob::CPtr mout2 = InferenceEngine::as<InferenceEngine::MemoryBlob>(outBlobFP32);
        ASSERT_NE(mout2, nullptr);
        auto lm2 = mout2->rmap();

        // debug to figure out the maximum value in output tensors:
        // std::cout << "Max in bfloat16 network by output " << outputName << ": " <<
        //      BFloat16Helpers::getMaxAbsValue(lm1.as<const float *>(), mout1->size()) << std::endl;
        // std::cout << "Max in fp32 network by output " << outputNameFP32 << ": " <<
        //     BFloat16Helpers::getMaxAbsValue(lm2.as<const float *>(), mout2->size()) << std::endl;
        FuncTestUtils::compareRawBuffers(lm1.as<const float *>(),
                                         lm2.as<const float *>(),
                                         mout1->size(), mout2->size(),
                                         FuncTestUtils::CompareType::ABS,
                                         threshold);
    }

    void Validate() override {
        LayerTestsCommon::Validate();

        //ValidateWithFP32Inference();

        // Stage2: verification of performance counters
        std::pair<std::string, std::string> wrongLayer =
                BFloat16Helpers::matchPerfCountPrecisionVsExpected(inferRequest.GetPerformanceCounts(), expectedPrecisions);
        if (wrongLayer.first != std::string("")) {
            std::string layerInPerfCounts = wrongLayer.first + " " + wrongLayer.second;
            std::string layerExpected = wrongLayer.first + " " + expectedPrecisions[wrongLayer.first];
            ASSERT_EQ(layerInPerfCounts, layerExpected);
        }
    }

    void SetUp() override {
        auto testParams = GetParam();
        auto netPrecision = std::get<1>(testParams);

        //Set specific bf16 configuration
        if (netPrecision == InferenceEngine::Precision::FP32) {
            configuration[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] = InferenceEngine::PluginConfigParams::YES;
        } else {
            configuration[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] = InferenceEngine::PluginConfigParams::NO;
        }
        configuration[InferenceEngine::PluginConfigParams::KEY_PERF_COUNT] = InferenceEngine::PluginConfigParams::YES;
#ifndef NDEBUG
        configuration[InferenceEngine::PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT] = "egraph_test";
#endif
        //Input precision is always fp32, because bf16 input is prohibited.
        inPrc = Precision::FP32;

        //Make graph
        function = createGraph(testParams);

        // set up safe threshold
        threshold = 0.02f;

        // filling of expected precision of layer execution defined by precision of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Add_1"] = "FP32";
        expectedPrecisions["Interp"] = "BF16";
    }

protected:
    std::map<std::string, std::string> expectedPrecisions;
};

TEST_P(InterpolateLayerTestBF16, CompareWithRefs) {
    Run();
};


const std::vector<InferenceEngine::Precision> prc = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16
};

const std::vector<std::vector<size_t>> inShapes = {
        {1, 20, 30, 30}
};

const std::vector<std::vector<size_t>> targetShapes = {
        {1, 20, 40, 40}
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
        ngraph::op::v4::Interpolate::InterpolateMode::linear,
        ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
        ngraph::op::v4::Interpolate::InterpolateMode::cubic,
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::nearest
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::simple,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
        ngraph::op::v4::Interpolate::NearestMode::floor,
        ngraph::op::v4::Interpolate::NearestMode::ceil,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestMode = {
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor
};

const std::vector<std::vector<size_t>> pads = {
//        {0, 0, 1, 1},
        {0, 0, 0, 0}
};

const std::vector<bool> antialias = {
// Not enabled in Inference Engine
//        true,
        false
};

const std::vector<double> cubeCoefs = {
        -0.75f
};

const auto interpolateCasesWithoutNearest = ::testing::Combine(
        ::testing::ValuesIn(modesWithoutNearest),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs));

const auto interpolateCases = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_CASE_P(bfloat16_Interpolate_Basic, InterpolateLayerTestBF16, ::testing::Combine(
        interpolateCasesWithoutNearest,
        ::testing::ValuesIn(prc),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    InterpolateLayerTestBF16::getTestCaseName);

INSTANTIATE_TEST_CASE_P(bfloat16_Interpolate_Nearest, InterpolateLayerTestBF16, ::testing::Combine(
        interpolateCases,
        ::testing::ValuesIn(prc),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    InterpolateLayerTestBF16::getTestCaseName);

} // namespace LayerTestsDefinitions