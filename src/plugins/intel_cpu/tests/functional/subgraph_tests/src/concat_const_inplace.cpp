// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {
// Subgraph:
/*
 *          Parameter    Constant[FP32/BF16]
 *                  \    /
 *                   \  /
 *               Transpose[FP32/BF16]
 *  Constant[FP32] /
 *        \      X  No Reorder
 *         \    /
 *        Concat (inPlace)[FP32/BF16]
 *           |
 *      Convolution [FP32/BF16]
 *           |
 *        Result[FP32/BF16]
 */

class ConcatConstantInPlaceTest : public testing::WithParamInterface<InferenceEngine::Precision>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferenceEngine::Precision> obj) {
        std::ostringstream result;
        result << "ConcatConstantInPlaceTest" << obj.param.name();
        return result.str();
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        if (Precision::BF16 == (inPrc = outPrc = this->GetParam()))
            configuration.insert({ PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES });
        else
            configuration.insert({ PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO });

        const std::vector<size_t> inputShape = {1, 3, 3, 11};
        ov::ParameterVector inputParams {std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, ov::Shape(inputShape))};

        auto transposeOrder = ngraph::opset8::Constant::create(ngraph::element::i32, {4}, {0, 3, 2, 1});
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(inputParams[0], transposeOrder);

        auto concatConstantInput = ngraph::opset8::Constant::create(ngraph::element::f32, {1, 1, 3, 3}, {10.0f});
        auto concat = ngraph::builder::makeConcat({concatConstantInput, transpose}, 1);

        // convolution
        std::vector<float> weightValuesFP32(12);
        ngraph::Shape convFilterShape = { 1, 12, 1, 1 };
//        weightValuesFP32.resize(12);
        FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
        auto weightsNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, convFilterShape, weightValuesFP32);
        std::shared_ptr<ngraph::Node> conv = std::make_shared<ngraph::opset1::Convolution>(
            concat, weightsNode, ngraph::Strides({ 1, 1 }), ngraph::CoordinateDiff({ 0, 0 }),
            ngraph::CoordinateDiff({ 0, 0 }), ngraph::Strides({ 1, 1 }), ngraph::op::PadType::EXPLICIT);
            conv->set_friendly_name("CONV");

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(conv)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "ConcatConstantInPlace");
    }
};

namespace {
    TEST_P(ConcatConstantInPlaceTest, smoke_ConcatConstantInPlaceTest_CPU) {
        Run();
        if (this->GetParam() == Precision::BF16)
            CheckNumberOfNodesWithType(executableNetwork, "Reorder", 3);
        else
            CheckNumberOfNodesWithType(executableNetwork, "Reorder", 2);
    }

INSTANTIATE_TEST_SUITE_P(smoke_ConcatConstantInPlaceTest_CPU, ConcatConstantInPlaceTest,
    testing::Values(Precision::FP32, Precision::BF16),
    ConcatConstantInPlaceTest::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
