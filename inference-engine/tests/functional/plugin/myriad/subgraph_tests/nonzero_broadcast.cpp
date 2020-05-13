// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>

namespace {

using TensorType  = ngraph::element::Type;
using TensorShape = ngraph::PartialShape;

using BroadcastExplicitTestParams = std::tuple<
        TensorType, TensorShape, LayerTestsUtils::TargetDevice>;

class NonZero_Broadcast : public testing::WithParamInterface<BroadcastExplicitTestParams>,
                          public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& tensorType  = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters);
        targetDevice = std::get<2>(GetParam());

        const auto tensorParam = std::make_shared<ngraph::opset3::Parameter>(
                tensorType, tensorShape);
        const auto nonZero = std::make_shared<ngraph::opset3::NonZero>(tensorParam);
        const auto shapeOfNonZero = std::make_shared<ngraph::opset3::ShapeOf>(nonZero);

        const auto broadcastConstant = std::make_shared<ngraph::opset3::Constant>(
                tensorType, ngraph::Shape{1}, 1);

        const auto axesMappingConstant = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::u64, ngraph::Shape{1}, 0);

        const auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(
                broadcastConstant, shapeOfNonZero, axesMappingConstant);

        const auto result = std::make_shared<ngraph::opset3::Result>(broadcast);

        function = std::make_shared<ngraph::Function>(
                ngraph::ResultVector{result},
                ngraph::ParameterVector{tensorParam},
                "NonZero-Broadcast");
    }
};

TEST_P(NonZero_Broadcast, CompareWithReference) {
    Run();
}
// Blocked by #-30913, #-30915
INSTANTIATE_TEST_CASE_P(DISABLED_DynamicBroadcast, NonZero_Broadcast, ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::Values(
                TensorShape{1000},
                TensorShape{4, 1000},
                TensorShape{3, 128, 256}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
