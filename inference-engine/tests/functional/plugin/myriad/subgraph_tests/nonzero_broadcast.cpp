// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

#include "vpu/private_plugin_config.hpp"

#include "../common/myriad_common_test_utils.hpp"
#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>

namespace {

using TensorType  = ngraph::element::Type;
using TensorShape = ngraph::Shape;

using BroadcastExplicitTestParams = std::tuple<
        TensorType, TensorShape, LayerTestsUtils::TargetDevice>;

class NonZero_Broadcast : public testing::WithParamInterface<BroadcastExplicitTestParams>,
                          public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
        configuration[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
        // DISABLE_REORDER is needed for Myriad2 cases
        if (CommonTestUtils::vpu::CheckMyriad2()) {
            configuration[VPU_CONFIG_KEY(DISABLE_REORDER)] = CONFIG_VALUE(YES);
        }

        const auto& parameters = GetParam();
        const auto& tensorType  = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters);
        targetDevice = std::get<2>(GetParam());

        const auto tensorParam = std::make_shared<ngraph::opset3::Parameter>(
                tensorType, tensorShape);
        const auto nonZero = std::make_shared<ngraph::opset3::NonZero>(tensorParam);
        const auto shapeOfNonZero = std::make_shared<ngraph::opset3::ShapeOf>(nonZero);

        const auto broadcastConstant = std::make_shared<ngraph::opset3::Constant>(
                tensorType, ngraph::Shape{tensorShape.size()}, 1);
        const auto axesMappingConstant = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::u64, ngraph::Shape{1}, 0);
        const auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(
                broadcastConstant, shapeOfNonZero, axesMappingConstant);

        const auto resultBroadcast = std::make_shared<ngraph::opset3::Result>(broadcast);
        const auto resultNonZero = std::make_shared<ngraph::opset3::Result>(nonZero->output(0));

        function = std::make_shared<ngraph::Function>(
                ngraph::ResultVector{resultBroadcast, resultNonZero},
                ngraph::ParameterVector{tensorParam},
                "NonZero-Broadcast");
    }
};

TEST_P(NonZero_Broadcast, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DynamicBroadcast, NonZero_Broadcast, ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::Values(
                TensorShape{1000},
                TensorShape{4, 1000},
                TensorShape{3, 128, 256}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
