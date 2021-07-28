// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph/opsets/opset6.hpp>

#include <vpu/private_plugin_config.hpp>

namespace {

using TensorType  = ngraph::element::Type_t;
using TensorShape = ngraph::Shape;

class UnsqueezeGather : public testing::WithParamInterface<std::tuple<TensorType, TensorShape, size_t, std::string>>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        const auto &parameters = GetParam();
        const auto &inType = std::get<0>(parameters);
        const auto &inShape = std::get<1>(parameters);
        const auto &axis = std::get<2>(parameters);
        targetDevice = std::get<3>(GetParam());

        const auto parameter = std::make_shared<ngraph::opset6::Parameter>(inType, inShape);

        const auto unsqueeze = std::make_shared<ngraph::opset6::Unsqueeze>(
                parameter,
                ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {axis}));

        const auto gather = std::make_shared<ngraph::opset6::Gather>(
                unsqueeze,
                ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}),
                ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {axis}));

        const auto relu = std::make_shared<ngraph::opset6::Relu>(gather);

        function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{relu},
                ngraph::ParameterVector{parameter},
                "unsqueeze-gather");
    }
};

TEST_P(UnsqueezeGather, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, UnsqueezeGather, testing::Combine(
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32),
        testing::Values(
                TensorShape{3, 128, 256}),
        testing::Values(0, 1, 2, 3),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)
));

} // namespace
