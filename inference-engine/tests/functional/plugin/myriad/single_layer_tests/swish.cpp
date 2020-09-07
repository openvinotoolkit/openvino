// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph_functions/builders.hpp>

#include <tuple>
#include <string>

using Parameters = std::tuple<
        ngraph::element::Type,
        ngraph::Shape,
        LayerTestsUtils::TargetDevice>;

namespace LayerTestsDefinitions {
class SwishLayerTest : public testing::WithParamInterface<Parameters>,
                       virtual public LayerTestsUtils::LayerTestsCommon {
    std::string getTestCaseName(testing::TestParamInfo <Parameters> obj) {
        ngraph::Shape inputShape;
        ngraph::element::Type inputType;
        std::string targetDevice;
        std::tie(inputType, inputShape, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "inPrc=" << inputType << "_";
        result << "targetDevice=" << targetDevice;

        return result.str();
    }

    void SetUp() {
        ngraph::Shape inputShape;
        ngraph::element::Type inputType;

        std::tie(inputType, inputShape, targetDevice) = GetParam();

        const auto params = ngraph::builder::makeParams(inputType, {inputShape});

        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(
                        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto swish = std::make_shared<ngraph::opset4::Swish>(paramOuts.at(0));

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(swish)};

        function = std::make_shared<ngraph::Function>(results, params, "swish");
    }
};
TEST_P(SwishLayerTest, CompareWithRefs) {
    Run();
}
INSTANTIATE_TEST_CASE_P(accuracy, SwishLayerTest,
                        ::testing::Combine(
                                ::testing::Values(ngraph::element::f16),
                                ::testing::Values(ngraph::Shape{50, 50}),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace LayerTestsDefinitions
