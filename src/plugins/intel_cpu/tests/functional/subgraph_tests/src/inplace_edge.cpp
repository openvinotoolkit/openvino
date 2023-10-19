// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {
// If a node (CumSum) with constant parents has several non-constant nodes (Eltwises) than the edge is broken.
// The fix is to check node type - is should be Input.
// Subgraph:
/*
 *            Constant  Constant
 *                 \    /
 *                  \  /
 *                 CumSum
 *  Parameter      /   \     Parameter
 *        \       /     \       /
 *         \     /       \     /
 *         Eltwise       Eltwise
 *               \       /  
 *                Eltwise
 *                   |
 *                 Result
 */

class NonInputInPlaceTest : public testing::WithParamInterface<InferenceEngine::Precision>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferenceEngine::Precision> obj) {
        std::ostringstream result;
        result << "NonInputInPlaceTest" << obj.param.name();
        return result.str();
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16.to_string()});
        inPrc = outPrc = this->GetParam();

        const std::vector<size_t> inputShape = {1, 11, 3, 3};
        ov::ParameterVector inputParams {std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, ov::Shape(inputShape)),
                                         std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, ov::Shape(inputShape))};

        auto cumsum_tensor = ngraph::opset8::Constant::create(ngraph::element::f32, inputShape, {10.0f});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i32, {}, {0});
        const auto cumsum = std::make_shared<ov::op::v0::CumSum>(cumsum_tensor, axis_node);

        auto eltwiseMul = ngraph::builder::makeEltwise(inputParams[0], cumsum, ngraph::helpers::EltwiseTypes::MULTIPLY);
        auto eltwiseAdd1 = ngraph::builder::makeEltwise(inputParams[1], cumsum, ngraph::helpers::EltwiseTypes::ADD);
        auto eltwiseAdd2 = ngraph::builder::makeEltwise(eltwiseAdd1, eltwiseMul, ngraph::helpers::EltwiseTypes::ADD);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(eltwiseAdd2)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "NonInputInPlaceT");
    }
};

namespace {
    TEST_P(NonInputInPlaceTest, CompareWithRefs) {
        Run();
    }

INSTANTIATE_TEST_SUITE_P(smoke_NonInputInPlaceTest_CPU, NonInputInPlaceTest,
    testing::Values(Precision::FP32, Precision::FP16),
    NonInputInPlaceTest::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
