// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {
// Subgraph:
/*
 *          Parameter    Constant
 *                  \    /
 *                   \  /
 *               Transpose
 *  Constant      /
 *        \      /
 *         \    /
 *        Concat (inPlace)
 *           |
 *           |
 *        Result
 */

class ConcatConstantInPlaceTest : virtual public LayerTestsUtils::LayerTestsCommon {
public:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        inPrc = outPrc = Precision::FP32;
        const std::vector<size_t> inputShape = {1, 384, 196};
        auto inputParams = ngraph::builder::makeParams(ngraph::element::f32, {inputShape, inputShape});

        auto transposeOrder = ngraph::opset8::Constant::create(ngraph::element::i32, {3}, {0, 2, 1});
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(inputParams[0], transposeOrder);

        auto concatConstantInput = ngraph::opset8::Constant::create(ngraph::element::f32, {1, 1, 384}, {10.0f});
        auto concat = ngraph::builder::makeConcat({concatConstantInput, transpose}, 1);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(concat)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "ConcatConstantInPlace");
    }
};

namespace {
    TEST_F(ConcatConstantInPlaceTest, smoke_ConcatConstantInPlaceTest_CPU) {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        Run();
    }
} // namespace
} // namespace SubgraphTestsDefinitions