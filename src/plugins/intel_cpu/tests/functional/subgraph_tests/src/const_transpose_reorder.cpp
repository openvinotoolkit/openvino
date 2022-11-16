// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace ov::test;
namespace SubgraphTestsDefinitions {
// Subgraph:
/*
 *      param
 *        |
 *       relu
 *        |
 *  ---------------
 *  |  const      |
 *  |   |---- transpose
 * transpose      |
 *      |    convolution
 *      |         |
 * convolution    |
 *      |         |
 *      |--------add
 *                |
 *              result
 *
 * when const node has more than 1 output, in GraphOptimizer::MergeTransposeAndReorder should not
 * remove the const node directly.
 */

class Const2OutsInTransposeReorderCheck : public SubgraphBaseTest {
protected:
    void SetUp() override {
        const std::vector<size_t> inputShape = {1, 12, 12, 64};
        const InferenceEngine::SizeVector kernel = {1, 1};
        const InferenceEngine::SizeVector stride = {1, 1};
        const InferenceEngine::SizeVector dilation = {1, 1};
        const std::vector<ptrdiff_t> padBegin = {0, 0};
        const std::vector<ptrdiff_t> padEnd = {0, 0};
        const size_t convOutChannels = 4;

        targetDevice = CommonTestUtils::DEVICE_CPU;
        targetStaticShapes.push_back({inputShape});

        auto inputParams = ngraph::builder::makeParams(ov::element::f32, {inputShape});
        const auto relu = std::make_shared<ov::opset8::Relu>(inputParams[0]);
        const auto transposeOrder = ov::opset8::Constant::create(ov::element::i32, {4}, {0, 3, 1, 2});
        const auto transpose1 = std::make_shared<ov::opset8::Transpose>(relu, transposeOrder);
        const auto conv1 = ngraph::builder::makeConvolution(transpose1, ov::element::f32, kernel, stride, padBegin,
                                                     padEnd, dilation, ngraph::op::PadType::AUTO, convOutChannels);
        const auto transpose2 = std::make_shared<ov::opset8::Transpose>(relu, transposeOrder);
        const auto conv2 = ngraph::builder::makeConvolution(transpose2, ov::element::f32, kernel, stride, padBegin,
                                                     padEnd, dilation, ngraph::op::PadType::AUTO, convOutChannels);
        const auto add = std::make_shared<ov::opset8::Add>(conv1, conv2);

        ov::ResultVector results{std::make_shared<ov::opset8::Result>(add->output(0))};
        function = std::make_shared<ov::Model>(results, inputParams, "transpose_check");
    }
};

TEST_F(Const2OutsInTransposeReorderCheck, smoke_CPU_Const2OutsInTransposeReorderCheck) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}
}  // namespace SubgraphTestsDefinitions