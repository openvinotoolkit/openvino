// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ngraph;

namespace SubgraphTestsDefinitions {
// Subgraph:
/*
 *              paramter1   parameter2
 *                    \       /
 *                     \     /
 *                     Concat (inPlace)
 *                    /   |   \
 *                   /    |    \
 *             Reorder Reorder Reorder (the reorder nodes are optimized and use inplace memory mode)
 *                /       |       \
 *               /        |        \
 *         Multiply    Multiply    Multiply
 *            /           |           \
 *           /            |            \
 *        Result        Result         Result
 */

class ConcatReorderInPlaceTest : virtual public LayerTestsUtils::LayerTestsCommon {
public:
    void SetUp() override {
        const std::vector<size_t> inputShape = {1, 100, 1, 1};
        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape)),
                                        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape))};
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{inputParams[0], inputParams[1]}, 1);
        const auto targetFormat = nhwc;
        auto mul1 = std::make_shared<ngraph::opset8::Multiply>(
            concat,
            ngraph::builder::makeConstant(ngraph::element::f32, Shape{1}, std::vector<float>{4}));
        mul1->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto mul2 = std::make_shared<ngraph::opset8::Multiply>(
            concat,
            ngraph::builder::makeConstant(ngraph::element::f32, Shape{1}, std::vector<float>{5}));
        mul2->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto mul3 = std::make_shared<ngraph::opset8::Multiply>(
            concat,
            ngraph::builder::makeConstant(ngraph::element::f32, Shape{1}, std::vector<float>{6}));
        mul3->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(mul1),
                                     std::make_shared<ngraph::opset8::Result>(mul2),
                                     std::make_shared<ngraph::opset8::Result>(mul3)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "ConcatReorderInPlace");
        targetDevice = ov::test::utils::DEVICE_CPU;
    }
};

namespace {
TEST_F(ConcatReorderInPlaceTest, smoke_ConcatReorderInPlace_CPU) {
    Run();
}
}  // namespace
}  // namespace SubgraphTestsDefinitions
