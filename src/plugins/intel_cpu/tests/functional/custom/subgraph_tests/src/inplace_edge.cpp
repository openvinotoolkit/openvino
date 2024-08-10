// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/eltwise.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {
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

class NonInputInPlaceTest : public testing::WithParamInterface<ElementType>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ElementType> obj) {
        std::ostringstream result;
        result << "NonInputInPlaceTest_inPrc=outPrc=" << obj.param;
        return result.str();
    }

    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
        const ov::Shape inputShape = {1, 11, 3, 3};
        targetStaticShapes = {{inputShape, inputShape}};
        ElementType prc = this->GetParam();

        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(prc, ov::Shape(inputShape)),
                                        std::make_shared<ov::op::v0::Parameter>(prc, ov::Shape(inputShape))};

        auto cumsum_tensor = ov::op::v0::Constant::create(prc, inputShape, {10.0f});
        auto axis_node = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
        const auto cumsum = std::make_shared<ov::op::v0::CumSum>(cumsum_tensor, axis_node);

        auto eltwiseMul = ov::test::utils::make_eltwise(inputParams[0], cumsum, ov::test::utils::EltwiseTypes::MULTIPLY);
        auto eltwiseAdd1 = ov::test::utils::make_eltwise(inputParams[1], cumsum, ov::test::utils::EltwiseTypes::ADD);
        auto eltwiseAdd2 = ov::test::utils::make_eltwise(eltwiseAdd1, eltwiseMul, ov::test::utils::EltwiseTypes::ADD);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseAdd2)};
        function = std::make_shared<ov::Model>(results, inputParams, "NonInputInPlaceT");
    }
};

namespace {
TEST_P(NonInputInPlaceTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_NonInputInPlaceTest_CPU,
                         NonInputInPlaceTest,
                         testing::Values(ov::element::f32, ov::element::f16),
                         NonInputInPlaceTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
