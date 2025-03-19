// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
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

class ConcatReorderInPlaceTest : virtual public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        const ov::Shape inputShape = {1, 100, 1, 1};
        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape),
                                        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape)};
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{inputParams[0], inputParams[1]}, 1);
        const auto targetFormat = nhwc;
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(
            concat,
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{1}, std::vector<float>{4}));
        mul1->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(
            concat,
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{1}, std::vector<float>{5}));
        mul2->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});
        auto mul3 = std::make_shared<ov::op::v1::Multiply>(
            concat,
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{1}, std::vector<float>{6}));
        mul3->get_rt_info() = CPUTestsBase::makeCPUInfo({targetFormat}, {targetFormat}, {});

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(mul1),
                                 std::make_shared<ov::op::v0::Result>(mul2),
                                 std::make_shared<ov::op::v0::Result>(mul3)};
        function = std::make_shared<ov::Model>(results, inputParams, "ConcatReorderInPlace");
        targetDevice = ov::test::utils::DEVICE_CPU;
    }
};

namespace {
TEST_F(ConcatReorderInPlaceTest, smoke_ConcatReorderInPlace_CPU) {
    run();
}
}  // namespace
}  // namespace test
}  // namespace ov
