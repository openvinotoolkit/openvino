// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/matmul.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

/**
 * Verify correctness of inplace StridedSlice on constant path.
 * Each FullyConnected Node uses overlaping part of the Constant Memory as weights
 * StridedSlice is expected to not execute but just to provide a partition
 * "view" on the input data.
 *
 *                          +------------+
 *                          |  Constant  |
 *                          +-----/\-----+
 *                               /  \
 *                              /    \
 *                             /      \
 *                 +----------v-+     +v-----------+
 *                 |StridedSlice|     |StridedSlice|
 * +------------+  |  inplace   |     |  inplace   |
 * |  Parameter |  | first 3/4  |     | last 3/4   |
 * +------+-----+  +------------+     +------------+
 *        |          /                  /
 *        +------------------+         /
 *        |        /         |        /
 *        |       /          |       /
 *    +---v------v---+   +---v------v---+
 *    |FullyConnected|   |FullyConnected|                 -
 *    +-------+------+   +-------+------+
 *            |                  |
 *     +------v-----+      +-----v-----+
 *     |   Result   |      |  Result   |
 *     +------------+      +-----------+
 *
 */

namespace ov {
namespace test {

class MultipleStridedSliceInPlace : public SubgraphBaseTest {
protected:
    void SetUp() override {
        const auto precision = ov::element::f32;
        const size_t K = 32;
        const ov::Shape inputShape = {8, K};
        targetStaticShapes = {{inputShape}};
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(precision, inputShape)};
        auto& input = params.front();

        const ov::Shape weightsShape = {4, K};
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::test::utils::create_and_fill_tensor(precision, weightsShape));
        ov::disable_constant_folding(weights);
        // default unused strides
        auto strides = std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{2}, std::vector<int>{1, 1});
        std::vector<int64_t> beginMask{0, 0}; // default unused
        std::vector<int64_t> endMask{0, 0}; // default unused
        // first 3/4 slice of the input Memory
        auto beginFirstPart   = std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{2}, std::vector<int32_t>{0, 0});
        auto endFirstPart     = std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{2}, std::vector<int32_t>{3, K});

        auto ss_first_part = std::make_shared<ov::op::v1::StridedSlice>(weights,
                                                                        beginFirstPart,
                                                                        endFirstPart,
                                                                        strides,
                                                                        beginMask,
                                                                        endMask);
        // last 3/4 slice of the input Memory
        auto beginSecondPart   = std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{2}, std::vector<int32_t>{1, 0});
        auto endSecondPart     = std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{2}, std::vector<int32_t>{4, K});

        auto ss_second_part = std::make_shared<ov::op::v1::StridedSlice>(weights,
                                                                         beginSecondPart,
                                                                         endSecondPart,
                                                                         strides,
                                                                         beginMask,
                                                                         endMask);
        ov::disable_constant_folding(ss_second_part);

        auto fc1 = std::make_shared<ov::op::v0::MatMul>(input, ss_first_part, false, true);
        auto fc2 = std::make_shared<ov::op::v0::MatMul>(input, ss_second_part, false, true);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(fc1), std::make_shared<ov::op::v0::Result>(fc2)};
        function = std::make_shared<ov::Model>(results, params, "MultipleInplaceStridedSlices");
    }
};

TEST_F(MultipleStridedSliceInPlace, smoke_CPU_InPlaceReshapeFromConstantCheck) {
    // ensure StridedSlices were not optimized out
    CPUTestUtils::CheckNumberOfNodesWithType(compiledModel, "StridedSlice", 2);
    // ensure no inplace conflict reorders is inserted
    CPUTestUtils::CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
    run();
}

}  // namespace test
}  // namespace ov
