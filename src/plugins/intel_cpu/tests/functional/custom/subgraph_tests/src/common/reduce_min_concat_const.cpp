// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"
#include "openvino/op/reduce_min.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// Regression test for GH#33255: ReduceMin after Concat(Parameter, Constant)
// returned the Constant values with wrong shape instead of the scalar minimum.
//
// Subgraph:
/*
 *  Parameter[i32,(7,)]   Constant[i32,(6,)]={0,-1,1,1,0,0}
 *                    \         /
 *                   Concat(axis=0)[i32,(13,)]
 *                          |
 *               ReduceMin(axes=[0], keep_dims=False)
 *                          |
 *                     Result[i32,()]
 */

class ReduceMinAfterConcatConstTest : virtual public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{7})};
        auto constData =
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{6}, std::vector<int32_t>{0, -1, 1, 1, 0, 0});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{params[0], constData}, 0);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
        auto reduceMin = std::make_shared<ov::op::v1::ReduceMin>(concat, axes, false);

        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(reduceMin)},
                                               params,
                                               "ReduceMinAfterConcatConst");
    }
};

namespace {
TEST_F(ReduceMinAfterConcatConstTest, smoke_ReduceMinAfterConcatConst) {
    run();
}
}  // namespace

}  // namespace test
}  // namespace ov
