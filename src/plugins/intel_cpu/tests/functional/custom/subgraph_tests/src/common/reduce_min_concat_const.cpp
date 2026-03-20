// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression test for GitHub issue #33255.

#include "openvino/op/concat.hpp"
#include "openvino/op/reduce_min.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class ReduceMinAfterConcatConstTest : public SubgraphBaseStaticTest {};

TEST_F(ReduceMinAfterConcatConstTest, smoke_ReduceMinAfterConcatConst) {
    targetDevice = ov::test::utils::DEVICE_CPU;

    const ov::Shape paramShape = {7};
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, paramShape)};

    auto constData =
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{6}, std::vector<int32_t>{0, -1, 1, 1, 0, 0});

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{params[0], constData}, 0);

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto reduceMin = std::make_shared<ov::op::v1::ReduceMin>(concat, axes, false);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(reduceMin)},
                                           params,
                                           "ReduceMinAfterConcatConst");

    run();
}

}  // namespace test
}  // namespace ov
