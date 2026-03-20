// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression test for GitHub issue #33255.

#include <gtest/gtest.h>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {

TEST(ReduceMinAfterConcatConstTest, smoke_ReduceMinAfterConcatConst) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{7});
    auto constData =
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{6}, std::vector<int32_t>{0, -1, 1, 1, 0, 0});
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{param, constData}, 0);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto reduceMin = std::make_shared<ov::op::v1::ReduceMin>(concat, axes, false);
    auto result = std::make_shared<ov::op::v0::Result>(reduceMin);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled = core->compile_model(model, "CPU");
    auto req = compiled.create_infer_request();

    // param input: {5, 3, 7, 2, 4, 8, 6}, const: {0, -1, 1, 1, 0, 0}
    // concatenated: {5, 3, 7, 2, 4, 8, 6, 0, -1, 1, 1, 0, 0}, min = -1
    std::vector<int32_t> inputData{5, 3, 7, 2, 4, 8, 6};
    req.set_input_tensor(ov::Tensor(ov::element::i32, ov::Shape{7}, inputData.data()));
    req.infer();

    auto output = req.get_output_tensor(0);
    ASSERT_EQ(output.get_shape(), ov::Shape({})) << "Expected scalar output, got shape " << output.get_shape();
    ASSERT_EQ(output.data<int32_t>()[0], -1) << "Expected minimum value -1";
}

}  // namespace test
}  // namespace ov
