// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace testing;
using namespace std;
using namespace ov;

TEST(TransformationTests, ConstFoldingMVN) {
    shared_ptr<ov::Model> fun(nullptr);
    {
        const auto in = make_shared<opset10::Parameter>(element::f32, Shape{6});
        const auto mvn_in = make_shared<opset10::Constant>(element::f32, Shape{6}, vector<float>(6, 0.0f));
        const auto axes = make_shared<opset10::Constant>(element::i32, Shape{1}, vector<int>{0});
        auto mvn = make_shared<opset10::MVN>(mvn_in, axes, false, 1e-9f, op::MVNEpsMode::OUTSIDE_SQRT);
        auto add = make_shared<opset10::Add>(in, mvn);

        fun = make_shared<ov::Model>(NodeVector{add}, ParameterVector{in});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.run_passes(fun);
    }
    shared_ptr<ov::Model> f_ref(nullptr);
    {
        const auto in = make_shared<opset10::Parameter>(element::f32, Shape{6});
        const auto mvn_const = make_shared<opset10::Constant>(element::f32, Shape{6}, vector<float>(6, 0.0f));
        auto add = make_shared<opset10::Add>(in, mvn_const);

        f_ref = make_shared<ov::Model>(NodeVector{add}, ParameterVector{in});
    }

    auto res = compare_functions(fun, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
