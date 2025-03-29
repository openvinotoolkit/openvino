// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_reset_no_sinking_attribute.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace std;
using namespace ov;

using namespace op::v0;
using namespace op::v1;

TEST(TransformationTests, ResetNoSinkingAttribute) {
    auto a = std::make_shared<Parameter>(element::f32, Shape{12, 3, 4, 8});
    auto b = std::make_shared<Parameter>(element::f32, Shape{12, 3, 4, 8});

    auto transpose_a = make_shared<Transpose>(a, Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
    auto transpose_b = make_shared<Transpose>(b, Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));

    auto add = std::make_shared<Add>(transpose_a, transpose_b);
    auto trans_after = make_shared<Transpose>(add, Constant::create(element::i64, Shape{4}, {1, 0, 2, 3}));
    auto model = std::make_shared<Model>(NodeVector{trans_after}, ParameterVector{a, b});

    mark_as_no_sinking_node(transpose_a);
    mark_as_no_sinking_node(transpose_b);
    mark_as_no_sinking_node(trans_after);

    const auto& ops = model->get_ordered_ops();
    const auto cnt_before = count_if(ops.begin(), ops.end(), [](const std::shared_ptr<Node>& node) {
        const auto& rt_info = node->get_rt_info();
        return rt_info.find(NoTransposeSinkingAttr::get_type_info_static()) != rt_info.end();
    });

    EXPECT_EQ(cnt_before, 3);
    ov::pass::Manager manager;
    manager.register_pass<pass::transpose_sinking::TSResetNoSinkingAttribute>();
    manager.run_passes(model);

    const auto cnt_after = count_if(ops.begin(), ops.end(), [](const std::shared_ptr<Node>& node) {
        const auto& rt_info = node->get_rt_info();
        return rt_info.find(NoTransposeSinkingAttr::get_type_info_static()) != rt_info.end();
    });

    EXPECT_EQ(cnt_after, 0);
}