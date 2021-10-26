// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "change_placeholder_types.hpp"
#include "transformations/rt_info/old_api_map_attribute.hpp"

#include <frontend_manager/frontend_manager.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/manager.hpp>

#include "gtest/gtest.h"

using namespace std;
using namespace ov;
using namespace opset8;
using namespace frontend::tf::pass;

TEST(ChangePlaceholderTypeTest, OldApiMapForI64Param) {
    auto p1 = make_shared<Parameter>(ngraph::element::i64, ngraph::Shape({1, 2, 3, 4}));
    auto p2 = make_shared<Parameter>(ngraph::element::i64, ngraph::Shape({1, 1, 1, 1}));
    auto add1 = make_shared<Add>(p1, p2);
    auto func = make_shared<ngraph::Function>(add1, ngraph::ParameterVector{p1, p2});

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>();
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map(p1));
    ASSERT_TRUE(has_old_api_map(p2));
    ASSERT_EQ(get_old_api_map(p1).get().get_type(), ngraph::element::i32);
    ASSERT_EQ(get_old_api_map(p1).get().get_order(), std::vector<uint64_t>({0, 1, 2, 3}));
    ASSERT_EQ(get_old_api_map(p2).get().get_type(), ngraph::element::i32);
    ASSERT_EQ(get_old_api_map(p2).get().get_order(), std::vector<uint64_t>({0, 1, 2, 3}));
}

TEST(ChangePlaceholderTypeTest, ExistingOldApiMapForU8Param) {
    auto p1 = make_shared<Parameter>(ngraph::element::u8, ngraph::Shape({3, 4, 5}));
    auto old_api_map = std::make_shared<ov::OldApiMap>(ov::OldApiMapAttr({1, 0, 2}, ov::element::u8));
    set_old_api_map(std::dynamic_pointer_cast<ov::Node>(p1), old_api_map->get());

    auto axes = make_shared<Constant>(ngraph::element::i64, ngraph::Shape{2}, vector<int64_t>{0, 1});
    auto sum = make_shared<ReduceSum>(p1, axes, true);
    auto func = make_shared<ngraph::Function>(sum, ngraph::ParameterVector{p1});

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>();
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map(p1));
    ASSERT_EQ(get_old_api_map(p1).get().get_type(), ngraph::element::f32);
    ASSERT_EQ(get_old_api_map(p1).get().get_order(), std::vector<uint64_t>({1, 0, 2}));
}

TEST(ChangePlaceholderTypeTest, OldApiMapForParamWithShapeOf) {
    auto p1 = make_shared<Parameter>(ngraph::element::f64, ngraph::Shape({2, 5}));
    auto p2 = make_shared<Parameter>(ngraph::element::f32, ngraph::Shape({1, 1, 1, 1}));
    auto shapeof = make_shared<ShapeOf>(p1);
    auto convert = make_shared<Convert>(shapeof, ov::element::f32);
    auto add = make_shared<Add>(convert, p2);
    auto func = make_shared<ngraph::Function>(add, ngraph::ParameterVector{p1, p2});

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>();
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map(p1));
    ASSERT_TRUE(!has_old_api_map(p2));
    ASSERT_EQ(get_old_api_map(p1).get().get_type(), ngraph::element::f32);
    ASSERT_EQ(get_old_api_map(p1).get().get_order(), std::vector<uint64_t>({0, 1}));
}

TEST(ChangePlaceholderTypeTest, OldApiMapForParamWithConvert) {
    auto p1 = make_shared<Parameter>(ngraph::element::f64, ngraph::Shape({2, 3, 4}));
    auto convert = make_shared<Convert>(p1, ov::element::f32);
    auto func = make_shared<ngraph::Function>(convert, ngraph::ParameterVector{p1});

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>();
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map(p1));
    ASSERT_EQ(get_old_api_map(p1).get().get_type(), ngraph::element::f32);
    ASSERT_EQ(get_old_api_map(p1).get().get_order(), std::vector<uint64_t>({0, 1, 2}));
}
