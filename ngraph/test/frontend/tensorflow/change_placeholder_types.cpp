// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "change_placeholder_types.hpp"

#include "gtest/gtest.h"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/old_api_map_attribute.hpp"

using namespace std;
using namespace ov;
using namespace opset8;
using namespace frontend::tf::pass;

TEST(ChangePlaceholderTypeTest, OldApiMapForI64Param) {
    auto p1 = make_shared<Parameter>(element::i64, Shape({1, 2, 3, 4}));
    auto p2 = make_shared<Parameter>(element::i64, Shape({1, 1, 1, 1}));
    auto add1 = make_shared<Add>(p1, p2);
    auto func = make_shared<Function>(add1, ParameterVector{p1, p2});

    pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>();
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map(p1));
    ASSERT_TRUE(has_old_api_map(p2));
    ASSERT_EQ(get_old_api_map(p1).get().get_type(), element::i32);
    ASSERT_EQ(get_old_api_map(p1).get().get_order(), std::vector<uint64_t>({0, 1, 2, 3}));
    ASSERT_EQ(get_old_api_map(p2).get().get_type(), element::i32);
    ASSERT_EQ(get_old_api_map(p2).get().get_order(), std::vector<uint64_t>({0, 1, 2, 3}));
}

TEST(ChangePlaceholderTypeTest, ExistingOldApiMapForU8Param) {
    auto p1 = make_shared<Parameter>(element::u8, Shape({3, 4, 5}));
    set_old_api_map(p1, OldApiMapAttr({1, 0, 2}, element::u8));

    auto axes = make_shared<Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});
    auto sum = make_shared<ReduceSum>(p1, axes, true);
    auto func = make_shared<Function>(sum, ParameterVector{p1});

    pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>();
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map(p1));
    ASSERT_EQ(get_old_api_map(p1).get().get_type(), element::f32);
    ASSERT_EQ(get_old_api_map(p1).get().get_order(), std::vector<uint64_t>({1, 0, 2}));
}

TEST(ChangePlaceholderTypeTest, OldApiMapForParamWithShapeOf) {
    auto p1 = make_shared<Parameter>(element::f64, Shape({2, 5}));
    auto p2 = make_shared<Parameter>(element::f32, Shape({1, 1, 1, 1}));
    auto shapeof = make_shared<ShapeOf>(p1);
    auto convert = make_shared<Convert>(shapeof, element::f32);
    auto add = make_shared<Add>(convert, p2);
    auto func = make_shared<Function>(add, ParameterVector{p1, p2});

    pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>();
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map(p1));
    ASSERT_TRUE(!has_old_api_map(p2));
    ASSERT_EQ(get_old_api_map(p1).get().get_type(), element::f32);
    ASSERT_EQ(get_old_api_map(p1).get().get_order(), std::vector<uint64_t>({0, 1}));
}

TEST(ChangePlaceholderTypeTest, OldApiMapForParamWithConvert) {
    auto p1 = make_shared<Parameter>(element::f64, Shape({2, 3, 4}));
    auto convert = make_shared<Convert>(p1, element::f32);
    auto func = make_shared<Function>(convert, ParameterVector{p1});

    pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>();
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map(p1));
    ASSERT_EQ(get_old_api_map(p1).get().get_type(), element::f32);
    ASSERT_EQ(get_old_api_map(p1).get().get_order(), std::vector<uint64_t>({0, 1, 2}));
}
