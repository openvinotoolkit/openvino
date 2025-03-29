// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/change_placeholder_types.hpp"

#include "gtest/gtest.h"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/old_api_map_element_type_attribute.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset8;
using namespace ov::pass;

using ParamNames = std::vector<std::string>;

TEST(ChangePlaceholderTypeTest, OldApiMapForI64ParamEmptyParamNames) {
    ParamNames params_with_custom_types;
    auto p1 = make_shared<Parameter>(element::i64, Shape({1, 2, 3, 4}));
    auto p2 = make_shared<Parameter>(element::i64, Shape({1, 1, 1, 1}));
    auto add1 = make_shared<Add>(p1, p2);
    auto func = make_shared<Model>(add1, ParameterVector{p1, p2});

    pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>(params_with_custom_types);
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map_element_type(p1));
    ASSERT_TRUE(has_old_api_map_element_type(p2));
    ASSERT_EQ(get_old_api_map_element_type(p1).value, element::i32);
    ASSERT_EQ(get_old_api_map_element_type(p2).value, element::i32);
}

TEST(ChangePlaceholderTypeTest, OldApiMapForParamWithShapeOfEmptyParamNames) {
    ParamNames params_with_custom_types;
    auto p1 = make_shared<Parameter>(element::f64, Shape({2, 5}));
    auto p2 = make_shared<Parameter>(element::f32, Shape({1, 1, 1, 1}));
    auto shapeof = make_shared<ShapeOf>(p1);
    auto convert = make_shared<Convert>(shapeof, element::f32);
    auto add = make_shared<Add>(convert, p2);
    auto func = make_shared<Model>(add, ParameterVector{p1, p2});

    pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>(params_with_custom_types);
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map_element_type(p1));
    ASSERT_FALSE(has_old_api_map_element_type(p2));
    ASSERT_EQ(get_old_api_map_element_type(p1).value, element::f32);
}

TEST(ChangePlaceholderTypeTest, OldApiMapForParamWithConvertEmptyParamNames) {
    ParamNames params_with_custom_types;
    auto p1 = make_shared<Parameter>(element::f64, Shape({2, 3, 4}));
    auto convert = make_shared<Convert>(p1, element::f32);
    auto func = make_shared<Model>(convert, ParameterVector{p1});

    pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>(params_with_custom_types);
    pass_manager.run_passes(func);

    ASSERT_TRUE(has_old_api_map_element_type(p1));
    ASSERT_EQ(get_old_api_map_element_type(p1).value, element::f32);
}

TEST(ChangePlaceholderTypeTest, OldApiMapForI64ParamWithUserParamNames) {
    // this test covers a case with user defined type for parameter
    // for which legacy type in OldApi map is not set
    ParamNames params_with_custom_types;
    auto p1 = make_shared<Parameter>(element::i64, Shape({1, 2, 3, 4}));
    p1->set_friendly_name("p1_param");
    auto p2 = make_shared<Parameter>(element::i64, Shape({1, 1, 1, 1}));
    auto add1 = make_shared<Add>(p1, p2);
    auto func = make_shared<Model>(add1, ParameterVector{p1, p2});

    // create user data types dictionary to overwrite legacy type
    params_with_custom_types.push_back(p1->get_friendly_name());

    pass::Manager pass_manager;
    pass_manager.register_pass<ChangePlaceholderTypes>(params_with_custom_types);
    pass_manager.run_passes(func);

    ASSERT_FALSE(has_old_api_map_element_type(p1));
    ASSERT_TRUE(has_old_api_map_element_type(p2));
    ASSERT_EQ(get_old_api_map_element_type(p2).value, element::i32);
}
