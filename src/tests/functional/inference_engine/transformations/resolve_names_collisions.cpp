// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/resolve_names_collisions.hpp"

#include "gtest/gtest.h"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"

TEST(ResolveNameCollisionsTest, FixGeneratedNames) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 3, 3});
    const auto gen_friendly_name = arg0->get_friendly_name();

    std::string name = "Parameter_";
    EXPECT_NE(std::string::npos, gen_friendly_name.find("Parameter_"));
    unsigned long long index = std::stoull(gen_friendly_name.substr(name.length()));
    name += std::to_string(++index);

    arg0->set_friendly_name(name);

    auto arg1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});

    auto concat = std::make_shared<ov::opset8::Concat>(ov::NodeVector{arg0, arg1}, 1);
    auto result1 = std::make_shared<ov::opset8::Result>(concat);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1}, ov::ParameterVector{arg0, arg1});

    EXPECT_EQ(name, arg0->get_friendly_name());
    EXPECT_EQ(arg1->get_friendly_name(), arg0->get_friendly_name());
    EXPECT_NE(arg1->get_friendly_name(), arg0->get_friendly_name() + "_2");

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);
    EXPECT_EQ(name, arg0->get_friendly_name());
    EXPECT_NE(arg1->get_friendly_name(), arg0->get_friendly_name());
    EXPECT_EQ(arg1->get_friendly_name(), arg0->get_friendly_name() + "_2");
}

TEST(ResolveNameCollisionsTest, DoNotFixFriendlyNamesForParameters) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 3, 3});
    const auto gen_friendly_name = arg0->get_friendly_name();

    arg0->set_friendly_name(gen_friendly_name);

    auto arg1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});
    arg1->set_friendly_name(gen_friendly_name);

    auto concat = std::make_shared<ov::opset8::Concat>(ov::NodeVector{arg0, arg1}, 1);
    auto result1 = std::make_shared<ov::opset8::Result>(concat);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1}, ov::ParameterVector{arg0, arg1});

    EXPECT_EQ(gen_friendly_name, arg0->get_friendly_name());
    EXPECT_EQ(arg1->get_friendly_name(), arg0->get_friendly_name());
    EXPECT_NE(arg1->get_friendly_name(), arg0->get_friendly_name() + "_2");

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);
    EXPECT_EQ(gen_friendly_name, arg0->get_friendly_name());
    EXPECT_EQ(arg1->get_friendly_name(), arg0->get_friendly_name());
    EXPECT_NE(arg1->get_friendly_name(), arg0->get_friendly_name() + "_2");
}

TEST(ResolveNameCollisionsTest, FixFriendlyNamesForInternalOperations) {
    auto arg0 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 3, 3});
    const auto gen_friendly_name = arg0->get_friendly_name();


    auto arg1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 3});

    auto concat1 = std::make_shared<ov::opset8::Concat>(ov::NodeVector{arg0, arg1}, 1);
    concat1->set_friendly_name("concat");
    auto concat = std::make_shared<ov::opset8::Concat>(ov::NodeVector{concat1, arg1}, 1);
    concat->set_friendly_name("concat");
    auto result1 = std::make_shared<ov::opset8::Result>(concat);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1}, ov::ParameterVector{arg0, arg1});

    EXPECT_EQ(concat->get_friendly_name(), concat1->get_friendly_name());

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::ResolveNameCollisions>();
    pass_manager.run_passes(model);
    EXPECT_NE(concat->get_friendly_name(), concat1->get_friendly_name());
}
