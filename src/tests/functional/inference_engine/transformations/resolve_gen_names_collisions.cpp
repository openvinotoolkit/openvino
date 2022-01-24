// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/resolve_gen_names_collisions.hpp"

#include "gtest/gtest.h"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"

TEST(ResolveGeneratedNameCollisionsTest, FixGeneratedNames) {
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
    pass_manager.register_pass<ov::pass::ResolveGeneratedNameCollisions>();
    pass_manager.run_passes(model);
    EXPECT_EQ(name, arg0->get_friendly_name());
    EXPECT_NE(arg1->get_friendly_name(), arg0->get_friendly_name());
    EXPECT_EQ(arg1->get_friendly_name(), arg0->get_friendly_name() + "_2");
}

TEST(ResolveGeneratedNameCollisionsTest, DoNotFixFriendlyNames) {
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
    pass_manager.register_pass<ov::pass::ResolveGeneratedNameCollisions>();
    pass_manager.run_passes(model);
    EXPECT_EQ(gen_friendly_name, arg0->get_friendly_name());
    EXPECT_EQ(arg1->get_friendly_name(), arg0->get_friendly_name());
    EXPECT_NE(arg1->get_friendly_name(), arg0->get_friendly_name() + "_2");
}
