// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

using namespace ov;
using namespace testing;

TEST(TransformationTests, ConvBiasFusion) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto weights = opset1::Constant::create(element::f32, Shape{3, 3, 1, 1}, {1});
        auto bias = opset1::Constant::create(element::f32, Shape{3, 1, 1}, {1});
        auto conv = std::make_shared<opset1::Convolution>(input1,
                                                          weights,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

        auto add = std::make_shared<opset1::Add>(conv, bias);
        add->set_friendly_name("add");

        f = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{input1});
    }

    std::unordered_map<std::string, std::string> pp;

    for (auto& op : f->get_ops()) {
        if (auto conv = ov::as_type_ptr<opset1::Convolution>(op)) {
            auto& rtInfo = conv->get_rt_info();
            rtInfo[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority("test");
            pp[op->get_friendly_name()] = "test";
        }
    }

    auto funcs = f->clone();

    for (auto& op : funcs->get_ops()) {
        if (auto conv = ov::as_type_ptr<opset1::Convolution>(op)) {
            ASSERT_TRUE(pp.find(op->get_friendly_name()) != pp.end());
            ASSERT_EQ(pp[op->get_friendly_name()], "test");
        }
    }
}
