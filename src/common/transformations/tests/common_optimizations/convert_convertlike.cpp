// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convertlike.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST(TransformationTests, ConvertConvertLike) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto like = opset8::Constant::create(element::i32, Shape{1}, {1});
        auto cvtlike = std::make_shared<opset8::ConvertLike>(data, like);

        f = std::make_shared<ov::Model>(NodeVector{cvtlike}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertConvertLike>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto cvt = std::make_shared<opset8::Convert>(data, element::i32);

        f_ref = std::make_shared<ov::Model>(NodeVector{cvt}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertConvertLike2) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto data2 = std::make_shared<opset8::Parameter>(element::i8, Shape{1});
        auto constant = opset8::Constant::create(element::i8, Shape{}, {1});
        auto like = std::make_shared<opset8::Add>(data2, constant);
        auto cvtlike = std::make_shared<opset8::ConvertLike>(data, like);

        f = std::make_shared<ov::Model>(NodeVector{cvtlike}, ParameterVector{data, data2});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertConvertLike>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto cvt = std::make_shared<opset8::Convert>(data, element::i8);

        f_ref = std::make_shared<ov::Model>(NodeVector{cvt}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertConvertLike_Negative) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto like = std::make_shared<opset8::Parameter>(element::dynamic, Shape{1});
        auto cvtlike = std::make_shared<opset8::ConvertLike>(data, like);

        f = std::make_shared<ov::Model>(NodeVector{cvtlike}, ParameterVector{data, like});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertConvertLike>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto like = std::make_shared<opset8::Parameter>(element::dynamic, Shape{1});
        auto cvtlike = std::make_shared<opset8::ConvertLike>(data, like);

        f_ref = std::make_shared<ov::Model>(NodeVector{cvtlike}, ParameterVector{data, like});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}