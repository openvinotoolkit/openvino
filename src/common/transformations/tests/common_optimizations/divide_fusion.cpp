// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/divide_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST(TransformationTests, DivideFusion) {
    std::shared_ptr<ov::Model> f, f_ref;
    {
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto pow_constant = opset1::Constant::create(element::f32, Shape{1}, {-1});
        auto pow = std::make_shared<opset1::Power>(data2, pow_constant);
        auto mul = std::make_shared<opset1::Multiply>(data1, pow);

        f = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data1, data2});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::DivideFusion>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto divide = std::make_shared<opset1::Divide>(data1, data2);

        f_ref = std::make_shared<ov::Model>(NodeVector{divide}, ParameterVector{data1, data2});
    }

    const auto res = FunctionsComparator::with_default()
                         .enable(FunctionsComparator::CONST_VALUES)
                         .enable(FunctionsComparator::ATTRIBUTES)
                         .compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, DivideFusionNegative) {
    std::shared_ptr<ov::Model> f, f_ref;
    {
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto pow_constant = opset1::Constant::create(element::f32, Shape{1}, {-1.01});
        auto pow = std::make_shared<opset1::Power>(data2, pow_constant);
        auto mul = std::make_shared<opset1::Multiply>(data1, pow);

        f = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data1, data2});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::DivideFusion>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto pow_constant = opset1::Constant::create(element::f32, Shape{1}, {-1.01});
        auto pow = std::make_shared<opset1::Power>(data2, pow_constant);
        auto mul = std::make_shared<opset1::Multiply>(data1, pow);

        f_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data1, data2});
    }

    const auto res = FunctionsComparator::with_default()
                         .enable(FunctionsComparator::CONST_VALUES)
                         .enable(FunctionsComparator::ATTRIBUTES)
                         .compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}
