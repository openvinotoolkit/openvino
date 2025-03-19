// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "transformations/common_optimizations/fused_names_cleanup.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

TEST(TransformationTests, FusedNamesCleanup) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto data = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});

        auto add1_const = opset9::Constant::create(element::f32, Shape{1}, {1.0});
        auto add2_const = opset9::Constant::create(element::f32, Shape{1}, {2.0});
        auto add1 = std::make_shared<opset9::Add>(add1_const, add2_const);
        auto add2 = std::make_shared<opset9::Add>(data, add1);
        model = std::make_shared<Model>(NodeVector{add2}, ParameterVector{data});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.run_passes(model);
        OV_ASSERT_NO_THROW(check_rt_info(model));

        manager.register_pass<ov::pass::FusedNamesCleanup>();
        manager.run_passes(model);
        ASSERT_THROW(check_rt_info(model), ov::Exception);
    }
    {
        auto data = std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});

        auto add_const = opset9::Constant::create(element::f32, Shape{1}, {3.0});
        auto add = std::make_shared<opset9::Add>(data, add_const);
        model_ref = std::make_shared<Model>(NodeVector{add}, ParameterVector{data});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    const FunctionsComparator::Result result = func_comparator(model, model_ref);
    ASSERT_TRUE(result.valid);
}
