// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_topk3.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

// check that the first output from the TopK-3 with I32 output indices is equal to the TopK-1 first output
TEST_F(TransformationTestsF, ConvertTopK3I32Output0) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{15, 20, 3});
        auto k = opset3::Constant::create(element::i64, Shape{}, {10});
        auto topk = std::make_shared<opset3::TopK>(input, k, 1, "min", "value", element::i32);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        model = std::make_shared<ov::Model>(OutputVector{topk->output(0)}, ParameterVector{input});
        manager.register_pass<ov::pass::ConvertTopK3>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{15, 20, 3});
        auto k = opset3::Constant::create(element::i64, Shape{}, {10});
        auto topk = std::make_shared<opset2::TopK>(input, k, 1, "min", "value", element::i32);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        model_ref = std::make_shared<ov::Model>(OutputVector{topk->output(0)}, ParameterVector{input});
    }
}

// check that the second output from the TopK-3 with I32 output indices is equal to the TopK-1 second output
TEST_F(TransformationTestsF, ConvertTopK3I32Output1) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{15, 20, 3});
        auto k = opset3::Constant::create(element::i64, Shape{}, {10});
        auto topk = std::make_shared<opset3::TopK>(input, k, 1, "min", "value", element::i32);

        // due to the 'compare_functions' limitation we will check only one output
        model = std::make_shared<ov::Model>(OutputVector{topk->output(1)}, ParameterVector{input});
        manager.register_pass<ov::pass::ConvertTopK3>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{15, 20, 3});
        auto k = opset3::Constant::create(element::i64, Shape{}, {10});
        auto topk = std::make_shared<opset2::TopK>(input, k, 1, "min", "value", element::i32);

        // due to the 'compare_functions' limitation we will check only one output
        model_ref = std::make_shared<ov::Model>(OutputVector{topk->output(1)}, ParameterVector{input});
    }
}

// check that the first output from the TopK-3 with I64 output indices is equal to the TopK-1 first output
TEST_F(TransformationTestsF, ConvertTopK3I64Output0) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{15, 20, 3});
        auto k = opset3::Constant::create(element::i64, Shape{}, {10});
        auto topk = std::make_shared<opset3::TopK>(input, k, 1, "min", "value", element::i64);

        // due to the 'compare_functions' limitation we will check only one output
        model = std::make_shared<ov::Model>(OutputVector{topk->output(0)}, ParameterVector{input});
        manager.register_pass<ov::pass::ConvertTopK3>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{15, 20, 3});
        auto k = opset3::Constant::create(element::i64, Shape{}, {10});
        auto topk = std::make_shared<opset2::TopK>(input, k, 1, "min", "value", element::i32);

        // due to the 'compare_functions' limitation we will check only one output
        model_ref = std::make_shared<ov::Model>(OutputVector{topk->output(0)}, ParameterVector{input});
    }
}

// check that the second output from the TopK-3 with I64 output indices is equal to the TopK-1 second output converted
// to I64
TEST_F(TransformationTestsF, ConvertTopK3I64Output1) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{15, 20, 3});
        auto k = opset3::Constant::create(element::i64, Shape{}, {10});
        auto topk = std::make_shared<opset3::TopK>(input, k, 1, "min", "value", element::i64);

        // due to the 'compare_functions' limitation we will check only one output
        model = std::make_shared<ov::Model>(OutputVector{topk->output(1)}, ParameterVector{input});
        manager.register_pass<ov::pass::ConvertTopK3>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{15, 20, 3});
        auto k = opset3::Constant::create(element::i64, Shape{}, {10});
        auto topk = std::make_shared<opset2::TopK>(input, k, 1, "min", "value", element::i32);
        auto convert = std::make_shared<opset2::Convert>(topk->output(1), element::i64);

        // due to the 'compare_functions' limitation we will check only one output
        model_ref = std::make_shared<ov::Model>(NodeVector{convert}, ParameterVector{input});
    }
}
