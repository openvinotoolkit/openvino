// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_downgrade.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ConvertGather7toGather1) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = std::make_shared<opset1::Parameter>(element::i32, Shape{2, 2});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {0});

        auto gather_v7 = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model = std::make_shared<ov::Model>(NodeVector{gather_v7}, ParameterVector{data, indices});
        manager.register_pass<ov::pass::ConvertGather7ToGather1>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = std::make_shared<opset1::Parameter>(element::i32, Shape{2, 2});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {0});

        auto gather_v1 = std::make_shared<opset1::Gather>(data, indices, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather_v1}, ParameterVector{data, indices});
    }
}

TEST_F(TransformationTestsF, ConvertGather7toGather1_nonzero_batch_dims) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = std::make_shared<opset1::Parameter>(element::i32, Shape{2, 2});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {1});

        auto gather_v7 = std::make_shared<opset7::Gather>(data, indices, axis, -1);

        model = std::make_shared<ov::Model>(NodeVector{gather_v7}, ParameterVector{data, indices});
        manager.register_pass<ov::pass::ConvertGather7ToGather1>();
    }
}

TEST_F(TransformationTestsF, ConvertGather8toGather7_param_indices) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = std::make_shared<opset1::Parameter>(element::i32, Shape{2, 2});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {1});
        int64_t batch_dims = 1;

        auto gather_v8 = std::make_shared<opset8::Gather>(data, indices, axis, batch_dims);

        model = std::make_shared<ov::Model>(NodeVector{gather_v8}, ParameterVector{data, indices});

        manager.register_pass<ov::pass::ConvertGather8ToGather7>();
    }
}

TEST_F(TransformationTestsF, ConvertGather8toGather7_const_indices) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = opset8::Constant::create(element::i32, Shape{2, 2}, {0, 1, 2, 0});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {1});
        int64_t batch_dims = 1;

        auto gather_v8 = std::make_shared<opset8::Gather>(data, indices, axis, batch_dims);

        model = std::make_shared<ov::Model>(NodeVector{gather_v8}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertGather8ToGather7>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = opset8::Constant::create(element::i32, Shape{2, 2}, {0, 1, 2, 0});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {1});
        int64_t batch_dims = 1;

        auto gather_v7 = std::make_shared<opset7::Gather>(data, indices, axis, batch_dims);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather_v7}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertGather8toGather7_negative_indices) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = opset8::Constant::create(element::i32, Shape{2, 2}, {2, 1, 0, -1});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {1});
        int64_t batch_dims = 1;

        auto gather_v8 = std::make_shared<opset8::Gather>(data, indices, axis, batch_dims);

        model = std::make_shared<ov::Model>(NodeVector{gather_v8}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertGather8ToGather7>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = opset8::Constant::create(element::i32, Shape{2, 2}, {2, 1, 0, 2});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {1});
        int64_t batch_dims = 1;

        auto gather_v7 = std::make_shared<opset7::Gather>(data, indices, axis, batch_dims);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather_v7}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertGather8toGather7_out_of_bound_indices) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = opset8::Constant::create(element::i32, Shape{2, 2}, {0, 1, 2, 3});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {1});
        int64_t batch_dims = 1;

        auto gather_v8 = std::make_shared<opset8::Gather>(data, indices, axis, batch_dims);

        model = std::make_shared<ov::Model>(NodeVector{gather_v8}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertGather8ToGather7>();
    }
}

TEST_F(TransformationTestsF, ConvertGather8toGather7_negative_axis) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = opset8::Constant::create(element::i32, Shape{2, 2}, {0, 1, 2, 0});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {-1});
        int64_t batch_dims = 1;

        auto gather_v8 = std::make_shared<opset8::Gather>(data, indices, axis, batch_dims);

        model = std::make_shared<ov::Model>(NodeVector{gather_v8}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertGather8ToGather7>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = opset8::Constant::create(element::i32, Shape{2, 2}, {0, 1, 2, 0});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {-1});
        int64_t batch_dims = 1;

        auto gather_v7 = std::make_shared<opset7::Gather>(data, indices, axis, batch_dims);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather_v7}, ParameterVector{data});
    }
}
