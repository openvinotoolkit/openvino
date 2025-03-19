// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_upgrade.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ConvertGather1toGather7) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = std::make_shared<opset1::Parameter>(element::i32, Shape{2, 2});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {0});

        auto gather_v1 = std::make_shared<opset1::Gather>(data, indices, axis);

        model = std::make_shared<ov::Model>(NodeVector{gather_v1}, ParameterVector{data, indices});
        manager.register_pass<ov::pass::ConvertGather1ToGather7>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = std::make_shared<opset1::Parameter>(element::i32, Shape{2, 2});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {0});

        auto gather_v7 = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather_v7}, ParameterVector{data, indices});
    }
}

TEST_F(TransformationTestsF, ConvertGather7toGather8) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = std::make_shared<opset1::Parameter>(element::i32, Shape{2, 2});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {1});
        int64_t batch_dims = 1;
        auto gather_v7 = std::make_shared<opset7::Gather>(data, indices, axis, batch_dims);

        model = std::make_shared<ov::Model>(NodeVector{gather_v7}, ParameterVector{data, indices});

        manager.register_pass<ov::pass::ConvertGather7ToGather8>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indices = std::make_shared<opset1::Parameter>(element::i32, Shape{2, 2});
        auto axis = opset1::Constant::create(element::i32, Shape{1}, {1});
        int64_t batch_dims = 1;

        auto gather_v8 = std::make_shared<opset8::Gather>(data, indices, axis, batch_dims);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather_v8}, ParameterVector{data, indices});
    }
}
