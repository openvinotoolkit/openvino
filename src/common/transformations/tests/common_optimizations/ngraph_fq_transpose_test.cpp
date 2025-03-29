// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/pull_transpose_through_fq.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, FQTransposeTest1) {
    {
        auto data = ov::op::v0::Constant::create(element::f32, Shape{1, 1, 3}, {1, 2, 3});
        auto input_low = ov::op::v0::Constant::create(element::f32, Shape{1}, {2});
        auto input_high = ov::op::v0::Constant::create(element::f32, Shape{1}, {3});
        auto output_low = ov::op::v0::Constant::create(element::f32, Shape{1}, {2});
        auto output_high = ov::op::v0::Constant::create(element::f32, Shape{1}, {3});
        auto transpose_order = ov::op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});

        auto fq = std::make_shared<ov::op::v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, 1);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(fq, transpose_order);

        model = std::make_shared<ov::Model>(ov::NodeVector{transpose}, ParameterVector{});

        manager.register_pass<ov::pass::PullTransposeThroughFQUp>();
        manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
            check_rt_info(f);
        });
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto data = ov::op::v0::Constant::create(element::f32, Shape{1, 3, 1}, {1, 2, 3});
        auto input_low = ov::op::v0::Constant::create(element::f32, Shape{1, 1, 1}, {2});
        auto input_high = ov::op::v0::Constant::create(element::f32, Shape{1, 1, 1}, {3});
        auto output_low = ov::op::v0::Constant::create(element::f32, Shape{1, 1, 1}, {2});
        auto output_high = ov::op::v0::Constant::create(element::f32, Shape{1, 1, 1}, {3});

        auto fq = std::make_shared<ov::op::v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, 1);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fq}, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, FQTransposeNegativeCase) {
    auto create_graph = []() -> std::shared_ptr<ov::Model> {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 3, 1});
        auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(data);
        auto input_low = ov::op::v0::Constant::create(element::f32, Shape{1}, {2});
        auto input_high = ov::op::v0::Constant::create(element::f32, Shape{1}, {3});
        auto output_low = ov::op::v0::Constant::create(element::f32, Shape{1}, {2});
        auto output_high = ov::op::v0::Constant::create(element::f32, Shape{1}, {3});
        auto transpose_order = ov::op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});

        auto fq =
            std::make_shared<ov::op::v0::FakeQuantize>(sigmoid, input_low, input_high, output_low, output_high, 1);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(fq, transpose_order);

        return std::make_shared<ov::Model>(ov::NodeVector{transpose}, ParameterVector{data});
    };
    model = create_graph();

    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::PullTransposeThroughFQUp>();
    manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
        check_rt_info(f);
    });

    model_ref = create_graph();
}
