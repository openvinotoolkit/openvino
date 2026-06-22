// Copyright (C) 2018-2026 Intel Corporation
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
#include "openvino/op/add.hpp"
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

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

TEST_F(TransformationTestsF, FQTransposeTest1) {
    {
        auto data = v0::Constant::create(element::f32, Shape{1, 1, 3}, {1, 2, 3});
        auto input_low = v0::Constant::create(element::f32, Shape{1}, {2});
        auto input_high = v0::Constant::create(element::f32, Shape{1}, {3});
        auto output_low = v0::Constant::create(element::f32, Shape{1}, {2});
        auto output_high = v0::Constant::create(element::f32, Shape{1}, {3});
        auto transpose_order = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});

        auto fq = std::make_shared<v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, 1);
        auto transpose = std::make_shared<v1::Transpose>(fq, transpose_order);

        model = std::make_shared<ov::Model>(ov::OutputVector{transpose}, ParameterVector{});

        manager.register_pass<ov::pass::PullTransposeThroughFQUp>();
        manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
            check_rt_info(f);
        });
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto data = v0::Constant::create(element::f32, Shape{1, 3, 1}, {1, 2, 3});
        auto input_low = v0::Constant::create(element::f32, Shape{1, 1, 1}, {2});
        auto input_high = v0::Constant::create(element::f32, Shape{1, 1, 1}, {3});
        auto output_low = v0::Constant::create(element::f32, Shape{1, 1, 1}, {2});
        auto output_high = v0::Constant::create(element::f32, Shape{1, 1, 1}, {3});

        auto fq = std::make_shared<v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, 1);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fq}, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, FQTransposePullUpCase) {
    // Positive case: transpose pulled up through FQ when FQ has single consumer.
    // Original graph: Parameter -> Sigmoid -> FQ(sigmoid, il, ih, ol, oh) -> Transpose{0,2,1} -> output
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, 3, 1});
        auto sigmoid = std::make_shared<v0::Sigmoid>(data);
        auto input_low = v0::Constant::create(element::f32, Shape{1}, {2});
        auto input_high = v0::Constant::create(element::f32, Shape{1}, {3});
        auto output_low = v0::Constant::create(element::f32, Shape{1}, {2});
        auto output_high = v0::Constant::create(element::f32, Shape{1}, {3});
        auto transpose_order = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});

        auto fq = std::make_shared<v0::FakeQuantize>(sigmoid, input_low, input_high, output_low, output_high, 1);
        auto transpose = std::make_shared<v1::Transpose>(fq, transpose_order);

        model = std::make_shared<ov::Model>(ov::OutputVector{transpose}, ParameterVector{data});
    }

    // Reference: transpose pulled up through FQ, with constant-folded range inputs
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, 3, 1});
        auto sigmoid = std::make_shared<v0::Sigmoid>(data);
        auto transpose_order = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
        auto sigmoid_transposed = std::make_shared<v1::Transpose>(sigmoid, transpose_order);

        // Constants after unsqueeze and transpose get folded to shape {1,1,1}
        auto il = v0::Constant::create(element::f32, Shape{1, 1, 1}, {2});
        auto ih = v0::Constant::create(element::f32, Shape{1, 1, 1}, {3});
        auto ol = v0::Constant::create(element::f32, Shape{1, 1, 1}, {2});
        auto oh = v0::Constant::create(element::f32, Shape{1, 1, 1}, {3});

        auto fq = std::make_shared<v0::FakeQuantize>(sigmoid_transposed, il, ih, ol, oh, 1);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fq}, ParameterVector{data});
    }

    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::PullTransposeThroughFQUp>();
    manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
        check_rt_info(f);
    });
}

TEST_F(TransformationTestsF, FQTransposeNegativeMultipleConsumers) {
    // Negative case: FQ has multiple consumers, so PullTransposeThroughFQUp should NOT apply.
    // The pass requires FQ to have exactly 1 consumer (pattern::consumers_count(1)).
    auto create_graph = []() -> std::shared_ptr<ov::Model> {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, 3, 1});
        auto input_low = v0::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = v0::Constant::create(element::f32, Shape{1}, {1});
        auto output_low = v0::Constant::create(element::f32, Shape{1}, {0});
        auto output_high = v0::Constant::create(element::f32, Shape{1}, {1});

        auto fq = std::make_shared<v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, 255);

        // FQ has 2 consumers: Transpose and Add
        auto transpose_order = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<v1::Transpose>(fq, transpose_order);

        auto add_const = v0::Constant::create(element::f32, Shape{1, 3, 1}, {0.1});
        auto add = std::make_shared<v1::Add>(fq, add_const);

        return std::make_shared<ov::Model>(ov::OutputVector{transpose, add}, ParameterVector{data});
    };

    model = create_graph();
    model_ref = model->clone();

    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::PullTransposeThroughFQUp>();
    manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
        check_rt_info(f);
    });
}

TEST_F(TransformationTestsF, FQTransposeNegativeParameterInput) {
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{1, 3, 1});
    auto input_low = v0::Constant::create(element::f32, Shape{1}, {2});
    auto input_high = v0::Constant::create(element::f32, Shape{1}, {3});
    auto output_low = v0::Constant::create(element::f32, Shape{1}, {2});
    auto output_high = v0::Constant::create(element::f32, Shape{1}, {3});
    auto transpose_order = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});

    auto fq = std::make_shared<v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, 1);

    auto transpose = std::make_shared<v1::Transpose>(fq, transpose_order);

    model = std::make_shared<ov::Model>(ov::OutputVector{transpose}, ParameterVector{data});
    model_ref = model->clone();

    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::PullTransposeThroughFQUp>();
    manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
        check_rt_info(f);
    });
}
