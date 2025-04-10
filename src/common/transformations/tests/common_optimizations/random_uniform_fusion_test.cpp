// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/random_uniform_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, RandomUniformMulFusing) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);

        auto mul_const = opset8::Constant::create(element::f32, Shape{1, 1, 1}, {30.0});
        auto mul = std::make_shared<opset8::Multiply>(ru, mul_const);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {30.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);

        model_ref = std::make_shared<ov::Model>(NodeVector{ru}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, RandomUniformAddFusing) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {30.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);

        auto add_const = opset8::Constant::create(element::f32, Shape{1, 1, 1, 1}, {-10.0});
        auto add = std::make_shared<opset8::Add>(ru, add_const);

        model = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{input});

        manager.register_pass<ov::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto ru_max_const = opset8::Constant::create(element::f32, Shape{}, {20.0});
        auto ru_min_const = opset8::Constant::create(element::f32, Shape{}, {-10.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, ru_min_const, ru_max_const, element::f32, 100, 200);

        model_ref = std::make_shared<ov::Model>(NodeVector{ru}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, RandomUniformWithConvertMulFusing) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto conv = std::make_shared<opset8::Convert>(ru, element::f16);

        auto mul_const = opset8::Constant::create(element::f16, Shape{}, {30.0});
        auto mul = std::make_shared<opset8::Multiply>(conv, mul_const);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {30.0});

        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto conv = std::make_shared<opset8::Convert>(ru, element::f16);

        model_ref = std::make_shared<ov::Model>(NodeVector{conv}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, RandomUniformWithConvertAddFusing) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {30.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto conv = std::make_shared<opset8::Convert>(ru, element::f16);

        auto add_const = opset8::Constant::create(element::f16, Shape{}, {-10.0});
        auto add = std::make_shared<opset8::Add>(conv, add_const);

        model = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{input});

        manager.register_pass<ov::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto ru_min_const = opset8::Constant::create(element::f32, Shape{}, {-10.0});
        auto ru_max_const = opset8::Constant::create(element::f32, Shape{}, {20.0});

        auto ru = std::make_shared<opset8::RandomUniform>(input, ru_min_const, ru_max_const, element::f32, 100, 200);
        auto conv = std::make_shared<opset8::Convert>(ru, element::f16);

        model_ref = std::make_shared<ov::Model>(NodeVector{conv}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, RandomUniformFusingInvalidRUType) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::i32, Shape{}, {0});
        auto max_const = opset8::Constant::create(element::i32, Shape{}, {100});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::i32, 100, 200);

        auto mul_const = opset8::Constant::create(element::i32, Shape{}, {30});
        auto mul = std::make_shared<opset8::Multiply>(ru, mul_const);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::i32, Shape{}, {0});
        auto max_const = opset8::Constant::create(element::i32, Shape{}, {100});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::i32, 100, 200);

        auto mul_const = opset8::Constant::create(element::i32, Shape{}, {30});
        auto mul = std::make_shared<opset8::Multiply>(ru, mul_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, RandomUniformFusingInvalidConstShape) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);

        auto mul_const = opset8::Constant::create(element::f32, Shape{3}, {30, 20, 15});
        auto mul = std::make_shared<opset8::Multiply>(ru, mul_const);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);

        auto mul_const = opset8::Constant::create(element::f32, Shape{3}, {30, 20, 15});
        auto mul = std::make_shared<opset8::Multiply>(ru, mul_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
    }
}
