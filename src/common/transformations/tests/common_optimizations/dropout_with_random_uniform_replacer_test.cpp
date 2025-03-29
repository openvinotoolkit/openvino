// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerCase1) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto add_const = opset8::Constant::create(element::f32, Shape{}, {30.0});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});

        manager.register_pass<ov::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto broadcast_const = opset8::Constant::create(element::f32, Shape{}, {0.5});
        auto broadcast = std::make_shared<opset8::Broadcast>(broadcast_const, input);

        auto add_const = opset8::Constant::create(element::f32, Shape{}, {30.0});
        auto add = std::make_shared<opset8::Add>(broadcast, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerCase2) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f16, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f16, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f16, 100, 200);
        auto add_const = opset8::Constant::create(element::f16, Shape{}, {1.0});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});

        manager.register_pass<ov::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto broadcast_const = opset8::Constant::create(element::f16, Shape{}, {0.5});
        auto broadcast = std::make_shared<opset8::Broadcast>(broadcast_const, input);

        auto add_const = opset8::Constant::create(element::f16, Shape{}, {1.0});
        auto add = std::make_shared<opset8::Add>(broadcast, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerWithConvert) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto convert = std::make_shared<opset8::Convert>(ru, element::f16);
        auto add_const = opset8::Constant::create(element::f16, Shape{}, {1.0});
        auto add = std::make_shared<opset8::Add>(convert, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});

        manager.register_pass<ov::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto broadcast_const = opset8::Constant::create(element::f32, Shape{}, {0.5});
        auto broadcast = std::make_shared<opset8::Broadcast>(broadcast_const, input);
        auto convert = std::make_shared<opset8::Convert>(broadcast, element::f16);

        auto add_const = opset8::Constant::create(element::f16, Shape{}, {1.0});
        auto add = std::make_shared<opset8::Add>(convert, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerAddConstNegative) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto add_const = opset8::Constant::create(element::f32, Shape{}, {0.5});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});

        manager.register_pass<ov::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto add_const = opset8::Constant::create(element::f32, Shape{}, {0.5});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerNonFloatRUNegative) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::i32, Shape{}, {0});
        auto max_const = opset8::Constant::create(element::i32, Shape{}, {100});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::i32, 100, 200);
        auto add_const = opset8::Constant::create(element::i32, Shape{}, {10});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});

        manager.register_pass<ov::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::i32, Shape{}, {0});
        auto max_const = opset8::Constant::create(element::i32, Shape{}, {100});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::i32, 100, 200);
        auto add_const = opset8::Constant::create(element::i32, Shape{}, {10});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerInvalidMinNegative) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {-2.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto add_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});

        manager.register_pass<ov::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {-2.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto add_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerInvalidMaxNegative) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.5});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto add_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});

        manager.register_pass<ov::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.5});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto add_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerInvalidAddConstRankNegative) {
    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto add_const = opset8::Constant::create(element::f32, Shape{3}, {1.0, 2.0, 3.0});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});

        manager.register_pass<ov::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<opset8::Parameter>(element::i32, Shape{3});
        auto min_const = opset8::Constant::create(element::f32, Shape{}, {0.0});
        auto max_const = opset8::Constant::create(element::f32, Shape{}, {1.0});
        auto ru = std::make_shared<opset8::RandomUniform>(input, min_const, max_const, element::f32, 100, 200);
        auto add_const = opset8::Constant::create(element::f32, Shape{3}, {1.0, 2.0, 3.0});
        auto add = std::make_shared<opset8::Add>(ru, add_const);
        auto floor = std::make_shared<opset8::Floor>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{floor}, ParameterVector{input});
    }
}
