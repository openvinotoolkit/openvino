// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "transformations/common_optimizations/gelu_fusion.hpp"

#include <gtest/gtest.h>
#include <math.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::op;
using namespace ov::opset12;

TEST_F(TransformationTestsF, GeluFusionPatternOne) {
    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});

        auto div_const = Constant::create(element::f32, Shape{1}, {M_SQRT2});
        auto add_const = Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const = Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<Divide>(data, div_const);
        auto erf = std::make_shared<Erf>(div);
        auto add = std::make_shared<Add>(erf, add_const);
        auto mul_first = std::make_shared<Multiply>(data, mul_const);
        auto mul = std::make_shared<Multiply>(mul_first, add);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfOne>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternOneF16) {
    {
        auto data = std::make_shared<Parameter>(element::f16, Shape{2, 2});

        auto div_const = Constant::create(element::f16, Shape{1}, {M_SQRT2});
        auto add_const = Constant::create(element::f16, Shape{1}, {1.0});
        auto mul_const = Constant::create(element::f16, Shape{1}, {0.5});

        auto div = std::make_shared<Divide>(data, div_const);
        auto erf = std::make_shared<Erf>(div);
        auto add = std::make_shared<Add>(erf, add_const);
        auto mul_first = std::make_shared<Multiply>(data, mul_const);
        auto mul = std::make_shared<Multiply>(mul_first, add);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfOne>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f16, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternTwo) {
    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});

        auto div_const = Constant::create(element::f32, Shape{1}, {M_SQRT2});
        auto add_const = Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const = Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<Divide>(data, div_const);
        auto erf = std::make_shared<Erf>(div);
        auto add = std::make_shared<Add>(erf, add_const);
        auto mul_first = std::make_shared<Multiply>(data, add);
        auto mul = std::make_shared<Multiply>(mul_first, mul_const);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfTwo>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternTwoF16) {
    {
        auto data = std::make_shared<Parameter>(element::f16, Shape{2, 2});

        auto div_const = Constant::create(element::f16, Shape{1}, {M_SQRT2});
        auto add_const = Constant::create(element::f16, Shape{1}, {1.0});
        auto mul_const = Constant::create(element::f16, Shape{1}, {0.5});

        auto div = std::make_shared<Divide>(data, div_const);
        auto erf = std::make_shared<Erf>(div);
        auto add = std::make_shared<Add>(erf, add_const);
        auto mul_first = std::make_shared<Multiply>(data, add);
        auto mul = std::make_shared<Multiply>(mul_first, mul_const);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfTwo>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f16, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternThree) {
    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});

        auto div_const = Constant::create(element::f32, Shape{1}, {M_SQRT2});
        auto add_const = Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const = Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<Divide>(data, div_const);
        auto erf = std::make_shared<Erf>(div);
        auto add = std::make_shared<Add>(erf, add_const);
        auto mul_first = std::make_shared<Multiply>(add, mul_const);
        auto mul = std::make_shared<Multiply>(data, mul_first);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfThree>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternThreeF16) {
    {
        auto data = std::make_shared<Parameter>(element::f16, Shape{2, 2});

        auto div_const = Constant::create(element::f16, Shape{1}, {M_SQRT2});
        auto add_const = Constant::create(element::f16, Shape{1}, {1.0});
        auto mul_const = Constant::create(element::f16, Shape{1}, {0.5});

        auto div = std::make_shared<Divide>(data, div_const);
        auto erf = std::make_shared<Erf>(div);
        auto add = std::make_shared<Add>(erf, add_const);
        auto mul_first = std::make_shared<Multiply>(add, mul_const);
        auto mul = std::make_shared<Multiply>(data, mul_first);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfThree>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f16, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternFour) {
    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});

        auto mul1_const = Constant::create(element::f32, Shape{1}, {1.0f / M_SQRT2});
        auto add_const = Constant::create(element::f32, Shape{1}, {0.5f});
        auto mul2_const = Constant::create(element::f32, Shape{1}, {0.5f});

        auto mul1 = std::make_shared<Multiply>(data, mul1_const);
        auto erf = std::make_shared<Erf>(mul1);
        auto mul2 = std::make_shared<Multiply>(erf, mul2_const);
        auto add = std::make_shared<Add>(mul2, add_const);
        auto mul3 = std::make_shared<Multiply>(data, add);

        model = std::make_shared<Model>(NodeVector{mul3}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfFour>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternFourF16) {
    {
        auto data = std::make_shared<Parameter>(element::f16, Shape{2, 2});

        auto mul1_const = Constant::create(element::f16, Shape{1}, {1.0f / M_SQRT2});
        auto add_const = Constant::create(element::f16, Shape{1}, {0.5f});
        auto mul2_const = Constant::create(element::f16, Shape{1}, {0.5f});

        auto mul1 = std::make_shared<Multiply>(data, mul1_const);
        auto erf = std::make_shared<Erf>(mul1);
        auto mul2 = std::make_shared<Multiply>(erf, mul2_const);
        auto add = std::make_shared<Add>(mul2, add_const);
        auto mul3 = std::make_shared<Multiply>(data, add);

        model = std::make_shared<Model>(NodeVector{mul3}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfFour>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f16, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternIncorrectDivConstValue) {
    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});

        auto div_const = Constant::create(element::f32, Shape{1}, {1.4149});
        auto add_const = Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const = Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<Divide>(data, div_const);
        auto erf = std::make_shared<Erf>(div);
        auto add = std::make_shared<Add>(erf, add_const);
        auto mul_first = std::make_shared<Multiply>(data, add);
        auto mul = std::make_shared<Multiply>(mul_first, mul_const);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfTwo>();
    }
}

TEST_F(TransformationTestsF, GeluFusionPatternTooShortDivConstValue) {
    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});

        auto div_const = Constant::create(element::f32, Shape{1}, {1.4142});
        auto add_const = Constant::create(element::f32, Shape{1}, {1.0});
        auto mul_const = Constant::create(element::f32, Shape{1}, {0.5});

        auto div = std::make_shared<Divide>(data, div_const);
        auto erf = std::make_shared<Erf>(div);
        auto add = std::make_shared<Add>(erf, add_const);
        auto mul_first = std::make_shared<Multiply>(data, add);
        auto mul = std::make_shared<Multiply>(mul_first, mul_const);

        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::GeluFusionWithErfTwo>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_equal_const_values) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data, op::GeluApproximationMode::TANH);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_params_no_conversion) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_param = std::make_shared<Parameter>(element::f32, Shape{1});
        auto pow = std::make_shared<Power>(input, pow_param);
        auto mul_0_param = std::make_shared<Parameter>(element::f32, Shape{1});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_param);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        auto mul_1_param = std::make_shared<Parameter>(element::f32, Shape{1});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_param);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_param = std::make_shared<Parameter>(element::f32, Shape{1});
        auto add_1 = std::make_shared<Add>(tanh, add_1_param);

        auto mul_2_param = std::make_shared<Parameter>(element::f32, Shape{1});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_param);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(
            NodeVector{mul_3},
            ParameterVector{input, pow_param, mul_0_param, mul_1_param, add_1_param, mul_2_param});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_pow_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f + 1.0e-8f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data, op::GeluApproximationMode::TANH);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_pow_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{2.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_mul_0_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.04515f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data, op::GeluApproximationMode::TANH);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_mul_0_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.4715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_mul_1_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        auto mul_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.7980868f});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data, op::GeluApproximationMode::TANH);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_mul_1_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(10.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_add_1_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f + 1.0e-8f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data, op::GeluApproximationMode::TANH);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_add_1_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{2.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_epsilon_mul_2_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.5f + 1.0e-8f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }

    {
        auto data = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto gelu = std::make_shared<Gelu>(data, op::GeluApproximationMode::TANH);
        model_ref = std::make_shared<Model>(NodeVector{gelu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GeluFusionTanhWithTanh_wrong_mul_2_value) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{2, 2});
        auto pow_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{3.0f});
        auto pow = std::make_shared<Power>(input, pow_constant);
        auto mul_0_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{0.044715f});
        auto mul_0 = std::make_shared<Multiply>(pow, mul_0_constant);
        auto add_0 = std::make_shared<Add>(input, mul_0);

        constexpr float pi = 3.141592653589793238462643383279502884f;
        auto mul_1_constant =
            std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{std::sqrt(2.0f / pi)});
        auto mul_1 = std::make_shared<Multiply>(add_0, mul_1_constant);

        auto tanh = std::make_shared<Tanh>(mul_1);

        auto add_1_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{1.0f});
        auto add_1 = std::make_shared<Add>(tanh, add_1_constant);

        auto mul_2_constant = std::make_shared<Constant>(element::f32, Shape{1}, std::vector<float>{5.0f});
        auto mul_2 = std::make_shared<Multiply>(add_1, mul_2_constant);

        auto mul_3 = std::make_shared<Multiply>(input, mul_2);

        model = std::make_shared<Model>(NodeVector{mul_3}, ParameterVector{input});
        manager.register_pass<ov::pass::GeluFusionWithTanh>();
    }
}

TEST_F(TransformationTestsF, FoldGeluOperation) {
    {
        auto param = std::make_shared<Parameter>(element::f32, Shape{1006, 2, 100, 3, 4096});
        auto const1 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{0.044715});

        auto mul1 = std::make_shared<Multiply>(param, const1);
        auto mul2 = std::make_shared<Multiply>(mul1, param);

        auto const2 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{1.0});
        auto add1 = std::make_shared<Add>(const2, mul2);

        auto const3 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{0.7978845608028654});
        auto mul3 = std::make_shared<Multiply>(param, const3);

        auto mul4 = std::make_shared<Multiply>(add1, mul3);
        auto tan = std::make_shared<Tanh>(mul4);

        auto const4 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{1.0});
        auto add2 = std::make_shared<Add>(tan, const4);

        auto const5 = Constant::create(element::f32, Shape{1, 1, 1}, std::vector<float>{0.5});
        auto mul5 = std::make_shared<Multiply>(param, const5);

        auto mul6 = std::make_shared<Multiply>(add2, mul5);

        auto result = std::make_shared<Result>(mul6);
        model = std::make_shared<Model>(NodeVector{result}, ParameterVector{param});

        manager.register_pass<ov::pass::GeluFusionWithTanhNoPower>();
    }

    {
        auto param = std::make_shared<Parameter>(element::f32, Shape{1006, 2, 100, 3, 4096});
        auto gelu = std::make_shared<Gelu>(param, GeluApproximationMode::TANH);
        auto result = std::make_shared<Result>(gelu);
        model_ref = std::make_shared<Model>(NodeVector{result}, ParameterVector{param});
    }
}
