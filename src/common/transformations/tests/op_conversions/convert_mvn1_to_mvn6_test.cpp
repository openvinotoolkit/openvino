// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_mvn1_to_mvn6.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ConvertMVN1ToMVN6) {
    {
        auto data = std::make_shared<opset2::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto mvn = std::make_shared<op::v0::MVN>(data, false, true, 1e-5);

        model = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertMVN1ToMVN6>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto axes_const = opset6::Constant::create(element::i64, Shape{2}, {2, 3});
        auto mvn = std::make_shared<op::v6::MVN>(data, axes_const, true, 1e-5f, op::MVNEpsMode::INSIDE_SQRT);

        model_ref = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertMVN1ToMVN6_across_channels) {
    {
        auto data = std::make_shared<opset2::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto mvn = std::make_shared<op::v0::MVN>(data, true, true, 1e-5);

        model = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertMVN1ToMVN6>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto axes_const = opset6::Constant::create(element::i64, Shape{3}, {1, 2, 3});
        auto mvn = std::make_shared<op::v6::MVN>(data, axes_const, true, 1e-5f, op::MVNEpsMode::INSIDE_SQRT);

        model_ref = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertMVN1ToMVN6_5D) {
    {
        auto data = std::make_shared<opset2::Parameter>(element::f32, Shape{1, 2, 3, 4, 5});
        auto mvn = std::make_shared<op::v0::MVN>(data, false, true, 1e-5);

        model = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});

        manager.register_pass<ov::pass::ConvertMVN1ToMVN6>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4, 5});
        auto axes_const = opset6::Constant::create(element::i64, Shape{3}, {2, 3, 4});
        auto mvn = std::make_shared<op::v6::MVN>(data, axes_const, true, 1e-5f, op::MVNEpsMode::INSIDE_SQRT);

        model_ref = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});
    }
}

namespace {
struct ConvertMVN1ToMVN6_OutOfFloat32Eps_params {
    double eps_d;
    float eps_f;
};

class ConvertMVN1ToMVN6_OutOfFloat32Eps : public testing::WithParamInterface<ConvertMVN1ToMVN6_OutOfFloat32Eps_params>,
                                          public TransformationTestsF {};

TEST_P(ConvertMVN1ToMVN6_OutOfFloat32Eps, Limits) {
    manager.register_pass<ov::pass::ConvertMVN1ToMVN6>();

    const auto& params = GetParam();
    {
        auto data = std::make_shared<opset2::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto mvn = std::make_shared<op::v0::MVN>(data, true, true, params.eps_d);
        model = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});
    }
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto axes_const = opset6::Constant::create(element::i64, Shape{3}, {1, 2, 3});
        auto mvn = std::make_shared<op::v6::MVN>(data, axes_const, true, params.eps_f, op::MVNEpsMode::INSIDE_SQRT);

        model_ref = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});
    }
}

const auto out_of_f32_epsilons =
    std::vector<ConvertMVN1ToMVN6_OutOfFloat32Eps_params>{{1e-39, std::numeric_limits<float>::min()},
                                                          {1e-3, 1e-3f},
                                                          {1e+39, std::numeric_limits<float>::max()}};
}  // namespace

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         ConvertMVN1ToMVN6_OutOfFloat32Eps,
                         ::testing::ValuesIn(out_of_f32_epsilons));
