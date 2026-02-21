// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/softmax_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/manager.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, SoftmaxDecompositionScalar) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
        auto softmax = std::make_shared<op::v8::Softmax>(data, -1);

        model = std::make_shared<ov::Model>(OutputVector{softmax}, ParameterVector{data});

        manager.register_pass<ov::pass::SoftmaxDecomposition>();
    }

    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
        auto one_const = op::v0::Constant::create(element::f32, Shape{}, {1});

        model_ref = std::make_shared<ov::Model>(OutputVector{one_const}, ParameterVector{data});
    }
}
