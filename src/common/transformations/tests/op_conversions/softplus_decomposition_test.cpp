// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/softplus_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, SoftPlusDecompositionFP32) {
    {
        auto data = std::make_shared<opset4::Parameter>(element::f32, Shape{3, 1, 2});
        auto softplus = std::make_shared<opset4::SoftPlus>(data);

        model = std::make_shared<ov::Model>(NodeVector{softplus}, ParameterVector{data});

        manager.register_pass<ov::pass::SoftPlusDecomposition>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{3, 1, 2});
        auto exp = std::make_shared<opset4::Exp>(input);
        auto add = std::make_shared<opset4::Add>(exp, opset4::Constant::create(element::f32, Shape{1}, {1.0}));
        auto log = std::make_shared<opset4::Log>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{log}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SoftPlusDecompositionFP16) {
    {
        auto data = std::make_shared<opset4::Parameter>(element::f16, Shape{3, 1, 2});
        auto softplus = std::make_shared<opset4::SoftPlus>(data);

        model = std::make_shared<ov::Model>(NodeVector{softplus}, ParameterVector{data});

        manager.register_pass<ov::pass::SoftPlusDecomposition>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, Shape{3, 1, 2});
        auto exp = std::make_shared<opset4::Exp>(input);
        auto add = std::make_shared<opset4::Add>(exp, opset4::Constant::create(element::f16, Shape{1}, {1.0}));
        auto log = std::make_shared<opset4::Log>(add);

        model_ref = std::make_shared<ov::Model>(NodeVector{log}, ParameterVector{input});
    }
}
