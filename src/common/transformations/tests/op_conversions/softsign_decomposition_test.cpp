// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/softsign_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset9.hpp"
#include "transformations/init_node_info.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, SoftSignDecomposition) {
    {
        auto data = std::make_shared<opset9::Parameter>(element::f32, Shape{3, 1, 2});
        auto softsign = std::make_shared<opset9::SoftSign>(data);

        model = std::make_shared<ov::Model>(NodeVector{softsign}, ParameterVector{data});

        manager.register_pass<ov::pass::SoftSignDecomposition>();
    }

    {
        auto input = std::make_shared<opset9::Parameter>(element::f32, Shape{3, 1, 2});
        auto abs = std::make_shared<opset9::Abs>(input);
        auto add = std::make_shared<opset9::Add>(abs, opset9::Constant::create(element::f32, Shape{1}, {1}));
        auto div = std::make_shared<opset9::Divide>(input, add);

        model_ref = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SoftSignDecompositionFP16) {
    {
        auto data = std::make_shared<opset9::Parameter>(element::f16, Shape{3, 1, 2});
        auto softsign = std::make_shared<opset9::SoftSign>(data);

        model = std::make_shared<ov::Model>(NodeVector{softsign}, ParameterVector{data});

        manager.register_pass<ov::pass::SoftSignDecomposition>();
    }

    {
        auto input = std::make_shared<opset9::Parameter>(element::f16, Shape{3, 1, 2});
        auto abs = std::make_shared<opset9::Abs>(input);
        auto add = std::make_shared<opset9::Add>(abs, opset9::Constant::create(element::f16, Shape{1}, {1}));
        auto div = std::make_shared<opset9::Divide>(input, add);

        model_ref = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});
    }
}
