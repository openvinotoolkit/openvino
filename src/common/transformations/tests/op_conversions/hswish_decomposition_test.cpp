// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/hswish_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, HSwishDecompositionTest) {
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hswish = std::make_shared<op::v4::HSwish>(input);

        model = std::make_shared<ov::Model>(NodeVector{hswish}, ParameterVector{input});

        manager.register_pass<ov::pass::HSwishDecomposition>();
    }

    {
        auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = op::v0::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<op::v1::Add>(input, add_constant);
        auto relu = std::make_shared<op::v0::Relu>(add);
        auto min_constant = op::v0::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<op::v1::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<op::v1::Multiply>(input, min);
        auto mul_constant = op::v0::Constant::create(element::f16, Shape{}, {0.1666666716});
        auto mul_second = std::make_shared<op::v1::Multiply>(mul_first, mul_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});
    }
}
