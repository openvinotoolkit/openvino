// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mish_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/softplus_to_mish_fusion.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

// LPT to openvino migration: temporary disabling unexpected not reproduced fails on CI:
// https://openvino-ci.intel.com/job/private-ci/job/ie/job/build-linux-ubuntu18_i386/478/
TEST_F(TransformationTestsF, MishFusing) {
    {
        auto input0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 1, 2});
        auto exp = std::make_shared<op::v0::Exp>(input0);
        auto input_const = op::v0::Constant::create(element::f32, Shape{1}, {-1});
        auto add = std::make_shared<op::v1::Add>(exp, input_const);
        auto log = std::make_shared<op::v0::Log>(add);
        auto tanh = std::make_shared<op::v0::Tanh>(log);
        auto mul = std::make_shared<op::v1::Multiply>(input0, tanh);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input0});

        manager.register_pass<ov::pass::MishFusion>();
    }

    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 1, 2});
        auto mish = std::make_shared<op::v4::Mish>(data);

        model_ref = std::make_shared<ov::Model>(NodeVector{mish}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, MishWithSoftPlusFusing) {
    {
        auto input0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 1, 2});
        auto softplus = std::make_shared<op::v4::SoftPlus>(input0);
        auto tanh = std::make_shared<op::v0::Tanh>(softplus);
        auto mul = std::make_shared<op::v1::Multiply>(input0, tanh);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input0});

        manager.register_pass<ov::pass::SoftPlusToMishFusion>();
    }

    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{3, 1, 2});
        auto mish = std::make_shared<op::v4::Mish>(data);

        model_ref = std::make_shared<ov::Model>(NodeVector{mish}, ParameterVector{data});
    }
}
