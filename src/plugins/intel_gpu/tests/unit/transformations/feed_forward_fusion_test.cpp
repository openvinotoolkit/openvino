// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <memory>

#include <openvino/pass/manager.hpp>
#include <openvino/core/model.hpp>
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/type/element_type.hpp"
#include <openvino/op/constant.hpp>
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/tanh.hpp"
#include "intel_gpu/op/feed_forward.hpp"

#include <plugin/transformations/feed_forward_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, FeedForwardTestFusion1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto add1 = std::make_shared<ov::op::v1::Add>(input1, input2);
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(add1, add1);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(add1, mul1);
        auto mul3_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.2});
        auto mul3 = std::make_shared<ov::op::v1::Multiply>(mul2, mul3_const);
        auto add2 = std::make_shared<ov::op::v1::Add>(add1, mul3);
        auto mul4_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.3});
        auto mul4 = std::make_shared<ov::op::v1::Multiply>(add2, mul4_const);
        auto tanh = std::make_shared<ov::op::v0::Tanh>(mul4);
        auto add3_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.4});
        auto add3 = std::make_shared<ov::op::v1::Add>(tanh, add3_const);
        auto mul5 = std::make_shared<ov::op::v1::Multiply>(add3, add1);
        auto mul6_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.5});
        auto mul6 = std::make_shared<ov::op::v1::Multiply>(mul5, mul6_const);
        model = std::make_shared<ov::Model>(ov::NodeVector{ mul6 }, ov::ParameterVector{ input1, input2 });
        manager.register_pass<FeedForwardFusion>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto add1 = std::make_shared<ov::op::v1::Add>(input1, input2);
        auto mul3_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.2});
        auto mul4_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.3});
        auto add3_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.4});
        auto mul6_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.5});
        auto feed_forawd = std::make_shared<op::FeedForward>(add1, mul3_const, mul4_const, add3_const, mul6_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ feed_forawd }, ov::ParameterVector{ input1, input2 });
    }
}
