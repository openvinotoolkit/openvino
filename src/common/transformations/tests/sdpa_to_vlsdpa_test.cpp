// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_vlsdpa.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;

TEST(SDPATOVLSDPATest, SDPANotPresent) {
    const auto p0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    const auto p1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    const auto add = std::make_shared<op::v1::Add>(p0, p1);
    const auto result = std::make_shared<op::v0::Result>(add);

    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{p0, p1});

    ov::pass::Manager manager;
    manager.register_pass<pass::SDPAToVLSDPA>();
    EXPECT_THROW(manager.run_passes(model), ov::Exception);
}