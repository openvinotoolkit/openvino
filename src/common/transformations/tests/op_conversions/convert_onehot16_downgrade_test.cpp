// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/convert_one_hot_v16_to_v1.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

namespace {
template <class TOp>
std::shared_ptr<ov::Model> create_model() {
    const std::vector<int64_t> indices{0, 1, 2};
    const float on_value = 1.123f;
    const float off_value = 0.321f;

    const auto indices_const = ov::op::v0::Constant::create(element::i64, Shape{3}, indices);
    const auto depth_const = ov::op::v0::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = ov::op::v0::Constant::create(element::f32, Shape{}, {on_value});
    const auto off_const = ov::op::v0::Constant::create(element::f32, Shape{}, {off_value});

    auto one_hot = std::make_shared<TOp>(indices_const, depth_const, on_const, off_const, -1);
    return std::make_shared<ov::Model>(one_hot, ParameterVector{});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertOneHot16To1) {
    manager.register_pass<ov::pass::ConvertOneHot16To1>();
    model = create_model<op::v16::OneHot>();
    model_ref = create_model<op::v1::OneHot>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
