// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/utils/utils.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "openvino/opsets/opset4.hpp"

using namespace ov;

namespace ov {
namespace op {
namespace util {

template <>
bool has_constant_value<bool>(const std::shared_ptr<Node>& node, const bool value, bool epsilon) {
    if (!node) {
        return false;
    }

    auto constant = ov::as_type_ptr<opset4::Constant>(node);
    if (!constant) {
        return false;
    }

    const bool is_scalar_or_single_elem = is_scalar(constant->get_shape()) || shape_size(constant->get_shape()) == 1;
    if (!is_scalar_or_single_elem) {
        return false;
    }

    const auto data = constant->cast_vector<bool>();
    if (data[0] != value) {
        return false;
    }

    return true;
}
}  // namespace util
}  // namespace op
}  // namespace ov

TEST(TransformationTests, HasConstantValueHelper) {
    auto float32_scalar = opset4::Constant::create(element::f32, Shape{}, {1.234f});
    ASSERT_TRUE(ov::op::util::has_constant_value<float>(float32_scalar, 1.234f));
    ASSERT_TRUE(ov::op::util::has_constant_value<float>(float32_scalar, 1.23f, 0.005f));
    ASSERT_FALSE(ov::op::util::has_constant_value<float>(float32_scalar, 1.23f, 0.003f));

    auto float32_1D = opset4::Constant::create(element::f32, Shape{1}, {1.234f});
    ASSERT_TRUE(ov::op::util::has_constant_value<float>(float32_scalar, 1.234f));

    auto int64_scalar = opset4::Constant::create(element::i64, Shape{}, {12});
    ASSERT_TRUE(ov::op::util::has_constant_value<int64_t>(int64_scalar, 12));

    auto bool_scalar = opset4::Constant::create(element::boolean, Shape{}, {true});
    ASSERT_TRUE(ov::op::util::has_constant_value<bool>(bool_scalar, true));

    ASSERT_FALSE(ov::op::util::has_constant_value<int8_t>(nullptr, 0));

    auto float32_2D = opset4::Constant::create(element::f32, Shape{2}, {1.2f, 3.4f});
    ASSERT_FALSE(ov::op::util::has_constant_value<float>(float32_2D, 1.2f));

    float32_scalar = opset4::Constant::create(element::f32, Shape{}, {1.234f});
    ASSERT_FALSE(ov::op::util::has_constant_value<float>(float32_scalar, 1.235f));

    float32_1D = opset4::Constant::create(element::f32, Shape{1}, {1.234f});
    ASSERT_FALSE(ov::op::util::has_constant_value<float>(float32_scalar, 1.235f));

    int64_scalar = opset4::Constant::create(element::i64, Shape{}, {12});
    ASSERT_FALSE(ov::op::util::has_constant_value<int64_t>(int64_scalar, 13));
}
