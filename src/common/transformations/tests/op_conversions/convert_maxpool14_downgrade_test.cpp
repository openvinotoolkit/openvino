// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/opsets/opset14.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/convert_maxpool14_to_maxpool8_downgrade.hpp"

namespace {

std::shared_ptr<ov::Model> create_v14_model(const ov::op::RoundingType rounding_type) {
    const auto input = std::make_shared<ov::opset14::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto max_pool_v14 =
        std::make_shared<ov::opset14::MaxPool>(input, strides, dilations, pads_begin, pads_end, kernel, rounding_type);

    max_pool_v14->set_friendly_name("max_pool_v14");

    return std::make_shared<ov::Model>(max_pool_v14->outputs(), ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_v8_model(const ov::op::RoundingType rounding_type) {
    const auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto max_pool_v8 =
        std::make_shared<ov::opset8::MaxPool>(input, strides, dilations, pads_begin, pads_end, kernel, rounding_type);

    max_pool_v8->set_friendly_name("max_pool_v8");

    return std::make_shared<ov::Model>(max_pool_v8->outputs(), ov::ParameterVector{input});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertMaxPool14ToMaxPool8_ceil_torch_to_ceil) {
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    model = create_v14_model(ov::op::RoundingType::CEIL_TORCH);
    model_ref = create_v8_model(ov::op::RoundingType::CEIL);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertMaxPool14ToMaxPool8_ceil_to_ceil) {
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    model = create_v14_model(ov::op::RoundingType::CEIL);
    model_ref = create_v8_model(ov::op::RoundingType::CEIL);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertMaxPool14ToMaxPool8_floor_to_floor) {
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    model = create_v14_model(ov::op::RoundingType::FLOOR);
    model_ref = create_v8_model(ov::op::RoundingType::FLOOR);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertMaxPool14ToMaxPool8_incorrect_version) {
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    model = create_v8_model(ov::op::RoundingType::CEIL);
}
