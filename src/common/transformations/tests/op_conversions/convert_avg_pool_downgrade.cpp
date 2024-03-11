// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_avg_pool_downgrade.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset14.hpp"
#include "openvino/pass/manager.hpp"

namespace {

std::shared_ptr<ov::Model> create_v14_model(const ov::op::RoundingType rounding_type) {
    const auto input = std::make_shared<ov::opset14::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto avg_pool_v14 = std::make_shared<ov::opset14::AvgPool>(input,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     kernel,
                                                                     true,
                                                                     rounding_type,
                                                                     ov::op::PadType::EXPLICIT);

    avg_pool_v14->set_friendly_name("avg_pool_v14");

    return std::make_shared<ov::Model>(avg_pool_v14->outputs(), ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_v1_model(const ov::op::RoundingType rounding_type) {
    const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
    const ov::Strides strides{1, 1}, dilations{1, 1};
    const ov::Shape pads_begin{1, 1}, pads_end{1, 1}, kernel{2, 2};

    const auto avg_pool_v1 = std::make_shared<ov::opset1::AvgPool>(input,
                                                                   strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   kernel,
                                                                   true,
                                                                   rounding_type,
                                                                   ov::op::PadType::EXPLICIT);

    avg_pool_v1->set_friendly_name("avg_pool_v1");

    return std::make_shared<ov::Model>(avg_pool_v1->outputs(), ov::ParameterVector{input});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_ceil_torch_to_ceil) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_model(ov::op::RoundingType::CEIL_TORCH);
    model_ref = create_v1_model(ov::op::RoundingType::CEIL);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_ceil_to_ceil) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_model(ov::op::RoundingType::CEIL);
    model_ref = create_v1_model(ov::op::RoundingType::CEIL);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_floor_to_floor) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v14_model(ov::op::RoundingType::FLOOR);
    model_ref = create_v1_model(ov::op::RoundingType::FLOOR);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertAvgPool14ToAvgPool1_incorrect_version) {
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    model = create_v1_model(ov::op::RoundingType::CEIL);
}
