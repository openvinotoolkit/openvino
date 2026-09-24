// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/mark_math_before_floor_to_keep_f16_rounding.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

TEST_F(TransformationTestsF, MarkMathBeforeFloorToKeepF16RoundingTest_CosFeedsFloor) {
    /*
    Cos directly feeding Floor is marked with disable_conversion(f16, f32)

        Param
          |
         Cos
          |
        Floor
    */
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1});
        auto cos = std::make_shared<ov::op::v0::Cos>(input);
        auto floor = std::make_shared<ov::op::v0::Floor>(cos);
        model = std::make_shared<ov::Model>(floor, ov::ParameterVector{input}, "model");
    }

    manager.register_pass<ov::pass::MarkMathBeforeFloorToKeepF16Rounding>();

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1});
        auto cos = std::make_shared<ov::op::v0::Cos>(input);
        auto floor = std::make_shared<ov::op::v0::Floor>(cos);
        ov::disable_conversion(cos, ov::element::f16, ov::element::f32);
        model_ref = std::make_shared<ov::Model>(floor, ov::ParameterVector{input}, "model_ref");
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkMathBeforeFloorToKeepF16RoundingTest_CosWithoutFloorIsNotMarked) {
    /*
    Cos not directly feeding a Floor is left unmarked

        Param
          |
         Cos
          |
        Relu
    */
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1});
        auto cos = std::make_shared<ov::op::v0::Cos>(input);
        auto relu = std::make_shared<ov::op::v0::Relu>(cos);
        model = std::make_shared<ov::Model>(relu, ov::ParameterVector{input}, "model");
    }

    manager.register_pass<ov::pass::MarkMathBeforeFloorToKeepF16Rounding>();

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1});
        auto cos = std::make_shared<ov::op::v0::Cos>(input);
        auto relu = std::make_shared<ov::op::v0::Relu>(cos);
        model_ref = std::make_shared<ov::Model>(relu, ov::ParameterVector{input}, "model_ref");
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, MarkMathBeforeFloorToKeepF16RoundingTest_SinFeedsFloor) {
    /*
    Sin directly feeding Floor is marked with disable_conversion(f16, f32) too

        Param
          |
         Sin
          |
        Floor
    */
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1});
        auto sin = std::make_shared<ov::op::v0::Sin>(input);
        auto floor = std::make_shared<ov::op::v0::Floor>(sin);
        model = std::make_shared<ov::Model>(floor, ov::ParameterVector{input}, "model");
    }

    manager.register_pass<ov::pass::MarkMathBeforeFloorToKeepF16Rounding>();

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1});
        auto sin = std::make_shared<ov::op::v0::Sin>(input);
        auto floor = std::make_shared<ov::op::v0::Floor>(sin);
        ov::disable_conversion(sin, ov::element::f16, ov::element::f32);
        model_ref = std::make_shared<ov::Model>(floor, ov::ParameterVector{input}, "model_ref");
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}
