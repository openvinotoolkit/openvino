// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/op_conversions/convert_nonzero_to_range.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertNonZeroToRangeStaticShape) {
    PartialShape data_shape{ 1, 128 };
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data);

        auto gather_indices = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto one_const = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto bcast = std::make_shared<opset8::Broadcast>(one_const, gather);
        auto nonzero = std::make_shared<opset8::NonZero>(bcast);
        auto res = std::make_shared<opset8::Result>(nonzero);

        function = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
        manager.register_pass<pass::ConvertNonZeroToRange>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data);

        auto gather_indices = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto squeeze_const = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto squeeze_before = std::make_shared<opset8::Squeeze>(gather, squeeze_const);

        auto start = opset8::Constant::create(element::i64, {}, { 0 });
        auto step = opset8::Constant::create(element::i64, {}, { 1 });
        auto range = std::make_shared<opset8::Range>(start, squeeze_before, step, element::i64);

        auto unsqueeze_const = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto unsqueeze_after = std::make_shared<opset8::Unsqueeze>(range, unsqueeze_const);
        auto res = std::make_shared<opset8::Result>(unsqueeze_after);

        function_ref = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
    }
}

TEST_F(TransformationTestsF, ConvertNonZeroToRangeDynamicShape) {
    PartialShape data_shape{ 1, -1 };
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data);

        auto gather_indices = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto one_const = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto bcast = std::make_shared<opset8::Broadcast>(one_const, gather);
        auto nonzero = std::make_shared<opset8::NonZero>(bcast);
        auto res = std::make_shared<opset8::Result>(nonzero);

        function = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
        manager.register_pass<pass::ConvertNonZeroToRange>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data);

        auto gather_indices = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto squeeze_const = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto squeeze_before = std::make_shared<opset8::Squeeze>(gather, squeeze_const);

        auto start = opset8::Constant::create(element::i64, {}, { 0 });
        auto step = opset8::Constant::create(element::i64, {}, { 1 });
        auto range = std::make_shared<opset8::Range>(start, squeeze_before, step, element::i64);

        auto unsqueeze_const = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto unsqueeze_after = std::make_shared<opset8::Unsqueeze>(range, unsqueeze_const);
        auto res = std::make_shared<opset8::Result>(unsqueeze_after);

        function_ref = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
    }
}

TEST_F(TransformationTestsF, ConvertNonZeroToRangeBcastWith3Inputs) {
    PartialShape data_shape{ 1, 128 };
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data);

        auto gather_indices = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto one_const = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto axes_mapping = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto bcast = std::make_shared<opset8::Broadcast>(one_const, gather, axes_mapping);
        auto nonzero = std::make_shared<opset8::NonZero>(bcast);
        auto res = std::make_shared<opset8::Result>(nonzero);

        function = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
        manager.register_pass<pass::ConvertNonZeroToRange>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data);

        auto gather_indices = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto squeeze_const = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto squeeze_before = std::make_shared<opset8::Squeeze>(gather, squeeze_const);

        auto start = opset8::Constant::create(element::i64, {}, { 0 });
        auto step = opset8::Constant::create(element::i64, {}, { 1 });
        auto range = std::make_shared<opset8::Range>(start, squeeze_before, step, element::i64);

        auto unsqueeze_const = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto unsqueeze_after = std::make_shared<opset8::Unsqueeze>(range, unsqueeze_const);
        auto res = std::make_shared<opset8::Result>(unsqueeze_after);

        function_ref = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
    }
}

TEST_F(TransformationTestsF, ConvertNonZeroToRangeBcastFromOpset1) {
    PartialShape data_shape{ 1, 128 };
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data);

        auto gather_indices = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto one_const = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto bcast = std::make_shared<opset1::Broadcast>(one_const, gather);
        auto nonzero = std::make_shared<opset8::NonZero>(bcast);
        auto res = std::make_shared<opset8::Result>(nonzero);

        function = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
        manager.register_pass<pass::ConvertNonZeroToRange>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data);

        auto gather_indices = opset8::Constant::create(element::i64, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto squeeze_const = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto squeeze_before = std::make_shared<opset8::Squeeze>(gather, squeeze_const);

        auto start = opset8::Constant::create(element::i64, {}, { 0 });
        auto step = opset8::Constant::create(element::i64, {}, { 1 });
        auto range = std::make_shared<opset8::Range>(start, squeeze_before, step, element::i64);

        auto unsqueeze_const = opset8::Constant::create(element::i64, { 1 }, { 0 });
        auto unsqueeze_after = std::make_shared<opset8::Unsqueeze>(range, unsqueeze_const);
        auto res = std::make_shared<opset8::Result>(unsqueeze_after);

        function_ref = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
    }
}

TEST_F(TransformationTestsF, ConvertNonZeroToRangei32Precision) {
    PartialShape data_shape{ 1, 128 };
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data, element::i32);

        auto gather_indices = opset8::Constant::create(element::i32, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i32, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto one_const = opset8::Constant::create(element::i32, { 1 }, { 1 });
        auto bcast = std::make_shared<opset1::Broadcast>(one_const, gather);
        auto nonzero = std::make_shared<opset8::NonZero>(bcast, element::i32);
        auto res = std::make_shared<opset8::Result>(nonzero);

        function = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
        manager.register_pass<pass::ConvertNonZeroToRange>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto shapeOf = std::make_shared<opset8::ShapeOf>(data, element::i32);

        auto gather_indices = opset8::Constant::create(element::i32, { 1 }, { 1 });
        auto gather_axis = opset8::Constant::create(element::i32, { 1 }, { 0 });
        auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

        auto squeeze_const = opset8::Constant::create(element::i32, { 1 }, { 0 });
        auto squeeze_before = std::make_shared<opset8::Squeeze>(gather, squeeze_const);

        auto start = opset8::Constant::create(element::i32, {}, { 0 });
        auto step = opset8::Constant::create(element::i32, {}, { 1 });
        auto range = std::make_shared<opset8::Range>(start, squeeze_before, step, element::i32);

        auto unsqueeze_const = opset8::Constant::create(element::i32, { 1 }, { 0 });
        auto unsqueeze_after = std::make_shared<opset8::Unsqueeze>(range, unsqueeze_const);
        auto res = std::make_shared<opset8::Result>(unsqueeze_after);

        function_ref = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
    }
}

TEST_F(TransformationTestsF, ConvertNonZeroToRangeZeroBcastedConst) {
    PartialShape data_shape{ 1, 128 };
    auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
    auto shapeOf = std::make_shared<opset8::ShapeOf>(data, element::i32);

    auto gather_indices = opset8::Constant::create(element::i32, { 1 }, { 1 });
    auto gather_axis = opset8::Constant::create(element::i32, { 1 }, { 0 });
    auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

    auto zero_const = opset8::Constant::create(element::i32, { 1 }, { 0 });
    auto bcast = std::make_shared<opset1::Broadcast>(zero_const, gather);
    auto nonzero = std::make_shared<opset8::NonZero>(bcast, element::i32);
    auto res = std::make_shared<opset8::Result>(nonzero);

    function = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
    manager.register_pass<pass::ConvertNonZeroToRange>();
}

TEST_F(TransformationTestsF, ConvertNonZeroToRange2DBcastTargetShape) {
    PartialShape data_shape{ 1, 128 };
    auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
    auto shapeOf = std::make_shared<opset8::ShapeOf>(data, element::i32);

    auto gather_indices = opset8::Constant::create(element::i32, { 2 }, { 0, 1 });
    auto gather_axis = opset8::Constant::create(element::i32, { 1 }, { 0 });
    auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

    auto one_const = opset8::Constant::create(element::i32, { 1 }, { 1 });
    auto bcast = std::make_shared<opset1::Broadcast>(one_const, gather);
    auto nonzero = std::make_shared<opset8::NonZero>(bcast, element::i32);
    auto res = std::make_shared<opset8::Result>(nonzero);

    function = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data });
    manager.register_pass<pass::ConvertNonZeroToRange>();
}

TEST_F(TransformationTestsF, ConvertNonZeroToRangeNonConstBcastData) {
    PartialShape data_shape{ 1, 128 };
    auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
    auto shapeOf = std::make_shared<opset8::ShapeOf>(data, element::i32);

    auto gather_indices = opset8::Constant::create(element::i32, { 1 }, { 1 });
    auto gather_axis = opset8::Constant::create(element::i32, { 1 }, { 0 });
    auto gather = std::make_shared<opset8::Gather>(shapeOf, gather_indices, gather_axis);

    auto data2 = std::make_shared<opset8::Parameter>(element::f32, PartialShape{ 1 });
    auto bcast = std::make_shared<opset1::Broadcast>(data2, gather);
    auto nonzero = std::make_shared<opset8::NonZero>(bcast, element::i32);
    auto res = std::make_shared<opset8::Result>(nonzero);

    function = std::make_shared<Function>(ResultVector{ res }, ParameterVector{ data, data2 });
    manager.register_pass<pass::ConvertNonZeroToRange>();
}
