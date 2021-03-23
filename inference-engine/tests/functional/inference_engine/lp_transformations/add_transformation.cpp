// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

#include <low_precision/add.hpp>
#include "lpt_ngraph_functions/add_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class AddTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precision1;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::element::Type precision2;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
        std::vector<float> constValues;
    };

    class Expected {
    public:
        ngraph::element::Type precision1;
        ngraph::builder::subgraph::DequantizationOperations dequantization1;
        ngraph::element::Type precision2;
        ngraph::builder::subgraph::DequantizationOperations dequantization2;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
        std::vector<float> constValues;
        std::string operationType;

        Expected() = default;

        Expected(const ngraph::element::Type& precision1,
                 ngraph::builder::subgraph::DequantizationOperations dequantization1,
                 const ngraph::element::Type& precision2,
                 ngraph::builder::subgraph::DequantizationOperations dequantization2,
                 ngraph::builder::subgraph::DequantizationOperations dequantizationAfter,
                 std::vector<float> constValues,
                 std::string operationType = "Add"): precision1(precision1), dequantization1(std::move(dequantization1)),
                                         precision2(precision2), dequantization2(std::move(dequantization2)),
                                         dequantizationAfter(std::move(dequantizationAfter)), constValues(std::move(constValues)),
                                         operationType(std::move(operationType)) {}
    };

    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    bool broadcast;
    int constInput;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
    std::string additionalLayer;
};

typedef std::tuple <
    ngraph::element::Type,
    AddTransformationTestValues
> AddTransformationParams;

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

class AddTransformation : public LayerTransformation, public testing::WithParamInterface<AddTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const AddTransformationTestValues& testValues = std::get<1>(GetParam());

        actualFunction = AddFunction::getOriginal(
            precision,
            testValues.inputShape,
            testValues.broadcast,
            testValues.params,
            testValues.actual.precision1,
            testValues.actual.dequantization1,
            testValues.actual.precision2,
            testValues.actual.dequantization2,
            testValues.constInput,
            testValues.actual.constValues,
            testValues.additionalLayer);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::AddTransformation, ngraph::opset1::Add>(
                low_precision::LayerTransformation::Params(testValues.params));
        transform.transform(actualFunction);

        referenceFunction = AddFunction::getReference(
            precision,
            testValues.inputShape,
            testValues.broadcast,
            testValues.params,
            testValues.expected.precision1,
            testValues.expected.dequantization1,
            testValues.expected.precision2,
            testValues.expected.dequantization2,
            testValues.expected.dequantizationAfter,
            // Constant operations after transformations are on 1 input only
            testValues.constInput == -1 ? -1 : 1,
            testValues.expected.constValues,
            testValues.additionalLayer,
            testValues.expected.operationType);
    }

    static std::string getTestCaseName(testing::TestParamInfo<AddTransformationParams> obj) {
        const element::Type precision = std::get<0>(obj.param);
        const AddTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            precision << "_" <<
            testValues.inputShape << "_" <<
            testValues.broadcast << "_" <<
            testValues.actual.precision1 << "_" <<
            testValues.actual.dequantization1 << "_" <<
            testValues.actual.precision2 << "_" <<
            testValues.actual.dequantization2 << "_" <<
            testValues.constInput << "_" <<
            testValues.actual.constValues << "_" <<
            testValues.additionalLayer;
        return result.str();
    }
};

TEST_P(AddTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> netPrecision = {
    element::f32,
    element::f16
};

const std::vector<AddTransformationTestValues> addTransformationTestValues = {
    // Multiply with zero on the first branch
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            { },
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { {1.f, 0.f, 2.f, 3.f} }},
            { }
        },
        {
            ngraph::element::f32,
            { },
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { {1.f, 0.f, 2.f, 3.f} }},
            { },
            { }
        },
        ""
    },
    // Multiply with zero on the second branch
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { {1.f, 0.f, 2.f, 3.f} }},
            ngraph::element::f32,
            { },
            { }
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { {1.f, 0.f, 2.f, 3.f} }},
            ngraph::element::f32,
            { },
            { },
            { }
        },
        ""
    },

    // Actual:
    //
    // Parameter           Parameter
    //   |U8                 |U8
    //   |                   |
    // Convert Constant    Convert  Constant
    //  \FP32  /FP32        \FP32   /FP32
    //   \    /              \     /
    //  Subtract  Constant  Subtract  Constant
    //     \FP32   /FP32       \FP32  /FP32
    //      \     /             \    /
    //      Multiply           Multiply
    //             \FP32      /FP32
    //              \        /
    //                 Add
    // Transformed:
    //
    // Parameter
    //   |U8
    //   |
    // Convert  Constant
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //      Multiply   Parameter
    //          \FP32  /U8
    //           \    /
    //            Add
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { 10.f }},
            ngraph::element::u8,
            { {ngraph::element::f32},  { 3.f }, { 5.f } },
            {}
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 8.5f }, { 2.f }},
            ngraph::element::u8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },

    // Actual:
    //
    // Parameter Constant Parameter Constant
    //  |U8      |U8        |U8      |U8
    //  |        |          |        |
    // Convert Convert    Convert  Convert
    //  \FP32  /FP32        \FP32   /FP32
    //   \    /              \     /
    //  Subtract  Constant  Subtract  Constant
    //     \FP32   /FP32       \FP32  /FP32
    //      \     /             \    /
    //      Multiply           Multiply
    //             \FP32      /FP32
    //              \        /
    //                 Add
    // Transformed:
    //
    // Parameter
    //   |U8
    //   |
    // Convert  Constant
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //      Multiply   Parameter
    //          \FP32  /U8
    //           \    /
    //            Add
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {7.f}, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true },
                { 10.f }
            },
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {3.f}, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true },
                { 5.f }
            },
            {}
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 8.5f }, { 2.f }},
            ngraph::element::u8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 2.f }, { 10.f }},
            ngraph::element::u8,
            { {ngraph::element::f32},  { }, { 5.f } },
            {}
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 2.f }, { 2.f }},
            ngraph::element::u8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { }, { 10.f }},
            ngraph::element::u8,
            { {ngraph::element::f32},  { }, { 5.f } },
            {}
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { }, { 2.f }},
            ngraph::element::u8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 2.f }, { }},
            ngraph::element::u8,
            { {ngraph::element::f32},  { }, { 5.f } },
            {}
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 2.f }, { 0.2f }},
            ngraph::element::u8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 2.f }, { }},
            ngraph::element::u8,
            { {ngraph::element::f32},  { 3.f }, { 5.f } },
            {}
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 17.f }, { 0.2f }},
            ngraph::element::u8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },

    // I8 + broadcast

    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        true,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { 7.f }, { 10.f }},
            ngraph::element::i8,
            { {ngraph::element::f32},  { 3.f }, { 5.f } },
            {}
        },
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { 8.5f }, { 2.f }},
            ngraph::element::i8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        true,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { 2.f }, { 10.f }},
            ngraph::element::i8,
            { {ngraph::element::f32},  { }, { 5.f } },
            {}
        },
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { 2.f }, { 2.f }},
            ngraph::element::i8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        true,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { }, { 10.f }},
            ngraph::element::i8,
            { {ngraph::element::f32},  { }, { 5.f } },
            {}
        },
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { }, { 2.f }},
            ngraph::element::i8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        true,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { 2.f }, { }},
            ngraph::element::i8,
            { {ngraph::element::f32},  { }, { 5.f } },
            {}
        },
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { 2.f }, { 0.2f }},
            ngraph::element::i8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        true,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { 2.f }, { }},
            ngraph::element::i8,
            { {ngraph::element::f32},  { 3.f }, { 5.f } },
            {}
        },
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { 17.f }, { 0.2f }},
            ngraph::element::i8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            {}
        },
        ""
    },

    {
        ngraph::element::f32,
        ngraph::Shape{4, 1},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { }, { {1.f, 2.f, 3.f, 4.f}, ngraph::element::f32, {4, 1}, true, 0ul }},
            ngraph::element::f32,
            {},
            { 5.f, 6.f, 7.f, 8.f }
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { }, { {1.f, 2.f, 3.f, 4.f}, ngraph::element::f32, {4, 1}, true, 0ul }},
            ngraph::element::f32,
            { {},  {}, {} },
            { {},  {}, {} },
            { 5.f, 6.f, 7.f, 8.f }
        },
        ""
    },

    // constant input: Add -> Subtract
    {
    ngraph::element::f32,
        ngraph::Shape{ 1, 2, 2, 2 },
        false,
        1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  {}, {5.f}},
            ngraph::element::i8,
            { {},  {}, {} },
            { 10.f, 5.f, 2.f, 4.f, 3.f, 12.f, 8.f, 14.f }
        },
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  { }, { }},
            ngraph::element::f32,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            { -2.f, -1.f, -0.4f, -0.8f, -0.6f, -2.4f, -1.6f, -2.8f },
            "Subtract"
        },
        ""
    },

    // constant input: Add -> Subtract
    {
        ngraph::element::f32,
        ngraph::Shape{1, 2, 2, 2},
        false,
        0,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::i8,
            { {},  {}, {}},
            ngraph::element::i8,
            { {ngraph::element::f32},  {}, { 5.f } },
            { 10.f, 5.f, 2.f, 4.f, 3.f, 12.f, 8.f, 14.f }
        },
        {
            ngraph::element::i8,
            { {ngraph::element::f32},  {}, {} },
            ngraph::element::f32,
            { {},  {}, { }},

            { {},  {}, {5.f} },
            { -2.f, -1.f, -0.4f, -0.8f, -0.6f, -2.4f, -1.6f, -2.8f },
            "Subtract"
        },
        "",
    },
    // convolution before FQ (choose that branch)
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { 10.f }},
            ngraph::element::u8,
            { {ngraph::element::f32},  { 3.f }, { 5.f } },
            {}
        },
        {
            ngraph::element::u8,
            { {},  {}, {} },
            ngraph::element::u8,
            { {ngraph::element::f32},  { 17.f }, { 0.5f }},
            { {},  {}, {10.f} },
            {}
        },
        "convolution"
    },
    // group convolution before FQ (choose that branch)
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        -1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            { {ngraph::element::f32},  { 7.f }, { 10.f }},
            ngraph::element::u8,
            { {ngraph::element::f32},  { 3.f }, { 5.f } },
            {}
        },
        {
            ngraph::element::u8,
            { {},  {}, {} },
            ngraph::element::u8,
            { {ngraph::element::f32},  { 17.f }, { 0.5f }},
            { {},  {}, {10.f} },
            {}
        },
        "group_convolution"
    },

    // Actual:
    //
    // Parameter          Parameter Constant
    //  |U8                 |U8      |U8
    //  |                   |        |
    // Convert Constant    Convert  Convert
    //  \FP32  /FP32        \FP32   /FP32
    //   \    /              \     /
    //  Subtract  Constant  Subtract  Constant
    //     \FP32   /FP32       \FP32  /FP32
    //      \     /             \    /
    //      Multiply           Multiply
    //             \FP32      /FP32
    //              \        /
    //                 Add
    // Transformed:
    //
    // Parameter
    //   |U8
    //   |
    // Convert  Constant
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //      Multiply
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        1,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                {7.f},
                { 10.f }
            },
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {3.f}, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true },
                { 5.f }
            },
            {10.f}
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32}, {}, {}},
            ngraph::element::u8,
            { },
            { {},  {}, {10.f} },
            {3.5f},
            "Subtract"
        },
        ""
    },

    // Actual:
    //
    // Constant Constant   Parameter
    //  |U8      |U8        |U8
    //  |        |          |
    // Convert Convert    Convert  Constant
    //  \FP32  /FP32        \FP32   /FP32
    //   \    /              \     /
    //  Subtract  Constant  Subtract  Constant
    //     \FP32   /FP32       \FP32  /FP32
    //      \     /             \    /
    //      Multiply           Multiply
    //             \FP32      /FP32
    //              \        /
    //                 Add
    // Transformed:
    //
    // Parameter
    //   |U8
    //   |
    // Convert  Constant
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //      Multiply
    {
        ngraph::element::f32,
        ngraph::Shape{1, 4, 16, 16},
        false,
        0,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {7.f}, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true },
                { 10.f }
            },
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { 3.f },
                { 5.f }
            },
            { 10.f }
        },
        {
            ngraph::element::u8,
            { {ngraph::element::f32}, {}, {}},
            ngraph::element::u8,
            { },
            { {},  {}, { 5.f } },
            { -3.f },
            "Subtract"
        },
        ""
    }
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    AddTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecision),
        ::testing::ValuesIn(addTransformationTestValues)),
    AddTransformation::getTestCaseName);
