// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

#include <transformations/low_precision/add.hpp>
#include "ngraph_functions/low_precision_transformations/add_function.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

using namespace testing;
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
    };

    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    bool broadcast;
    int constInput;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

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

class AddTransformation : public LayerTransformation, public testing::WithParamInterface<AddTransformationTestValues> {
public:
    void SetUp() override {
        const AddTransformationTestValues testValues = GetParam();

        actualFunction = AddFunction::getOriginal(
            testValues.precision,
            testValues.inputShape,
            testValues.broadcast,
            testValues.params,
            testValues.actual.precision1,
            testValues.actual.dequantization1,
            testValues.actual.precision2,
            testValues.actual.dequantization2,
            testValues.constInput,
            testValues.actual.constValues);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::AddTransformation, ngraph::opset1::Add>(
            low_precision::LayerTransformation::Params(testValues.params));
        transform.transform(actualFunction);

        referenceFunction = AddFunction::getReference(
            testValues.precision,
            testValues.inputShape,
            testValues.broadcast,
            testValues.params,
            testValues.expected.precision1,
            testValues.expected.dequantization1,
            testValues.expected.precision2,
            testValues.expected.dequantization2,
            testValues.expected.dequantizationAfter,
            testValues.constInput,
            testValues.expected.constValues);
    }

    static std::string getTestCaseName(testing::TestParamInfo<AddTransformationTestValues> obj) {
        const AddTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.precision << "_" <<
            testValues.inputShape << "_" <<
            testValues.broadcast << "_" <<
            testValues.actual.precision1 << "_" <<
            testValues.actual.dequantization1 << "_" <<
            testValues.actual.precision2 << "_" <<
            testValues.actual.dequantization2 << "_" <<
            testValues.constInput << "_" <<
            testValues.actual.constValues;
        return result.str();
    }
};

TEST_P(AddTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<AddTransformationTestValues> addTransformationTestValues = {
    // U8
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
    },

    // constant input
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
            ngraph::element::i8,
            { {},  {}, {} },
            { {},  {}, {5.f} },
            { 2.f, 1.f, 0.4f, 0.8f, 0.6f, 2.4f, 1.6f, 2.8f }
        },
    },
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
            { {},  { }, { }},
            ngraph::element::i8,
            { {ngraph::element::f32},  {}, {} },
            { {},  {}, {5.f} },
            { 2.f, 1.f, 0.4f, 0.8f, 0.6f, 2.4f, 1.6f, 2.8f }
        },
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    AddTransformation,
    ::testing::ValuesIn(addTransformationTestValues),
    AddTransformation::getTestCaseName);
