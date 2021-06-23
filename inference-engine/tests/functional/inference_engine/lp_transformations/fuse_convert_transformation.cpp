// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include "low_precision/fuse_convert.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/fuse_convert_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class FuseConvertTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    ngraph::Shape inputShape;
    bool constInput;
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

class FuseConvertTransformation : public LayerTransformation, public testing::WithParamInterface<FuseConvertTransformationTestValues> {
public:
    void SetUp() override {
        const FuseConvertTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::FuseConvertFunction::get(
                testValues.inputShape,
                testValues.actual.inputPrecision,
                testValues.actual.dequantization,
                testValues.constInput);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::FuseConvertTransformation, ngraph::opset1::Convert>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::FuseConvertFunction::get(
                testValues.inputShape,
                testValues.expected.inputPrecision,
                testValues.expected.dequantization,
                testValues.constInput);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FuseConvertTransformationTestValues> obj) {
        const FuseConvertTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
               testValues.inputShape << "_" <<
               testValues.actual.inputPrecision << "_" <<
               testValues.actual.dequantization << "_" <<
               testValues.constInput;
        return result.str();
    }
};

const std::vector<FuseConvertTransformationTestValues> testValues = {
    // fuse to subtract
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32 },
                {1.f},
                {0.45f}
            }
        },
        {
            ngraph::element::u8,
            {
                {},
                DequantizationOperations::Subtract({1.f}, ngraph::element::f32).setConstantPrecision(ngraph::element::f32),
                {0.45f}
            }
        }
    },
    // fuse to multiply
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32 },
                {},
                {0.45f}
            }
        },
        {
            ngraph::element::u8,
            {
                {},
                {},
                DequantizationOperations::Multiply({0.45f}, ngraph::element::f32).setConstantPrecision(ngraph::element::f32)
            }
        }
    },
    // fuse to const
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        true,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32 },
                {1.f},
                {0.45f}
            }
        },
        {
            ngraph::element::f32,
            {
                {},
                {1.f},
                {0.45f}
            }
        }
    },
    // Convert with unexpected precision
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {{ ngraph::element::i32 }, {}, {3.f}}
        },
        {
            ngraph::element::f32,
            {{ ngraph::element::i32 }, {}, {3.f}}
        }
    },
};

TEST_P(FuseConvertTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FuseConvertTransformation,
    ::testing::ValuesIn(testValues),
    FuseConvertTransformation::getTestCaseName);
