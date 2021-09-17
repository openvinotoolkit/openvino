// Copyright (C) 2018-2021 Intel Corporation
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

#include <low_precision/fold_convert.hpp>
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class FoldConvertTransformationTestValues {
public:
    TestTransformationParams params;
    ngraph::element::Type precision;
    ngraph::builder::subgraph::DequantizationOperations dequantizationActual;
    ngraph::builder::subgraph::DequantizationOperations dequantizationExpected;
};

typedef std::tuple<
    ngraph::PartialShape,
    FoldConvertTransformationTestValues> FoldConvertTransformationParams;

class FoldConvertTransformation : public LayerTransformation, public testing::WithParamInterface<FoldConvertTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const FoldConvertTransformationTestValues testValues = std::get<1>(GetParam());

        const auto createFunction = [](
            const ngraph::element::Type precision,
            const ngraph::PartialShape& inputShape,
            const ngraph::builder::subgraph::DequantizationOperations& dequantization) -> std::shared_ptr<ngraph::Function> {
            auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
            std::shared_ptr<ngraph::Node> output = makeDequantization(input, dequantization);
            output->set_friendly_name("output");

            return std::make_shared<ngraph::Function>(
                ngraph::ResultVector{ std::make_shared<ngraph::opset1::Result>(output) },
                ngraph::ParameterVector{ input },
                "FoldConvertTransformation");
        };
        actualFunction = createFunction(testValues.precision, inputShape, testValues.dequantizationActual);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::FoldConvertTransformation, ngraph::opset1::Add>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = createFunction(testValues.precision, inputShape, testValues.dequantizationExpected);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FoldConvertTransformationParams> obj) {
        const ngraph::PartialShape inputShape = std::get<0>(obj.param);
        const FoldConvertTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            testValues.precision << "_" <<
            inputShape << "_" <<
            testValues.dequantizationActual << "_" <<
            testValues.dequantizationExpected;
        return result.str();
    }
};

TEST_P(FoldConvertTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::PartialShape> inputShapes = {
    {1, 4, 16, 16},
    {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
    PartialShape::dynamic()
};

const std::vector<FoldConvertTransformationTestValues> testValues = {
    // Actual:
    //
    // Parameter Constant
    //  |U8      |U8
    //  |        |
    // Convert  Convert
    //  \FP32   /FP32
    //   \     /
    //  Subtract   Constant
    //     \FP32    /FP32
    //      \      /
    //      Multiply
    //
    // Transformed:
    //
    // Parameter
    //   |U8
    //   |
    // Convert   Constant
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //       Multiply
    {
        LayerTransformation::createParamsU8I8(),
        ngraph::element::f32,
        {
            {ngraph::element::f32},
            { {7.f}, ngraph::element::f32, {}, false, 1, ngraph::element::u8, true },
            { 10.f }
        },
        {
            {ngraph::element::f32},
            { {7.f}, ngraph::element::f32, {}, false, 1 },
            { 10.f }
        }
    },

    // Actual:
    //
    // Constant Parameter
    //  |U8      |U8
    //  |        |
    // Convert  Convert
    //  \FP32   /FP32
    //   \     /
    //  Subtract   Constant
    //     \FP32    /FP32
    //      \      /
    //      Multiply
    //
    // Transformed:
    //
    //           Parameter
    //            |U8
    //            |
    // Constant  Convert
    //   \FP32   /FP32
    //    \     /
    //   Subtract    Constant
    //      \FP32    /FP32
    //       \      /
    //       Multiply
    {
        LayerTransformation::createParamsU8I8(),
        ngraph::element::f32,
        {
            {ngraph::element::f32},
            { {7.f}, ngraph::element::f32, {}, false, 0, ngraph::element::u8, true },
            { 10.f }
        },
        {
            {ngraph::element::f32},
            { {7.f}, ngraph::element::f32, {}, false, 0 },
            { 10.f }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FoldConvertTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues)),
    FoldConvertTransformation::getTestCaseName);
} // namespace
