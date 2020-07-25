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
#include <transformations/low_precision/relu.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/relu_function.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

// #include <ngraph/pass/visualize_tree.hpp>

namespace {

using namespace testing;
using namespace ngraph::pass;

class ReluTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::Shape shape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

class ReluTransformationParams {
public:
    ngraph::Shape shape;
    ReluTransformationTestValues testValues;
};

class ReluTransformation : public LayerTransformation, public testing::WithParamInterface<ReluTransformationTestValues> {
public:
    void SetUp() override {
        const ReluTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::ReluFunction::getOriginal(
            testValues.shape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        // VisualizeTree("C:\\Projects\\temp\\test.actual").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::ReluTransformation, ngraph::opset1::MatMul>(testValues.params);
        transformer.transform(actualFunction);

        // VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        referenceFunction = ngraph::builder::subgraph::ReluFunction::getReference(
            testValues.shape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);

        // VisualizeTree("C:\\Projects\\temp\\test.reference").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ referenceFunction });
    }

    static std::string getTestCaseName(testing::TestParamInfo<ReluTransformationTestValues> obj) {
        const ReluTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result << testValues.shape;
        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function> actualFunction;
    std::shared_ptr<ngraph::Function> referenceFunction;
};

TEST_P(ReluTransformation, CompareFunctions) {
     InitNodeInfo().run_on_function(actualFunction);
     actualFunction->validate_nodes_and_infer_types();
     auto res = compare_functions(referenceFunction, actualFunction);
     ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 16, 16 }
};

const std::vector<ReluTransformationTestValues> testValues = {
    {
        ngraph::Shape({ 1, 3, 16, 16 }),
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        }
    },
    // {
    //    ngraph::Shape({ 1, 3, 16, 16 }),
    //    LayerTransformation::createParamsU8I8(),
    //    {
    //        ngraph::element::u8,
    //        {{ngraph::element::f32}, { 128 }, {0.1f}}
    //    },
    //    {
    //        ngraph::element::u8,
    //        {{}, { 128 }, {}},
    //        ngraph::element::f32,
    //        {{}, {}, {0.1f}}
    //    }
    // }
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    ReluTransformation,
    ::testing::ValuesIn(testValues),
    ReluTransformation::getTestCaseName);

} // namespace
