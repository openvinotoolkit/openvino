// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/transformer.hpp>
#include <low_precision/mat_mul.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/mat_mul_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace {

using namespace testing;
using namespace ngraph::pass;

class SeparateInStandaloneBranchTransformationTestValues {
public:
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::element::Type precisionBefore;
    ngraph::builder::subgraph::DequantizationOperations dequantization;
};

inline std::ostream& operator << (std::ostream& out, const SeparateInStandaloneBranchTransformationTestValues& testValues) {
    return out << "_" << testValues.dequantization;
}

typedef std::tuple<
    ngraph::Shape,
    SeparateInStandaloneBranchTransformationTestValues> SeparateInStandaloneBranchTransformationParams;

class SeparateInStandaloneBranchTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<SeparateInStandaloneBranchTransformationParams> {
public:
    void SetUp() override {
        const ngraph::Shape shape = std::get<0>(GetParam());
        const SeparateInStandaloneBranchTransformationTestValues testValues = std::get<1>(GetParam());

        const auto createActualFunction = [](
            const ngraph::element::Type precision,
            const ngraph::Shape& inputShape,
            const ngraph::builder::subgraph::DequantizationOperations& dequantizations) -> std::shared_ptr<ngraph::Function> {
            const std::shared_ptr<ngraph::opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
            const auto relu = std::make_shared<ngraph::opset1::Relu>(input);
            const auto dequantizationsNode = ngraph::builder::subgraph::makeDequantization(relu, dequantizations);

            const std::shared_ptr<ngraph::Node> reshape1 = std::make_shared<ngraph::opset1::Reshape>(
                dequantizationsNode,
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::i32, ngraph::Shape{ 2 }, std::vector<double>({0, -1})),
                true);
            reshape1->set_friendly_name("reshape1");

            const std::shared_ptr<ngraph::Node> reshape2 = std::make_shared<ngraph::opset1::Reshape>(
                dequantizationsNode,
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::i32, ngraph::Shape{ 2 }, std::vector<double>({0, -1})),
                true);
            reshape2->set_friendly_name("reshape2");

            return std::make_shared<ngraph::Function>(
                ngraph::ResultVector{
                    std::make_shared<ngraph::opset1::Result>(reshape1),
                    std::make_shared<ngraph::opset1::Result>(reshape2)
                },
                std::vector<std::shared_ptr<ngraph::op::Parameter>> { input },
                "SeparateInStandaloneBranchTransformation");
        };
        actualFunction = createActualFunction(testValues.precisionBefore, shape, testValues.dequantization);
        const auto result = actualFunction->get_results()[0];
        ngraph::pass::low_precision::NetworkHelper::separateInStandaloneBranch(result->get_input_node_shared_ptr(0));

        const auto createReferenceFunction = [](
            const ngraph::element::Type precision,
            const ngraph::Shape& inputShape,
            const ngraph::builder::subgraph::DequantizationOperations& dequantization) -> std::shared_ptr<ngraph::Function> {
            const std::shared_ptr<ngraph::opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
            const auto relu = std::make_shared<ngraph::opset1::Relu>(input);

            const std::shared_ptr<ngraph::Node> reshape1 = std::make_shared<ngraph::opset1::Reshape>(
                ngraph::builder::subgraph::makeDequantization(relu, dequantization),
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::i32, ngraph::Shape{ 2 }, std::vector<double>({0, -1})),
                true);
            reshape1->set_friendly_name("reshape1");

            const std::shared_ptr<ngraph::Node> reshape2 = std::make_shared<ngraph::opset1::Reshape>(
                ngraph::builder::subgraph::makeDequantization(relu, dequantization),
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::i32, ngraph::Shape{ 2 }, std::vector<double>({0, -1})),
                true);
            reshape2->set_friendly_name("reshape2");

            return std::make_shared<ngraph::Function>(
                ngraph::ResultVector{
                    std::make_shared<ngraph::opset1::Result>(reshape1),
                    std::make_shared<ngraph::opset1::Result>(reshape2)
                },
                std::vector<std::shared_ptr<ngraph::op::Parameter>> { input },
                "SeparateInStandaloneBranchTransformation");
        };
        referenceFunction = createReferenceFunction(testValues.precisionBefore, shape, testValues.dequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SeparateInStandaloneBranchTransformationParams> obj) {
        ngraph::Shape shapes;
        SeparateInStandaloneBranchTransformationTestValues testValues;
        std::tie(shapes, testValues) = obj.param;

        std::stringstream ss;
        ss << shapes << "_" << "_" << testValues;
        return ss.str();
    }
};

TEST_P(SeparateInStandaloneBranchTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 },
    { 4, 3, 9, 9 }
};

std::vector<SeparateInStandaloneBranchTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(true),
        ngraph::element::u8,
        { ngraph::element::f32, { 127.f }, { 0.02f } }
    },
    {
        LayerTransformation::createParamsU8U8(),
        ngraph::element::u8,
        { ngraph::element::f32, { 127.f }, {} }
    },
    {
        LayerTransformation::createParamsU8U8().setSupportAsymmetricQuantization(true),
        ngraph::element::u8,
        {
            ngraph::element::f32,
            { {127.f}, ngraph::element::f32, {}, true, 1ul, ngraph::element::u8, true},
            { 0.02f }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    SeparateInStandaloneBranchTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    SeparateInStandaloneBranchTransformation::getTestCaseName);

} // namespace
