// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/round_with_tolerance_function.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "low_precision/network_helper.hpp"


namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class RoundWithToleranceTestValues {
public:
    ngraph::element::Type inputPrecision;
    ngraph::Shape inputShape;
    ngraph::builder::subgraph::DequantizationOperations actualDequantization;
    ngraph::builder::subgraph::DequantizationOperations referenceDequantization;
};



class RoundWithToleranceTransformation : public LayerTransformation, public testing::WithParamInterface<RoundWithToleranceTestValues> {
public:
    void SetUp() override {
        const auto testValues = this->GetParam();

        actualFunction = ngraph::builder::subgraph::RoundWithToleranceFunction::getOriginal(
            testValues.inputPrecision,
            testValues.inputShape,
            testValues.actualDequantization);

        const auto lastNode = actualFunction->get_output_op(0)->get_input_node_shared_ptr(0);
        const auto dequantization = ngraph::pass::low_precision::NetworkHelper::getDequantization(lastNode);
        const auto subtractConstant = dequantization.subtract->get_input_node_shared_ptr(1);
        const auto roundedConst = ngraph::pass::low_precision::NetworkHelper::roundWithTolerance(
            subtractConstant,
            testValues.inputPrecision);

        if (roundedConst->get_element_type() == testValues.inputPrecision) {
            const auto replacement = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(dequantization.data, roundedConst);
            ngraph::pass::low_precision::NetworkHelper::copyInfo(dequantization.subtract, replacement);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(replacement, dequantization.convert->get_element_type());
            replace_node(dequantization.subtract, replacement);
        }

        referenceFunction = ngraph::builder::subgraph::RoundWithToleranceFunction::getReference(
            testValues.inputPrecision,
            testValues.inputShape,
            testValues.referenceDequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<RoundWithToleranceTestValues> obj) {
        const auto testValues = obj.param;

        std::ostringstream result;
        result << testValues.inputPrecision << "_"
               << testValues.actualDequantization << "_"
               << testValues.referenceDequantization;
        return result.str();
    }
};

std::vector<RoundWithToleranceTestValues> testValues = {
    // U8: acceptable tolerance, rounded sub constant
    {
        ngraph::element::u8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { 128.34f }, { 0.1f } },
        { {}, { { 128.f }, ngraph::element::f32 }, { 0.1f } }
    },
    // acceptable tolerance, rounded sub constant
    {
        ngraph::element::u8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { { 128.34f, 64.17f, 31.78f } }, { { 0.1f, 0.1f, 0.1f } } },
        { {}, { { 128.f, 64.f, 32.f }, ngraph::element::f32 }, { { 0.1f, 0.1f, 0.1f } } }
    },
    // original constant
    {
        ngraph::element::u8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { 128.36f }, { 0.1f } },
        { { ngraph::element::f32 }, { 128.36f }, { 0.1f } }
    },
    // original constant
    {
        ngraph::element::u8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { { 128.36f, 64.52f, 31.64f } }, { { 0.1f, 0.1f, 0.1f } } },
        { { ngraph::element::f32 }, { { 128.36f, 64.52f, 31.64f } }, { { 0.1f, 0.1f, 0.1f } } }
    },
    // subtract values > 256, original constant
    {
        ngraph::element::u8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { 300.f }, { 0.1f } },
        { { ngraph::element::f32 }, { 300.f }, { 0.1f } }
    },
    // subtract values < 0, original constant
    {
        ngraph::element::u8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { -128.f }, { 0.1f } },
        { { ngraph::element::f32 }, { -128.f }, { 0.1f } }
    },
    // I8: acceptable tolerance, rounded sub constant
    {
        ngraph::element::i8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { 126.66f }, { 0.1f } },
        { {}, { { 127.f }, ngraph::element::f32 }, { 0.1f } }
    },
    // acceptable tolerance, rounded sub constant
    {
        ngraph::element::i8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { { 127.34f, 32.25f, -128.f } }, { { 0.1f, 0.1f, 0.1f } } },
        { {}, { { 127.f, 32.f, -128.f }, ngraph::element::f32 }, { { 0.1f, 0.1f, 0.1f } } }
    },
    // original constant
    {
        ngraph::element::i8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { 64.6f }, { 0.1f } },
        { { ngraph::element::f32 }, { 64.6f }, { 0.1f } }
    },
    // original constant
    {
        ngraph::element::i8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { { 128.36f, 64.52f, -52.36f } }, { { 0.1f, 0.1f, 0.1f } } },
        { { ngraph::element::f32 }, { { 128.36f, 64.52f, -52.36f } }, { { 0.1f, 0.1f, 0.1f } } }
    },
    // values > 127, original constant
    {
        ngraph::element::i8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { 129.f }, { 0.1f } },
        { { ngraph::element::f32 }, { 129.f }, { 0.1f } }
    },
    // values < -128, original constant
    {
        ngraph::element::i8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { -129.f }, { 0.1f } },
        { { ngraph::element::f32 }, { -129.f }, { 0.1f } }
    },
};

TEST_P(RoundWithToleranceTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_CASE_P(
    LPT,
    RoundWithToleranceTransformation,
    ::testing::ValuesIn(testValues),
    RoundWithToleranceTransformation::getTestCaseName);
} // namespace
