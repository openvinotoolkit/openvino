// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/round_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "low_precision/network_helper.hpp"


namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class RoundTestValues {
public:
    ngraph::element::Type inputPrecision;
    ngraph::Shape inputShape;
    ngraph::builder::subgraph::DequantizationOperations actualDequantization;
    ngraph::builder::subgraph::DequantizationOperations referenceDequantization;
};



class RoundTransformation : public LayerTransformation, public testing::WithParamInterface<RoundTestValues> {
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
        const auto roundedConst = ngraph::pass::low_precision::NetworkHelper::round(
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

    static std::string getTestCaseName(testing::TestParamInfo<RoundTestValues> obj) {
        const auto testValues = obj.param;

        std::ostringstream result;
        result << testValues.inputPrecision << "_"
               << testValues.actualDequantization << "_"
               << testValues.referenceDequantization;
        return result.str();
    }
};

std::vector<RoundTestValues> testValues = {
    {
        ngraph::element::u8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { 125.5f }, { 0.1f } },
        { {}, { { 126.f }, ngraph::element::f32 }, { 0.1f } }
    },
    {
        ngraph::element::u8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { { 128.3f, 64.5f, 31.7f } }, { { 0.1f, 0.1f, 0.1f } } },
        { {}, { { 128.f, 65.f, 32.f }, ngraph::element::f32 }, { { 0.1f, 0.1f, 0.1f } } }
    },
    {
        ngraph::element::i8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { 126.6f }, { 0.1f } },
        { {}, { { 127.f }, ngraph::element::f32 }, { 0.1f } }
    },
    {
        ngraph::element::i8,
        ngraph::Shape{ 1, 3, 16, 16 },
        { { ngraph::element::f32 }, { { 126.5f, 32.25f, -127.5f } }, { { 0.1f, 0.1f, 0.1f } } },
        { {}, { { 127.f, 32.f, -128.f }, ngraph::element::f32 }, { { 0.1f, 0.1f, 0.1f } } }
    },
};

TEST_P(RoundTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(
    LPT,
    RoundTransformation,
    ::testing::ValuesIn(testValues),
    RoundTransformation::getTestCaseName);
} // namespace
