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
#include <legacy/transformations/convert_opset1_to_legacy/convert_mul_or_add_finally.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

#include "lpt_ngraph_functions/convert_mul_or_add_finally_with_dequantization_function.hpp"

using namespace testing;
using namespace ngraph::pass;

namespace {

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
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

class ConvertMulOrAddFinallyTransformationWithDequantizationTestValues {
public:
    std::vector<float> multiplyConstValues;
    ngraph::Shape inputShape;
    ngraph::element::Type inputPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
};

using TestValuesType = ConvertMulOrAddFinallyTransformationWithDequantizationTestValues;

class ConvertMulOrAddFinallyTransformationWithDequantization : public LayerTransformation, public testing::WithParamInterface<TestValuesType> {
public:
    void SetUp() override {
        using namespace ngraph::builder::subgraph;
        const ConvertMulOrAddFinallyTransformationWithDequantizationTestValues testValues = GetParam();

        actualFunction = ConvertMulOrAddWithDequantizationFunction::getOriginal(testValues.inputShape,
                                                                                testValues.inputPrecision,
                                                                                testValues.multiplyConstValues);

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertMulOrAddFinally>();
        manager.register_pass<ngraph::pass::ConstantFolding>();

        manager.run_passes(actualFunction);

        referenceFunction = ConvertMulOrAddWithDequantizationFunction::getReference(testValues.inputShape,
                                                                                    testValues.inputPrecision,
                                                                                    testValues.multiplyConstValues);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvertMulOrAddFinallyTransformationWithDequantizationTestValues> obj) {
        const ConvertMulOrAddFinallyTransformationWithDequantizationTestValues testValues = obj.param;
        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(testValues.inputPrecision, testValues.inputShape, testValues.params) << "_" <<
            testValues.multiplyConstValues;
        return result.str();
    }
};

TEST_P(ConvertMulOrAddFinallyTransformationWithDequantization, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

std::vector<ConvertMulOrAddFinallyTransformationWithDequantizationTestValues> testValues = {
    {
        { -1.0 },
        { 1, 1000 },
        ngraph::element::f32,
        LayerTransformation::createParamsU8I8()
    },
    {
        { 128.0 },
        { 1, 10 },
        ngraph::element::f32,
        LayerTransformation::createParamsU8I8()
    },
    {
        { -64.5 },
        { 1, 10 },
        ngraph::element::i8,
        LayerTransformation::createParamsU8I8()
    },
    {
        { 1.2 },
        { 1, 100 },
        ngraph::element::u8,
        LayerTransformation::createParamsI8I8()
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConvertMulOrAddFinallyTransformationWithDequantization,
    ::testing::ValuesIn(testValues),
    ConvertMulOrAddFinallyTransformationWithDequantization::getTestCaseName);
} // namespace
