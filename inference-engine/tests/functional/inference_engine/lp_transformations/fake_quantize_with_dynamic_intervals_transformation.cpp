// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/low_precision/fake_quantize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FakeQuantizeWithDynamicIntervalsTransformationTestValues {
public:
    low_precision::LayerTransformation::Params params;
    bool inputLowConst;
    bool inpuHighConst;
    bool outputLowConst;
    bool outputHighConst;
};

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

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeWithDynamicIntervalsTransformationTestValues& testValue) {
    return out << "_" <<
        testValue.inputLowConst << "_" <<
        testValue.inpuHighConst << "_" <<
        testValue.outputLowConst << "_" <<
        testValue.outputHighConst;
}

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    FakeQuantizeWithDynamicIntervalsTransformationTestValues> FakeQuantizeTransformationParams;

class FakeQuantizeWithDynamicIntervalsTransformation : public LayerTransformation, public testing::WithParamInterface<FakeQuantizeTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const FakeQuantizeWithDynamicIntervalsTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = get(precision, shape, testValues.inputLowConst, testValues.inpuHighConst, testValues.outputLowConst, testValues.outputHighConst);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::FakeQuantizeTransformation, ngraph::opset1::FakeQuantize>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = get(precision, shape, testValues.inputLowConst, testValues.inpuHighConst, testValues.outputLowConst, testValues.outputHighConst);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        FakeQuantizeWithDynamicIntervalsTransformationTestValues testValues;
        std::tie(precision, shape, testValues) = obj.param;

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << testValues;
        return result.str();
    }

private:
    std::shared_ptr<ngraph::Node> makeTranspose(const ngraph::element::Type precision, const ngraph::Shape shape, const std::vector<float> low) {
        const auto constant = std::make_shared<opset1::Constant>(precision, shape, low);
        const auto transposeConstant = std::make_shared<opset1::Constant>(element::u32, Shape{ 4 }, std::vector<size_t>{1, 2, 3, 0});
        const auto transpose = std::make_shared<opset1::Transpose>(constant, transposeConstant);
        return transpose;
    }

    std::shared_ptr<ngraph::Node> makeConstant(const ngraph::element::Type precision, const ngraph::Shape shape, const std::vector<float> low) {
        return std::make_shared<opset1::Constant>(precision, shape, low);;
    }

    std::shared_ptr<ngraph::Function> get(
        ngraph::element::Type precision,
        ngraph::Shape inputShape,
        const bool inputLowConst,
        const bool inpuHighConst,
        const bool outputLowConst,
        const bool outputHighConst) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        input->set_friendly_name("input");

        const auto constantPresition = element::f32;
        const auto constantShape = Shape{ 1, 1, 1, 1 };
        const std::vector<float> low = { 0.f };
        const std::vector<float> high = { 1.f };

        const auto inputLow = inputLowConst ?
            makeConstant(constantPresition, constantShape, low) :
            makeTranspose(constantPresition, constantShape, low);

        const auto inputHigh = inputLowConst ?
            makeConstant(constantPresition, constantShape, high) :
            makeTranspose(constantPresition, constantShape, high);

        const auto outputLow = outputLowConst ?
            makeConstant(constantPresition, constantShape, low) :
            makeTranspose(constantPresition, constantShape, low);

        const auto outputHigh = outputHighConst ?
            makeConstant(constantPresition, constantShape, high) :
            makeTranspose(constantPresition, constantShape, high);

        const auto levels = 256ul;

        auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(input, inputLow, inputHigh, outputLow, outputHigh, levels);
        fakeQuantize->set_friendly_name("fakeQuantize");

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeWithDynamicIntervalsTransformation");
    }
};

TEST_P(FakeQuantizeWithDynamicIntervalsTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::i32,
    ngraph::element::f16
};

const std::vector<FakeQuantizeWithDynamicIntervalsTransformationTestValues> fakeQuantizeTransformationTestValues = {
    { LayerTransformation::createParamsU8I8(), false, false, false, true },
    { LayerTransformation::createParamsU8I8(), true, false, false, false }
};

const std::vector<ngraph::Shape> shapes = { { 1, 32, 72, 48 } };

INSTANTIATE_TEST_CASE_P(
    LPT,
    FakeQuantizeWithDynamicIntervalsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(fakeQuantizeTransformationTestValues)),
    FakeQuantizeWithDynamicIntervalsTransformation::getTestCaseName);
