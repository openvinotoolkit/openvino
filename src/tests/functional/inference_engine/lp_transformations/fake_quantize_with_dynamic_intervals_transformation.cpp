// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <low_precision/fake_quantize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class FakeQuantizeWithDynamicIntervalsTransformationTestValues {
public:
    TestTransformationParams params;
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
            std::dynamic_pointer_cast<ngraph::Node>(std::make_shared<opset1::Constant>(constantPresition, constantShape, low)) :
            std::make_shared<ngraph::opset1::Parameter>(constantPresition, constantShape);

        const auto inputHigh = inputLowConst ?
            std::dynamic_pointer_cast<ngraph::Node>(std::make_shared<opset1::Constant>(constantPresition, constantShape, high)) :
            std::make_shared<ngraph::opset1::Parameter>(constantPresition, constantShape);

        const auto outputLow = outputLowConst ?
            std::dynamic_pointer_cast<ngraph::Node>(std::make_shared<opset1::Constant>(constantPresition, constantShape, low)) :
            std::make_shared<ngraph::opset1::Parameter>(constantPresition, constantShape);

        const auto outputHigh = outputHighConst ?
            std::dynamic_pointer_cast<ngraph::Node>(std::make_shared<opset1::Constant>(constantPresition, constantShape, high)) :
            std::make_shared<ngraph::opset1::Parameter>(constantPresition, constantShape);

        const auto levels = 256ul;

        auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(input, inputLow, inputHigh, outputLow, outputHigh, levels);
        fakeQuantize->set_friendly_name("fakeQuantize");

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };

        ngraph::ParameterVector inputs{ input };
        if (as_type_ptr<ngraph::opset1::Parameter>(inputLow)) {
            inputs.push_back(as_type_ptr<ngraph::opset1::Parameter>(inputLow));
        }
        if (as_type_ptr<ngraph::opset1::Parameter>(inputHigh)) {
            inputs.push_back(as_type_ptr<ngraph::opset1::Parameter>(inputHigh));
        }
        if (as_type_ptr<ngraph::opset1::Parameter>(outputLow)) {
            inputs.push_back(as_type_ptr<ngraph::opset1::Parameter>(outputLow));
        }
        if (as_type_ptr<ngraph::opset1::Parameter>(outputHigh)) {
            inputs.push_back(as_type_ptr<ngraph::opset1::Parameter>(outputHigh));
        }

        return std::make_shared<ngraph::Function>(results, inputs, "FakeQuantizeWithDynamicIntervalsTransformation");
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
    { LayerTransformation::createParamsU8I8(), true, false, false, false },
    { LayerTransformation::createParamsU8I8(), false, false, false, false }
};

const std::vector<ngraph::Shape> shapes = { { 1, 32, 72, 48 } };

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FakeQuantizeWithDynamicIntervalsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(fakeQuantizeTransformationTestValues)),
    FakeQuantizeWithDynamicIntervalsTransformation::getTestCaseName);
