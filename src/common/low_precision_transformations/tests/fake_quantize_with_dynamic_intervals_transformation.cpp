// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "low_precision/fake_quantize.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

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
    ov::element::Type,
    ov::Shape,
    FakeQuantizeWithDynamicIntervalsTransformationTestValues> FakeQuantizeTransformationParams;

class FakeQuantizeWithDynamicIntervalsTransformation : public LayerTransformation, public testing::WithParamInterface<FakeQuantizeTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::Shape shape = std::get<1>(GetParam());
        const FakeQuantizeWithDynamicIntervalsTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = get(precision, shape, testValues.inputLowConst, testValues.inpuHighConst, testValues.outputLowConst, testValues.outputHighConst);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::FakeQuantizeTransformation, ov::op::v0::FakeQuantize>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = get(precision, shape, testValues.inputLowConst, testValues.inpuHighConst, testValues.outputLowConst, testValues.outputHighConst);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeTransformationParams> obj) {
        ov::element::Type precision;
        ov::Shape shape;
        FakeQuantizeWithDynamicIntervalsTransformationTestValues testValues;
        std::tie(precision, shape, testValues) = obj.param;

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << testValues;
        return result.str();
    }

private:
    std::shared_ptr<ov::Model> get(
        ov::element::Type precision,
        ov::Shape inputShape,
        const bool inputLowConst,
        const bool inpuHighConst,
        const bool outputLowConst,
        const bool outputHighConst) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
        input->set_friendly_name("input");

        const auto constantPresition = element::f32;
        const auto constantShape = Shape{ 1, 1, 1, 1 };
        const std::vector<float> low = { 0.f };
        const std::vector<float> high = { 1.f };

        const auto inputLow = inputLowConst ?
            std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Constant>(constantPresition, constantShape, low)) :
            std::make_shared<ov::op::v0::Parameter>(constantPresition, constantShape);

        const auto inputHigh = inputLowConst ?
            std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Constant>(constantPresition, constantShape, high)) :
            std::make_shared<ov::op::v0::Parameter>(constantPresition, constantShape);

        const auto outputLow = outputLowConst ?
            std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Constant>(constantPresition, constantShape, low)) :
            std::make_shared<ov::op::v0::Parameter>(constantPresition, constantShape);

        const auto outputHigh = outputHighConst ?
            std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Constant>(constantPresition, constantShape, high)) :
            std::make_shared<ov::op::v0::Parameter>(constantPresition, constantShape);

        const auto levels = 256ul;

        auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(input, inputLow, inputHigh, outputLow, outputHigh, levels);
        fakeQuantize->set_friendly_name("fakeQuantize");

        ov::ResultVector results{ std::make_shared<ov::op::v0::Result>(fakeQuantize) };

        ov::ParameterVector inputs{ input };
        if (as_type_ptr<ov::op::v0::Parameter>(inputLow)) {
            inputs.push_back(as_type_ptr<ov::op::v0::Parameter>(inputLow));
        }
        if (as_type_ptr<ov::op::v0::Parameter>(inputHigh)) {
            inputs.push_back(as_type_ptr<ov::op::v0::Parameter>(inputHigh));
        }
        if (as_type_ptr<ov::op::v0::Parameter>(outputLow)) {
            inputs.push_back(as_type_ptr<ov::op::v0::Parameter>(outputLow));
        }
        if (as_type_ptr<ov::op::v0::Parameter>(outputHigh)) {
            inputs.push_back(as_type_ptr<ov::op::v0::Parameter>(outputHigh));
        }

        return std::make_shared<ov::Model>(results, inputs, "FakeQuantizeWithDynamicIntervalsTransformation");
    }
};

TEST_P(FakeQuantizeWithDynamicIntervalsTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    ov::element::i32,
    ov::element::f16
};

const std::vector<FakeQuantizeWithDynamicIntervalsTransformationTestValues> fakeQuantizeTransformationTestValues = {
    { LayerTransformation::createParamsU8I8(), false, false, false, true },
    { LayerTransformation::createParamsU8I8(), true, false, false, false },
    { LayerTransformation::createParamsU8I8(), false, false, false, false }
};

const std::vector<ov::Shape> shapes = { { 1, 32, 72, 48 } };

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    FakeQuantizeWithDynamicIntervalsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(fakeQuantizeTransformationTestValues)),
    FakeQuantizeWithDynamicIntervalsTransformation::getTestCaseName);
