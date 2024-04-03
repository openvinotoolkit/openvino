// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mish.hpp"

#include <gtest/gtest.h>

#include <random>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct MishParams {
    template <class IT>
    MishParams(const ov::PartialShape& dynamicShape,
               const ov::Shape& inputShape,
               const ov::element::Type& iType,
               const std::vector<IT>& iValues,
               const std::vector<IT>& oValues,
               const std::string& test_name = "")
        : dynamicShape(dynamicShape),
          inputShape(inputShape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(inputShape, iType, iValues)),
          refData(CreateTensor(inputShape, iType, oValues)),
          testcaseName(test_name) {}

    ov::PartialShape dynamicShape;
    ov::PartialShape inputShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferenceMishLayerTest : public testing::TestWithParam<MishParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
        auto params = GetParam();
        function = CreateFunction(params.dynamicShape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MishParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dShape=" << param.dynamicShape << "_";
        result << "iShape=" << param.inputShape << "_";
        result << "iType=" << param.inType << "_";
        if (param.testcaseName != "") {
            result << "oType=" << param.outType << "_";
            result << param.testcaseName;
        } else {
            result << "oType=" << param.outType;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape, const element::Type& input_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto Mish = std::make_shared<op::v4::Mish>(in);
        return std::make_shared<ov::Model>(NodeVector{Mish}, ParameterVector{in});
    }
};

TEST_P(ReferenceMishLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<MishParams> generateMishFloatParams(const PartialShape& dynamicShape,
                                                const Shape& staticShape,
                                                const std::string& test_name = "") {
    using T = typename element_type_traits<IN_ET>::value_type;

    // generate input tensor (with possible type conversion)
    auto staticSize = shape_size(staticShape);
    std::vector<T> expected;
    std::vector<T> input;
    {
        std::mt19937 gen{0};  // use fixed seed for reproducibility of the test
        std::normal_distribution<> d{0.0, 20.0};

        for (auto i = staticSize; i > 0; i--) {
            auto x = static_cast<T>(d(gen));
            auto y = static_cast<T>(static_cast<double>(x) * std::tanh(std::log(1.0 + std::exp(x))));
            input.push_back(x);
            expected.push_back(y);
        }
    }

    std::vector<MishParams> mishParams;

    if (test_name != "") {
        mishParams = {MishParams(dynamicShape, staticShape, IN_ET, input, expected, test_name)};
    } else {
        mishParams = {MishParams(dynamicShape, staticShape, IN_ET, input, expected)};
    }
    return mishParams;
}

std::vector<MishParams> generateMishCombinedParams() {
    const std::vector<std::vector<MishParams>> mishTypeParams{
        generateMishFloatParams<element::Type_t::f32>({2, 5}, {2, 5}),
        generateMishFloatParams<element::Type_t::f32>({2, 3, 4, 5}, {2, 3, 4, 5}),
        generateMishFloatParams<element::Type_t::f32>(PartialShape::dynamic(), {2, 3, 4, 5}),
        generateMishFloatParams<element::Type_t::f32>({2, Dimension::dynamic(), 4, 5},
                                                      {2, 3, 4, 5},
                                                      "dimensionDynamic"),
        generateMishFloatParams<element::Type_t::f16>({2, 5}, {2, 5}),
        generateMishFloatParams<element::Type_t::f16>({2, 3, 4, 5}, {2, 3, 4, 5}),
        generateMishFloatParams<element::Type_t::f16>(PartialShape::dynamic(), {2, 3, 4, 5}),
        generateMishFloatParams<element::Type_t::f16>({2, Dimension::dynamic(), 4, 5},
                                                      {2, 3, 4, 5},
                                                      "dimensionDynamic")};
    std::vector<MishParams> combinedParams;

    for (const auto& params : mishTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Mish_With_Hardcoded_Refs,
                         ReferenceMishLayerTest,
                         testing::ValuesIn(generateMishCombinedParams()),
                         ReferenceMishLayerTest::getTestCaseName);

}  // namespace
