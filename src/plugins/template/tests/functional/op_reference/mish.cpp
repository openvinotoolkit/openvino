// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mish.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct MishParams {
    template <class IT>
    MishParams(const ov::PartialShape& parameterShape,
               const ov::Shape& tensorShape,
               const ov::element::Type& iType,
               const std::vector<IT>& iValues,
               const std::vector<IT>& oValues,
               const std::string& test_name = "")
        : parameterShape(parameterShape),
          tensorShape(tensorShape),
          inType(iType),
          inputData(CreateTensor(tensorShape, iType, iValues)),
          refData(CreateTensor(tensorShape, iType, oValues)),
          testcaseName(test_name) {}

    ov::PartialShape parameterShape;
    ov::Shape tensorShape;
    ov::element::Type inType;
    ov::Tensor inputData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferenceMishLayerTest : public testing::TestWithParam<MishParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.parameterShape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MishParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "dShape=" << param.parameterShape << "_";
        result << "iShape=" << param.tensorShape << "_";
        result << "iType=" << param.inType;
        if (!param.testcaseName.empty())
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape, const element::Type& input_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto Mish = std::make_shared<op::v4::Mish>(in);
        return std::make_shared<Model>(NodeVector{Mish}, ParameterVector{in});
    }
};

TEST_P(ReferenceMishLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
MishParams generateMishFloatParams(const PartialShape& parameterShape,
                                   const Shape& tensorShape,
                                   const std::string& test_name = "") {
    using T = typename element_type_traits<IN_ET>::value_type;

    // generate input tensor (with possible type conversion)
    const auto staticSize = shape_size(tensorShape);
    std::vector<T> expected;
    std::vector<T> input;
    {
        std::mt19937 gen{0};  // use fixed seed for reproducibility of the test
        std::normal_distribution<> d{0.0, 20.0};

        for (auto i = staticSize; i > 0; i--) {
            const auto x = static_cast<T>(d(gen));
            const auto y = static_cast<T>(x * std::tanh(std::log(std::exp(x) + T{1})));
            input.push_back(x);
            expected.push_back(y);
        }
    }

    return MishParams{parameterShape, tensorShape, IN_ET, input, expected, test_name};
}

std::vector<MishParams> generateMishCombinedParams() {
    return std::vector<MishParams>{generateMishFloatParams<element::Type_t::f32>({2, 5}, {2, 5}),
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
}

INSTANTIATE_TEST_SUITE_P(smoke_Mish_With_Hardcoded_Refs,
                         ReferenceMishLayerTest,
                         testing::ValuesIn(generateMishCombinedParams()),
                         ReferenceMishLayerTest::getTestCaseName);

}  // namespace
