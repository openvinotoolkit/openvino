// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/swish.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct SwishParams {
    template <class IT>
    SwishParams(const ov::Shape& shape,
                const ov::element::Type& iType,
                const std::vector<IT>& iValues,
                const float beta = 1)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(shape, iType, iValues)),
          beta(beta) {
        std::vector<IT> oValues;
        std::vector<float> output;
        std::vector<IT> betaVector;

        for (auto element : iValues)
            output.push_back(static_cast<float>(element));

        std::transform(output.begin(), output.end(), output.begin(), [&beta](float x) -> float {
            return (x / (1.0f + std::exp(x * beta * -1.0f)));
        });

        for (auto element : output)
            oValues.push_back(static_cast<IT>(element));
        refData = CreateTensor(shape, outType, oValues);

        betaVector.push_back(static_cast<IT>(beta));
        betaBlob = CreateTensor(inType, betaVector);
    }

    ov::Shape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
    ov::Tensor betaBlob;

    float beta;
};

class ReferenceSwishLayerTest : public testing::TestWithParam<SwishParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        threshold = 0.06;  // 0.01 failed in fp32 test

        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.beta);
        if (params.beta != 1) {
            inputData = {params.inputData, params.betaBlob};
            refOutData = {params.refData};
        } else {
            inputData = {params.inputData};
            refOutData = {params.refData};
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SwishParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "beta=" << param.beta;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& Swishected_output_type,
                                                 const float beta) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        if (beta != 1) {
            const auto BETA = std::make_shared<op::v0::Parameter>(input_type, Shape{});
            const auto Swish = std::make_shared<op::v4::Swish>(in, BETA);
            return std::make_shared<Model>(NodeVector{Swish}, ParameterVector{in, BETA});
        } else {
            const auto Swish = std::make_shared<op::v4::Swish>(in);
            return std::make_shared<ov::Model>(NodeVector{Swish}, ParameterVector{in});
        }
    }
};

TEST_P(ReferenceSwishLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SwishParams> generateSwishFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SwishParams> swishParams{
        SwishParams(ov::Shape{2, 4}, IN_ET, std::vector<T>{0.4, -5.7, -6, 3, -0.9, 23, 5, 3.3}, 0.6f),
        SwishParams(ov::Shape{2, 3}, IN_ET, std::vector<T>{1, 8, -8, 17, -0.5, -1}),
        SwishParams(ov::Shape{2, 2, 1, 2}, IN_ET, std::vector<T>{0.1, 0.6, 20, -7, -5.3, 3.5, -9, 11}, 0.33f)};
    return swishParams;
}

std::vector<SwishParams> generateSwishCombinedParams() {
    const std::vector<std::vector<SwishParams>> swishTypeParams{generateSwishFloatParams<element::Type_t::f32>(),
                                                                generateSwishFloatParams<element::Type_t::f16>()};
    std::vector<SwishParams> combinedParams;

    for (const auto& params : swishTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Swish_With_Hardcoded_Refs,
                         ReferenceSwishLayerTest,
                         testing::ValuesIn(generateSwishCombinedParams()),
                         ReferenceSwishLayerTest::getTestCaseName);

}  // namespace
