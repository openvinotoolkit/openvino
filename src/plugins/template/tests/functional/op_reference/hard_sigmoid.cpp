// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hard_sigmoid.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct HardSigmoidParams {
    template <class IT>
    HardSigmoidParams(const ov::Shape& shape,
                      const ov::element::Type& iType,
                      const std::vector<IT>& iValues,
                      const std::vector<IT>& oValues,
                      const float alpha,
                      const float beta)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(shape, iType, iValues)),
          refData(CreateTensor(shape, iType, oValues)),
          alpha(alpha),
          beta(beta) {}

    ov::Shape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
    float alpha;
    float beta;
};

class ReferenceHardSigmoidLayerTest : public testing::TestWithParam<HardSigmoidParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.alpha, params.beta);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<HardSigmoidParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "alpha=" << param.alpha << "_";
        result << "beta=" << param.beta;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type,
                                                 const float alphaData,
                                                 const float betaData) {
        std::vector<float> alphaArray;
        std::vector<float> betaArray;
        alphaArray.push_back(alphaData);
        betaArray.push_back(betaData);
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto alpha = ov::op::v0::Constant::create(input_type, Shape{}, {alphaData});
        const auto beta = ov::op::v0::Constant::create(input_type, Shape{}, {betaData});
        const auto HardSigmoid = std::make_shared<op::v0::HardSigmoid>(in, alpha, beta);
        return std::make_shared<ov::Model>(NodeVector{HardSigmoid}, ParameterVector{in});
    }
};

TEST_P(ReferenceHardSigmoidLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<HardSigmoidParams> generateHardSigmoidFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<HardSigmoidParams> hardSigmoidParams{
        HardSigmoidParams(ov::Shape{3},
                          IN_ET,
                          std::vector<T>{-1.0f, 0.0f, 1.0f},
                          std::vector<T>{0.1f, 0.6f, 1.f},
                          0.5,
                          0.6),
        HardSigmoidParams(ov::Shape{2, 5},
                          IN_ET,
                          std::vector<T>{-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 0.5f, -0.2f, 6.0f, 8.0f, 0.1f},
                          std::vector<T>{0.0f, 0.3f, 0.5f, 0.7f, 1.0f, 0.6f, 0.46f, 1.0f, 1.0f, 0.52f},
                          0.2,
                          0.5)};
    return hardSigmoidParams;
}

std::vector<HardSigmoidParams> generateHardSigmoidCombinedParams() {
    const std::vector<std::vector<HardSigmoidParams>> hardSigmoidTypeParams{
        generateHardSigmoidFloatParams<element::Type_t::f32>(),
        generateHardSigmoidFloatParams<element::Type_t::f16>()};
    std::vector<HardSigmoidParams> combinedParams;

    for (const auto& params : hardSigmoidTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_HardSigmoid_With_Hardcoded_Refs,
                         ReferenceHardSigmoidLayerTest,
                         testing::ValuesIn(generateHardSigmoidCombinedParams()),
                         ReferenceHardSigmoidLayerTest::getTestCaseName);

}  // namespace
