// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hsigmoid.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct HSigmoidParams {
    template <class IT>
    HSigmoidParams(const ov::PartialShape& shape,
                   const ov::element::Type& iType,
                   const std::vector<IT>& iValues,
                   const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)) {}

    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceHSigmoidLayerTest : public testing::TestWithParam<HSigmoidParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<HSigmoidParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& HSigmoidected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto HSigmoid = std::make_shared<op::v5::HSigmoid>(in);
        return std::make_shared<ov::Model>(NodeVector{HSigmoid}, ParameterVector{in});
    }
};

TEST_P(ReferenceHSigmoidLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<HSigmoidParams> generateHSigmoidFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<HSigmoidParams> hSigmoidParams{HSigmoidParams(
        ov::PartialShape{13},
        IN_ET,
        std::vector<T>{-10.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 10.f},
        std::vector<
            T>{0.f, 0.f, 0.f, 0.f, 0.16666667f, 0.33333333f, 0.5f, 0.66666667f, 0.83333333f, 1.f, 1.f, 1.f, 1.f})};
    return hSigmoidParams;
}

std::vector<HSigmoidParams> generateHSigmoidCombinedParams() {
    const std::vector<std::vector<HSigmoidParams>> hSigmoidTypeParams{
        generateHSigmoidFloatParams<element::Type_t::f32>(),
        generateHSigmoidFloatParams<element::Type_t::f16>()};
    std::vector<HSigmoidParams> combinedParams;

    for (const auto& params : hSigmoidTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_HSigmoid_With_Hardcoded_Refs,
                         ReferenceHSigmoidLayerTest,
                         testing::ValuesIn(generateHSigmoidCombinedParams()),
                         ReferenceHSigmoidLayerTest::getTestCaseName);

}  // namespace
