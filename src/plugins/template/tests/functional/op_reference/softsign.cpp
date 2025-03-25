// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softsign.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct SoftSignParams {
    template <class IT>
    SoftSignParams(const ov::PartialShape& shape,
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

class ReferenceSoftSignLayerTest : public testing::TestWithParam<SoftSignParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SoftSignParams>& obj) {
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
                                                 const element::Type& SoftSign_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto SoftSign = std::make_shared<op::v9::SoftSign>(in);
        return std::make_shared<ov::Model>(NodeVector{SoftSign}, ParameterVector{in});
    }
};

TEST_P(ReferenceSoftSignLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SoftSignParams> generateSoftSignFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SoftSignParams> ParamsVector{SoftSignParams(ov::PartialShape{4},
                                                            IN_ET,
                                                            std::vector<T>{-1.0, 0.0, 1.0, 15.0},
                                                            std::vector<T>{-0.5, 0.0, 0.5, 0.9375})};
    return ParamsVector;
}

std::vector<SoftSignParams> generateSoftSignCombinedParams() {
    const std::vector<std::vector<SoftSignParams>> SoftSignTypeParams{
        generateSoftSignFloatParams<element::Type_t::f32>(),
        generateSoftSignFloatParams<element::Type_t::f16>(),
        generateSoftSignFloatParams<element::Type_t::bf16>()};
    std::vector<SoftSignParams> combinedParams;

    for (const auto& params : SoftSignTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SoftSign_With_Hardcoded_Refs,
                         ReferenceSoftSignLayerTest,
                         testing::ValuesIn(generateSoftSignCombinedParams()),
                         ReferenceSoftSignLayerTest::getTestCaseName);

}  // namespace
