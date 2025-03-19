// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct AbsParams {
    template <class IT>
    AbsParams(const PartialShape& shape,
              const element::Type& iType,
              const std::vector<IT>& iValues,
              const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)) {}

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceAbsLayerTest : public testing::TestWithParam<AbsParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AbsParams>& obj) {
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
                                                 const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto log = std::make_shared<op::v0::Abs>(in);
        return std::make_shared<Model>(NodeVector{log}, ParameterVector{in});
    }
};

TEST_P(ReferenceAbsLayerTest, AbsWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<AbsParams> generateParamsForAbsFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AbsParams> params{AbsParams(ov::PartialShape{4},
                                            IN_ET,
                                            std::vector<T>{1.f, -2.f, 0.f, -4.75f},
                                            std::vector<T>{1.f, 2.f, 0.f, 4.75f})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<AbsParams> generateParamsForAbsInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AbsParams> params{
        AbsParams(ov::PartialShape{4}, IN_ET, std::vector<T>{1, -2, 0, -4}, std::vector<T>{1, 2, 0, 4})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<AbsParams> generateParamsForAbsUInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AbsParams> params{
        AbsParams(ov::PartialShape{4}, IN_ET, std::vector<T>{1, 2, 0, 4}, std::vector<T>{1, 2, 0, 4})};
    return params;
}

std::vector<AbsParams> generateCombinedParamsForAbs() {
    const std::vector<std::vector<AbsParams>> allTypeParams{generateParamsForAbsFloat<element::Type_t::f32>(),
                                                            generateParamsForAbsFloat<element::Type_t::f16>(),
                                                            generateParamsForAbsFloat<element::Type_t::bf16>(),
                                                            generateParamsForAbsInt<element::Type_t::i64>(),
                                                            generateParamsForAbsInt<element::Type_t::i32>(),
                                                            generateParamsForAbsUInt<element::Type_t::u64>(),
                                                            generateParamsForAbsUInt<element::Type_t::u32>()};

    std::vector<AbsParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Abs_With_Hardcoded_Refs,
                         ReferenceAbsLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForAbs()),
                         ReferenceAbsLayerTest::getTestCaseName);

}  // namespace
