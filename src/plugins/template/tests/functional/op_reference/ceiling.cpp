// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ceiling.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct CeilingParams {
    template <class IT>
    CeilingParams(const PartialShape& shape,
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

class ReferenceCeilingLayerTest : public testing::TestWithParam<CeilingParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CeilingParams>& obj) {
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
        const auto ceiling = std::make_shared<op::v0::Ceiling>(in);
        return std::make_shared<Model>(NodeVector{ceiling}, ParameterVector{in});
    }
};

TEST_P(ReferenceCeilingLayerTest, CeilingWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<CeilingParams> generateParamsForCeilingFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<CeilingParams> params{CeilingParams(ov::PartialShape{4},
                                                    IN_ET,
                                                    std::vector<T>{-2.5f, -2.0f, 0.3f, 4.8f},
                                                    std::vector<T>{-2.0f, -2.0f, 1.0f, 5.0f})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<CeilingParams> generateParamsForCeilingInt64() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<CeilingParams> params{CeilingParams(ov::PartialShape{3},
                                                    IN_ET,
                                                    std::vector<T>{0, 1, 0x4000000000000001},
                                                    std::vector<T>{0, 1, 0x4000000000000001})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<CeilingParams> generateParamsForCeilingInt32() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<CeilingParams> params{CeilingParams(ov::PartialShape{4},
                                                    IN_ET,
                                                    std::vector<T>{2, 136314888, 0x40000010, 0x40000001},
                                                    std::vector<T>{2, 136314888, 0x40000010, 0x40000001})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<CeilingParams> generateParamsForCeilingInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<CeilingParams> params{CeilingParams(ov::PartialShape{4},
                                                    IN_ET,
                                                    std::vector<T>{2, 64, 0x40, 0x01},
                                                    std::vector<T>{2, 64, 0x40, 0x01})};
    return params;
}

std::vector<CeilingParams> generateCombinedParamsForCeiling() {
    const std::vector<std::vector<CeilingParams>> allTypeParams{generateParamsForCeilingFloat<element::Type_t::f32>(),
                                                                generateParamsForCeilingFloat<element::Type_t::f16>(),
                                                                generateParamsForCeilingInt64<element::Type_t::i64>(),
                                                                generateParamsForCeilingInt32<element::Type_t::i32>(),
                                                                generateParamsForCeilingInt<element::Type_t::i16>(),
                                                                generateParamsForCeilingInt<element::Type_t::i8>(),
                                                                generateParamsForCeilingInt64<element::Type_t::u64>(),
                                                                generateParamsForCeilingInt32<element::Type_t::u32>(),
                                                                generateParamsForCeilingInt<element::Type_t::u16>(),
                                                                generateParamsForCeilingInt<element::Type_t::u8>()};

    std::vector<CeilingParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Ceiling_With_Hardcoded_Refs,
                         ReferenceCeilingLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForCeiling()),
                         ReferenceCeilingLayerTest::getTestCaseName);

}  // namespace
