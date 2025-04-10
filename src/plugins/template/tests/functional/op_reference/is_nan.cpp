// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_nan.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct IsNaNParams {
    template <class IT, class OT>
    IsNaNParams(const PartialShape& shape,
                const element::Type& iType,
                const element::Type& oType,
                const std::vector<IT>& iValues,
                const std::vector<OT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(oType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(oType, oValues)) {}

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceIsNaNLayerTest : public testing::TestWithParam<IsNaNParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<IsNaNParams>& obj) {
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
        const auto is_nan = std::make_shared<op::v10::IsNaN>(in);
        return std::make_shared<Model>(NodeVector{is_nan}, ParameterVector{in});
    }
};

TEST_P(ReferenceIsNaNLayerTest, IsNaNWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<IsNaNParams> generateParamsForIsNaNFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<element::Type_t::boolean>::value_type;

    std::vector<IsNaNParams> params{IsNaNParams(ov::PartialShape{8},
                                                IN_ET,
                                                element::Type_t::boolean,
                                                std::vector<T>{std::numeric_limits<T>::infinity(),
                                                               0.0000f,
                                                               std::numeric_limits<T>::max(),
                                                               -0.5000f,
                                                               -std::numeric_limits<T>::infinity(),
                                                               1.0000f,
                                                               std::numeric_limits<T>::min(),
                                                               std::nanf("")},
                                                std::vector<U>{false, false, false, false, false, false, false, true})};
    return params;
}

std::vector<IsNaNParams> generateCombinedParamsForIsNaN() {
    const std::vector<std::vector<IsNaNParams>> allTypeParams{
        generateParamsForIsNaNFloat<element::Type_t::f64>(),
        generateParamsForIsNaNFloat<element::Type_t::f32>(),
        generateParamsForIsNaNFloat<element::Type_t::f16>(),
        generateParamsForIsNaNFloat<element::Type_t::bf16>(),
    };

    std::vector<IsNaNParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_IsNaN_With_Hardcoded_Refs,
                         ReferenceIsNaNLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForIsNaN()),
                         ReferenceIsNaNLayerTest::getTestCaseName);
}  // namespace
