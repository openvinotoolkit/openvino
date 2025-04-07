// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_finite.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct IsFiniteParams {
    template <class IT, class OT>
    IsFiniteParams(const PartialShape& shape,
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

class ReferenceIsFiniteLayerTest : public testing::TestWithParam<IsFiniteParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<IsFiniteParams>& obj) {
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
        const auto is_finite = std::make_shared<op::v10::IsFinite>(in);
        return std::make_shared<Model>(NodeVector{is_finite}, ParameterVector{in});
    }
};

TEST_P(ReferenceIsFiniteLayerTest, IsFiniteWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<IsFiniteParams> generateParamsForIsFiniteFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<element::Type_t::boolean>::value_type;

    std::vector<IsFiniteParams> params{
        IsFiniteParams(ov::PartialShape{8},
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
                       std::vector<U>{false, true, true, true, false, true, true, false})};
    return params;
}

std::vector<IsFiniteParams> generateCombinedParamsForIsFinite() {
    const std::vector<std::vector<IsFiniteParams>> allTypeParams{
        generateParamsForIsFiniteFloat<element::Type_t::f64>(),
        generateParamsForIsFiniteFloat<element::Type_t::f32>(),
        generateParamsForIsFiniteFloat<element::Type_t::f16>(),
        generateParamsForIsFiniteFloat<element::Type_t::bf16>(),
    };

    std::vector<IsFiniteParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_IsFinite_With_Hardcoded_Refs,
                         ReferenceIsFiniteLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForIsFinite()),
                         ReferenceIsFiniteLayerTest::getTestCaseName);
}  // namespace
