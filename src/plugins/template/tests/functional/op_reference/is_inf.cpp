// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_inf.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct IsInfParams {
    template <class IT, class OT>
    IsInfParams(const std::string testcaseName,
                const PartialShape& shape,
                const element::Type& iType,
                const element::Type& oType,
                const std::vector<IT>& iValues,
                const std::vector<OT>& oValues,
                op::v10::IsInf::Attributes& attrs)
        : testcaseName(testcaseName),
          pshape(shape),
          inType(iType),
          outType(oType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(oType, oValues)),
          attrs(attrs) {}

    std::string testcaseName;
    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
    op::v10::IsInf::Attributes attrs;
};

class ReferenceIsInfLayerTest : public testing::TestWithParam<IsInfParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.attrs);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<IsInfParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (!param.testcaseName.empty()) {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type,
                                                 op::v10::IsInf::Attributes attrs) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto is_inf = std::make_shared<op::v10::IsInf>(in, attrs);
        return std::make_shared<Model>(NodeVector{is_inf}, ParameterVector{in});
    }
};

TEST_P(ReferenceIsInfLayerTest, IsInfWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<IsInfParams> generateParamsForIsInfDefault() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<element::Type_t::boolean>::value_type;
    op::v10::IsInf::Attributes attrs{};
    std::vector<IsInfParams> params{IsInfParams("IsInfDefault",
                                                ov::PartialShape{8},
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
                                                std::vector<U>{true, false, false, false, true, false, false, false},
                                                attrs)};
    return params;
}

template <element::Type_t IN_ET>
std::vector<IsInfParams> generateParamsForIsInfPositive() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<element::Type_t::boolean>::value_type;
    op::v10::IsInf::Attributes attrs{};
    attrs.detect_negative = false;
    std::vector<IsInfParams> params{IsInfParams("IsInfPositiveOnly",
                                                ov::PartialShape{8},
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
                                                std::vector<U>{true, false, false, false, false, false, false, false},
                                                attrs)};
    return params;
}

template <element::Type_t IN_ET>
std::vector<IsInfParams> generateParamsForIsInfNegative() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<element::Type_t::boolean>::value_type;
    op::v10::IsInf::Attributes attrs{};
    attrs.detect_positive = false;
    std::vector<IsInfParams> params{IsInfParams("IsInfNegativeOnly",
                                                ov::PartialShape{8},
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
                                                std::vector<U>{false, false, false, false, true, false, false, false},
                                                attrs)};
    return params;
}

template <element::Type_t IN_ET>
std::vector<IsInfParams> generateParamsForIsInfNone() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<element::Type_t::boolean>::value_type;
    op::v10::IsInf::Attributes attrs{};
    attrs.detect_negative = false;
    attrs.detect_positive = false;
    std::vector<IsInfParams> params{IsInfParams("IsInfDetectNone",
                                                ov::PartialShape{8},
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
                                                std::vector<U>{false, false, false, false, false, false, false, false},
                                                attrs)};
    return params;
}

std::vector<IsInfParams> generateCombinedParamsForIsInf() {
    const std::vector<std::vector<IsInfParams>> allTypeParams{
        generateParamsForIsInfDefault<element::Type_t::f64>(),
        generateParamsForIsInfDefault<element::Type_t::f32>(),
        generateParamsForIsInfDefault<element::Type_t::f16>(),
        generateParamsForIsInfDefault<element::Type_t::bf16>(),
        generateParamsForIsInfPositive<element::Type_t::f32>(),
        generateParamsForIsInfPositive<element::Type_t::f64>(),
        generateParamsForIsInfPositive<element::Type_t::f16>(),
        generateParamsForIsInfPositive<element::Type_t::bf16>(),
        generateParamsForIsInfNegative<element::Type_t::f64>(),
        generateParamsForIsInfNegative<element::Type_t::f32>(),
        generateParamsForIsInfNegative<element::Type_t::f16>(),
        generateParamsForIsInfNegative<element::Type_t::bf16>(),
        generateParamsForIsInfNone<element::Type_t::f64>(),
        generateParamsForIsInfNone<element::Type_t::f32>(),
        generateParamsForIsInfNone<element::Type_t::f16>(),
        generateParamsForIsInfNone<element::Type_t::bf16>(),
    };

    std::vector<IsInfParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_IsInf_With_Hardcoded_Refs,
                         ReferenceIsInfLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForIsInf()),
                         ReferenceIsInfLayerTest::getTestCaseName);
}  // namespace
