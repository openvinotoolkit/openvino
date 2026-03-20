// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erfinv.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ErfInvParams {
    template <class IT>
    ErfInvParams(const ov::PartialShape& shape,
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

class ReferenceErfInvLayerTest : public testing::TestWithParam<ErfInvParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ErfInvParams>& obj) {
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
                                                 const element::Type& output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto erfinv = std::make_shared<op::v16::ErfInv>(in);
        return std::make_shared<ov::Model>(OutputVector{erfinv}, ParameterVector{in});
    }
};

TEST_P(ReferenceErfInvLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ErfInvParams> generateErfInvFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    // erfinv(erf(x)) = x for known values:
    //   erf(0.0)   = 0.0,   erfinv(0.0)   = 0.0
    //   erf(0.5)   ≈ 0.5205 → erfinv(0.5205) ≈ 0.5
    //   erf(-0.5)  ≈ -0.5205
    //   erfinv(1)  = +inf, erfinv(-1) = -inf
    std::vector<ErfInvParams> params{
        ErfInvParams(ov::PartialShape{4},
                     IN_ET,
                     std::vector<T>{T(0.0f), T(0.5f), T(-0.5f), T(0.9f)},
                     std::vector<T>{T(0.0f), T(0.4769362762044699f), T(-0.4769362762044699f), T(1.1630871536766743f)}),
    };
    return params;
}

std::vector<ErfInvParams> generateErfInvCombinedParams() {
    const std::vector<std::vector<ErfInvParams>> typeParams{
        generateErfInvFloatParams<element::Type_t::f32>(),
        generateErfInvFloatParams<element::Type_t::f16>(),
        generateErfInvFloatParams<element::Type_t::bf16>()};
    std::vector<ErfInvParams> combinedParams;
    for (const auto& p : typeParams) {
        combinedParams.insert(combinedParams.end(), p.begin(), p.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ErfInv_With_Hardcoded_Refs,
                         ReferenceErfInvLayerTest,
                         testing::ValuesIn(generateErfInvCombinedParams()),
                         ReferenceErfInvLayerTest::getTestCaseName);

}  // namespace
