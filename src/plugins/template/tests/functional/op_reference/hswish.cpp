// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hswish.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct HSwishParams {
    template <class IT>
    HSwishParams(const ov::Shape& shape,
                 const ov::element::Type& iType,
                 const std::vector<IT>& iValues,
                 const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(shape, iType, iValues)),
          refData(CreateTensor(shape, iType, oValues)) {}

    ov::Shape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceHSwishLayerTest : public testing::TestWithParam<HSwishParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<HSwishParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& HSwishected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto HSwish = std::make_shared<op::v4::HSwish>(in);
        return std::make_shared<ov::Model>(NodeVector{HSwish}, ParameterVector{in});
    }
};

TEST_P(ReferenceHSwishLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<HSwishParams> generateHSwishFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<HSwishParams> hSwishParams{
        HSwishParams(ov::Shape{2, 3},
                     IN_ET,
                     std::vector<T>{1.f, 8.f, -8.f, 17.f, -0.5f, -1.f},
                     std::vector<T>{0.66666667f, 8.f, 0.f, 17.f, -0.20833333f, -0.33333333f}),
        HSwishParams(ov::Shape{2, 2, 1, 2},
                     IN_ET,
                     std::vector<T>{0.1f, 0.6f, 20.f, -7.f, -5.3f, 3.5f, -9.f, 11.f},
                     std::vector<T>{0.05166667f, 0.36f, 20.f, 0.f, 0.f, 3.5f, 0.f, 11.f})};
    return hSwishParams;
}

std::vector<HSwishParams> generateHSwishCombinedParams() {
    const std::vector<std::vector<HSwishParams>> hSwishTypeParams{generateHSwishFloatParams<element::Type_t::f32>(),
                                                                  generateHSwishFloatParams<element::Type_t::f16>()};
    std::vector<HSwishParams> combinedParams;

    for (const auto& params : hSwishTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_HSwish_With_Hardcoded_Refs,
                         ReferenceHSwishLayerTest,
                         testing::ValuesIn(generateHSwishCombinedParams()),
                         ReferenceHSwishLayerTest::getTestCaseName);

}  // namespace
