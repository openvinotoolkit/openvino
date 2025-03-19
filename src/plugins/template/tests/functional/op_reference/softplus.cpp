// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softplus.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct SoftPlusParams {
    template <class IT>
    SoftPlusParams(const ov::PartialShape& shape,
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

class ReferenceSoftPlusLayerTest : public testing::TestWithParam<SoftPlusParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SoftPlusParams>& obj) {
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
                                                 const element::Type& SoftPlusected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto SoftPlus = std::make_shared<op::v4::SoftPlus>(in);
        return std::make_shared<ov::Model>(NodeVector{SoftPlus}, ParameterVector{in});
    }
};

TEST_P(ReferenceSoftPlusLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SoftPlusParams> generateSoftPlusFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SoftPlusParams> softPlusParams{SoftPlusParams(ov::PartialShape{4},
                                                              IN_ET,
                                                              std::vector<T>{-1.0, 0.0, 1.0, 20.0},
                                                              std::vector<T>{0.31326166, 0.69314718, 1.3132616, 20.0})};
    return softPlusParams;
}

std::vector<SoftPlusParams> generateSoftPlusCombinedParams() {
    const std::vector<std::vector<SoftPlusParams>> softPlusTypeParams{
        generateSoftPlusFloatParams<element::Type_t::f32>(),
        generateSoftPlusFloatParams<element::Type_t::f16>(),
        generateSoftPlusFloatParams<element::Type_t::bf16>()};
    std::vector<SoftPlusParams> combinedParams;

    for (const auto& params : softPlusTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SoftPlus_With_Hardcoded_Refs,
                         ReferenceSoftPlusLayerTest,
                         testing::ValuesIn(generateSoftPlusCombinedParams()),
                         ReferenceSoftPlusLayerTest::getTestCaseName);

}  // namespace
