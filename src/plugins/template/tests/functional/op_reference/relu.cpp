// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/relu.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ReluParams {
    template <class IT>
    ReluParams(const ov::Shape& shape,
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

class ReferenceReluLayerTest : public testing::TestWithParam<ReluParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ReluParams>& obj) {
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
                                                 const element::Type& Reluected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto Relu = std::make_shared<op::v0::Relu>(in);
        return std::make_shared<ov::Model>(NodeVector{Relu}, ParameterVector{in});
    }
};

TEST_P(ReferenceReluLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ReluParams> generateReluFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ReluParams> reluParams{
        ReluParams(ov::Shape{2, 5},
                   IN_ET,
                   std::vector<T>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5},
                   std::vector<T>{1, 8, 0, 17, 0, 1, 8, 0, 17, 0}),
        ReluParams(ov::Shape{2, 2, 2, 2},
                   IN_ET,
                   std::vector<T>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1},
                   std::vector<T>{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1})};
    return reluParams;
}

template <element::Type_t IN_ET>
std::vector<ReluParams> generateReluIntParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ReluParams> reluParams{ReluParams(ov::Shape{2, 5},
                                                  IN_ET,
                                                  std::vector<T>{1, 8, -8, 17, -2, 1, 8, -8, 17, -1},
                                                  std::vector<T>{1, 8, 0, 17, 0, 1, 8, 0, 17, 0})};
    return reluParams;
}

template <element::Type_t IN_ET>
std::vector<ReluParams> generateReluUintParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ReluParams> reluParams{ReluParams(ov::Shape{2, 5},
                                                  IN_ET,
                                                  std::vector<T>{1, 8, 17, 1, 8, 17, 1, 8, 17, 0},
                                                  std::vector<T>{1, 8, 17, 1, 8, 17, 1, 8, 17, 0})};
    return reluParams;
}

std::vector<ReluParams> generateReluCombinedParams() {
    const std::vector<std::vector<ReluParams>> reluTypeParams{generateReluFloatParams<element::Type_t::f32>(),
                                                              generateReluFloatParams<element::Type_t::f16>(),
                                                              generateReluIntParams<element::Type_t::i64>(),
                                                              generateReluIntParams<element::Type_t::i32>(),
                                                              generateReluUintParams<element::Type_t::u64>(),
                                                              generateReluUintParams<element::Type_t::u32>()};
    std::vector<ReluParams> combinedParams;

    for (const auto& params : reluTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Relu_With_Hardcoded_Refs,
                         ReferenceReluLayerTest,
                         testing::ValuesIn(generateReluCombinedParams()),
                         ReferenceReluLayerTest::getTestCaseName);

}  // namespace
