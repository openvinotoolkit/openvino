// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/sigmoid.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct SigmoidParams {
    template <class IT>
    SigmoidParams(const ov::PartialShape& shape, const ov::element::Type& iType, const std::vector<IT>& iValues, const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)) {}

    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::runtime::Tensor inputData;
    ov::runtime::Tensor refData;
};

class ReferenceSigmoidLayerTest : public testing::TestWithParam<SigmoidParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SigmoidParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& Sigmoidected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto Sigmoid = std::make_shared<op::v0::Sigmoid>(in);
        return std::make_shared<ov::Function>(NodeVector {Sigmoid}, ParameterVector {in});
    }
};

TEST_P(ReferenceSigmoidLayerTest, CompareWithRefs) {
    Exec();
}


template <element::Type_t IN_ET>
std::vector<SigmoidParams> generateSigmoidFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    float x1 = 1.0f;
    float x2 = 4.0f;
    float sigma1 = 1.0f / (1.0f + std::exp(-x1));
    float sigma2 = 1.0f / (1.0f + std::exp(-x2));

    std::vector<SigmoidParams> sigmoidParams {
        SigmoidParams(ov::PartialShape {1, 1, 2, 2},
                    IN_ET,
                    std::vector<T>{x1, x2, x1, x2},
                    std::vector<T>{sigma1, sigma2, sigma1, sigma2}),
        SigmoidParams(ov::PartialShape {1, 1, 4},
                    IN_ET,
                    std::vector<T>{x1, x2, x1, x2},
                    std::vector<T>{sigma1, sigma2, sigma1, sigma2})
    };
    return sigmoidParams;
}

std::vector<SigmoidParams> generateSigmoidCombinedParams() {
    const std::vector<std::vector<SigmoidParams>> sigmoidTypeParams {
        generateSigmoidFloatParams<element::Type_t::f32>(),
        generateSigmoidFloatParams<element::Type_t::f16>()
        };
    std::vector<SigmoidParams> combinedParams;

    for (const auto& params : sigmoidTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Sigmoid_With_Hardcoded_Refs, ReferenceSigmoidLayerTest,
    testing::ValuesIn(generateSigmoidCombinedParams()), ReferenceSigmoidLayerTest::getTestCaseName);

} // namespace