// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selu.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct SeluParams {
    template <class IT>
    SeluParams(const ov::PartialShape& shape,
               const ov::element::Type& iType,
               const std::vector<IT>& iValues,
               const std::vector<IT>& oValues,
               const ov::Shape& alphaShape,
               const ov::Shape& lambdaShape,
               const std::vector<IT>& alphaValues,
               const std::vector<IT>& lambdaValues,
               const std::string& test_name = "")
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)),
          alphaShape(alphaShape),
          lambdaShape(lambdaShape),
          alpha(CreateTensor(iType, alphaValues)),
          lambda(CreateTensor(iType, lambdaValues)),
          testcaseName(test_name) {}

    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
    ov::Shape alphaShape;
    ov::Shape lambdaShape;
    ov::Tensor alpha;
    ov::Tensor lambda;
    std::string testcaseName;
};

class ReferenceSeluLayerTest : public testing::TestWithParam<SeluParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData, params.alpha, params.lambda};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SeluParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "alpha=" << param.alpha.data() << "_";
        if (param.testcaseName != "") {
            result << "lambda=" << param.lambda.data() << "_";
            result << param.testcaseName;
        } else {
            result << "lambda=" << param.lambda.data();
        }

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const SeluParams& params) {
        const auto in = std::make_shared<op::v0::Parameter>(params.inType, params.pshape);
        const auto alpha = std::make_shared<op::v0::Parameter>(params.inType, params.alphaShape);
        const auto lambda = std::make_shared<op::v0::Parameter>(params.inType, params.lambdaShape);
        const auto Selu = std::make_shared<op::v0::Selu>(in, alpha, lambda);
        return std::make_shared<ov::Model>(NodeVector{Selu}, ParameterVector{in, alpha, lambda});
    }
};

TEST_P(ReferenceSeluLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SeluParams> generateSeluFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SeluParams> seluParams{SeluParams(ov::PartialShape{2},
                                                  IN_ET,
                                                  std::vector<T>{-1, 3},
                                                  std::vector<T>{-1.1113307, 3.152103},
                                                  ov::Shape{1},
                                                  ov::Shape{1},
                                                  std::vector<T>{1.67326324},
                                                  std::vector<T>{1.05070098}),
                                       SeluParams(ov::PartialShape{4},
                                                  IN_ET,
                                                  std::vector<T>{-1.0, 0.0, 1.0, 2.0},
                                                  std::vector<T>{-1.1113307, 0., 1.050701, 2.101402},
                                                  ov::Shape{1},
                                                  ov::Shape{1},
                                                  std::vector<T>{1.67326324},
                                                  std::vector<T>{1.05070098}),
                                       SeluParams(ov::PartialShape{1},
                                                  IN_ET,
                                                  std::vector<T>{112.0},
                                                  std::vector<T>{117.67851},
                                                  ov::Shape{1},
                                                  ov::Shape{1},
                                                  std::vector<T>{1.67326324},
                                                  std::vector<T>{1.05070098}),
                                       SeluParams(ov::PartialShape{3},
                                                  IN_ET,
                                                  std::vector<T>{-3.0, -12.5, -7.0},
                                                  std::vector<T>{-1.6705687, -1.7580928, -1.7564961},
                                                  ov::Shape{1},
                                                  ov::Shape{1},
                                                  std::vector<T>{1.67326324},
                                                  std::vector<T>{1.05070098})};
    return seluParams;
}

std::vector<SeluParams> generateSeluCombinedParams() {
    const std::vector<std::vector<SeluParams>> seluTypeParams{generateSeluFloatParams<element::Type_t::f32>(),
                                                              generateSeluFloatParams<element::Type_t::f16>(),
                                                              generateSeluFloatParams<element::Type_t::bf16>()};
    std::vector<SeluParams> combinedParams;

    for (const auto& params : seluTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Selu_With_Hardcoded_Refs,
                         ReferenceSeluLayerTest,
                         testing::ValuesIn(generateSeluCombinedParams()),
                         ReferenceSeluLayerTest::getTestCaseName);

}  // namespace
