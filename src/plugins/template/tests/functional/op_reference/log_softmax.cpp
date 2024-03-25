// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log_softmax.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct LogSoftmaxParams {
    template <class IT>
    LogSoftmaxParams(const ov::Shape& shape,
                     const ov::element::Type& iType,
                     const std::vector<IT>& iValues,
                     const std::vector<IT>& oValues,
                     const int64_t axis)
        : axis(axis),
          pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(shape, iType, iValues)),
          refData(CreateTensor(shape, iType, oValues)) {}

    int64_t axis = 0;

    ov::Shape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceLogSoftmaxLayerTest : public testing::TestWithParam<LogSoftmaxParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.axis);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<LogSoftmaxParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "axis=" << param.axis;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type,
                                                 const int64_t axis) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto LogSoftmax = std::make_shared<op::v5::LogSoftmax>(in, axis);
        return std::make_shared<ov::Model>(NodeVector{LogSoftmax}, ParameterVector{in});
    }
};

TEST_P(ReferenceLogSoftmaxLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<LogSoftmaxParams> generateLogSoftmaxFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<LogSoftmaxParams> logSoftmaxParams{
        LogSoftmaxParams(ov::Shape{1}, IN_ET, std::vector<T>{1}, std::vector<T>{0}, 0),
        LogSoftmaxParams(ov::Shape{2, 4},
                         IN_ET,
                         std::vector<T>{0, 1, 2, 3, 10000, 10001, 10002, 10003},
                         std::vector<T>{-10000., -10000., -10000., -10000., 0., 0., 0., 0.},
                         0),
        LogSoftmaxParams(
            ov::Shape{2, 4},
            IN_ET,
            std::vector<T>{0, 1, 2, 3, 10000, 10001, 10002, 10003},
            std::vector<
                T>{-3.4401896, -2.4401896, -1.4401897, -0.4401897, -3.4401896, -2.4401896, -1.4401897, -0.4401897},
            1),
        LogSoftmaxParams(
            ov::Shape{2, 4},
            IN_ET,
            std::vector<T>{0, 1, 2, 3, 10000, 10001, 10002, 10003},
            std::vector<
                T>{-3.4401896, -2.4401896, -1.4401897, -0.4401897, -3.4401896, -2.4401896, -1.4401897, -0.4401897},
            -1),
        LogSoftmaxParams(ov::Shape{2, 4},
                         IN_ET,
                         std::vector<T>{0, 1, 2, 3, 10000, 10001, 10002, 10003},
                         std::vector<T>{-10000., -10000., -10000., -10000., 0., 0., 0., 0.},
                         -2),
        LogSoftmaxParams(ov::Shape{3, 2, 3},
                         IN_ET,
                         std::vector<T>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8},
                         std::vector<T>{-12.0024818,
                                        -12.0024818,
                                        -12.0024818,
                                        -12.0024818,
                                        -12.0024818,
                                        -12.0024818,
                                        -6.00248181,
                                        -6.00248181,
                                        -6.00248181,
                                        -6.00248181,
                                        -6.00248181,
                                        -6.00248181,
                                        -2.48181414e-03,
                                        -2.48181414e-03,
                                        -2.48181414e-03,
                                        -2.48181414e-03,
                                        -2.48181414e-03,
                                        -2.48181414e-03},
                         0),
        LogSoftmaxParams(ov::Shape{3, 2, 3},
                         IN_ET,
                         std::vector<T>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8},
                         std::vector<T>{-3.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -0.04858735},
                         1),
        LogSoftmaxParams(ov::Shape{3, 2, 3},
                         IN_ET,
                         std::vector<T>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8},
                         std::vector<T>{-2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596},
                         2),
        LogSoftmaxParams(ov::Shape{3, 2, 3},
                         IN_ET,
                         std::vector<T>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8},
                         std::vector<T>{-2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596,
                                        -2.40760596,
                                        -1.40760596,
                                        -0.40760596},
                         -1),
        LogSoftmaxParams(ov::Shape{3, 2, 3},
                         IN_ET,
                         std::vector<T>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8},
                         std::vector<T>{-3.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -3.04858735,
                                        -0.04858735,
                                        -0.04858735,
                                        -0.04858735},
                         -2),
        LogSoftmaxParams(ov::Shape{3, 2, 3},
                         IN_ET,
                         std::vector<T>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8},
                         std::vector<T>{-12.0024818,
                                        -12.0024818,
                                        -12.0024818,
                                        -12.0024818,
                                        -12.0024818,
                                        -12.0024818,
                                        -6.00248181,
                                        -6.00248181,
                                        -6.00248181,
                                        -6.00248181,
                                        -6.00248181,
                                        -6.00248181,
                                        -2.48181414e-03,
                                        -2.48181414e-03,
                                        -2.48181414e-03,
                                        -2.48181414e-03,
                                        -2.48181414e-03,
                                        -2.48181414e-03},
                         -3)};
    return logSoftmaxParams;
}

std::vector<LogSoftmaxParams> generateLogSoftmaxCombinedParams() {
    const std::vector<std::vector<LogSoftmaxParams>> logSoftmaxTypeParams{
        generateLogSoftmaxFloatParams<element::Type_t::f32>(),
        generateLogSoftmaxFloatParams<element::Type_t::f16>()};
    std::vector<LogSoftmaxParams> combinedParams;

    for (const auto& params : logSoftmaxTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_With_Hardcoded_Refs,
                         ReferenceLogSoftmaxLayerTest,
                         testing::ValuesIn(generateLogSoftmaxCombinedParams()),
                         ReferenceLogSoftmaxLayerTest::getTestCaseName);

}  // namespace
