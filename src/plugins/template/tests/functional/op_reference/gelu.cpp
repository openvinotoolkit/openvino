// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gelu.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GeluParams {
    template <class IT>
    GeluParams(const ov::PartialShape& shape,
               const ov::element::Type& iType,
               const std::vector<IT>& iValues,
               const std::vector<IT>& oValues,
               const ov::op::GeluApproximationMode mode)
        : mode(mode),
          pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)) {}

    ov::op::GeluApproximationMode mode = ov::op::GeluApproximationMode::ERF;
    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceGeluV0LayerTest : public testing::TestWithParam<GeluParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.mode);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<GeluParams>& obj) {
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
                                                 const element::Type& expected_output_type,
                                                 const op::GeluApproximationMode mode) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto Gelu = std::make_shared<op::v0::Gelu>(in);
        return std::make_shared<ov::Model>(NodeVector{Gelu}, ParameterVector{in});
    }
};

class ReferenceGeluV7LayerTest : public testing::TestWithParam<GeluParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.mode);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<GeluParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "ApproxMode=" << param.mode;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type,
                                                 const op::GeluApproximationMode mode) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto Gelu = std::make_shared<op::v7::Gelu>(in, mode);
        return std::make_shared<ov::Model>(NodeVector{Gelu}, ParameterVector{in});
    }
};

TEST_P(ReferenceGeluV0LayerTest, CompareWithRefs) {
    Exec();
}
TEST_P(ReferenceGeluV7LayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<GeluParams> generateGeluV0FloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<GeluParams> geluParams{
        GeluParams(
            ov::PartialShape{8},
            IN_ET,
            std::vector<T>{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0},
            std::vector<
                T>{-0.00012636185, -0.0040495098, -0.04550028, -0.15865529, 0.0, 0.8413447, 1.9544997, 2.9959507},
            op::GeluApproximationMode::ERF),
        GeluParams(ov::PartialShape{3},
                   IN_ET,
                   std::vector<T>{-0.5, 0.1, 0.4},
                   std::vector<T>{-0.15426877, 0.05398279, 0.2621686},
                   op::GeluApproximationMode::ERF)};
    return geluParams;
}

template <element::Type_t IN_ET>
std::vector<GeluParams> generateGeluV7FloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<GeluParams> geluParams{
        GeluParams(
            ov::PartialShape{8},
            IN_ET,
            std::vector<T>{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0},
            std::vector<
                T>{-0.00012636185, -0.0040495098, -0.04550028, -0.15865529, 0.0, 0.8413447, 1.9544997, 2.9959507},
            op::GeluApproximationMode::ERF),
        GeluParams(
            ov::PartialShape{8},
            IN_ET,
            std::vector<T>{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0},
            std::vector<
                T>{-0.00012636185, -0.0040495098, -0.04550028, -0.15865529, 0.0, 0.8413447, 1.9544997, 2.9959507},
            op::GeluApproximationMode::TANH),
        GeluParams(ov::PartialShape{3},
                   IN_ET,
                   std::vector<T>{-0.5, 0.1, 0.4},
                   std::vector<T>{-0.15426877, 0.05398279, 0.2621686},
                   op::GeluApproximationMode::ERF),
        GeluParams(ov::PartialShape{3},
                   IN_ET,
                   std::vector<T>{-0.5, 0.1, 0.4},
                   std::vector<T>{-0.15428599, 0.053982753, 0.262161165},
                   op::GeluApproximationMode::TANH)};
    return geluParams;
}

std::vector<GeluParams> generateGeluV0CombinedParams() {
    const std::vector<std::vector<GeluParams>> geluTypeParams{generateGeluV0FloatParams<element::Type_t::f32>(),
                                                              generateGeluV0FloatParams<element::Type_t::f16>()};
    std::vector<GeluParams> combinedParams;

    for (const auto& params : geluTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

std::vector<GeluParams> generateGeluV7CombinedParams() {
    const std::vector<std::vector<GeluParams>> geluTypeParams{generateGeluV7FloatParams<element::Type_t::f32>(),
                                                              generateGeluV7FloatParams<element::Type_t::f16>()};
    std::vector<GeluParams> combinedParams;

    for (const auto& params : geluTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Gelu_v2_With_Hardcoded_Refs,
                         ReferenceGeluV0LayerTest,
                         testing::ValuesIn(generateGeluV0CombinedParams()),
                         ReferenceGeluV0LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Gelu_v7_With_Hardcoded_Refs,
                         ReferenceGeluV7LayerTest,
                         testing::ValuesIn(generateGeluV7CombinedParams()),
                         ReferenceGeluV7LayerTest::getTestCaseName);

}  // namespace
