// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mvn.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "common_test_utils/common_utils.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace reference_tests;

// ------------------------------ V0 ------------------------------

struct MVN1Params {
    MVN1Params(const reference_tests::Tensor& paramInput,
               const ov::AxisSet& paramReductionAxes,
               const bool paramAcrossChannels,
               const bool paramNormalizeVariance,
               const double paramEps,
               const reference_tests::Tensor& paramExpected)
        : input(paramInput),
          reductionAxes(paramReductionAxes),
          acrossChannels(paramAcrossChannels),
          normalizeVariance(paramNormalizeVariance),
          eps(paramEps),
          expected(paramExpected) {}
    reference_tests::Tensor input;
    ov::AxisSet reductionAxes;
    bool acrossChannels;
    bool normalizeVariance;
    double eps;
    reference_tests::Tensor expected;
};

class ReferenceMVN1LayerTest : public testing::TestWithParam<MVN1Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input,
                                  params.reductionAxes,
                                  params.acrossChannels,
                                  params.normalizeVariance,
                                  params.eps);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MVN1Params>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape;
        result << "_iType=" << param.input.type;
        if (!param.reductionAxes.empty()) {
            result << "_reductionAccess=" << ov::test::utils::vec2str(param.reductionAxes.to_vector());
        } else {
            result << "_acrossChannels=" << (param.acrossChannels ? "TRUE" : "FALSE");
        }
        result << "_normalizeVariance=" << (param.normalizeVariance ? "TRUE" : "FALSE");
        result << "_eps=" << param.eps;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& input,
                                                 const ov::AxisSet& reductionAxes,
                                                 const bool acrossChannels,
                                                 const bool normalizeVariance,
                                                 const double eps) {
        const auto in = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        auto mvn = std::make_shared<op::v0::MVN>(in, acrossChannels, normalizeVariance, eps);
        if (!reductionAxes.empty()) {
            mvn = std::make_shared<op::v0::MVN>(in, reductionAxes, normalizeVariance, eps);
        }
        return std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{in});
    }
};

TEST_P(ReferenceMVN1LayerTest, CompareWithHardcodedRefs) {
    Exec();
}

const ov::AxisSet emptyReductionAxes{};

INSTANTIATE_TEST_SUITE_P(
    smoke_MVN1_With_Hardcoded_Refs,
    ReferenceMVN1LayerTest,
    ::testing::Values(
        // across_channels=false, variance=false
        MVN1Params(reference_tests::Tensor{{1, 3, 3, 3},
                                           ov::element::f32,
                                           std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5,
                                                              6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
                   emptyReductionAxes,
                   false,
                   false,
                   1e-9,
                   reference_tests::Tensor{{1, 3, 3, 3},
                                           ov::element::f32,
                                           std::vector<float>{-4, -3, -2, -1, 0,  1,  2,  3,  4, -4, -3, -2, -1, 0,
                                                              1,  2,  3,  4,  -4, -3, -2, -1, 0, 1,  2,  3,  4}}),
        // across_channels=true, variance=false
        MVN1Params(
            reference_tests::Tensor{{1, 3, 2, 2},
                                    ov::element::f32,
                                    std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3}},
            emptyReductionAxes,
            true,
            false,
            1e-9,
            reference_tests::Tensor{
                {1, 3, 2, 2},
                ov::element::f32,
                std::vector<float>{-3.25, -2.25, -1.25, -0.25, 0.75, 1.75, 2.75, 3.75, 4.75, -3.25, -2.25, -1.25}}),
        // across_channels=false, variance=true
        MVN1Params(
            reference_tests::Tensor{{1, 3, 3, 3}, ov::element::f32, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9}},
            emptyReductionAxes,
            false,
            true,
            1e-9,
            reference_tests::Tensor{
                {1, 3, 3, 3},
                ov::element::f32,
                std::vector<float>{-1.5491934,  -1.161895, -0.7745967, -0.38729835, 0.,         0.38729835,  0.7745967,
                                   1.161895,    1.5491934, -1.5491934, -1.161895,   -0.7745967, -0.38729835, 0.,
                                   0.38729835,  0.7745967, 1.161895,   1.5491934,   -1.5491934, -1.161895,   -0.7745967,
                                   -0.38729835, 0.,        0.38729835, 0.7745967,   1.161895,   1.5491934}}),
        // across_channels=true, variance=true
        MVN1Params(
            reference_tests::Tensor{{1, 3, 3, 3}, ov::element::f32, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9}},
            emptyReductionAxes,
            true,
            true,
            1e-9,
            reference_tests::Tensor{
                {1, 3, 3, 3},
                ov::element::f32,
                std::vector<float>{-1.5491934,  -1.161895, -0.7745967, -0.38729835, 0.,         0.38729835,  0.7745967,
                                   1.161895,    1.5491934, -1.5491934, -1.161895,   -0.7745967, -0.38729835, 0.,
                                   0.38729835,  0.7745967, 1.161895,   1.5491934,   -1.5491934, -1.161895,   -0.7745967,
                                   -0.38729835, 0.,        0.38729835, 0.7745967,   1.161895,   1.5491934}}),
        // reductionAxes, variance=false
        MVN1Params(
            reference_tests::Tensor{{1, 3, 2, 2},
                                    ov::element::f32,
                                    std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3}},
            {1, 2, 3},
            false,
            false,
            1e-9,
            reference_tests::Tensor{
                {1, 3, 2, 2},
                ov::element::f32,
                std::vector<float>{-3.25, -2.25, -1.25, -0.25, 0.75, 1.75, 2.75, 3.75, 4.75, -3.25, -2.25, -1.25}}),
        // reductionAxes, variance=true
        MVN1Params(
            reference_tests::Tensor{{1, 3, 3, 3}, ov::element::f32, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9}},
            {2, 3},
            false,
            true,
            1e-9,
            reference_tests::Tensor{
                {1, 3, 3, 3},
                ov::element::f32,
                std::vector<float>{-1.5491934,  -1.161895, -0.7745967, -0.38729835, 0.,         0.38729835,  0.7745967,
                                   1.161895,    1.5491934, -1.5491934, -1.161895,   -0.7745967, -0.38729835, 0.,
                                   0.38729835,  0.7745967, 1.161895,   1.5491934,   -1.5491934, -1.161895,   -0.7745967,
                                   -0.38729835, 0.,        0.38729835, 0.7745967,   1.161895,   1.5491934}})),
    ReferenceMVN1LayerTest::getTestCaseName);

// ------------------------------ V6 ------------------------------

struct MVN6Params {
    MVN6Params(const reference_tests::Tensor& paramInput,
               const reference_tests::Tensor& paramReductionAxes,
               const bool paramNormalizeVariance,
               const double paramEps,
               const op::MVNEpsMode mode,
               const reference_tests::Tensor& paramExpected)
        : input(paramInput),
          reductionAxes(paramReductionAxes),
          normalizeVariance(paramNormalizeVariance),
          eps(paramEps),
          epsMode(mode),
          expected(paramExpected) {}
    reference_tests::Tensor input;
    reference_tests::Tensor reductionAxes;
    bool normalizeVariance;
    double eps;
    op::MVNEpsMode epsMode;
    reference_tests::Tensor expected;
};

class ReferenceMVN6LayerTest : public testing::TestWithParam<MVN6Params>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function =
            CreateFunction(params.input, params.reductionAxes, params.normalizeVariance, params.eps, params.epsMode);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MVN6Params>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape;
        result << "_iType=" << param.input.type;
        result << "_reductionAccess=" << ov::test::utils::vec2str(param.reductionAxes.shape);
        result << "_normalizeVariance=" << (param.normalizeVariance ? "TRUE" : "FALSE");
        result << "_eps=" << param.eps;
        result << "_eps_mode=" << param.epsMode;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& input,
                                                 const reference_tests::Tensor& reductionAxes,
                                                 const bool normalizeVariance,
                                                 const double eps,
                                                 const op::MVNEpsMode epsMode) {
        std::vector<int64_t> dataVector(reductionAxes.shape[0]);
        const auto in = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        const auto refBuffer = reductionAxes.data.data<const std::int64_t>();
        for (size_t i = 0; i < dataVector.size(); ++i) {
            dataVector[i] = refBuffer[i];
        }
        const auto axes = std::make_shared<op::v0::Constant>(reductionAxes.type, reductionAxes.shape, dataVector);
        auto mvn = std::make_shared<op::v6::MVN>(in, axes, normalizeVariance, static_cast<float>(eps), epsMode);
        return std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{in});
    }
};

TEST_P(ReferenceMVN6LayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_MVN6_With_Hardcoded_Refs,
    ReferenceMVN6LayerTest,
    ::testing::Values(
        // variance=false, OUTSIDE_SQRT
        MVN6Params(reference_tests::Tensor{{1, 3, 3, 3},
                                           ov::element::f32,
                                           std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5,
                                                              6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
                   reference_tests::Tensor{Shape{2}, ov::element::i64, std::vector<int64_t>{2, 3}},
                   false,
                   1e-9,
                   op::MVNEpsMode::OUTSIDE_SQRT,
                   reference_tests::Tensor{{1, 3, 3, 3},
                                           ov::element::f32,
                                           std::vector<float>{-4, -3, -2, -1, 0,  1,  2,  3,  4, -4, -3, -2, -1, 0,
                                                              1,  2,  3,  4,  -4, -3, -2, -1, 0, 1,  2,  3,  4}}),
        // variance=true, OUTSIDE_SQRT
        MVN6Params(
            reference_tests::Tensor{{1, 3, 3, 3}, ov::element::f32, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9}},
            reference_tests::Tensor{Shape{2}, ov::element::i64, std::vector<int64_t>{2, 3}},
            true,
            1e-9,
            op::MVNEpsMode::OUTSIDE_SQRT,
            reference_tests::Tensor{
                {1, 3, 3, 3},
                ov::element::f32,
                std::vector<float>{-1.5491934,  -1.161895, -0.7745967, -0.38729835, 0.,         0.38729835,  0.7745967,
                                   1.161895,    1.5491934, -1.5491934, -1.161895,   -0.7745967, -0.38729835, 0.,
                                   0.38729835,  0.7745967, 1.161895,   1.5491934,   -1.5491934, -1.161895,   -0.7745967,
                                   -0.38729835, 0.,        0.38729835, 0.7745967,   1.161895,   1.5491934}}),
        // variance=true, INSIDE_SQRT
        MVN6Params(
            reference_tests::Tensor{{1, 3, 3, 3}, ov::element::f32, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       1, 2, 3, 4, 5, 6, 7, 8, 9}},
            reference_tests::Tensor{Shape{2}, ov::element::i64, std::vector<int64_t>{2, 3}},
            true,
            1e-9,
            op::MVNEpsMode::INSIDE_SQRT,
            reference_tests::Tensor{
                {1, 3, 3, 3},
                ov::element::f32,
                std::vector<float>{-1.5491934,  -1.161895, -0.7745967, -0.38729835, 0.,         0.38729835,  0.7745967,
                                   1.161895,    1.5491934, -1.5491934, -1.161895,   -0.7745967, -0.38729835, 0.,
                                   0.38729835,  0.7745967, 1.161895,   1.5491934,   -1.5491934, -1.161895,   -0.7745967,
                                   -0.38729835, 0.,        0.38729835, 0.7745967,   1.161895,   1.5491934}}),
        // variance=true, another reductionAxes, OUTSIDE_SQRT
        MVN6Params(
            reference_tests::Tensor{{1, 3, 3, 3}, ov::element::f32, std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                        1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                        1, 2, 3, 4, 5, 6, 7, 8, 9})},
            reference_tests::Tensor{Shape{3}, ov::element::i64, std::vector<int64_t>({1, 2, 3})},
            true,
            1e-9,
            op::MVNEpsMode::OUTSIDE_SQRT,
            reference_tests::Tensor{
                {1, 3, 3, 3},
                ov::element::f32,
                std::vector<float>{-1.5491934,  -1.161895, -0.7745967, -0.38729835, 0.,         0.38729835,  0.7745967,
                                   1.161895,    1.5491934, -1.5491934, -1.161895,   -0.7745967, -0.38729835, 0.,
                                   0.38729835,  0.7745967, 1.161895,   1.5491934,   -1.5491934, -1.161895,   -0.7745967,
                                   -0.38729835, 0.,        0.38729835, 0.7745967,   1.161895,   1.5491934}})),
    ReferenceMVN6LayerTest::getTestCaseName);
