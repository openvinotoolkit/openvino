// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <tuple>
#include <gtest/gtest.h>
#include <ngraph/op/parameter.hpp>
#include <ngraph/ops.hpp>
#include <ie_precision.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include "../gna_matcher.hpp"

using namespace InferenceEngine;

struct EltwiseTestParams {
    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
    bool has_reshape_before_eltwise = false;
    template <class T>
    void decodeGtestParams(const T &);
};

using GNAEltwiseTestParams = std::tuple<
    decltype(EltwiseTestParams::precision),
    decltype(EltwiseTestParams::has_reshape_before_eltwise)>;

template <>
inline void EltwiseTestParams::decodeGtestParams<GNAEltwiseTestParams>(const GNAEltwiseTestParams & params) {
    std::tie(precision, has_reshape_before_eltwise) = params;
}

class GNAEltwiseTest : public GNATest<>, public testing::WithParamInterface<GNAEltwiseTestParams>, public EltwiseTestParams {
 public:

    static std::string getTestName(const testing::TestParamInfo<GNAEltwiseTestParams>& params) {
        EltwiseTestParams tp;
        tp.decodeGtestParams(params.param);

        std::stringstream test_name;
        test_name << tp.precision << (tp.has_reshape_before_eltwise ? "_with_reshapes" : "");

        return test_name.str();
    }

 protected:

    void SetUp() override {
        decodeGtestParams(GetParam());
    }

    std::shared_ptr<ngraph::Function> buildNgraphModel() {
        auto input1 = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 32});
        const_cast<std::string&>(input1->get_name()) = "input1";

        auto input2 = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, 32});
        const_cast<std::string&>(input2->get_name()) = "input2";

        auto weights = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{ 32, 32 },  {1});
        auto biases = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{32},  {0});
        std::shared_ptr<ngraph::op::Op> FC1 = std::make_shared<ngraph::op::FullyConnected>(input1, weights, biases, ngraph::Shape{ 1, 32});
        std::shared_ptr<ngraph::op::Op> FC2 = std::make_shared<ngraph::op::FullyConnected>(input2, weights, biases, ngraph::Shape{ 1, 32});

        if (has_reshape_before_eltwise) {
            auto reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                           ngraph::Shape{2},
                                                                           std::vector<size_t>{1, 32});

            FC1 = std::make_shared<ngraph::op::v1::Reshape>(FC1, reshape_pattern, false);
            FC2 = std::make_shared<ngraph::op::v1::Reshape>(FC2, reshape_pattern, false);
        }

        auto add = std::make_shared<ngraph::op::v1::Add>(FC1, FC2);

        auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{input1, input2});

        return function;
    }
};

TEST_P(GNAEltwiseTest, FourBytesInputsViaReshape) {

    if (precision == InferenceEngine::Precision::FP32) {

        std::vector<float> expected_result(32, 96.0f);
        std::vector<float> input1(32, 1.0f);
        std::vector<float> input2(32, 2.0f);

        assert_that().onInferNgraphModel(buildNgraphModel())
            .inNotCompactMode()
            .gna()
            .propagate_forward()
            .onCPU()
            .called_with()
            .input("input1", input1)
            .input("input2", input2)
            .equals_to(expected_result);
    } else {
        assert_that().onInferNgraphModel(buildNgraphModel())
            .inNotCompactMode()
            .gna()
            .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_0"), 1.0f)
            .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_1"), 1.0f)
            .withGNAConfig(GNA_CONFIG_KEY(PRECISION), Precision(precision).name())
            .propagate_forward()
            .called();
    }
}

INSTANTIATE_TEST_SUITE_P(
    GNALayerTests,
    GNAEltwiseTest,
    ::testing::Combine(
        ::testing::Values(Precision::FP32, Precision::I16, Precision::I8),
        ::testing::Values(true, false)),
    GNAEltwiseTest::getTestName);
