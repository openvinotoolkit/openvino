// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "gna_matcher.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>

namespace detail {
    template<typename T>
    std::shared_ptr<ngraph::Function> CreateActivationFunction(const ngraph::Shape& input_shape) {
        auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto f = std::make_shared<T>(input_params);
        auto result = std::make_shared<ngraph::opset8::Result>(f);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    }

    template<typename T>
    std::shared_ptr<ngraph::Function> CreateActivationFunction(const ngraph::Shape& input_shape, double min, double max);

    template<>
    std::shared_ptr<ngraph::Function> CreateActivationFunction<ngraph::opset8::Clamp>(const ngraph::Shape& input_shape, double min, double max) {
        auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto f = std::make_shared<ngraph::opset8::Clamp>(input_params, min, max);
        auto result = std::make_shared<ngraph::opset8::Result>(f);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
    }
}

class PWLAproximationTest : public GNATest<> {
protected:
    void SetUp() override  {
    }
};
using namespace GNATestIRs;

// Recursive Algorithm
// Precision Threshold

TEST_F(PWLAproximationTest, forTanhOnRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Tanh>({1, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActTanh)
                                    .pwl_quantization_precision_threshold(0.0053);
}

TEST_F(PWLAproximationTest, forSigmoidOnRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Sigmoid>({1, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActSigmoid)
                                    .pwl_quantization_precision_threshold(0.0027);
}

TEST_F(PWLAproximationTest, forReLUonRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Relu>({1, 1, 10, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActRelu)
                                    .pwl_quantization_precision_threshold(0.0001);
}

TEST_F(PWLAproximationTest, forLeakyReLUonRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(LeakyReLUActivationModel())
                                .gna()
                                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActLeakyRelu)
                                .pwl_quantization_precision_threshold(0.0003);
}

TEST_F(PWLAproximationTest, DISABLED_forIdentityOnRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(IdentityActivationModel())
                                .gna()
                                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActIdentity)
                                .pwl_quantization_precision_threshold(0.0003);
}

TEST_F(PWLAproximationTest, forClampOnRecursiveAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Clamp>({1, 10}, -50, 50))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActKaldiLstmClipping)
                                    .pwl_quantization_precision_threshold(0.0001);
}

TEST_F(PWLAproximationTest, forTanhOnUniformAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Tanh>({1, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .withUniformPWLAlgo()
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActTanh)
                                    .pwl_quantization_precision_threshold(0.0009);
}

TEST_F(PWLAproximationTest, forSigmoidOnUniformAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Sigmoid>({1, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .withUniformPWLAlgo()
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActSigmoid)
                                    .pwl_quantization_precision_threshold(0.0004);
}

TEST_F(PWLAproximationTest, DISABLED_forIdentityOnUniformAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferModel(IdentityActivationModel())
                                .gna()
                                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActIdentity)
                                .pwl_quantization_precision_threshold(0.0003);
}

TEST_F(PWLAproximationTest, forClampOnUniformAlgoWithPrecisionThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Clamp>({1, 10}, -50, 50))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .withUniformPWLAlgo()
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActKaldiLstmClipping)
                                    .pwl_quantization_precision_threshold(0.0001);
}

TEST_F(PWLAproximationTest, forSigmoidonRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Sigmoid>({1, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActSigmoid)
                                    .pwl_quantization_segments_threshold(12);
}

TEST_F(PWLAproximationTest, forTanhonRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Tanh>({1, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActTanh)
                                    .pwl_quantization_segments_threshold(12);
}

TEST_F(PWLAproximationTest, forReLUonRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Relu>({1, 1, 10, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActRelu)
                                    .pwl_quantization_segments_threshold(4);
}

TEST_F(PWLAproximationTest, forLeakyReLUonRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(LeakyReLUActivationModel())
                                .gna()
                                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActLeakyRelu)
                                .pwl_quantization_segments_threshold(4);
}

TEST_F(PWLAproximationTest, DISABLED_forIdentityOnRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(IdentityActivationModel())
                                .gna()
                                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActIdentity)
                                .pwl_quantization_segments_threshold(3);
}

TEST_F(PWLAproximationTest, forClampOnRecursiveAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Clamp>({1, 10}, -50, 50))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActKaldiLstmClipping)
                                    .pwl_quantization_segments_threshold(3);
}

TEST_F(PWLAproximationTest, forSigmoidonUniformAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Sigmoid>({1, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .withUniformPWLAlgo()
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActSigmoid)
                                    .pwl_quantization_segments_threshold(65);
}

TEST_F(PWLAproximationTest, forTanhonUniformAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Tanh>({1, 10}))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .withUniformPWLAlgo()
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActTanh)
                                    .pwl_quantization_segments_threshold(65);
}

TEST_F(PWLAproximationTest, DISABLED_forIdentityOnUniformAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferModel(IdentityActivationModel())
                                .gna()
                                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                .withUniformPWLAlgo()
                                .propagate_forward()
                                .called_with()
                                .pwl_quantization_activation(DnnActivationType::kActIdentity)
                                .pwl_quantization_segments_threshold(3);
}

TEST_F(PWLAproximationTest, forClampOnUniformAlgoWithSegmentThresholdIsSuccess) {
    assert_that().onInferNgraphModel(detail::CreateActivationFunction<ngraph::opset8::Clamp>({1, 10}, -50, 50))
                                    .gna()
                                    .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                                    .withUniformPWLAlgo()
                                    .propagate_forward()
                                    .called_with()
                                    .pwl_quantization_activation(DnnActivationType::kActKaldiLstmClipping)
                                    .pwl_quantization_segments_threshold(3);
}
