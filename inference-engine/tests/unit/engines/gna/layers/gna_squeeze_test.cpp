// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <single_layer_common.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/ops.hpp>
#include <ie_precision.hpp>
#include <debug.h>
#include "../gna_matcher.hpp"

typedef struct {
    std::vector<size_t> input_shape;
    std::vector<size_t> squeeze_indices;
} SqueezeCaseParam;

using SqueezeTestParam = std::tuple<InferenceEngine::Precision, SqueezeCaseParam>;

class GNASqueezeTest : public GNATest<>,
                       public testing::WithParamInterface<SqueezeTestParam> {
protected:
    const float MUL_VALUE = -1.0f;

    std::shared_ptr<ngraph::Function> getNgraphModel(const SqueezeCaseParam& param) {
        const std::size_t input_dim = InferenceEngine::details::product(param.input_shape);

        auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, input_dim});

        auto reshape1_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{param.input_shape.size()},
                                                                       param.input_shape);
        auto reshape1 = std::make_shared<ngraph::op::v1::Reshape>(input, reshape1_pattern, false);

        auto squeeze_axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                   ngraph::Shape{param.squeeze_indices.size()},
                                                                   param.squeeze_indices);
        auto squeeze = std::make_shared<ngraph::op::v0::Squeeze>(reshape1, squeeze_axes);

        auto reshape2_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{2},
                                                                       std::vector<size_t>{1, input_dim});
        auto reshape2 = std::make_shared<ngraph::op::v1::Reshape>(squeeze, reshape2_pattern, false);

        auto weights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{1},
                                                              std::vector<float>{MUL_VALUE});
        auto mul = std::make_shared<ngraph::op::v1::Multiply>(reshape2, weights);

        auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
        return function;
    }
};

TEST_P(GNASqueezeTest, SqueezeTest) {
    SqueezeTestParam test_param = GetParam();

    const InferenceEngine::Precision precision = std::get<0>(test_param);
    const SqueezeCaseParam param = std::get<1>(test_param);

    std::size_t input_dim = InferenceEngine::details::product(param.input_shape);

    if (precision == InferenceEngine::Precision::FP32) {
        std::vector<float> input_data(input_dim);
        std::iota(input_data.begin(), input_data.end(), 1.0);

        std::vector<float> expected_result(input_dim);
        for (std::size_t i = 0; i < expected_result.size(); i++) {
            expected_result[i] = input_data[i] * MUL_VALUE;
        }

        assert_that().onInferNgraphModel(getNgraphModel(param))
                .inNotCompactMode()
                .gna()
                .propagate_forward()
                .onCPU()
                .called_with_input(input_data)
                .equals_to(expected_result);
    } else {
        assert_that().onInferNgraphModel(getNgraphModel(param))
                .inNotCompactMode()
                .gna()
                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                .withGNAConfig(GNA_CONFIG_KEY(PRECISION), "I16")
                .propagate_forward()
                .called();
    }
}

const SqueezeCaseParam gna_squeeze_test_params[] = {
        {{1, 1, 3}, {0, 1}},
        {{1, 1, 3}, {0}},
        {{1, 1, 3}, {1}},
        {{1, 3, 1}, {0, 2}},
        {{1, 3, 1}, {0}},
        {{1, 3, 1}, {2}},
        {{3, 1, 1}, {1, 2}},
        {{3, 1, 1}, {1}},
        {{3, 1, 1}, {2}},
        {{4, 1, 3, 1}, {1, 3}},
        {{4, 1, 1, 3}, {1, 2}},
        {{1, 4, 1, 3}, {0, 2}},
        {{1, 3, 5, 2, 1}, {0, 4}},
        {{3, 1, 2, 4, 4, 3}, {1}},
        {{1, 1, 1, 1, 1, 3}, {0, 1, 2, 3, 4}},
        {{1, 1, 1, 1, 1, 3}, {1, 3}}
};

INSTANTIATE_TEST_CASE_P(
    GNALayerTests, GNASqueezeTest,
        ::testing::Combine(
                ::testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I16),
                ::testing::ValuesIn(gna_squeeze_test_params)));
