// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <tuple>
#include <gtest/gtest.h>
#include <single_layer_common.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/ops.hpp>
#include <ie_precision.hpp>
#include <debug.h>
#include "../gna_matcher.hpp"

using namespace InferenceEngine::details;

typedef struct {
    std::vector<size_t> input_shape;
    std::vector<size_t> squeeze_indices;
} SqueezeCaseParam;

using SqueezeTestParam = std::tuple<InferenceEngine::Precision, bool, SqueezeCaseParam>;

class GNASqueezeTest_ : public GNATest<>,
                       public testing::WithParamInterface<SqueezeTestParam> {
 public:
    static std::string getTestName(const testing::TestParamInfo<SqueezeTestParam>& params) {
        std::stringstream test_name;
        test_name << std::get<0>(params.param) << "_";
        test_name << (std::get<1>(params.param) ? "squeeze" : "unsqueeze") << "_";
        test_name << std::get<2>(params.param).input_shape << "_";
        test_name << std::get<2>(params.param).squeeze_indices;
        return test_name.str();
    }
protected:
    const float MUL_VALUE = -1.0f;

    template <class SqueezeType>
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
        auto squeeze = std::make_shared<SqueezeType>(reshape1, squeeze_axes);

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

    void runtest() {
#if GNA_LIB_VER == 1
        if (strcmp(::testing::UnitTest::GetInstance()->current_test_info()->name(),
            "SqueezeTest/I16_squeeze_[3 1 2 4 4 3]_[1]") == 0) {
            GTEST_SKIP();
        }
#endif

        InferenceEngine::Precision precision;
        bool is_squeeze;
        SqueezeCaseParam param;
        std::tie(precision, is_squeeze, param) = GetParam();

        std::size_t input_dim = InferenceEngine::details::product(param.input_shape);

        auto buildNgraphFunction = [&]() {
            if (is_squeeze) {
                return getNgraphModel<ngraph::op::v0::Squeeze>(param);
            }
            return getNgraphModel<ngraph::op::v0::Unsqueeze>(param);
        };

        if (precision == InferenceEngine::Precision::FP32) {
            std::vector<float> input_data(input_dim);
            std::iota(input_data.begin(), input_data.end(), 1.0);

            std::vector<float> expected_result(input_dim);
            for (std::size_t i = 0; i < expected_result.size(); i++) {
                expected_result[i] = input_data[i] * MUL_VALUE;
            }

            assert_that().onInferNgraphModel(buildNgraphFunction())
                .inNotCompactMode()
                .gna()
                .propagate_forward()
                .onCPU()
                .called_with_input(input_data)
                .equals_to(expected_result);
        } else {
            assert_that().onInferNgraphModel(buildNgraphFunction())
                .inNotCompactMode()
                .gna()
                .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
                .withGNAConfig(GNA_CONFIG_KEY(PRECISION), "I16")
                .propagate_forward()
                .called();
        }
    }
};

class GNASqueezeTest : public GNASqueezeTest_{};


TEST_P(GNASqueezeTest, SqueezeTest) {
   runtest();
}

class GNAUnsqueezeTest : public GNASqueezeTest_{};


TEST_P(GNAUnsqueezeTest, UnsqueezeTest) {
    runtest();
}

static const SqueezeCaseParam gna_squeeze_test_params[] = {
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

INSTANTIATE_TEST_SUITE_P(
    GNALayerTests, GNASqueezeTest,
    ::testing::Combine(
        ::testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I16),
        ::testing::Values(true),
        ::testing::ValuesIn(gna_squeeze_test_params)),
        GNAUnsqueezeTest::getTestName);

INSTANTIATE_TEST_SUITE_P(
    GNALayerTests, GNAUnsqueezeTest,
    ::testing::Combine(
        ::testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I16),
        ::testing::Values(false),
        ::testing::ValuesIn(gna_squeeze_test_params)),
        GNAUnsqueezeTest::getTestName);
