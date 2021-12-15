// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <single_layer_common.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/ops.hpp>
#include <ie_precision.hpp>
#include "../gna_matcher.hpp"

using GNAAlignFilterTestParams  = std::tuple<InferenceEngine::Precision, std::size_t, std::size_t>;
using namespace GNAPluginNS;

class GNAAlignFilterTest : public GNATest<>,
                             public testing::WithParamInterface<GNAAlignFilterTestParams> {
 public:

    static std::string getTestName(const testing::TestParamInfo<GNAAlignFilterTestParams>& params) {
        std::string test_name;
        test_name += "fast_";
        test_name += "concat_of(" + std::to_string(std::get<1>(params.param));
        test_name += "_" + std::to_string(std::get<2>(params.param));
        test_name += ")_on_";
        test_name += std::get<0>(params.param).name();
        return test_name;
    }

 protected:

    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
    std::size_t concat_inputs[2];

    void SetUp() override {
        std::tie(precision, concat_inputs[0], concat_inputs[1]) = GetParam();
    }

    std::shared_ptr<ngraph::Function> getNgraphModel() {
        auto input0 = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, concat_inputs[0]});
        auto input1 = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{1, concat_inputs[1]});

        auto relu0 = std::make_shared<ngraph::op::v0::Relu>(input0);
        auto relu1 = std::make_shared<ngraph::op::v0::Relu>(input1);

        auto concat = std::make_shared<ngraph::op::Concat>(ngraph::NodeVector{relu0, relu1}, 1);

        auto relu3 = std::make_shared<ngraph::op::v0::Relu>(concat);

        auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu3}, ngraph::ParameterVector{input0, input1});
        return function;
    }
};

TEST_P(GNAAlignFilterTest, concatWith_2_Inputs_Small_mem_footprint) {

    auto ngraf = getNgraphModel();
    if (precision == InferenceEngine::Precision::FP32) {
        GTEST_SKIP() << "FP32 case - won't produce gna primitives";
    }

    // calc expected weight size
    size_t expected_affine_size = 0;
    size_t expected_copy_layers = 0;

    auto getFastAffineFilterParams = [](size_t sz) -> std::pair<size_t, size_t> {
        //align first input by 8
        auto copy_N = sz > 32 ? 1 : 0; // number of copy layers
        auto firstFilter_frac = sz % 32;
        auto firstFilter_N = ALIGN(firstFilter_frac, 8);

        return {copy_N, firstFilter_N   * firstFilter_frac};
    };

    auto getNumCopyElements = [&getFastAffineFilterParams](size_t sz) {
        return getFastAffineFilterParams(sz).first;
    };
    auto getsNumFilterWeights = [&getFastAffineFilterParams](size_t sz) {
        return getFastAffineFilterParams(sz).second;
    };

    expected_copy_layers = getNumCopyElements(concat_inputs[0]);
    expected_affine_size = getsNumFilterWeights(concat_inputs[0]);

    // calculation size for second filter
    auto offset = ALIGN(concat_inputs[0], 32) - 32;
    auto zerolen = concat_inputs[0] - offset;
    auto second_output_len = zerolen + concat_inputs[1];

    expected_affine_size += second_output_len  * ALIGN(concat_inputs[1], 8);

    assert_that().onInferNgraphModel(ngraf)
        .inNotCompactMode()
        .withGNAConfig(std::string(GNA_CONFIG_KEY(SCALE_FACTOR)) + "_0", 1.0f)
        .withGNAConfig(std::string(GNA_CONFIG_KEY(SCALE_FACTOR)) + "_1", 1.0f)
        .withGNAConfig(GNA_CONFIG_KEY(PRECISION), precision.name())
        .gna()
        .affine_weights()
        .size()
        .equals_to(expected_affine_size)
        .And()
        .copy_inserted_into_nnet()
        .times(expected_copy_layers);
}

TEST_P(GNAAlignFilterTest, concatWith_2_Inputs_accurate) {
    auto ngraf = getNgraphModel();
    if (precision == InferenceEngine::Precision::FP32) {
        std::vector<std::vector<float>> input_data;
        float start_value = 1.0;

        for (auto dim : concat_inputs) {
            if (dim > 0) {
                input_data.push_back(std::vector<float>(dim));

                std::iota(input_data.back().begin(), input_data.back().end(), start_value);
                start_value += dim;
            }
        }

        std::vector<float> expected_result(static_cast<size_t>(start_value - 1));
        start_value = 1.0;
        std::iota(expected_result.begin(), expected_result.end(), start_value);
        assert_that().onInferNgraphModel(ngraf)
            .inNotCompactMode()
            .gna()
            .propagate_forward()
            .onCPU()
            .called_with()
            .input(ngraf->get_parameters().at(0)->get_name(), input_data[0])
            .input(ngraf->get_parameters().at(1)->get_name(), input_data[1])
            .equals_to(expected_result);
    } else {
        assert_that().onInferNgraphModel(ngraf)
            .inNotCompactMode()
            .gna()
            .withGNAConfig(std::string(GNA_CONFIG_KEY(SCALE_FACTOR)) + "_0", 1.0f)
            .withGNAConfig(std::string(GNA_CONFIG_KEY(SCALE_FACTOR)) + "_1", 1.0f)
            .withGNAConfig(GNA_CONFIG_KEY(PRECISION), "I16")
            .propagate_forward()
            .called();
    }
}

INSTANTIATE_TEST_SUITE_P(
    GNALayerTests,
    GNAAlignFilterTest,
    testing::Combine(
    testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I16),
    // Size of first Split layer output
    testing::Values(31, 49),
    // Size of second Split layer output
    testing::Values(31, 73)),
    GNAAlignFilterTest::getTestName);
