// Copyright (C) 2018-2020 Intel Corporation
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

using GNAAlignFilterTestParams  = std::tuple<InferenceEngine::Precision, GNAPluginNS::Policy::ConcatAlignment, std::size_t, std::size_t>;
using namespace GNAPluginNS;

class GNAAlignFilterTest : public GNATest<>,
                             public testing::WithParamInterface<GNAAlignFilterTestParams> {
 public:

    static std::string getTestName(const testing::TestParamInfo<GNAAlignFilterTestParams>& params) {
        std::stringstream test_name;
        test_name << std::get<1>(params.param);
        test_name << "_concat_of[" << std::get<2>(params.param);
        test_name << "_" << std::get<3>(params.param);
        test_name << "]_on_";
        test_name << std::get<0>(params.param).name();
        return test_name.str();
    }

 protected:

    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
    std::size_t concat_inputs[2];
    GNAPluginNS::Policy::ConcatAlignment alignmentPolicy;

    void SetUp() override {
        std::tie(precision, alignmentPolicy, concat_inputs[0], concat_inputs[1]) = GetParam();
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

        return {copy_N, firstFilter_N  * firstFilter_frac};
    };

    auto getNumCopyElements = [&getFastAffineFilterParams](size_t sz) {
        return getFastAffineFilterParams(sz).first;
    };
    auto getsNumFilterWeights = [&getFastAffineFilterParams](size_t sz) {
        return getFastAffineFilterParams(sz).second;
    };
    auto getsStandaloneFilterWeights = [](size_t input_sz, size_t output_sz) {
        return ALIGN(input_sz, 8) * output_sz;
    };

    bool has1stFilter   = (concat_inputs[0] % 32) != 0  ? 1 : 0;
    bool has2ndFilter  = has1stFilter;

    switch(alignmentPolicy) {
        case  Policy::ConcatAlignment::ENABLED : {
            //align first input by 8
            auto firstFilter = ALIGN(concat_inputs[0], 8) * concat_inputs[0];
            //align first input by 8
            auto extraLeftElementsForSecond = concat_inputs[0] + 32 - ALIGN(concat_inputs[0], 32);

            auto secondFilter = ALIGN(concat_inputs[1], 8) * (extraLeftElementsForSecond + concat_inputs[1]);

            expected_affine_size = (has1stFilter ? firstFilter : 0) + (has2ndFilter ? secondFilter : 0);
            break;
        }
        case   Policy::ConcatAlignment::FAST_ZERO_OFFSET : {
            if (has1stFilter) {
                expected_copy_layers = getNumCopyElements(concat_inputs[0]);
                expected_affine_size = getsNumFilterWeights(concat_inputs[0]);
            }

            // calculation size for second filter
            if (has2ndFilter) {
                auto offset = ALIGN(concat_inputs[0], 32) - 32;
                auto zerolen = concat_inputs[0] - offset;
                auto second_output_len = zerolen + concat_inputs[1];

                expected_affine_size += second_output_len * ALIGN(concat_inputs[1], 8);
            }
            break;
        }
        case Policy::ConcatAlignment::FAST:  {
            if (has1stFilter) {
                expected_copy_layers = getNumCopyElements(concat_inputs[0]);
                expected_affine_size = getsNumFilterWeights(concat_inputs[0]);
            }

            // calculation size for second concat input
            if (has2ndFilter) {
                //first filter
                auto offset_sz = ALIGN(concat_inputs[0], 32) - concat_inputs[0];
                auto input_sz = std::min(concat_inputs[1], offset_sz);
                auto output_sz = 32;

                //second filter - weights are shared
                auto remained_size = concat_inputs[1] - input_sz;
                auto numfilters2 = remained_size / 32;
                auto input2_sz = ALIGN(32 + input_sz, 8);
                auto output2_sz = 32;

                //third filter
                auto input3_sz = input_sz + remained_size % 32;
                auto output3_sz = remained_size % 32;

                expected_affine_size
                    += getsStandaloneFilterWeights(input_sz, output_sz)
                    + (numfilters2 ? getsStandaloneFilterWeights(input2_sz, output2_sz) : 0)
                    + getsStandaloneFilterWeights(input3_sz, output3_sz);
            }
            break;
        }
        default : {
            FAIL() << "unsupported align policy: " << alignmentPolicy;
        }
    }

    assert_that().onInferNgraphModel(ngraf)
        .inNotCompactMode()
        .withPolicy(alignmentPolicy)
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
            .withPolicy(alignmentPolicy)
            .called_with()
            .input(ngraf->get_parameters().at(0)->get_name(), input_data[0])
            .input(ngraf->get_parameters().at(1)->get_name(), input_data[1])
            .equals_to(expected_result);
    } else {
        assert_that().onInferNgraphModel(ngraf)
            .inNotCompactMode()
            .gna()
            .withPolicy(alignmentPolicy)
            .withGNAConfig(std::string(GNA_CONFIG_KEY(SCALE_FACTOR)) + "_0", 1.0f)
            .withGNAConfig(std::string(GNA_CONFIG_KEY(SCALE_FACTOR)) + "_1", 1.0f)
            .withGNAConfig(GNA_CONFIG_KEY(PRECISION), precision.name())
            .propagate_forward()
            .called();
    }
}

INSTANTIATE_TEST_CASE_P(
    GNALayerTests,
    GNAAlignFilterTest,
    testing::Combine(
    testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I16, InferenceEngine::Precision::I8),
    //fast or not fast alignment policy
    testing::Values(
        Policy::ConcatAlignment::ENABLED,
        Policy::ConcatAlignment::FAST_ZERO_OFFSET,
        Policy::ConcatAlignment::FAST),
    // Size of first Split layer output
    testing::Values(64, 32, 31, 49),
    // Size of second Split layer output
    testing::Values(64, 32, 31, 73, 65)),
    GNAAlignFilterTest::getTestName);
