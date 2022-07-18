// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/adaptive_pooling.hpp>

#include <cstddef>
#include <string>

using namespace cldnn;
using namespace ::tests;

template<typename T>
struct AdaptiveAvgPoolingParams {
    format inputFormat;
    tensor inputTensor;
    std::vector<T> inputs;
    tensor outputTensor;
    std::vector<T> outputs;
};

template<typename T>
struct adaptive_avg_pooling_test
        : public ::testing::TestWithParam<AdaptiveAvgPoolingParams<T> > {
public:
    void test() {
        auto data_type = type_to_data_type<T>::value;
        AdaptiveAvgPoolingParams<T> params = testing::TestWithParam<AdaptiveAvgPoolingParams<T> >::GetParam();
        auto &engine = get_test_engine();

        auto input = engine.allocate_memory({data_type, params.inputFormat, params.inputTensor});

        set_values(input, params.inputs);

        const std::string input_id = "adaptive_avg_input_id";
        const std::string adaptive_avg_pooling_id = "adaptive_avg_pooling_id";
        topology topology;
        topology.add(input_layout(input_id, input->get_layout()));

        topology.add(adaptive_pooling(adaptive_avg_pooling_id, input_id, params.outputTensor));

        network network(engine, topology);

        network.set_input_data(input_id, input);

        auto result = network.execute();

        auto out_mem = result.at(adaptive_avg_pooling_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.outputTensor.count(), out_ptr.size());
        for (size_t i = 0; i < params.outputs.size(); ++i) {
            EXPECT_NEAR(params.outputs[i], out_ptr[i], 0.005) << "at i = " << i;
        }
    }
};

template<typename T>
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template<typename T>
std::vector<AdaptiveAvgPoolingParams<T>> generateAdaptiveAvgPoolingParams() {
    std::vector<AdaptiveAvgPoolingParams<T>> result = {
            {format::bfyx,
                    tensor(1, 2, 7, 3),
                    getValues<T>({0, 4, 1, 3, -2, -5, -2, -2, 1, -3, 1, -3, -4, 0, -2, 1, -1, -2, 3, -1, -3,
                                  -1, -2, 3, 4, -3, -4, 1, 2, 0, -4, -5, -2, -2, -3, 2, 3, 1, -5, 2, -4, -2}),
                    tensor(1, 2, 3, 3),
                    getValues<T>({1.66666663,
                                  0.66666669,
                                  -3.,
                                  -1.33333337,
                                  -1.66666663,
                                  -2.33333325,
                                  -0.66666669,
                                  0.,
                                  -0.33333334,

                                  0.,
                                  1.33333337,
                                  -2.,
                                  -0.66666669,
                                  -3.66666675,
                                  -2.33333325,
                                  2.,
                                  -0.66666669,
                                  -1.33333337})
            },
            {format::bfyx,
                    tensor(1, 3, 10, 7),
                    getValues<T>({-2, -3, -4, 3, -5, 4, 0, -4, -2, -4, -5, 0, -3, 0, -2, 0, 0, -5, -4, -1, 3, -1, 0, -1,
                                  0, -2, 0, 4, 1, 4, 0, -1, -4, 2, -2, -5, -1, -1, -2, 1, 2, -2, -1, 2, 0, -1, 0, -5,
                                  4, 4, 3, 0, -4, -4, -4, -2, 0, 1, -2, -1, 4, -2, -4, 1, -1, -3, -4, -1, 1, -4,

                                  -2, -4, -5, 0, -4, 3, 4, -5, -4, -2, 0, 2, -4, -3, 3, -1, 1, -4, -5, 4, 2, -5, 2, -3,
                                  0, 4, 3, 3, 1, 2, -1, -4, 1, -3, -3, -2, 3, 4, -2, -5, 1, 4, 4, -2, 2, 1, -5, -2,
                                  -5, 1, 1, -2, -3, -3, -1, -5, 1, -3, -5, -3, -4, -1, 4, -3, 4, -1, 4, 3, 1, 4,

                                  -2, -4, -4, 4, -3, 4, 2, -3, -2, 4, -3, 0, 1, -4, 4, 4, 0, 3, -1, 3, 3, -5, 0, 3,
                                  -3, 1, -2, 4, -5, -5, 1, 0, -1, 0, -3, -2, 0, -3, 3, -2, -2, 0, -3, 4, -1, 2, -2, 2,
                                  -3, -1, -4, -2, 0, 2, 0, 2, 0, -3, 4, 3, -5, -3, -5, 1, -5, -3, -5, 4, -3, 3}),
                    tensor(1, 3, 3, 3),
                    getValues<T>({-1.08333337, -0.25000000, -0.91666669, -0.08333334, -0.66666669,
                                  0.75000000, -0.41666666, -1.33333337, -0.58333331,

                                  -1.66666663, 0.58333331, -0.16666667, -0.33333334, -0.41666666,
                                  -0.16666667, -0.33333334, -0.66666669, -0.75000000,

                                  -0.91666669, 0.83333331, -0.16666667, 0., -0.25000000,
                                  -1.16666663, -1.41666663, -0.41666666, -0.08333334})
            },
            {format::bfzyx,
                    tensor(2, 2, 3, 3, 3),
                    getValues<T>(
                            {-5, 1, -3, -4, 4, -4, 3, -3, -1, 0, 0, -2, -4, 2, 0, -4, -5, -2, -4, -4, 0, -2, 3, -3, 4,
                             -1, -4,
                             -1, -1, -5, 4, -1, -2, -3, 0, 4, -1, -5, -4, 1, 1, 4, -5, -5, -5, 4, -3, -3, -3, 4, 0, -3,
                             -5, 1,
                             4, 2, 1, -5, -5, 1, 0, -4, -1, 2, -4, -2, 4, 3, 1, -3, -3, -2, -4, -3, -3, 3, -1, 1, 2, 2,
                             -4,
                             -5, -4, 1, 3, -4, -1, 2, 4, -5, 0, 1, -2, 0, 0, -2, 3, -2, -5, -3, -5, -2, -1, 3, -2, 4, 3,
                             -3}),
                    tensor(2, 2, 2, 2, 2),
                    getValues<T>({-0.750, -0.250, -1.375, -1.125, -1.125, -0.500, -0.875, -1.250,
                                  -0.375, -1.625, -1., -0.500, -0.250, -0.750, -1.875, -0.625,
                                  0.125, -0.375, -1.625, -1.250, 0., -1., 0.875, -0.375,
                                  -1.125, -1.375, 0.750, -1.875, -0.625, -1.125, 1.250, -1.}),
            },
    };
    return result;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<AdaptiveAvgPoolingParams<T> > &param) {
        std::stringstream buf;
        buf << " input tensor " << param.param.inputTensor.to_string()
            << " output tensor " << param.param.outputTensor.to_string();
        return buf.str();
    }
};

using adaptive_avg_pooling_test_f32 = adaptive_avg_pooling_test<float>;
using adaptive_avg_pooling_test_f16 = adaptive_avg_pooling_test<half_t>;

TEST_P(adaptive_avg_pooling_test_f32, adaptive_avg_pooling_test_f32) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(adaptive_avg_pooling_test_f16, adaptive_avg_pooling_test_f16) {
    ASSERT_NO_FATAL_FAILURE(test());
}


INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f32,
                         adaptive_avg_pooling_test_f32,
                         ::testing::ValuesIn(generateAdaptiveAvgPoolingParams<float>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f16,
                         adaptive_avg_pooling_test_f16,
                         ::testing::ValuesIn(generateAdaptiveAvgPoolingParams<half_t>()),
                         PrintToStringParamName());
