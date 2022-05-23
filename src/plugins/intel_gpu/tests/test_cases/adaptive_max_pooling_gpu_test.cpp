// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/adaptive_pooling.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>

#include <cstddef>
#include <string>

using namespace cldnn;
using namespace ::tests;

template<typename T>
struct AdaptiveMaxPoolingParams {
    format inputFormat;
    tensor inputTensor;
    std::vector<T> inputs;
    tensor outputTensor;
    std::vector<T> outputs;
    std::vector<int32_t> output_indices;
};

template<typename T>
struct adaptive_max_pooling_test
        : public ::testing::TestWithParam<AdaptiveMaxPoolingParams<T> > {
public:
    void test() {
        auto data_type = type_to_data_type<T>::value;
        AdaptiveMaxPoolingParams<T> params = testing::TestWithParam<AdaptiveMaxPoolingParams<T> >::GetParam();
        auto &engine = get_test_engine();

        auto input = engine.allocate_memory({data_type, params.inputFormat, params.inputTensor});
        auto indices_output = engine.allocate_memory({data_types::i32, params.inputFormat, params.outputTensor});

        set_values(input, params.inputs);

        const std::string input_id = "adaptive_max_input_id";
        const std::string adaptive_max_pooling_id = "adaptive_max_pooling_id";
        const std::string output_indices_id = "output_indices_id";
        topology topology;
        topology.add(input_layout(input_id, input->get_layout()));
        topology.add(mutable_data(output_indices_id, indices_output));

        topology.add(adaptive_pooling(adaptive_max_pooling_id, input_id, params.outputTensor, output_indices_id,
                                      data_types::i32));

        network network(engine, topology);

        network.set_input_data(input_id, input);

        auto result = network.execute();

        auto out_mem = result.at(adaptive_max_pooling_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());
        cldnn::mem_lock<int32_t> out_indices(indices_output, get_test_stream());

        ASSERT_EQ(params.outputTensor.count(), out_ptr.size());
        ASSERT_EQ(params.outputTensor.count(), out_indices.size());
        for (size_t i = 0; i < params.outputs.size(); ++i) {
            EXPECT_NEAR(params.outputs[i], out_ptr[i], 0.005) << "at i = " << i;
            EXPECT_EQ(params.output_indices[i], out_indices[i]);
        }
    }
};

template<typename T>
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template<typename T>
std::vector<AdaptiveMaxPoolingParams<T>> generateAdaptiveMaxPoolingParams() {
    std::vector<AdaptiveMaxPoolingParams<T>> result = {
            {format::bfyx,
                    tensor(2, 3, 1, 7),
                    getValues<T>({0, 4, 1, 3, -2, -5, -2, -2, 1, -3, 1, -3, -4, 0, -2, 1, -1, -2, 3, -1, -3,
                                  -1, -2, 3, 4, -3, -4, 1, 2, 0, -4, -5, -2, -2, -3, 2, 3, 1, -5, 2, -4, -2}),
                    tensor(2, 3, 1, 3),
                    getValues<T>({4,
                                  3,
                                  -2,
                                  1,
                                  1,
                                  0,
                                  1,
                                  3,
                                  3,
                                  3,
                                  4,
                                  1,
                                  2,
                                  -2,
                                  -2,
                                  3,
                                  2,
                                  2}),
                    std::vector<int32_t>{1,
                                         3,
                                         4,
                                         1,
                                         3,
                                         6,
                                         1,
                                         4,
                                         4,
                                         2,
                                         3,
                                         6,
                                         0,
                                         4,
                                         4,
                                         1,
                                         4,
                                         4}
            },
            {format::bfyx,
                    tensor(1, 3, 10, 7),
                    getValues<T>({0, -2, -5, -5, 2, 3, 2, -3, 1, -2, -4, -1, -1, -1, 2, -4, 3, -5, -1, -1, 1, 2, 4, -2,
                                  -3, -2, 0, -5, 2, -4, -1, -4, 4, 2, 1, -2, 2, -3, 0, 1, -3, 3, -1, 4, 0, 2, 0, 3,
                                  4, -4, 1, 4, -1, -5, -2, 4, -3, 3, 2, 1, 0, 4, 2, -5, 2, -5, -2, -1, 4, 2,

                                  0, 4, -2, 0, -5, -3, 4, -4, -2, -2, 2, 1, 4, 3, 2, -5, -4, -4, 0, 1, 4, -4, -3, 3,
                                  3, 4, -2, -3, -4, -2, 0, 1, -1, 3, -2, 2, 0, -3, -1, -1, 0, 0, 2, 2, -2, 1, -3, 1,
                                  2, 4, 3, -5, -4, 1, -4, 2, 0, -2, -5, 2, -3, -2, -3, -4, 2, -2, -4, 2, -4, -3,

                                  1, -5, -1, -5, 2, 1, 3, 4, 3, 0, -5, 4, -3, -4, -1, 2, -4, 2, 0, -5, -3, 0, 2, -3,
                                  -5, 3, -2, -1, -5, -4, -5, 0, -5, -1, -3, 3, 3, -4, -3, -4, -5, 4, -1, 1, -1, -4, 1,
                                  -3,
                                  -4, -1, -2, -3, -5, 2, 2, -5, 1, 1, -5, -4, 0, 2, 4, 2, 0, 2, 4, 0, -5, 2}),
                    tensor(1, 3, 3, 3),
                    getValues<T>({4, 3, 3, 4, 4, 4, 4, 4, 4,
                                  4, 4, 4, 4, 4, 4, 3, 2, 4,
                                  4, 3, 4, 4, 3, 3, 4, 4, 4}),
                    std::vector<int32_t>{22, 5, 16, 22, 43, 48, 43, 43, 48,
                                         1, 6, 6, 20, 25, 49, 50, 43, 49,
                                         11, 6, 7, 41, 25, 36, 41, 66, 66}
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
                    getValues<T>({4, 4, 4, 4, 3, 3, 4, 3,
                                  4, 4, 4, 4, 4, 4, 4, 4,
                                  4, 3, 4, 3, 4, 3, 4, 3,
                                  3, 1, 4, 4, 3, 3, 4, 3}),
                    std::vector<int32_t>{4, 4, 4, 4, 22, 22, 24, 22,
                                         3, 14, 3, 8, 18, 14, 22, 14,
                                         0, 13, 12, 13, 12, 13, 12, 13,
                                         3, 2, 7, 7, 22, 22, 24, 22}
            },
    };
    return result;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<AdaptiveMaxPoolingParams<T> > &param) {
        std::stringstream buf;
        buf << " input tensor " << param.param.inputTensor.to_string()
            << " output tensor " << param.param.outputTensor.to_string();
        return buf.str();
    }
};

using adaptive_max_pooling_test_f32 = adaptive_max_pooling_test<float>;
using adaptive_max_pooling_test_f16 = adaptive_max_pooling_test<half_t>;

TEST_P(adaptive_max_pooling_test_f32, adaptive_max_pooling_test_f32) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(adaptive_max_pooling_test_f16, adaptive_max_pooling_test_f16) {
    ASSERT_NO_FATAL_FAILURE(test());
}


INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f32,
                         adaptive_max_pooling_test_f32,
                         ::testing::ValuesIn(generateAdaptiveMaxPoolingParams<float>()),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f16,
                         adaptive_max_pooling_test_f16,
                         ::testing::ValuesIn(generateAdaptiveMaxPoolingParams<half_t>()),
                         PrintToStringParamName());
