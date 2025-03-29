// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/random_uniform.hpp>
#include <string>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

/**
 * Specific Random Uniform params to define the tests. Input and output should be the same type
 */
template<typename T>
struct RandomUniformParams {
    ov::Shape output_shape;
    T min_val;
    T max_val;
    uint64_t global_seed;
    uint64_t op_seed;
    std::vector<T> expected_out;
};

template<typename T>
struct random_uniform_gpu_test : public ::testing::TestWithParam<RandomUniformParams<T> > {
public:
    void test(bool is_caching_test) {

        auto data_type = ov::element::from<T>();
        RandomUniformParams<T> params = testing::TestWithParam<RandomUniformParams<T> >::GetParam();
        auto &engine = get_test_engine();

        auto format = format::get_default_format(params.output_shape.size());
        auto shape = engine.allocate_memory(
                {{1, 1, 1, static_cast<long int>(params.output_shape.size())}, ov::element::Type_t::i32, format});
        auto min_val = engine.allocate_memory(layout(data_type, format::bfyx, {1, 1, 1, 1}));
        auto max_val = engine.allocate_memory(layout(data_type, format::bfyx, {1, 1, 1, 1}));

        std::vector<int32_t> out_shapes;
        for (auto x : params.output_shape)
            out_shapes.push_back(static_cast<int32_t>(x));

        set_values(shape, out_shapes);
        set_values(min_val, {params.min_val});
        set_values(max_val, {params.max_val});

        topology topology;
        topology.add(
                random_uniform("random_uniform", { input_info("shape"), input_info("min_val"), input_info("max_val") }, data_type, params.global_seed,
                               params.op_seed, params.output_shape));
        topology.add(input_layout("shape", shape->get_layout()));
        topology.add(input_layout("min_val", min_val->get_layout()));
        topology.add(input_layout("max_val", max_val->get_layout()));
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        cldnn::network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("shape", shape);
        net->set_input_data("min_val", min_val);
        net->set_input_data("max_val", max_val);

        auto result = net->execute();

        auto out_mem = result.at("random_uniform").get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.expected_out.size(), out_ptr.size());
        for (size_t i = 0; i < params.expected_out.size(); ++i) {
            ASSERT_NEAR(params.expected_out[i], out_ptr[i], 0.0001) << "at i = " << i;
        }
    }
};

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<RandomUniformParams<T> > &param) {
        std::stringstream buf;
        buf << "output_tensor_" << param.param.output_shape
            << "_min_value_" << param.param.min_val
            << "_max_value_" << param.param.max_val
            << "_global_seed_" << param.param.global_seed
            << "_op_seed_" << param.param.op_seed;
        return buf.str();
    }

};

template<>
std::string PrintToStringParamName::operator()(const testing::TestParamInfo<RandomUniformParams<ov::float16> > &param) {
    std::stringstream buf;
    buf << "output_tensor_" << param.param.output_shape
        << "_min_value_" << static_cast<float>(param.param.min_val)
        << "_max_value_" << static_cast<float>(param.param.max_val)
        << "_global_seed_" << param.param.global_seed
        << "_op_seed_" << param.param.op_seed;
    return buf.str();
}

using random_uniform_gpu_test_i32 = random_uniform_gpu_test<int32_t>;
using random_uniform_gpu_test_i64 = random_uniform_gpu_test<int64_t>;
using random_uniform_gpu_test_f32 = random_uniform_gpu_test<float>;
using random_uniform_gpu_test_f16 = random_uniform_gpu_test<ov::float16>;

TEST_P(random_uniform_gpu_test_i32, random_int32) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(random_uniform_gpu_test_i64, random_int64) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}


TEST_P(random_uniform_gpu_test_f32, random_f32) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(random_uniform_gpu_test_f16, random_f16) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

INSTANTIATE_TEST_SUITE_P(smoke_random_uniform_int32,
                         random_uniform_gpu_test_i32,
                         ::testing::Values(
                                 RandomUniformParams<int32_t>{ov::Shape{1, 1, 3, 2}, 50, 100, 80, 100,
                                                              std::vector<int32_t>{
                                                                      65, 70, 56,
                                                                      59, 82, 92
                                                              }}
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_random_uniform_int64,
                         random_uniform_gpu_test_i64,
                         ::testing::Values(
                                 RandomUniformParams<int64_t>{ov::Shape{1, 1, 3, 4, 5}, -2600, 3700, 755,
                                                              951,
                                                              {
                                                                      2116L, -1581L, 2559L, -339L, -1660L, 519L, 90L,
                                                                      2027L, -210L, 3330L, 1831L, -1737L,
                                                                      2683L, 2661L, 3473L, 1220L, 3534L, -2384L, 2199L,
                                                                      1935L, 499L, 2861L, 2743L, 3223L,
                                                                      -531L, -836L, -65L, 3435L, 632L, 1765L, 2613L,
                                                                      1891L, 1698L, 3069L, 169L, -792L,
                                                                      -32L, 2976L, -1552L, -2588L, 3327L, -1756L, 2637L,
                                                                      -1084L, 3567L, -778L, -1465L, 2967L,
                                                                      1242L, 2672L, -1585L, -2271L, 3536L, -1502L, 400L,
                                                                      2241L, 3126L, 908L, 1073L, -2110L}}
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_random_uniform_f32,
                         random_uniform_gpu_test_f32,
                         ::testing::Values(
                                 RandomUniformParams<float>{ov::Shape{1, 1, 3, 3}, 0.0, 1.0, 150, 10,
                                                            {
                                                                    0.7011236, 0.30539632, 0.93931055,
                                                                    0.9456035, 0.11694777, 0.50770056,
                                                                    0.5197197, 0.22727466, 0.991374
                                                            }
                                 },
                                 RandomUniformParams<float>{ov::Shape{3, 3}, 0.0, 1.0, 150, 10,
                                                            {
                                                                    0.7011236, 0.30539632, 0.93931055,
                                                                    0.9456035, 0.11694777, 0.50770056,
                                                                    0.5197197, 0.22727466, 0.991374
                                                            }
                                 }
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_random_uniform_f16,
                         random_uniform_gpu_test_f16,
                         ::testing::Values(
                                 RandomUniformParams<ov::float16>{ov::Shape{1, 1, 4, 2, 3}, ov::float16(-1.5),
                                                             ov::float16(-1.0), 150, 10,
                                                             {ov::float16(-1.19726562), ov::float16(-1.09667969),
                                                              ov::float16(-1.08398438), ov::float16(-1.30859375),
                                                              ov::float16(-1.48242188), ov::float16(-1.45898438),
                                                              ov::float16(-1.22851562), ov::float16(-1.08300781),
                                                              ov::float16(-1.33203125), ov::float16(-1.14062500),
                                                              ov::float16(-1.42285156), ov::float16(-1.43554688),
                                                              ov::float16(-1.32617188), ov::float16(-1.06542969),
                                                              ov::float16(-1.29296875), ov::float16(-1.21386719),
                                                              ov::float16(-1.21289062), ov::float16(-1.03027344),
                                                              ov::float16(-1.17187500), ov::float16(-1.08886719),
                                                              ov::float16(-1.08789062), ov::float16(-1.43359375),
                                                              ov::float16(-1.17773438), ov::float16(-1.16992188)}
                                 }
                         ),
                         PrintToStringParamName());

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(random_uniform_gpu_test_i32, random_int32_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(random_uniform_gpu_test_i64, random_int64_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(random_uniform_gpu_test_f32, random_f32_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}
#endif
TEST_P(random_uniform_gpu_test_f16, random_f16_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}
