// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "openvino/reference/adaptive_avg_pool.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/adaptive_pooling.hpp>

#include <cstddef>
#include <string>

using namespace cldnn;
using namespace ::tests;

namespace {
struct AdaptiveAvgPoolingParams {
    tensor inputTensor;
    tensor outputTensor;
};

using AdaptiveAvgPoolingParamsWithLayout = std::tuple<
    AdaptiveAvgPoolingParams,
    format::type,   // source (plain) layout - bfyx or bfzyx
    format::type,   // target (blocked) layout
    bool            // is_caching_test
>;

const std::vector<format::type> layouts_2d = {
    format::bfyx,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv16,
    format::bs_fs_yx_bsv32_fsv32
};

const std::vector<format::type> layouts_3d = {
    format::bfzyx,
    format::b_fs_zyx_fsv16,
    format::b_fs_zyx_fsv32,
    format::bs_fs_zyx_bsv16_fsv32,
    format::bs_fs_zyx_bsv16_fsv16,
    format::bs_fs_zyx_bsv32_fsv32,
    format::bs_fs_zyx_bsv32_fsv16
};

template<typename T>
std::vector<T> getValues(const std::vector<float>& values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

ov::Shape tensorToShape(const tensor& t, const format f)
{
    std::vector<int> vec(cldnn::format::dimension(f));
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = t.sizes()[i];
    }
    std::reverse(vec.begin() + 2, vec.end());

    return ov::Shape(vec.begin(), vec.end());
}

template<typename T>
void generateTestData(const AdaptiveAvgPoolingParams& p, const format fmt, const std::vector<float>& random_inputs, std::vector<T>& inputs, std::vector<T>& outputs) {
    std::vector<float> out(p.outputTensor.count());

    const auto inShape = tensorToShape(p.inputTensor, fmt);
    const auto outShape = tensorToShape(p.outputTensor, fmt);

    ov::reference::adaptive_avg_pool<float>(random_inputs.data(), out.data(), inShape, outShape);

    inputs = getValues<T>(random_inputs);
    outputs = getValues<T>(out);
}

template <typename T> float getError();

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<ov::float16>() {
    return 0.5;
}

struct PrintToStringParamName {
    std::string operator()(const testing::TestParamInfo<AdaptiveAvgPoolingParamsWithLayout> &param) {
        std::stringstream buf;
        AdaptiveAvgPoolingParams p;
        format::type plain_layout;
        format::type target_layout;
        bool is_caching_test;
        std::tie(p, plain_layout, target_layout, is_caching_test) = param.param;
        buf << " input tensor " << p.inputTensor.to_string()
            << " output tensor " << p.outputTensor.to_string()
            << " plain layout " << plain_layout
            << " target layout " << target_layout
            << " is_caching_test " << is_caching_test;
        return buf.str();
    }
};
};  // namespace

template<typename T>
struct adaptive_avg_pooling_test
        : public ::testing::TestWithParam<AdaptiveAvgPoolingParamsWithLayout> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

public:
    void test() {
        const auto data_type = ov::element::from<T>();
        AdaptiveAvgPoolingParams params;
        format::type plain_layout;
        format::type target_layout;
        bool is_caching_test;
        std::tie(params, plain_layout, target_layout, is_caching_test) = this->GetParam();

        std::vector<T> input_data;
        std::vector<T> expected;
        const std::vector<float> random_input_data = rg.generate_random_1d<float>(params.inputTensor.count(), -127, 127, 8);
        generateTestData<T>(params, plain_layout, random_input_data, input_data, expected);
        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({data_type, plain_layout, params.inputTensor});

        set_values(input, input_data);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("input_reordered", input_info("input"), target_layout, data_type));
        topology.add(adaptive_pooling("adaptive_avg_pooling_blocked", input_info("input_reordered"), params.outputTensor));
        topology.add(reorder("adaptive_avg_pooling", input_info("adaptive_avg_pooling_blocked"), plain_layout, data_type));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto result = network->execute();

        auto out_mem = result.at("adaptive_avg_pooling").get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.outputTensor.count(), out_ptr.size());
        ASSERT_EQ(params.outputTensor.count(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_NEAR(expected[i], out_ptr[i], getError<T>())
                << "i = " << i << ", format=" << fmt_to_str(target_layout);
        }
    }
};


using adaptive_avg_pooling_test_f32 = adaptive_avg_pooling_test<float>;
using adaptive_avg_pooling_test_f16 = adaptive_avg_pooling_test<ov::float16>;

TEST_P(adaptive_avg_pooling_test_f32, adaptive_avg_pooling_test_f32) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(adaptive_avg_pooling_test_f16, adaptive_avg_pooling_test_f16) {
    ASSERT_NO_FATAL_FAILURE(test());
}

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f32_2d,
                         adaptive_avg_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveAvgPoolingParams>{
                                        { tensor(1, 2, 7, 3), tensor(1, 2, 3, 3) },
                                        { tensor(2, 3, 7, 3), tensor(2, 3, 3, 3) },
                                        { tensor(1, 3, 7, 7), tensor(1, 3, 7, 7) },
                                    }),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f32_3d,
                         adaptive_avg_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveAvgPoolingParams>{
                                        { tensor(2, 2, 7, 3, 3), tensor(2, 2, 2, 2, 2) },
                                        { tensor(2, 2, 8, 5, 4), tensor(2, 2, 3, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f16_2d,
                         adaptive_avg_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveAvgPoolingParams>{
                                        { tensor(1, 2, 7, 3), tensor(1, 2, 3, 3) },
                                        { tensor(2, 3, 7, 3), tensor(2, 3, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f16_3d,
                         adaptive_avg_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveAvgPoolingParams>{
                                        { tensor(2, 2, 7, 3, 3), tensor(2, 2, 2, 2, 2) },
                                        { tensor(2, 2, 8, 5, 4), tensor(2, 2, 3, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_2d_all_formats,
                         adaptive_avg_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveAvgPoolingParams>{
                                        { tensor(20, 20, 7, 3), tensor(20, 20, 3, 3) },
                                        { tensor(32, 32, 7, 3), tensor(32, 32, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(layouts_2d),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_3d_all_formats,
                         adaptive_avg_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveAvgPoolingParams>{
                                        { tensor(20, 20, 7, 3, 3), tensor(20, 20, 3, 3, 2) },
                                        { tensor(32, 32, 7, 3, 3), tensor(32, 32, 3, 3, 2) },
                                    }),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(layouts_3d),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(export_import,
                         adaptive_avg_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveAvgPoolingParams>{
                                        { tensor(1, 2, 7, 3), tensor(1, 2, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(true)),
                         PrintToStringParamName());
