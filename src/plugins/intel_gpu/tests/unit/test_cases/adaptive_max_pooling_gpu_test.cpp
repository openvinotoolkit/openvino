// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "openvino/reference/adaptive_avg_pool.hpp"
#include "openvino/reference/adaptive_max_pool.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/adaptive_pooling.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>

#include <cstddef>
#include <string>

using namespace cldnn;
using namespace ::tests;

namespace {
struct AdaptiveMaxPoolingParams {
    tensor inputTensor;
    tensor outputTensor;
};

using AdaptiveMaxPoolingParamsWithLayout = std::tuple<
    AdaptiveMaxPoolingParams,
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
void generateTestData(const AdaptiveMaxPoolingParams& p, const format fmt, const std::vector<float>& random_inputs,
                      std::vector<T>& inputs, std::vector<T>& outputs, std::vector<int32_t>& indices) {
    std::vector<float> out(p.outputTensor.count());
    std::vector<int32_t> ind(p.outputTensor.count());

    const auto inShape = tensorToShape(p.inputTensor, fmt);
    const auto outShape = tensorToShape(p.outputTensor, fmt);

    ov::reference::adaptive_max_pool<float, int32_t>(random_inputs.data(), out.data(), ind.data(), inShape, outShape);

    inputs = getValues<T>(random_inputs);
    outputs = getValues<T>(out);
    indices = ind;
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
    std::string operator()(const testing::TestParamInfo<AdaptiveMaxPoolingParamsWithLayout>& param) {
        std::stringstream buf;

        const auto& [p, plain_layout, target_layout, is_caching_test] = param.param;
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
struct adaptive_max_pooling_test
        : public ::testing::TestWithParam<AdaptiveMaxPoolingParamsWithLayout> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

public:
    void test() {
        const auto data_type = ov::element::from<T>();

        const auto& [params, plain_layout, target_layout, is_caching_test] = this->GetParam();
        const bool need_reorder = target_layout != plain_layout;

        std::vector<T> input_data;
        std::vector<T> expected;
        std::vector<int32_t> expected_indices;
        auto random_in = rg.generate_random_1d<float>(params.inputTensor.count(), -127, 127, 8);
        generateTestData<T>(params, plain_layout, random_in, input_data, expected, expected_indices);
        auto& engine = get_test_engine();

        auto input_mem = engine.allocate_memory({data_type, plain_layout, params.inputTensor});
        const layout indices_layout{data_types::i32, target_layout, params.outputTensor};
        auto indices_mem = engine.allocate_memory(indices_layout);

        set_values(input_mem, input_data);

        const std::string input_data_id = "adaptive_max_input_id";
        const std::string adaptive_max_pooling_id = "adaptive_max_pooling_id";
        const std::string indices_id = "indices_id";
        topology topology;
        topology.add(input_layout(input_data_id, input_mem->get_layout()));
        topology.add(mutable_data(indices_id, indices_mem));

        std::string input_id = input_data_id;
        if (need_reorder) {
            const std::string reorder_input_id = input_data_id + "_reordered";
            topology.add(reorder(reorder_input_id, input_info(input_data_id), target_layout, data_type));
            input_id = reorder_input_id;
        }

        topology.add(adaptive_pooling(adaptive_max_pooling_id, input_info(input_id), params.outputTensor, indices_id,
                                      data_types::i32));

        std::string result_id = adaptive_max_pooling_id;
        if (need_reorder) {
            const primitive_id reorder_result_id = adaptive_max_pooling_id + "_reordered";
            topology.add(reorder(reorder_result_id, adaptive_max_pooling_id, plain_layout, data_type));
            result_id = reorder_result_id;
        }

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data(input_data_id, input_mem);

        auto result = network->execute();

        auto out_mem = result.at(result_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.outputTensor.count(), out_ptr.size());
        ASSERT_EQ(params.outputTensor.count(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_NEAR(expected[i], out_ptr[i], getError<T>())
                << "i = " << i << ", format=" << fmt_to_str(target_layout);
        }

        if (is_caching_test)
            return;

        const auto block_sizes = format::traits(target_layout).block_sizes;
        const auto index_offset = std::accumulate(block_sizes.begin(), block_sizes.end(), 1,
                                                  [](int total, const std::pair<size_t, int>& b) {
                                                      return total * b.second;
                                                  }
        );

        const auto get_reordered_indices_mem = [&, plain_layout = plain_layout]() {
            cldnn::topology reorder_topology;
            reorder_topology.add(input_layout("indices", indices_layout));
            reorder_topology.add(reorder("plane_indices", input_info("indices"), plain_layout, data_types::i32));
            cldnn::network reorder_net{engine, reorder_topology, get_test_default_config(engine)};
            reorder_net.set_input_data("indices", indices_mem);
            const auto second_output_result = reorder_net.execute();
            const auto plane_indices_mem = second_output_result.at("plane_indices").get_memory();
            return plane_indices_mem;
        };

        cldnn::mem_lock<int32_t> indices_ptr(need_reorder ? get_reordered_indices_mem() : indices_mem, get_test_stream());
        ASSERT_EQ(params.outputTensor.count(), indices_ptr.size());
        ASSERT_EQ(params.outputTensor.count(), expected_indices.size());
        for (size_t i = 0; i < expected_indices.size(); ++i) {
            ASSERT_EQ(index_offset * expected_indices[i], indices_ptr[i])
                << "i = " << i << ", format=" << fmt_to_str(target_layout);
        }
    }
};


using adaptive_max_pooling_test_f32 = adaptive_max_pooling_test<float>;
using adaptive_max_pooling_test_f16 = adaptive_max_pooling_test<ov::float16>;

TEST_P(adaptive_max_pooling_test_f32, adaptive_max_pooling_test_f32) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(adaptive_max_pooling_test_f16, adaptive_max_pooling_test_f16) {
    ASSERT_NO_FATAL_FAILURE(test());
}

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f32_2d,
                         adaptive_max_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveMaxPoolingParams>{
                                        { tensor(1, 2, 7, 3), tensor(1, 2, 3, 3) },
                                        { tensor(2, 3, 7, 3), tensor(2, 3, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f32_3d,
                         adaptive_max_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveMaxPoolingParams>{
                                        { tensor(2, 2, 7, 3, 3), tensor(2, 2, 2, 2, 2) },
                                        { tensor(2, 2, 8, 5, 4), tensor(2, 2, 3, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f16_2d,
                         adaptive_max_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveMaxPoolingParams>{
                                        { tensor(1, 2, 7, 3), tensor(1, 2, 3, 3) },
                                        { tensor(2, 3, 7, 3), tensor(2, 3, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f16_3d,
                         adaptive_max_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveMaxPoolingParams>{
                                        { tensor(2, 2, 7, 3, 3), tensor(2, 2, 2, 2, 2) },
                                        { tensor(2, 2, 8, 5, 4), tensor(2, 2, 3, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_2d_all_formats,
                         adaptive_max_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveMaxPoolingParams>{
                                        { tensor(20, 20, 7, 3), tensor(20, 20, 3, 3) },
                                        { tensor(32, 32, 7, 3), tensor(32, 32, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(layouts_2d),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_3d_all_formats,
                         adaptive_max_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveMaxPoolingParams>{
                                        { tensor(20, 20, 7, 3, 3), tensor(20, 20, 3, 3, 2) },
                                        { tensor(32, 32, 7, 3, 3), tensor(32, 32, 3, 3, 2) },
                                    }),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(layouts_3d),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(export_import,
                         adaptive_max_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<AdaptiveMaxPoolingParams>{
                                        { tensor(1, 2, 7, 3), tensor(1, 2, 3, 3) },
                                    }),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(true)),
                         PrintToStringParamName());
