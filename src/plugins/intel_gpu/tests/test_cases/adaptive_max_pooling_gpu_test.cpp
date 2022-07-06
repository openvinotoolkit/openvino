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

namespace {
template<typename T>
struct AdaptiveMaxPoolingParams {
    tensor inputTensor;
    std::vector<T> inputs;
    tensor outputTensor;
    std::vector<T> outputs;
    std::vector<int32_t> output_indices;
};

template<typename T>
using AdaptiveMaxPoolingParamsWithLayout = std::tuple<
    AdaptiveMaxPoolingParams<T>,
    format::type,   // source (plain) layout - bfyx or bfzyx
    format::type    // target (blocked) layout
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

template<typename T>
std::vector<AdaptiveMaxPoolingParams<T>> generateAdaptiveMaxPoolingParams2D() {
    static const std::vector<AdaptiveMaxPoolingParams<T>> result = {
        {
            tensor(2, 3, 1, 7),
            getValues<T>({0, 4, 1, 3, -2, -5, -2, -2, 1, -3, 1, -3, -4, 0, -2, 1, -1, -2, 3, -1, -3,
                          -1, -2, 3, 4, -3, -4, 1, 2, 0, -4, -5, -2, -2, -3, 2, 3, 1, -5, 2, -4, -2}),
            tensor(2, 3, 1, 3),
            getValues<T>({4, 3, -2, 1, 1, 0, 1, 3, 3, 3, 4, 1, 2, -2, -2, 3, 2, 2}),
            std::vector<int32_t>{1, 3, 4, 1, 3, 6, 1, 4, 4, 2, 3, 6, 0, 4, 4, 1, 4, 4}
        },
        {
            tensor(1, 3, 10, 7),
            getValues<T>(
                    {0, -2, -5, -5, 2, 3, 2, -3, 1, -2, -4, -1, -1, -1, 2, -4, 3, -5, -1, -1, 1, 2, 4, -2,
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
        }
    };
    return result;
}

template<typename T>
std::vector<AdaptiveMaxPoolingParams<T>> generateAdaptiveMaxPoolingParams3D() {
    static const std::vector<AdaptiveMaxPoolingParams<T>> result = {
        {
            tensor(2, 2, 3, 3, 3),
            getValues<T>(
                    {-5, 1, -3, -4, 4, -4, 3, -3, -1, 0, 0, -2, -4, 2, 0, -4, -5, -2, -4, -4, 0, -2, 3, -3,
                     4,
                     -1, -4,
                     -1, -1, -5, 4, -1, -2, -3, 0, 4, -1, -5, -4, 1, 1, 4, -5, -5, -5, 4, -3, -3, -3, 4, 0,
                     -3,
                     -5, 1,
                     4, 2, 1, -5, -5, 1, 0, -4, -1, 2, -4, -2, 4, 3, 1, -3, -3, -2, -4, -3, -3, 3, -1, 1, 2,
                     2,
                     -4,
                     -5, -4, 1, 3, -4, -1, 2, 4, -5, 0, 1, -2, 0, 0, -2, 3, -2, -5, -3, -5, -2, -1, 3, -2,
                     4, 3,
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
        }
    };
    return result;
}


struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<AdaptiveMaxPoolingParamsWithLayout<T>>& param) {
        std::stringstream buf;
        AdaptiveMaxPoolingParams<T> p;
        format::type plain_layout;
        format::type target_layout;
        std::tie(p, plain_layout, target_layout) = param.param;
        buf << " input tensor " << p.inputTensor.to_string()
            << " output tensor " << p.outputTensor.to_string()
            << " plain layout " << plain_layout
            << " target layout " << target_layout;
        return buf.str();
    }
};
};  // namespace

template<typename T>
struct adaptive_max_pooling_test
        : public ::testing::TestWithParam<AdaptiveMaxPoolingParamsWithLayout<T>> {
public:
    void test() {
        const auto data_type = type_to_data_type<T>::value;
        AdaptiveMaxPoolingParams<T> params;
        format::type plain_layout;
        format::type target_layout;
        std::tie(params, plain_layout, target_layout) = this->GetParam();
        const bool need_reorder = target_layout != plain_layout;

        auto& engine = get_test_engine();

        auto input_mem = engine.allocate_memory({data_type, plain_layout, params.inputTensor});
        const layout indices_layout{data_types::i32, target_layout, params.outputTensor};
        auto indices_mem = engine.allocate_memory(indices_layout);

        set_values(input_mem, params.inputs);

        const std::string input_data_id = "adaptive_max_input_id";
        const std::string adaptive_max_pooling_id = "adaptive_max_pooling_id";
        const std::string indices_id = "indices_id";
        topology topology;
        topology.add(input_layout(input_data_id, input_mem->get_layout()));
        topology.add(mutable_data(indices_id, indices_mem));

        std::string input_id = input_data_id;
        if (need_reorder) {
            const std::string reorder_input_id = input_data_id + "_reordered";
            topology.add(reorder(reorder_input_id, input_data_id, target_layout, data_type));
            input_id = reorder_input_id;
        }

        topology.add(adaptive_pooling(adaptive_max_pooling_id, input_id, params.outputTensor, indices_id,
                                      data_types::i32));

        std::string result_id = adaptive_max_pooling_id;
        if (need_reorder) {
            const primitive_id reorder_result_id = adaptive_max_pooling_id + "_reordered";
            topology.add(reorder(reorder_result_id, adaptive_max_pooling_id, plain_layout, data_type));
            result_id = reorder_result_id;
        }

        network network(engine, topology);

        network.set_input_data(input_data_id, input_mem);

        auto result = network.execute();

        auto out_mem = result.at(result_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.outputTensor.count(), out_ptr.size());
        for (size_t i = 0; i < params.outputs.size(); ++i) {
            EXPECT_NEAR(params.outputs[i], out_ptr[i], 0.005) << "at i = " << i;
        }

        const auto& expected_indices = params.output_indices;
        const auto block_sizes = format::traits(target_layout).block_sizes;
        const auto index_offset = std::accumulate(block_sizes.begin(), block_sizes.end(), 1u,
                                                  [](size_t total, const std::pair<size_t, int>& b) {
                                                      return total * b.second;
                                                  }
        );

        const auto get_reordered_indices_mem = [&]() {
            cldnn::topology reorder_topology;
            reorder_topology.add(input_layout("indices", indices_layout));
            reorder_topology.add(reorder("plane_indices", "indices", plain_layout, data_types::i32));
            cldnn::network reorder_net{engine, reorder_topology};
            reorder_net.set_input_data("indices", indices_mem);
            const auto second_output_result = reorder_net.execute();
            const auto plane_indices_mem = second_output_result.at("plane_indices").get_memory();
            return plane_indices_mem;
        };

        cldnn::mem_lock<int32_t> indices_ptr(need_reorder ? get_reordered_indices_mem() : indices_mem, get_test_stream());
        ASSERT_EQ(params.outputTensor.count(), indices_ptr.size());
        for (size_t i = 0; i < expected_indices.size(); ++i) {
            EXPECT_EQ(index_offset * expected_indices[i], indices_ptr[i]) << "at i = " << i;
        }
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


INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f32_2d,
                         adaptive_max_pooling_test_f32,
                         ::testing::Combine(
                            ::testing::ValuesIn(generateAdaptiveMaxPoolingParams2D<float>()),
                            ::testing::Values(format::bfyx),
                            ::testing::ValuesIn(layouts_2d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f32_3d,
                         adaptive_max_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateAdaptiveMaxPoolingParams3D<float>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(layouts_3d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f16_2d,
                         adaptive_max_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateAdaptiveMaxPoolingParams2D<half_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(layouts_2d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_max_pooling_test_f16_3d,
                         adaptive_max_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateAdaptiveMaxPoolingParams2D<half_t>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(layouts_3d)),
                         PrintToStringParamName());
