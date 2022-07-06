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

namespace {
template<typename T>
struct AdaptiveAvgPoolingParams {
    tensor inputTensor;
    std::vector<T> inputs;
    tensor outputTensor;
    std::vector<T> outputs;
};

template<typename T>
using AdaptiveAvgPoolingParamsWithLayout = std::tuple<
    AdaptiveAvgPoolingParams<T>,
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
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template<typename T>
std::vector<AdaptiveAvgPoolingParams<T>> generateAdaptiveAvgPoolingParams2D() {
    static const std::vector<AdaptiveAvgPoolingParams<T>> result = {
        {
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
        {
            tensor(1, 3, 10, 7),
            getValues<T>(
                    {-2, -3, -4, 3, -5, 4, 0, -4, -2, -4, -5, 0, -3, 0, -2, 0, 0, -5, -4, -1, 3, -1, 0, -1,
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
        }
    };
    return result;
}

template<typename T>
std::vector<AdaptiveAvgPoolingParams<T>> generateAdaptiveAvgPoolingParams3D() {
    static const std::vector<AdaptiveAvgPoolingParams<T>> result = {
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
             getValues<T>({-0.750, -0.250, -1.375, -1.125, -1.125, -0.500, -0.875, -1.250,
                           -0.375, -1.625, -1., -0.500, -0.250, -0.750, -1.875, -0.625,
                           0.125, -0.375, -1.625, -1.250, 0., -1., 0.875, -0.375,
                           -1.125, -1.375, 0.750, -1.875, -0.625, -1.125, 1.250, -1.}),
        }
};
    return result;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<AdaptiveAvgPoolingParamsWithLayout<T> > &param) {
        std::stringstream buf;
        AdaptiveAvgPoolingParams<T> p;
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
struct adaptive_avg_pooling_test
        : public ::testing::TestWithParam<AdaptiveAvgPoolingParamsWithLayout<T> > {
public:
    void test() {
        const auto data_type = type_to_data_type<T>::value;
        AdaptiveAvgPoolingParams<T> params;
        format::type plain_layout;
        format::type target_layout;
        std::tie(params, plain_layout, target_layout) = this->GetParam();
        const bool need_reorder = target_layout != plain_layout;

        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({data_type, plain_layout, params.inputTensor});

        set_values(input, params.inputs);

        const std::string input_data_id = "adaptive_avg_input_id";
        const std::string adaptive_avg_pooling_id = "adaptive_avg_pooling_id";
        topology topology;
        topology.add(input_layout(input_data_id, input->get_layout()));

        std::string input_id = input_data_id;
        if (need_reorder) {
            const std::string reorder_input_id = input_data_id + "_reordered";
            topology.add(reorder(reorder_input_id, input_data_id, target_layout, data_type));
            input_id = reorder_input_id;
        }

        topology.add(adaptive_pooling(adaptive_avg_pooling_id, input_id, params.outputTensor));

        std::string result_id = adaptive_avg_pooling_id;
        if (need_reorder) {
            const primitive_id reorder_result_id = adaptive_avg_pooling_id + "_reordered";
            topology.add(reorder(reorder_result_id, adaptive_avg_pooling_id, plain_layout, data_type));
            result_id = reorder_result_id;
        }

        network network(engine, topology);

        network.set_input_data(input_data_id, input);

        auto result = network.execute();

        auto out_mem = result.at(result_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.outputTensor.count(), out_ptr.size());
        for (size_t i = 0; i < params.outputs.size(); ++i) {
            EXPECT_NEAR(params.outputs[i], out_ptr[i], 0.005) << "at i = " << i;
        }
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


INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f32_2d,
                         adaptive_avg_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateAdaptiveAvgPoolingParams2D<float>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(layouts_2d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f32_3d,
                         adaptive_avg_pooling_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateAdaptiveAvgPoolingParams3D<float>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(layouts_3d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f16_2d,
                         adaptive_avg_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateAdaptiveAvgPoolingParams2D<half_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(layouts_2d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_avg_pooling_test_f16_3d,
                         adaptive_avg_pooling_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateAdaptiveAvgPoolingParams2D<half_t>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(layouts_3d)),
                         PrintToStringParamName());
