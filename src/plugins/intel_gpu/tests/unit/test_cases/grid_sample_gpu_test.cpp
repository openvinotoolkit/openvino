// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "intel_gpu/primitives/grid_sample.hpp"
#include "test_utils/test_utils.h"

#ifdef RUN_ALL_MODEL_CACHING_TESTS
    #define RUN_CACHING_TEST false, true
#else
    #define RUN_CACHING_TEST false
#endif

using namespace cldnn;
using namespace ::tests;

namespace {

template <class TD, class TG>
struct grid_sample_test_inputs {
    std::vector<tensor::value_type> data_shape;
    std::vector<TD> data;
    std::vector<tensor::value_type> grid_shape;
    std::vector<TG> grid;
    GridSampleOp::Attributes attributes;
    std::vector<TD> expected_values;
    std::string test_name;
};

template <class TD, class TG>
using grid_sample_test_params = std::tuple<grid_sample_test_inputs<TD, TG>, format::type, bool>;

template <class T>
float getError();

template <>
float getError<float>() {
    return 0.001f;
}

template <>
float getError<ov::float16>() {
    return 0.5f;
}

template <class TD, class TG>
struct grid_sample_gpu_test : public testing::TestWithParam<grid_sample_test_params<TD, TG>> {
public:
    void test() {
        const auto& [p, fmt, is_caching_test] = testing::TestWithParam<grid_sample_test_params<TD, TG>>::GetParam();

        auto& engine = get_test_engine();
        const auto data_data_type = ov::element::from<TD>();
        const auto grid_data_type = ov::element::from<TG>();
        const auto plane_format = format::bfyx;

        const layout data_layout(data_data_type, plane_format, tensor(plane_format, p.data_shape));
        auto data = engine.allocate_memory(data_layout);
        set_values(data, p.data);

        const layout grid_layout(grid_data_type, plane_format, tensor(plane_format, p.grid_shape));
        auto grid = engine.allocate_memory(grid_layout);
        set_values(grid, p.grid);

        topology topology;
        topology.add(input_layout("data", data->get_layout()));
        topology.add(input_layout("grid", grid->get_layout()));
        topology.add(reorder("reordered_data", input_info("data"), fmt, data_data_type));
        topology.add(reorder("reordered_grid", input_info("grid"), fmt, grid_data_type));
        topology.add(grid_sample("grid_sample", { input_info("reordered_data"), input_info("reordered_grid") }, p.attributes));
        topology.add(reorder("plane_grid_sample", input_info("grid_sample"), plane_format, data_data_type));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        network->set_input_data("data", data);
        network->set_input_data("grid", grid);
        const auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), std::size_t(1));
        ASSERT_EQ(outputs.begin()->first, "plane_grid_sample");

        auto output = outputs.at("plane_grid_sample").get_memory();
        cldnn::mem_lock<TD> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), p.expected_values.size());
        for (size_t i = 0; i < output_ptr.size(); ++i) {
            ASSERT_NEAR(p.expected_values[i], output_ptr[i], getError<TD>());
        }
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<grid_sample_test_params<TD, TG>>& info) {
        const auto& [p, fmt, is_caching_test] = info.param;

        std::ostringstream result;
        result << "TestName=" << p.test_name << ";";
        result << "Format=" << fmt_to_str(fmt) << ";";
        result << "Cached=" << bool_to_str(is_caching_test) << ";";
        return result.str();
    }
};

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getNearestParamsOddDimensionsInnerGrids() {
    return {
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::ZEROS},
         {8, 8, 8, 8, 2, 12, 4, 14, 1, 11, 5, 15},
         "nearest_zeros_noalign_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::ZEROS},
         {8, 8, 8, 8, 2, 12, 4, 14, 1, 11, 5, 15},
         "nearest_zeros_align_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::BORDER},
         {8, 8, 8, 8, 2, 12, 4, 14, 1, 11, 5, 15},
         "nearest_border_noalign_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::BORDER},
         {8, 8, 8, 8, 2, 12, 4, 14, 1, 11, 5, 15},
         "nearest_border_align_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::REFLECTION},
         {8, 8, 8, 8, 2, 12, 4, 14, 1, 11, 5, 15},
         "nearest_reflection_noalign_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::REFLECTION},
         {8, 8, 8, 8, 2, 12, 4, 14, 1, 11, 5, 15},
         "nearest_reflection_align_odd_dims_inner"},
    };
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getNearestParamsOddDimensionsOuterGrids() {
    return {
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::ZEROS},
         {0, 0, 0, 0, 0, 0, 0},
         "nearest_zeros_noalign_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::ZEROS},
         {0, 0, 0, 0, 0, 0, 0},
         "nearest_zeros_align_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::BORDER},
         {1, 11, 11, 14, 15, 10, 5},
         "nearest_border_noalign_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::BORDER},
         {1, 6, 11, 14, 15, 10, 5},
         "nearest_border_align_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::REFLECTION},
         {8, 14, 1, 4, 14, 6, 5},
         "nearest_reflection_noalign_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::REFLECTION},
         {8, 9, 6, 4, 14, 6, 4},
         "nearest_reflection_align_odd_dims_outer"},
    };
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getNearestParamsEvenDimensionsInnerGrids() {
    return {
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::ZEROS},
         {2, 14, 5, 17, 0, 0, 9, 16},
         "nearest_zeros_noalign_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::ZEROS},
         {8, 14, 11, 17, 19, 6, 9, 16},
         "nearest_zeros_align_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::BORDER},
         {2, 14, 5, 17, 19, 6, 9, 16},
         "nearest_border_noalign_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::BORDER},
         {8, 14, 11, 17, 19, 6, 9, 16},
         "nearest_border_align_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::REFLECTION},
         {2, 14, 5, 17, 19, 6, 9, 16},
         "nearest_reflection_noalign_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::REFLECTION},
         {8, 14, 11, 17, 19, 6, 9, 16},
         "nearest_reflection_align_even_dims_inner"},
    };
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getBilinearParamsOddDimensionsInnerGrids() {
    return {
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::ZEROS},
         {7, 8.5f, 7.5f, 9, 3, 10.5f, 5.5f, 13, 0.25f, 2.75f, 1.25f, 3.75f},
         "bilinear_zeros_noalign_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {true, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::ZEROS},
         {7.3f, 8.3f, 7.7f, 8.7f, 4.5f, 9.5f, 6.5f, 11.5f, 1, 11, 5, 15},
         "bilinear_zeros_align_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::BORDER},
         {7, 8.5f, 7.5f, 9, 3, 10.5f, 5.5f, 13, 1, 11, 5, 15},
         "bilinear_border_noalign_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {true, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::BORDER},
         {7.3f, 8.3f, 7.7f, 8.7f, 4.5f, 9.5f, 6.5f, 11.5f, 1, 11, 5, 15},
         "bilinear_border_align_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::REFLECTION},
         {7, 8.5f, 7.5f, 9, 3, 10.5f, 5.5f, 13, 1, 11, 5, 15},
         "bilinear_reflection_noalign_odd_dims_inner"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 3, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f, -0.5f, -0.5f, 0.5f,
          0.5f,  -0.5f, 0.5f,  0.5f, -1.f, -1.f,  -1.f, 1.f,  1.f,   -1.f,  1.f,   1.f},
         {true, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::REFLECTION},
         {7.3f, 8.3f, 7.7f, 8.7f, 4.5f, 9.5f, 6.5f, 11.5f, 1, 11, 5, 15},
         "bilinear_reflection_align_odd_dims_inner"},
    };
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getBilinearParamsOddDimensionsOuterGrids() {
    return {
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::ZEROS},
         {0, 0, 0, 0, 0, 0, 0},
         "bilinear_zeros_noalign_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {true, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::ZEROS},
         {0, 0, 0, 0, 0, 0, 1.9880099f},
         "bilinear_zeros_align_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::BORDER},
         {1, 8.775f, 11, 14.25f, 15, 8.725f, 5},
         "bilinear_border_noalign_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {true, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::BORDER},
         {1, 7.85f, 11, 14, 15, 9.15f, 5},
         "bilinear_border_align_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::REFLECTION},
         {5.9999995f, 11.9f, 2.7000031f, 5.1250005f, 13.75f, 4.725f, 4.7475f},
         "bilinear_reflection_noalign_odd_dims_outer"},
        {{1, 1, 3, 5},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {1, 1, 7, 2},
         {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
         {true, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::REFLECTION},
         {6.7f, 10.75f, 3.800002f, 6.25f, 13.099999f, 5.15f, 4.4030004f},
         "bilinear_reflection_align_odd_dims_outer"},
    };
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getBilinearParamsEvenDimensionsInnerGrids() {
    return {
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::ZEROS},
         {5, 17, 8, 20, 4.75f, 1.5f, 11, 14},
         "bilinear_zeros_noalign_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {true, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::ZEROS},
         {6.75f, 15.75f, 9.25f, 18.25f, 19, 6, 11.35f, 13.65f},
         "bilinear_zeros_align_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::BORDER},
         {5, 17, 8, 20, 19, 6, 11, 14},
         "bilinear_border_noalign_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {true, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::BORDER},
         {6.75f, 15.75f, 9.25f, 18.25f, 19, 6, 11.35f, 13.65f},
         "bilinear_border_align_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::REFLECTION},
         {5, 17, 8, 20, 19, 6, 11, 14},
         "bilinear_reflection_noalign_even_dims_inner"},
        {{1, 1, 4, 6},
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
         {1, 1, 8, 2},
         {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -1.f, 1.f, 1.f, -1.f, -0.1f, -0.1f, 0.1f, 0.1f},
         {true, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::REFLECTION},
         {6.75f, 15.75f, 9.25f, 18.25f, 19, 6, 11.35f, 13.65f},
         "bilinear_reflection_align_even_dims_inner"},
    };
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getBicubicParams() {
    return {
        {{1, 1, 4, 7},
         {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 3, 5, 3, 2, 1, 1, 2, 5, 9, 5, 2, 1},
         {1, 4, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f,  -0.5f, -0.5f, 0.5f,   0.5f,   -0.5f,  0.5f, 0.5f,
          -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f, -1.75f, 0.7f,  1.33f, -1.11f, 0.965f, 1.007f, 21.f, 37.f},
         {false, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::ZEROS},
         {2.6663566f,
          3.527928f,
          2.6663566f,
          3.527928f,
          1.6318359f,
          2.7156982f,
          1.6318359f,
          2.7156982f,
          0.6378987f,
          0.57033366f,
          0.6378987f,
          0.57033366f,
          0,
          -0.01507522f,
          0.25528803f,
          0},
         "bicubic_zeros_noalign"},
        {{1, 1, 4, 7},
         {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 3, 5, 3, 2, 1, 1, 2, 5, 9, 5, 2, 1},
         {1, 4, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f,  -0.5f, -0.5f, 0.5f,   0.5f,   -0.5f,  0.5f, 0.5f,
          -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f, -1.75f, 0.7f,  1.33f, -1.11f, 0.965f, 1.007f, 21.f, 37.f},
         {true, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::ZEROS},
         {2.7887204f,
          3.4506166f,
          2.7887204f,
          3.4506166f,
          1.8481445f,
          2.7364502f,
          1.8481445f,
          2.7364502f,
          1.2367951f,
          1.3602872f,
          1.2367951f,
          1.3602872f,
          0,
          0.00650583f,
          1.1182348f,
          0},
         "bicubic_zeros_align"},
        {{1, 1, 4, 7},
         {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 3, 5, 3, 2, 1, 1, 2, 5, 9, 5, 2, 1},
         {1, 4, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f,  -0.5f, -0.5f, 0.5f,   0.5f,   -0.5f,  0.5f, 0.5f,
          -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f, -1.75f, 0.7f,  1.33f, -1.11f, 0.965f, 1.007f, 21.f, 37.f},
         {false, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::BORDER},
         {2.6663566f,
          3.527928f,
          2.6663566f,
          3.527928f,
          1.5380859f,
          2.4677734f,
          1.5380859f,
          2.4677734f,
          1.0089612f,
          0.91871876f,
          1.0089612f,
          0.91871876f,
          1,
          1,
          0.8902873f,
          1},
         "bicubic_border_noalign"},
        {{1, 1, 4, 7},
         {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 3, 5, 3, 2, 1, 1, 2, 5, 9, 5, 2, 1},
         {1, 4, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f,  -0.5f, -0.5f, 0.5f,   0.5f,   -0.5f,  0.5f, 0.5f,
          -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f, -1.75f, 0.7f,  1.33f, -1.11f, 0.965f, 1.007f, 21.f, 37.f},
         {true, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::BORDER},
         {2.7887204f,
          3.4506166f,
          2.7887204f,
          3.4506166f,
          1.8129883f,
          2.623291f,
          1.8129883f,
          2.623291f,
          1.0363026f,
          1.1486388f,
          1.0363026f,
          1.1486388f,
          1,
          1.0000064f,
          1.0641243f,
          1},
         "bicubic_border_align"},
        {{1, 1, 4, 7},
         {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 3, 5, 3, 2, 1, 1, 2, 5, 9, 5, 2, 1},
         {1, 4, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f,  -0.5f, -0.5f, 0.5f,   0.5f,   -0.5f,  0.5f, 0.5f,
          -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f, -1.75f, 0.7f,  1.33f, -1.11f, 0.965f, 1.007f, 21.f, 37.f},
         {false, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::REFLECTION},
         {2.6663566f,
          3.527928f,
          2.6663566f,
          3.527928f,
          1.5380859f,
          2.4677734f,
          1.5380859f,
          2.4677734f,
          1.0150609f,
          0.904375f,
          1.0150609f,
          0.904375f,
          5.48851f,
          0.898316f,
          0.8237547f,
          0.8125f},
         "bicubic_reflection_noalign"},
        {{1, 1, 4, 7},
         {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 3, 5, 3, 2, 1, 1, 2, 5, 9, 5, 2, 1},
         {1, 4, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f,  -0.5f, -0.5f, 0.5f,   0.5f,   -0.5f,  0.5f, 0.5f,
          -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f, -1.75f, 0.7f,  1.33f, -1.11f, 0.965f, 1.007f, 21.f, 37.f},
         {true, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::REFLECTION},
         {2.7887204f,
          3.4506166f,
          2.7887204f,
          3.4506166f,
          1.7745361f,
          2.6518555f,
          1.7745361f,
          2.6518555f,
          1.0085088f,
          1.0307077f,
          1.0085088f,
          1.0307077f,
          5.5649586f,
          1.0553409f,
          1.0011607f,
          1},
         "bicubic_reflection_align"},
    };
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getBicubicBatchesParams() {
    return {
        {{2, 2, 4, 3},
         {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
         {2, 2, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f,  -0.5f, -0.5f, 0.5f,   0.5f,   -0.5f,  0.5f, 0.5f,
          -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f, -1.75f, 0.7f,  1.33f, -1.11f, 0.965f, 1.007f, 21.f, 37.f},
         {true, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::BORDER},
         {6.0096254f, 6.7048755f, 6.2951245f, 6.9903746f, 3.4101562f, 8.402344f,  4.5976562f, 9.589844f,
          18.009624f, 18.704876f, 18.295124f, 18.990376f, 15.410156f, 20.402344f, 16.597656f, 21.589844f,
          25.415281f, 33.735218f, 27.26478f,  35.58472f,  32.884f,    26.852259f, 35.996872f, 36.f,
          37.41528f,  45.735218f, 39.264782f, 47.58472f,  44.884f,    38.852257f, 47.996872f, 48.f},
         "bicubic_border_align_batches"},
        {{2, 2, 4, 3},
         {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
         {2, 2, 4, 2},
         {-0.1f, -0.1f, -0.1f, 0.1f, 0.1f, -0.1f, 0.1f, 0.1f, -0.5f,  -0.5f, -0.5f, 0.5f,   0.5f,   -0.5f,  0.5f, 0.5f,
          -0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f, -1.75f, 0.7f,  1.33f, -1.11f, 0.965f, 1.007f, 21.f, 37.f},
         {false, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::REFLECTION},
         {5.8170314f, 6.7650313f, 6.2349687f, 7.182969f,  2.4101562f, 8.972656f,  4.0273438f, 10.589844f,
          17.81703f,  18.765032f, 18.234968f, 19.18297f,  14.410156f, 20.972656f, 16.027344f, 22.589844f,
          24.356874f, 34.301876f, 26.698126f, 36.643124f, 34.304035f, 26.55013f,  36.74749f,  36.75f,
          36.356876f, 46.301876f, 38.698124f, 48.643124f, 46.304035f, 38.55013f,  48.74749f,  48.75f},
         "bicubic_reflection_noalign_batches"},
    };
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getCornerCaseData1x1Params() {
    return {
        {{1, 1, 1, 1},
         {7},
         {1, 1, 5, 2},
         {1, -1, 0, 0, -1, 0, 0.5f, 0.5f, 2, -4},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::ZEROS},
         {1.75f, 7, 3.5f, 3.9375f, 0},
         "bilinear_zeros_no_align_data1x1"},
        {{1, 1, 1, 1},
         {7},
         {1, 1, 5, 2},
         {1, -1, 0, 0, -1, 0, 0.5f, 0.5f, 2, -4},
         {false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::ZEROS},
         {7, 7, 7, 7, 0},
         "nearest_zeros_no_align_data1x1"},
        {{1, 1, 1, 1},
         {7},
         {1, 1, 5, 2},
         {1, -1, 0, 0, -1, 0, 0.5f, 0.5f, 2, -4},
         {false, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::ZEROS},
         {2.4677734f, 7, 4.15625f, 5.4073334f, 0},
         "bicubic_zeros_no_align_data1x1"},
        {{1, 1, 1, 1},
         {7},
         {1, 1, 5, 2},
         {1, -1, 0, 0, -1, 0, 0.5f, 0.5f, 2, -4},
         {true, GridSampleOp::InterpolationMode::BICUBIC, GridSampleOp::PaddingMode::ZEROS},
         {7, 7, 7, 7, 7},
         "bicubic_zeros_align_data1x1"},
        {{1, 1, 1, 1},
         {7},
         {1, 1, 5, 2},
         {1, -1, 0, 0, -1, 0, 0.5f, 0.5f, 2, -4},
         {false, GridSampleOp::InterpolationMode::BILINEAR, GridSampleOp::PaddingMode::REFLECTION},
         {7, 7, 7, 7, 7},
         "bilinear_reflection_noalign_data1x1"},
        {{1, 1, 1, 1},
         {7},
         {1, 1, 5, 2},
         {1, -1, 0, 0, -1, 0, 0.5f, 0.5f, 2, -4},
         {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::BORDER},
         {7, 7, 7, 7, 7},
         "nearest_border_align_data1x1"},
    };
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getParamsToCheckLayouts() {
    auto nearestParamsOddDimsInner = getNearestParamsOddDimensionsInnerGrids<TD, TG>();
    auto bilinearParamsOddDimsInner = getBilinearParamsOddDimensionsInnerGrids<TD, TG>();
    auto bicubicParams = getBicubicParams<TD, TG>();

    auto all = std::move(nearestParamsOddDimsInner);
    std::move(bilinearParamsOddDimsInner.begin(), bilinearParamsOddDimsInner.end(), std::back_inserter(all));
    std::move(bicubicParams.begin(), bicubicParams.end(), std::back_inserter(all));

    return all;
}

template <class TD, class TG>
std::vector<grid_sample_test_inputs<TD, TG>> getParamsToCheckLogic() {
    auto nearestParamsOddDimsOuter = getNearestParamsOddDimensionsOuterGrids<TD, TG>();
    auto nearestParamsEvenDimsInner = getNearestParamsEvenDimensionsInnerGrids<TD, TG>();
    auto bilinearParamsOddDimsOuter = getBilinearParamsOddDimensionsOuterGrids<TD, TG>();
    auto bilinearParamsEvenDimsInner = getBilinearParamsEvenDimensionsInnerGrids<TD, TG>();
    auto bicubicBatchesParams = getBicubicBatchesParams<TD, TG>();
    auto cornerCaseParams = getCornerCaseData1x1Params<TD, TG>();

    auto all = std::move(nearestParamsOddDimsOuter);
    std::move(nearestParamsEvenDimsInner.begin(), nearestParamsEvenDimsInner.end(), std::back_inserter(all));
    std::move(bilinearParamsOddDimsOuter.begin(), bilinearParamsOddDimsOuter.end(), std::back_inserter(all));
    std::move(bilinearParamsEvenDimsInner.begin(), bilinearParamsEvenDimsInner.end(), std::back_inserter(all));
    std::move(bicubicBatchesParams.begin(), bicubicBatchesParams.end(), std::back_inserter(all));
    std::move(cornerCaseParams.begin(), cornerCaseParams.end(), std::back_inserter(all));

    return all;
}

const std::vector<format::type> layout_formats = {
    format::bfyx,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv32,
    format::bs_fs_yx_bsv32_fsv16,
};

using grid_sample_gpu_test_float_float = grid_sample_gpu_test<float, float>;
TEST_P(grid_sample_gpu_test_float_float, test) {
    ASSERT_NO_FATAL_FAILURE(test());
}

using grid_sample_gpu_test_FLOAT16_FLOAT16 = grid_sample_gpu_test<ov::float16, ov::float16>;
TEST_P(grid_sample_gpu_test_FLOAT16_FLOAT16, test) {
    ASSERT_NO_FATAL_FAILURE(test());
}

INSTANTIATE_TEST_SUITE_P(smoke_grid_sample_gpu_test_float_float,
                         grid_sample_gpu_test_float_float,
                         testing::Combine(testing::ValuesIn(getParamsToCheckLayouts<float, float>()),
                                          testing::ValuesIn(layout_formats),
                                          testing::Values(RUN_CACHING_TEST)),
                         grid_sample_gpu_test_float_float::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(smoke_grid_sample_gpu_test_FLOAT16_FLOAT16,
                         grid_sample_gpu_test_FLOAT16_FLOAT16,
                         testing::Combine(testing::ValuesIn(getParamsToCheckLogic<ov::float16, ov::float16>()),
                                          testing::Values(format::bfyx),
                                          testing::Values(RUN_CACHING_TEST)),
                         grid_sample_gpu_test_FLOAT16_FLOAT16::PrintToStringParamName);

#ifndef RUN_ALL_MODEL_CACHING_TESTS
INSTANTIATE_TEST_SUITE_P(smoke_grid_sample_gpu_test_FLOAT16_FLOAT16_cached,
                         grid_sample_gpu_test_FLOAT16_FLOAT16,
                         testing::Combine(testing::ValuesIn(getNearestParamsOddDimensionsOuterGrids<ov::float16, ov::float16>()),
                                          testing::Values(format::bfyx),
                                          testing::Values(true)),
                         grid_sample_gpu_test_FLOAT16_FLOAT16::PrintToStringParamName);
#endif

class grid_sample_gpu_dynamic : public ::testing::TestWithParam<grid_sample_test_params<float, float>> {};
TEST_P(grid_sample_gpu_dynamic, basic) {
    const auto& [p, fmt, is_caching_test] = testing::TestWithParam<grid_sample_test_params<float, float>>::GetParam();

    auto& engine = get_test_engine();
    const auto data_data_type = data_types::f32;
    const auto grid_data_type = data_types::f32;
    const auto plane_format = format::bfyx;

    auto in0_layout = layout{{-1, -1, -1, -1}, data_data_type, fmt};
    auto in1_layout = layout{{-1, -1, -1, 2}, grid_data_type, fmt};

    ov::Shape in0_shape = ov::Shape();
    ov::Shape in1_shape = ov::Shape();
    for (auto shape : p.data_shape) {
        in0_shape.push_back(shape);
    }
    for (auto shape : p.grid_shape) {
        in1_shape.push_back(shape);
    }
    auto input0 = engine.allocate_memory(layout{ov::PartialShape(in0_shape), data_data_type, fmt});
    auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), grid_data_type, fmt});

    set_values(input0, p.data);
    set_values(input1, p.grid);

    topology topology;
    topology.add(input_layout("data", in0_layout));
    topology.add(input_layout("grid", in1_layout));
    topology.add(reorder("reordered_data", input_info("data"), fmt, data_data_type));
    topology.add(reorder("reordered_grid", input_info("grid"), fmt, grid_data_type));
    topology.add(grid_sample("grid_sample", { input_info("reordered_data"), input_info("reordered_grid") }, p.attributes));
    topology.add(reorder("plane_grid_sample", input_info("grid_sample"), plane_format, data_data_type));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("data", input0);
    network.set_input_data("grid", input1);

    const auto outputs = network.execute();
    auto output = outputs.at("plane_grid_sample").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), p.expected_values.size());
    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_NEAR(p.expected_values[i], output_ptr[i], getError<float>());
    }
}

const std::vector<format::type> dynamic_layout_formats = {
    format::bfyx
};

std::vector<grid_sample_test_inputs<float, float>> dynamicGridSampleTestInputs() {
    return {
        {
            {1, 1, 3, 5},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            {1, 1, 7, 2},
            {-10.1f, -9.7f, -7.55f, 0.37f, -77.f, 11.56f, 0.5f, 2.55f, 1.7f, 1.1f, 3.f, -0.17f, 1.301f, -1.001f},
            {true, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::BORDER},
            {1, 6, 11, 14, 15, 10, 5},
            "nearest_border_align_odd_dims_outer"
        },
    };
}

INSTANTIATE_TEST_SUITE_P(smoke_grid_sample_gpu_dynamic_test,
                         grid_sample_gpu_dynamic,
                         testing::Combine(testing::ValuesIn(dynamicGridSampleTestInputs()),
                                          testing::ValuesIn(dynamic_layout_formats),
                                          testing::Values(false)));

}  // namespace
