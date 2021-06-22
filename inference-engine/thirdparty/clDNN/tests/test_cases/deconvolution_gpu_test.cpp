// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils/test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/deconvolution.hpp>
#include <cldnn/primitives/crop.hpp>
#include <cldnn/primitives/reorder.hpp>
#include <cldnn/primitives/data.hpp>

namespace cldnn {
template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace ::tests;

template <typename InputT>
struct deconvolution_traits {
    using accumulator_type = InputT;
};

template <>
struct deconvolution_traits<uint8_t> {
    using accumulator_type = int;
};

template <>
struct deconvolution_traits<int8_t> {
    using accumulator_type = int;
};

template <>
struct deconvolution_traits<FLOAT16> {
    using accumulator_type = float;
};

template<typename T>
T kahan_summation(std::vector<T> &input) {
    T sum = 0;
    T c = 0;
    for (T x : input) {
        T y = x - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template <typename InputT, typename WeightsT, typename OutputT, typename AccumulatorT = typename deconvolution_traits<InputT>::accumulator_type>
VVVF<OutputT> reference_deconvolution(
    const VVVVF<InputT>& input,    // fyx dimensions order
    const VVVVF<WeightsT>& weights,
    float bias,
    tensor stride,
    tensor offset,
    size_t input_f_start
) {
    auto ifm = weights.size();
    auto filter_z = static_cast<int>(weights[0].size());
    auto filter_y = static_cast<int>(weights[0][0].size());
    auto filter_x = static_cast<int>(weights[0][0][0].size());

    auto in_z = static_cast<int>(input[0].size());
    auto in_y = static_cast<int>(input[0][0].size());
    auto in_x = static_cast<int>(input[0][0][0].size());

    auto stride_x = stride.spatial[0];
    auto stride_y = stride.spatial[1];
    auto stride_z = stride.spatial[2];

    auto offset_x = offset.spatial[0];
    auto offset_y = offset.spatial[1];
    auto offset_z = offset.spatial[2];

    int out_x = 2 * offset_x + (in_x - 1) * stride_x + filter_x;
    int out_y = 2 * offset_y + (in_y - 1) * stride_y + filter_y;
    int out_z = 2 * offset_z + (in_z - 1) * stride_z + filter_z;
    VVVF<OutputT> output(static_cast<size_t>(out_z), VVF<OutputT>(static_cast<size_t>(out_y), VF<OutputT>(static_cast<size_t>(out_x))));

    for (int oz = 0; oz < out_z; ++oz) {
        for (int oy = 0; oy < out_y; ++oy) {
            for (int ox = 0; ox < out_x; ++ox) {
                VF<AccumulatorT> values;
                for (int fz = 0; fz < filter_z; ++fz) {
                    int iz = oz - filter_z + 1 - offset_z + fz;
                    if (iz < 0 || iz >= in_z * stride_z || iz % stride_z != 0)
                        continue;
                    iz = iz / stride_z;

                    for (int fy = 0; fy < filter_y; ++fy) {
                        int iy = oy - filter_y + 1 - offset_y + fy;
                        if (iy < 0 || iy >= in_y * stride_y || iy % stride_y != 0)
                            continue;
                        iy = iy / stride_y;

                        for (int fx = 0; fx < filter_x; ++fx) {
                            int ix = ox - filter_x + 1 - offset_x + fx;
                            if (ix < 0 || ix >= in_x * stride_x || ix % stride_x != 0)
                                continue;
                            ix = ix / stride_x;

                            for (size_t ifi = 0; ifi < ifm; ++ifi) {
                                auto in_val = input[input_f_start + ifi][iz][iy][ix];
                                auto wei_val = weights[ifi][filter_z - fz - 1][filter_y - fy - 1][filter_x - fx - 1];
                                values.push_back(static_cast<AccumulatorT>(in_val) * static_cast<AccumulatorT>(wei_val));
                            }
                        }
                    }
                }
                output[oz][oy][ox] = static_cast<OutputT>(kahan_summation<AccumulatorT>(values)) + static_cast<OutputT>(bias);
            }
        }
    }
    return output;
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 3x3
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  -14    5     2.25
    //   18    0.75  7.25
    //   23    42.5  15.5

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, no_bias_basic_wsiz2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 3x3
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  no bias
    //
    //
    //  Output:
    //  -14    5     2.25
    //   18    0.75  7.25
    //   23    42.5  15.5

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx,{ 1, 1, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        deconvolution("deconv", "input", { "weights" })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -16.f, 3.f, 0.25f,
        16.f, -1.25f, 5.25f,
        21.f, 40.5f, 13.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_nopad_bfyx) {    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 3x3
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  -14    5     2.25
    //   18    0.75  7.25
    //   23    42.5  15.5

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 1x1
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  0.75

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 1, 1 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_FLOAT_EQ(0.75f, output_ptr[0]);
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride2_nopad) {
    //  Filter : 2x2
    //  Input  : 2x2
    //  Output : 1x1
    //  Stride : 2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  0.75

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -15.f, 5.f, 0.f, 1.25f,
        29.f, 13.f, 2.75f, 1.75,
        -11.f, 4.f, -17.f, 5.5f,
        22.f, 10.f, 32.5f, 14.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride4_pad2) {
    //  Filter : 3x3
    //  Input  : 2x2
    //  Output : 1x1
    //  Stride : 4x4
    //  Pad    : 2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5   1
    //   3.5 1.5   2
    //   3   4     5
    //
    //  Bias
    //  0
    //
    //  Output:
    //  40   0    1.5
    //  0    0    0
    //  6    0   -18

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 1, 3, 3 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f });
    set_values(biases, { 0.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 4, 4 }, { 0, 0, -2, -2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        40.f, 0.f, 1.5f,
        0.f, 0.f, 0.f,
        6.f, 0.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 1.f, 0.5f, 3.f, 6.f, 2.f, 9.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 0.5f, 4.5f, 22.f,
        13.f, 5.f, -17.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2x2_in2x2x1x1_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x1
    //  Output : 2x2x1x1
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  f0:-2   2
    //  f0: 7  -0.5
    //  f1:-2   2
    //  f1: 7  -0.5
    //
    //  Bias
    //  1  5
    //
    //  Output:
    //  f0: -3   4.5
    //  f0: 13   -17
    //  f1: 1    8.5
    //  f1: 17 - 13

    auto& engine = get_test_engine();
    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::yxio, { 2, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.f, -2.f, 2.f, 2.f, 7.f, 7.f, -0.5f, -0.5f });
    set_values(biases, { 1.0f, 5.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 1.f, 4.5f, 8.5f,
        13.f, 17.f, -17.f, -13.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1_input_padding) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Out Padding   : 1x1
    //  Input Padding : 2x1 (with reorder)
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx,{ 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", input->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "reorder", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2x2_in2x2x1x1_stride2_pad1_input_padding) {
    //  Filter : 2x2
    //  Input  : 2x2x1x1
    //  Output : 2x2x1x1
    //  Stride : 2x2
    //  Out Padding   : 1x1
    //  Input Padding : 2x1 (with reorder)
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  f0:-2   2
    //  f0: 7  -0.5
    //  f1:-2   2
    //  f1: 7  -0.5
    //
    //  Bias
    //  1  5
    //
    //  Output:
    //  f0: -3   4.5
    //  f0: 13   -17
    //  f1: 1    8.5
    //  f1: 17 - 13

    auto& engine = get_test_engine();
    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::yxio,{ 2, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.f, -2.f, 2.f, 2.f, 7.f, 7.f, -0.5f, -0.5f });
    set_values(biases, { 1.0f, 5.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", input->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "reorder", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 1.f, 4.5f, 8.5f,
        13.f, 17.f, -17.f, -13.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_yxfb_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f16_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_yxfb_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx,{ 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    set_values(input, { FLOAT16(8.f), FLOAT16(0.5f),
                        FLOAT16(6.f), FLOAT16(9.f),

                        FLOAT16(1.f), FLOAT16(3.f),
                        FLOAT16(2.f), FLOAT16(4.f) });
    set_values(weights, { -2.f, 2.f,
                          7.f, -0.5f});
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<uint16_t> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2) {
    //  Filter : 2x2x2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter1
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Filter2
    //  -4   1
    //  -9  -7
    //
    //  Bias
    //  -1
    //
    //  Output:
    //  -3    4.5    -8   -28
    //   13  -17     1    -17

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f, -4.f, 1.f, -9.f, -7.f });
    set_values(biases, { 1.0f, -1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        -8.f, -28.f, 1.f, -17.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group2) {
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, {
        -2.f, 2.f, 7.f, -0.5f,
        -4.f, 1.f, -9.f, -7.f
    });
    set_values(biases, { 1.0f, -1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        -8.f, -28.f, 1.f, -17.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group16) {
    //  Test for depthwise separable optimization, there are 16 joined weights and biases (group 16)
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2_depthwise_sep_opt

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 16, 2, 2 } });
    set_values(input,
    { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f
    });

    topology topology(input_layout("input", input->get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;

    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(16), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 16, 1, 1 } });

    set_values(weights,
        {
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f
        }
    );
    set_values(biases, { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f });
    topology.add(
        data("weights", weights),
        data("bias", biases)
    );

    topology.add(deconvolution("deconv", "input", { "weights" }, { "bias" }, 16, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group16_ofm2) {
    //  Test for depthwise separable optimization, there are 16 joined weights and biases (group 16)
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2_depthwise_sep_opt_ofm2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 16, 2, 2 } });
    set_values(input,
    { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f,
        8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f
    });

    topology topology(input_layout("input", input->get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;

    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(16), batch(2), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 32, 1, 1 } });

    set_values(weights,
        {
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
            -2.f, 2.f, 7.f, -0.5f, -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f, -4.f, 1.f, -9.f, -7.f,
        }
    );

    set_values(biases,
        {
            1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
            1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f
        }
    );

    topology.add(
        data("weights", weights),
        data("bias", biases)
    );

    topology.add(deconvolution("deconv", "input", { "weights" }, { "bias" }, 16, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
        -3.f, 4.5f, 13.f, -17.f,-3.f, 4.5f, 13.f, -17.f, -8.f, -28.f, 1.f, -17.f, -8.f, -28.f, 1.f, -17.f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x6x1x1_bfyx_stride2_pad1_group2_ofm3) {
    //  data is similar as in basic_wsiz2x2_in1x6x1x1_bfyx_stride2_pad1_split2_ofm3

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 4, 1, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(3), feature(2), spatial(1, 1)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 6, 1, 1 } });

    set_values(input, {
        1.5f, 0.5f, 2.0f, -1.0f
    });
    set_values(weights, {
        -2.0f, 1.0f, 1.0f, 3.0f, 0.5f, 8.0f,
        4.0f, -4.0f, 2.0f, 0.5f, -0.5f, 3.0f
    });
    set_values(biases, {
        1.0f, 5.0f, 3.0f,
        -1.0f, 2.5f, 2.0f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 1, 1 }, { 0, 0, 0, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -1.5f, 8.0f, 7.75f, 11.0f, 6.0f, -2.0f
    };
    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}
TEST(deconvolution_f32_fw_gpu, basic3D_wsiz2x2x1_in1x1x2x2x1_nopad) {
    //  Filter : 2x2x1
    //  Input  : 2x2x1
    //  Output : 3x3x1
    //
    //  Input:
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  -14    5     2.25
    //   18    0.75  7.25
    //   23    42.5  15.5

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oizyx,{ 1, 1, 2, 2, 1 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1,1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic3D_wsiz3x3x3_in1x1x4x4x4_nopad) {
    //  Filter : 3x3x3
    //  Input  : 3x3x3
    //  Output : 6x6x6
    //
    //  Input:
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //
    //
    //  Filter
    //  1  1  1
    //  1  1  1
    //  1  1  1
    //
    //  1  1  1
    //  1  1  1
    //  1  1  1
    //
    //  1  1  1
    //  1  1  1
    //  1  1  1
    //
    //
    //  Output:
    //
    //  1  2  3  3  2  1
    //  2  4  6  6  4  2
    //  3  6  9  9  6  3
    //  3  6  9  9  6  3
    //  2  4  6  6  4  2
    //  1  2  3  3  2  1
    //
    //  2   4   6   6   4  2
    //  4   8  12  12   8  4
    //  6  12  18  18  12  6
    //  6  12  18  18  12  6
    //  4   8  12  12   8  4
    //  2   4   6   6   4  2
    //
    //  3   6   9   9   6  3
    //  6  12  18  18  12  6
    //  9  18  27  27  18  9
    //  9  18  27  27  18  9
    //  6  12  18  18  12  6
    //  3   6   9   9   6  3
    //
    //  3   6   9   9   6  3
    //  6  12  18  18  12  6
    //  9  18  27  27  18  9
    //  9  18  27  27  18  9
    //  6  12  18  18  12  6
    //  3   6   9   9   6  3
    //
    //  2   4   6   6   4  2
    //  4   8  12  12   8  4
    //  6  12  18  18  12  6
    //  6  12  18  18  12  6
    //  4   8  12  12   8  4
    //  2   4   6   6   4  2
    //
    //  1  2  3  3  2  1
    //  2  4  6  6  4  2
    //  3  6  9  9  6  3
    //  3  6  9  9  6  3
    //  2  4  6  6  4  2
    //  1  2  3  3  2  1
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 1, 4, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oizyx,{ 1, 1, 3, 3, 3 } });

    set_values(input,
    {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    });
    set_values(weights, {
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f,
        1.0f,  1.0f, 1.0f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        deconvolution("deconv", "input", { "weights" })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f,

        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        4.0f, 8.0f, 12.0f, 12.0f, 8.0f, 4.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        4.0f, 8.0f, 12.0f, 12.0f, 8.0f, 4.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,

        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        9.0f, 18.0f, 27.0f, 27.0f, 18.0f, 9.0f,
        9.0f, 18.0f, 27.0f, 27.0f, 18.0f, 9.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,

        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        9.0f, 18.0f, 27.0f, 27.0f, 18.0f, 9.0f,
        9.0f, 18.0f, 27.0f, 27.0f, 18.0f, 9.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,

        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        4.0f, 8.0f, 12.0f, 12.0f, 8.0f, 4.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        6.0f, 12.0f, 18.0f, 18.0f, 12.0f, 6.0f,
        4.0f, 8.0f, 12.0f, 12.0f, 8.0f, 4.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,

        1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        3.0f, 6.0f, 9.0f, 9.0f, 6.0f, 3.0f,
        2.0f, 4.0f, 6.0f, 6.0f, 4.0f, 2.0f,
        1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic3D_wsiz2x2x2_in1x1x2x2x2_stride2_nopad) {
    //  Filter : 2x2x2
    //  Input  : 2x2x2
    //  Output : 1x1
    //  Stride : 2x2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //  -2   0.5
    //   3.5 1.5
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oizyx,{ 1, 1, 2, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, -2.0f, 0.5f, 3.5f, 1.5f });
    //set_values(input, { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f });
    //set_values(weights, { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        deconvolution("deconv", "input", { "weights" }, { 1,1,2,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -16.f, 4.f, -1.f, 0.25f,
        28.f, 12.f, 1.75f, 0.75f,
        -12.f, 3.f, -18.f, 4.5f,
        21.f, 9.f, 31.5f, 13.5f,
        -16.f, 4.f, -1.f, 0.25f,
        28.f, 12.f, 1.75f, 0.75f,
        -12.f, 3.f, -18.f, 4.5f,
        21.f, 9.f, 31.5f, 13.5f,
        -16.f, 4.f, -1.f, 0.25f,
        28.f, 12.f, 1.75f, 0.75f,
        -12.f, 3.f, -18.f, 4.5f,
        21.f, 9.f, 31.5f, 13.5f,
        -16.f, 4.f, -1.f, 0.25f,
        28.f, 12.f, 1.75f, 0.75f,
        -12.f, 3.f, -18.f, 4.5f,
        21.f, 9.f, 31.5f, 13.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic3D_wsiz2x2x2_in1x1x2x2x2_stride2_pad1) {
    //  Filter : 2x2x2
    //  Input  : 2x2x2
    //  Output : 1x1
    //  Stride : 2x2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //  -2   0.5
    //   3.5 1.5
    //
    //  Output:
    //  12 1.75
    //   3 -18
    //  12 1.75
    //   3 -18

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oizyx,{ 1, 1, 2, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, -2.0f, 0.5f, 3.5f, 1.5f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        deconvolution("deconv", "input", { "weights" }, { 1,1,2,2,2 }, { 0, 0, -1, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        12.f, 1.75f, 3.f, -18.f,
        12.f, 1.75f, 3.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }

}

TEST(deconvolution_f16_gpu, basic_k9x9_s2x2_pad4x4) {
    //  Filter : 1x32x9x9
    //  Input  : 1x32x16x16
    //  Stride : 2x2
    //  Pad    : 4x4

    //auto& engine = get_test_engine();
    auto& engine = get_test_engine();

    VVVVF<FLOAT16> input_rnd = generate_random_4d<FLOAT16>(1, 32, 16, 16, -2, 2);
    VF<FLOAT16> input_rnd_vec = flatten_4d<FLOAT16>(format::bfyx, input_rnd);
    VVVVF<FLOAT16> filter_rnd = generate_random_4d<FLOAT16>(1, 32, 9, 9, -1, 1);
    VF<FLOAT16> filter_rnd_vec = flatten_4d<FLOAT16>(format::bfyx, filter_rnd);
    VF<FLOAT16> bias_rnd = generate_random_1d<FLOAT16>(1, -1, 1);
    VF<float> filter_rnd_f32_vec, bias_f32_rnd;

    for (unsigned int i = 0; i < filter_rnd_vec.size(); i++)
        filter_rnd_f32_vec.push_back(float(filter_rnd_vec[i]));

    for (unsigned int i = 0; i < bias_rnd.size(); i++)
        bias_f32_rnd.push_back(float(bias_rnd[i]));

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 32, 16, 16 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::oiyx, { 1, 32, 9, 9 } });
    auto biases = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 1, 1 } });
    auto weights_f32 = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 32, 9, 9 } });
    auto biases_f32 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, input_rnd_vec);
    set_values(weights, filter_rnd_vec);
    set_values(biases, bias_rnd);
    set_values(weights_f32, filter_rnd_f32_vec);
    set_values(biases_f32, bias_f32_rnd);

    topology topology_ref(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -4, -4 }, tensor{ 1, 1, 32, 32 })
    );

    network network_ref(engine, topology_ref);
    network_ref.set_input_data("input", input);

    auto outputs_ref = network_ref.execute();
    EXPECT_EQ(outputs_ref.size(), size_t(1));
    EXPECT_EQ(outputs_ref.begin()->first, "deconv");
    auto output_ref_prim = outputs_ref.begin()->second.get_memory();
    cldnn::mem_lock<FLOAT16> output_ref_ptr(output_ref_prim, get_test_stream());

    std::vector<FLOAT16> output_vec_ref;
    for (unsigned int i = 0; i < output_ref_prim->get_layout().count(); i++) {
        output_vec_ref.push_back(output_ref_ptr[i]);
    }

    topology topology_act(
        input_layout("input_act", input->get_layout()),
        data("weights_f32", weights_f32),
        data("biases_f32", biases_f32),
        deconvolution("deconv_act", "input_act", { "weights_f32" }, { "biases_f32" }, { 1, 1, 2, 2 }, { 0, 0, -4, -4 }),
        reorder("out", "deconv_act", format::bfyx, data_types::f16)
    );

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network_act(engine, topology_act, options);
    network_act.set_input_data("input_act", input);

    auto outputs_act = network_act.execute();
    EXPECT_EQ(outputs_act.size(), size_t(1));
    EXPECT_EQ(outputs_act.begin()->first, "out");
    auto output_act_prim = outputs_act.begin()->second.get_memory();
    cldnn::mem_lock<FLOAT16> output_act_ptr(output_act_prim, get_test_stream());

    std::vector<float> output_vec;
    for (unsigned int i = 0; i < output_act_prim->get_layout().count(); i++) {
        float x = float_round(output_act_ptr[i]), y = float_round(output_vec_ref[i]);
        EXPECT_NEAR(x, y, 1e-0f);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_b_fs_yx_fsv16_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::yxio, { 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(biases, { 1.0f });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -3.f, 4.5f, 13.f, -17.f,
            .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f16_fw_gpu, basic_wsiz2x2_in2x2x1x2_b_fs_yx_fsv16_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //  Pad    : 1x1
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Bias
    //  1
    //
    //  Output:
    //  -3    4.5    0.5   22
    //   13  -17     5    -7

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::oiyx,{ 1, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f16, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { FLOAT16(8.f), FLOAT16(0.5f),
                        FLOAT16(6.f), FLOAT16(9.f),

                        FLOAT16(1.f), FLOAT16(3.f),
                        FLOAT16(2.f), FLOAT16(4.f) });
    set_values(weights, { FLOAT16(-2.f), FLOAT16(2.f),
                          FLOAT16(7.f), FLOAT16(-0.5f)});
    set_values(biases, { FLOAT16(1.0f) });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f16)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<uint16_t> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -3.f, 4.5f, 13.f, -17.f,
            .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_b_fs_yx_fsv16_stride2_pad1_group2) {
    //  data is similar as in basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, {
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f
    });
    set_values(biases, { 1.0f, -1.0f });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -3.f, 4.5f, 13.f, -17.f,
            -8.f, -28.f, 1.f, -17.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in1x2x2x2_b_fs_yx_fsv16_stride2_pad1_b_fs_yx_fsv16_dw) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, {
            -2.f, 2.f, 7.f, -0.5f,
            -4.f, 1.f, -9.f, -7.f
    });
    set_values(biases, { 0.0f, 0.0f });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -4.f, 3.5f, 12.f, -18.f,
            -7.f, -27.f, 2.f, -16.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_nopad_b_fs_yx_fsv16_dw) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,  8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f,  -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f, 2.0f });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            reorder("input_fsv16", "input", format::b_fs_yx_fsv16, data_types::f32),
            deconvolution("deconv", "input_fsv16", { "weights" }, { "biases" }, 2, { 1,1,1,1 }, { 0, 0, 0, 0 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -14.f, 5.f, 2.25f,
            18.f, 0.75f, 7.25f,
            23.f, 42.5f, 15.5f,

            -14.f, 5.f, 2.25f,
            18.f, 0.75f, 7.25f,
            23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_pad1_b_fs_yx_fsv16_dw) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f});
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f,
                          -2.0f, 0.5f, 3.5f, 1.5f});
    set_values(biases, { 2.0f, 2.0f });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            reorder("input_fsv16", "input", format::b_fs_yx_fsv16, data_types::f32),
            deconvolution("deconv", "input_fsv16", { "weights" }, { "biases" }, 2, { 1, 1, 1, 1 }, { 0, 0, -1, -1 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    EXPECT_FLOAT_EQ(0.75f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.75f, output_ptr[1]);
}


TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride2_nopad_b_fs_yx_fsv16_dw) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(2, 2)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f});
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f,
                          -2.0f, 0.5f, 3.5f, 1.5f});
    set_values(biases, { 1.0f, 1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1,1,2,2 }),
        reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -15.f, 5.f, 0.f, 1.25f,
        29.f, 13.f, 2.75f, 1.75,
        -11.f, 4.f, -17.f, 5.5f,
        22.f, 10.f, 32.5f, 14.5f,


        -15.f, 5.f, 0.f, 1.25f,
        29.f, 13.f, 2.75f, 1.75,
        -11.f, 4.f, -17.f, 5.5f,
        22.f, 10.f, 32.5f, 14.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride4_pad2_b_fs_yx_fsv16_dw) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(3, 3)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f});
    set_values(weights, { -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f,
                          -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f});
    set_values(biases, { 0.0f, 0.0f });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 4, 4 }, { 0, 0, -2, -2 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f,

            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, basic_wsiz2x2_in2x2x1x1_stride4_pad2_b_fs_yx_fsv16_dw_batch2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::goiyx, tensor(group(2), batch(1), feature(1), spatial(3, 3)) });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f,

                        8.f, 0.5f, 6.f, 9.f,
                        8.f, 0.5f, 6.f, 9.f});
    set_values(weights, { -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f,
                          -2.0f, 0.5f, 1.f, 3.5f, 1.5f, 2.f, 3.f, 4.f, 5.f});
    set_values(biases, { 0.0f, 0.0f });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "input", { "weights" }, { "biases" }, 2, { 1, 1, 4, 4 }, { 0, 0, -2, -2 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::b_fs_yx_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f,

            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f,



            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f,

            40.f, 0.f, 1.5f,
            0.f, 0.f, 0.f,
            6.f, 0.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_f32_fw_gpu, bs_fs_zyx_bsv16_fsv16_wsiz2x2x2_in1x1x2x2x2_stride2_nopad) {
    //  Batch : 32
    //  Filter : 2x2x2
    //  Input  : 2x2x2
    //  Output : 1x1
    //  Stride : 2x2x2
    //
    //  Input:
    //  8  0.5
    //  6  9
    //  8  0.5
    //  6  9
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //  -2   0.5
    //   3.5 1.5
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 32, 1, 2, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oizyx,{ 1, 1, 2, 2, 2 } });

    std::vector<float> input_single_batch = { 8.f, 0.5f, 6.f, 9.f, 8.f, 0.5f, 6.f, 9.f };
    std::vector<float> input_batched;
    for (size_t i = 0; i < 32; i++) {
        for (size_t j = 0; j < 8; j++) {
            input_batched.push_back(input_single_batch[j]);
        }
    }

    set_values(input, input_batched);
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, -2.0f, 0.5f, 3.5f, 1.5f });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            deconvolution("deconv", "input", { "weights" }, { 1,1,2,2,2 }, { 0, 0, -1, -1, -1 }),
            reorder("out", "deconv", format::bfzyx, data_types::f32)
    );

    cldnn::build_options options;
    implementation_desc impl = { format::bs_fs_zyx_bsv16_fsv16, "" };
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(build_option::force_implementations({ {"deconv", impl} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            12.f, 1.75f, 3.f, -18.f,
            12.f, 1.75f, 3.f, -18.f
    };

    for (size_t b = 0; b < 32; b++) {
        for (size_t i = 0; i < expected_output_vec.size(); i++) {
            ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[b*expected_output_vec.size() + i]) << " b = " << b << " i = " << i;
        }
    }
}

TEST(deconvolution_f16_fw_gpu, basic_wsiz2x2_in1x2x2x2_fs_b_yx_fsv32_stride1_pad1_replace_to_conv) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx,{ 2, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f16, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { FLOAT16(8.f), FLOAT16(0.5f), FLOAT16(6.f), FLOAT16(9.f),
                        FLOAT16(1.f), FLOAT16(3.f), FLOAT16(2.f), FLOAT16(4.f)
                        });
    set_values(weights, {
            FLOAT16(-2.f), FLOAT16(2.f), FLOAT16(7.f), FLOAT16(-0.5f),
            FLOAT16(-4.f), FLOAT16(1.f), FLOAT16(-9.f), FLOAT16(-7.f)
    });
    set_values(biases, { FLOAT16(1.0f), FLOAT16(-1.0f) });

    topology topology(
            input_layout("input", input->get_layout()),
            reorder("reorder", "input", format::fs_b_yx_fsv32, data_types::f16),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", "reorder", { "weights" }, { "biases" }, 1, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }),
            reorder("out", "deconv", format::bfyx, data_types::f32)
    );

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -15.f, 16.f, 2.f, 45.f, -5.5f, 18.75f, 43.f, 61.f, -3.5f,
            -33.f, 5.f, -0.5f, -97.f, -91.5f, 4.5f, -55.f, -124.f, -64.f,
            -1.f, -3.f, 7.f, 4.f, 17.5f, 7.5f, 15.f, 28.f, -1.f,
            -5.f, -12.f, 2.f, -18.f, -49.f, -18.f, -19.f, -51.f, -29.f,
    };
    ASSERT_EQ(expected_output_vec.size(), output_prim->count());

    for (size_t i = 0; i < expected_output_vec.size(); i++) {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]) << " index=" << i;
    }
}

struct deconvolution_random_test_params {
    data_types input_type;
    format::type input_format;
    tensor input_size;
    data_types weights_type;
    format::type weights_format;
    tensor weights_size;
    tensor strides;
    tensor input_offset;
    bool with_bias;
    data_types output_type;
    cldnn::implementation_desc deconv_desc;

    static std::string print_params(const testing::TestParamInfo<deconvolution_random_test_params>& param_info) {
        auto& param = param_info.param;
        auto to_string_neg = [](int v) {
            if (v >= 0) {
                return std::to_string(v);
            } else {
                return "m" + std::to_string(-v);
            }
        };

        auto print_tensor = [&](const tensor& size) {
            return to_string_neg(size.batch[0]) + "x" +
                to_string_neg(size.feature[0]) + "x" +
                to_string_neg(size.spatial[0]) + "x" +
                to_string_neg(size.spatial[1]) + "x" +
                to_string_neg(size.spatial[2]);
        };

        // construct a readable name
        return "in_" + dt_to_str(param.input_type) +
            "_" + fmt_to_str(param.input_format) +
            "_" + print_tensor(param.input_size) +
            "_wei_" + dt_to_str(param.weights_type) +
            "_" + fmt_to_str(param.weights_format) +
            "_" + print_tensor(param.weights_size) +
            (param.with_bias ? "_bias" : "") +
            "_s_" + print_tensor(param.strides) +
            "_off_" + print_tensor(param.input_offset) +
            "_out_" + dt_to_str(param.output_type) +
            (!param.deconv_desc.kernel_name.empty() ? "_kernel_" + param.deconv_desc.kernel_name : "") +
            (param.deconv_desc.output_format != format::any ? "_fmt_" + fmt_to_str(param.deconv_desc.output_format) : "");
    }
};

template <typename T>
struct typed_comparator {
    static ::testing::AssertionResult compare(const char* lhs_expr, const char* rhs_expr, T ref, T val) {
        return ::testing::internal::EqHelper::Compare(lhs_expr, rhs_expr, ref, val);
    }
};

template <>
struct typed_comparator<float> {
    static ::testing::AssertionResult compare(const char* lhs_expr, const char* rhs_expr, float ref, float val) {
        return ::testing::internal::CmpHelperFloatingPointEQ<float>(lhs_expr, rhs_expr, ref, val);
    }
};

template <>
struct typed_comparator<FLOAT16> {
    static ::testing::AssertionResult compare(const char* lhs_expr, const char* rhs_expr, FLOAT16 ref, FLOAT16 val) {
        double abs_error = std::abs(0.05 * (double)ref);
        return ::testing::internal::DoubleNearPredFormat(lhs_expr, rhs_expr, "5 percent", (double)ref, (double)val, abs_error);
    }
};

template <typename T>
struct type_test_ranges {
    static constexpr int min = -1;
    static constexpr int max = 1;
};

template <>
struct type_test_ranges<uint8_t> {
    static constexpr int min = 0;
    static constexpr int max = 255;
};

template <>
struct type_test_ranges<int8_t> {
    static constexpr int min = -127;
    static constexpr int max = 127;
};

#define TYPED_ASSERT_EQ(ref, val)                                                       \
    ASSERT_PRED_FORMAT2(typed_comparator<decltype(ref)>::compare, ref, val)

#define TYPED_EXPECT_EQ(ref, val)                                                       \
    EXPECT_PRED_FORMAT2(typed_comparator<decltype(ref)>::compare, ref, val)

template <typename InputT, typename WeightsT, typename OutputT>
class deconvolution_random_test_base {
public:
    template <typename T>
    void set_memory(cldnn::memory::ptr mem, const VVVVVF<T>& data) {
        cldnn::mem_lock<T> ptr(mem, get_test_stream());

        auto b = data.size();
        auto f = data[0].size();
        auto z = data[0][0].size();
        auto y = data[0][0][0].size();
        auto x = data[0][0][0][0].size();

        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t zi = 0; zi < z; ++zi) {
                    for (size_t yi = 0; yi < y; ++yi) {
                        for (size_t xi = 0; xi < x; ++xi) {
                            auto coords = cldnn::tensor(batch(bi), feature(fi), spatial(xi, yi, zi, 0));
                            auto offset = mem->get_layout().get_linear_offset(coords);
                            ptr[offset] = data[bi][fi][zi][yi][xi];
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void set_memory_weights(cldnn::memory::ptr mem, const VVVVVVF<T>& data) {
        cldnn::mem_lock<T> ptr(mem, get_test_stream());

        auto g = data.size();
        auto b = data[0].size();
        auto f = data[0][0].size();
        auto z = data[0][0][0].size();
        auto y = data[0][0][0][0].size();
        auto x = data[0][0][0][0][0].size();

        for (size_t gi = 0; gi < g; ++gi) {
            for (size_t bi = 0; bi < b; ++bi) {
                for (size_t fi = 0; fi < f; ++fi) {
                    for (size_t zi = 0; zi < z; ++zi) {
                        for (size_t yi = 0; yi < y; ++yi) {
                            for (size_t xi = 0; xi < x; ++xi) {
                                auto coords = cldnn::tensor(group(gi), batch(bi), feature(fi), spatial(xi, yi, zi, 0));
                                auto offset = mem->get_layout().get_linear_offset(coords);
                                ptr[offset] = data[gi][bi][fi][zi][yi][xi];
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    VVVVVF<T> generate_random(cldnn::tensor size) {
        return generate_random_5d<T>(
            size.batch[0],
            size.feature[0],
            size.spatial[2],
            size.spatial[1],
            size.spatial[0],
            type_test_ranges<T>::min,
            type_test_ranges<T>::max);
    }

    template <typename T>
    VVVVVVF<T> generate_random_weights(cldnn::tensor size) {
        return generate_random_6d<T>(
            size.group[0],
            size.batch[0],
            size.feature[0],
            size.spatial[2],
            size.spatial[1],
            size.spatial[0],
            type_test_ranges<T>::min,
            type_test_ranges<T>::max);
    }

    void run(cldnn::engine& eng, const deconvolution_random_test_params& params, cldnn::build_options build_opts) {
        uint32_t groups = params.weights_size.group[0];
        size_t ifm = params.weights_size.feature[0];
        size_t ofm = params.weights_size.batch[0];

        auto input_data = generate_random<InputT>(params.input_size);
        auto weights_data = generate_random_weights<WeightsT>(params.weights_size);

        auto in_layout = cldnn::layout(cldnn::type_to_data_type<InputT>::value, params.input_format, params.input_size);
        auto wei_layout = cldnn::layout(cldnn::type_to_data_type<WeightsT>::value, params.weights_format, params.weights_size);

        auto wei_mem = eng.allocate_memory(wei_layout);
        auto in_mem = eng.allocate_memory(in_layout);

        this->set_memory_weights(wei_mem, weights_data);
        this->set_memory(in_mem, input_data);

        auto topo = cldnn::topology(
            cldnn::input_layout("input", in_layout),
            cldnn::data("weights", wei_mem)
        );

        VF<OutputT> bias_data;

        if (params.with_bias) {
            auto bias_size = cldnn::tensor(feature(params.weights_size.batch[0] * params.weights_size.group[0]));
            auto bias_lay = cldnn::layout(cldnn::type_to_data_type<OutputT>::value, cldnn::format::bfyx, bias_size);
            auto bias_mem = eng.allocate_memory(bias_lay);
            bias_data = generate_random_1d<OutputT>(bias_lay.size.feature[0], -1, 1);
            set_values(bias_mem, bias_data);
            topo.add(cldnn::data("bias", bias_mem));
            topo.add(cldnn::deconvolution("deconv", "input", { "weights" }, { "bias" }, groups, params.strides, params.input_offset));
        } else {
            topo.add(cldnn::deconvolution("deconv", "input", { "weights" }, groups, params.strides, params.input_offset));
        }

        if (!params.deconv_desc.kernel_name.empty() || params.deconv_desc.output_format != cldnn::format::any) {
            build_opts.set_option(cldnn::build_option::force_implementations({ { "deconv", params.deconv_desc } }));
        }

        auto net = cldnn::network(eng, topo, build_opts);
        net.set_input_data("input", in_mem);

        auto result = net.execute();

        std::string kernel;
        for (auto i : net.get_primitives_info()) {
            if (i.original_id == "deconv")
                kernel = i.kernel_id;
        }

        auto out_mem = result.at("deconv").get_memory();

        // Compare results
        {
            cldnn::mem_lock<OutputT> ptr(out_mem, get_test_stream());

            auto b = static_cast<size_t>(out_mem->get_layout().size.batch[0]);
            auto of = static_cast<size_t>(out_mem->get_layout().size.feature[0]);

            for (size_t bi = 0; bi < b; ++bi) {
                for (size_t fi = 0; fi < of; ++fi) {
                    size_t group = fi / ofm;
                    auto reference = reference_deconvolution<InputT, WeightsT, OutputT>(
                        input_data[bi],
                        weights_data[group][fi % ofm],
                        bias_data.empty() ? 0.f : static_cast<float>(bias_data[fi]),
                        params.strides,
                        params.input_offset,
                        group * ifm);

                    ASSERT_EQ(reference.size(), out_mem->get_layout().size.spatial[2]);
                    ASSERT_EQ(reference[0].size(), out_mem->get_layout().size.spatial[1]);
                    ASSERT_EQ(reference[0][0].size(), out_mem->get_layout().size.spatial[0]);

                    for (size_t zi = 0; zi < reference.size(); zi++) {
                        for (size_t yi = 0; yi < reference[0].size(); yi++) {
                            for (size_t xi = 0; xi < reference[0][0].size(); xi++) {
                                auto ref_val = reference[zi][yi][xi];
                                auto out_coords = cldnn::tensor(batch(bi), feature(fi), spatial(xi, yi, zi, 0));
                                auto out_offset = out_mem->get_layout().get_linear_offset(out_coords);
                                auto out_val = ptr[out_offset];
                                TYPED_ASSERT_EQ(ref_val, out_val)
                                    << "at b=" << bi << ", f=" << fi << ", z=" << zi << ", y=" << yi << ", x=" << xi << std::endl
                                    << "  kernel: " << kernel;
                            }
                        }
                    }
                }
            }
        }
    }
};

#undef TYPED_ASSERT_EQ
#undef TYPED_EXPECT_EQ

class deconvolution_random_test : public testing::TestWithParam<deconvolution_random_test_params> {
protected:
    void SetUp() override {
        build_opts.set_option(cldnn::build_option::optimize_data(true));
    }

    void run() {
        auto params = GetParam();
        switch (params.input_type) {
        case data_types::f32:
            run_typed_in<float>();
            break;
        case data_types::f16:
            run_typed_in<FLOAT16>();
            break;
        case data_types::i8:
            run_typed_in<int8_t>();
            break;
        case data_types::u8:
            run_typed_in<uint8_t>();
            break;
        default:
            break;
        }
    }

    cldnn::build_options build_opts;

private:
    template <typename InputT, typename WeightsT, typename OutputT>
    void run_typed() {
        auto& params = GetParam();
        deconvolution_random_test_base<InputT, WeightsT, OutputT> test;
        test.run(get_test_engine(), params, build_opts);
    }

    template <typename InputT, typename WeightsT>
    void run_typed_in_wei() {
        auto& params = GetParam();
        switch (params.output_type) {
        case data_types::f32:
            run_typed<InputT, WeightsT, float>();
            break;
        case data_types::f16:
            run_typed<InputT, WeightsT, FLOAT16>();
            break;
        default:
            break;
        }
    }

    template <typename InputT>
    void run_typed_in() {
        auto& params = GetParam();
        switch (params.weights_type) {
        case data_types::f32:
            run_typed_in_wei<InputT, float>();
            break;
        case data_types::f16:
            run_typed_in_wei<InputT, FLOAT16>();
            break;
        case data_types::i8:
            run_typed_in_wei<InputT, int8_t>();
            break;
        case data_types::u8:
            run_typed_in_wei<InputT, uint8_t>();
            break;
        default:
            break;
        }
    }
};

class deconvolution_random_test_params_generator : public std::vector<deconvolution_random_test_params> {
public:
    using self = deconvolution_random_test_params_generator;
    self& add(const deconvolution_random_test_params& params) {
        push_back(params);
        return *this;
    }

    self& add_smoke_2d(data_types in_dt, data_types wei_dt, data_types out_dt, format::type in_fmt, format::type out_fmt) {
        std::vector<int> batches = { 1, 2 };
        for (auto b : batches) {
            // 1x1
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7}, wei_dt, format::oiyx, {15, 15, 1, 1}, tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7}, wei_dt, format::oiyx, {15, 15, 1, 1}, {1, 1, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            // 3x3
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7}, wei_dt, format::oiyx, {15, 15, 3, 3}, tensor(1), {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7}, wei_dt, format::oiyx, {15, 15, 3, 3}, {1, 1, 2, 2}, {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            // Grouped
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 8, 7, 7}, wei_dt, format::goiyx, tensor(group(2), batch(16), feature(4), spatial(1, 1)), tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 8, 7, 7}, wei_dt, format::goiyx, tensor(group(2), batch(16), feature(4), spatial(1, 1)), {1, 1, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 8, 7, 7}, wei_dt, format::goiyx, tensor(group(2), batch(16), feature(4), spatial(3, 3)), tensor(1), {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 8, 7, 7}, wei_dt, format::goiyx, tensor(group(2), batch(16), feature(4), spatial(3, 3)), {1, 1, 2, 2}, {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            // Depthwise
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 16, 7, 7}, wei_dt, format::goiyx, tensor(group(16), spatial(1, 1)), tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 16, 7, 7}, wei_dt, format::goiyx, tensor(group(16), spatial(1, 1)), {1, 1, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 16, 7, 7}, wei_dt, format::goiyx, tensor(group(16), spatial(3, 3)), tensor(1), {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 16, 7, 7}, wei_dt, format::goiyx, tensor(group(16), spatial(3, 3)), {1, 1, 2, 2}, {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });

        }
        return *this;
    }

    self& add_smoke_3d(data_types in_dt, data_types wei_dt, data_types out_dt, format::type in_fmt, format::type out_fmt) {
        std::vector<int> batches = { 1, 2 };
        for (auto b : batches) {
            // 1x1
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7, 7}, wei_dt, format::oizyx, {15, 15, 1, 1, 1}, tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7, 7}, wei_dt, format::oizyx, {15, 15, 1, 1, 1}, {1, 1, 2, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            // 3x3
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7, 7}, wei_dt, format::oizyx, {15, 15, 3, 3, 3}, tensor(1), {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7, 7}, wei_dt, format::oizyx, {15, 15, 3, 3, 3}, {1, 1, 2, 2, 2}, {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
            // Grouped
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 8, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(2), batch(16), feature(4), spatial(1, 1, 1)), tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 8, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(2), batch(16), feature(4), spatial(1, 1, 1)), {1, 1, 2, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 8, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(2), batch(16), feature(4), spatial(3, 3, 3)), tensor(1), {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 8, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(2), batch(16), feature(4), spatial(3, 3, 3)), {1, 1, 2, 2, 2}, {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
            // Depthwise
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 16, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(16), spatial(1, 1, 1)), tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 16, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(16), spatial(1, 1, 1)), {1, 1, 2, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 16, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(16), spatial(3, 3, 3)), tensor(1), {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 16, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(16), spatial(3, 3, 3)), {1, 1, 2, 2, 2}, {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
        }
        return *this;
    }

    self& add_extra_2d(data_types in_dt, data_types wei_dt, data_types out_dt, format::type in_fmt, format::type out_fmt) {
        std::vector<int> batches = { 1, 2, 16 };
        for (auto b : batches) {
            // 1x1
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 1, 1}, tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 1, 1}, {1, 1, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            // 3x3
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 3, 3}, tensor(1), {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 3, 3}, {1, 1, 2, 2}, {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            // Asymmetric weights
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 3, 2}, tensor(1), {0, 0, 0, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 3, 2}, {1, 1, 2, 2}, {0, 0, 0, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            // Uneven groups
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17}, wei_dt, format::goiyx, tensor(group(3), batch(7), feature(9), spatial(1, 1)), tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17}, wei_dt, format::goiyx, tensor(group(3), batch(7), feature(9), spatial(1, 1)), {1, 1, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17}, wei_dt, format::goiyx, tensor(group(3), batch(7), feature(9), spatial(3, 3)), tensor(1), {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17}, wei_dt, format::goiyx, tensor(group(3), batch(7), feature(9), spatial(3, 3)), {1, 1, 2, 2}, {0, 0, -1, -1, 0}, true, out_dt, implementation_desc{out_fmt, ""} });
        }
        return *this;
    }

    self& add_extra_3d(data_types in_dt, data_types wei_dt, data_types out_dt, format::type in_fmt, format::type out_fmt) {
        std::vector<int> batches = { 1, 2, 16 };
        for (auto b : batches) {
            // 1x1
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 1, 1, 1}, tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 1, 1, 1}, {1, 1, 2, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            // 3x3
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 3, 3, 3}, tensor(1), {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 3, 3, 3}, {1, 1, 2, 2, 2}, {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
            // Asymmetric weights
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 3, 2, 4}, tensor(1), {0, 0, 0, -1, -2}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 3, 2, 4}, {1, 1, 2, 2, 2}, {0, 0, 0, -1, -2}, true, out_dt, implementation_desc{out_fmt, ""} });
            // Uneven groups
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17, 11}, wei_dt, format::goizyx, tensor(group(3), batch(7), feature(9), spatial(1, 1, 1)), tensor(1), tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17, 11}, wei_dt, format::goizyx, tensor(group(3), batch(7), feature(9), spatial(1, 1, 1)), {1, 1, 2, 2, 2}, tensor(0), true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17, 11}, wei_dt, format::goizyx, tensor(group(3), batch(7), feature(9), spatial(3, 3, 3)), tensor(1), {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17, 11}, wei_dt, format::goizyx, tensor(group(3), batch(7), feature(9), spatial(3, 3, 3)), {1, 1, 2, 2, 2}, {0, 0, -1, -1, -1}, true, out_dt, implementation_desc{out_fmt, ""} });
        }
        return *this;
    }

    self& add_all_2d(data_types in_dt, data_types wei_dt, data_types out_dt, format::type in_fmt, format::type out_fmt) {
        return add_smoke_2d(in_dt, wei_dt, out_dt, in_fmt, out_fmt)
            .add_extra_2d(in_dt, wei_dt, out_dt, in_fmt, out_fmt);
    }

    self& add_all_3d(data_types in_dt, data_types wei_dt, data_types out_dt, format::type in_fmt, format::type out_fmt) {
        return add_smoke_3d(in_dt, wei_dt, out_dt, in_fmt, out_fmt)
            .add_extra_3d(in_dt, wei_dt, out_dt, in_fmt, out_fmt);
    }
};

TEST_P(deconvolution_random_test, basic) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke, deconvolution_random_test, testing::ValuesIn(
    deconvolution_random_test_params_generator()
    .add_smoke_2d(data_types::f32, data_types::f32, data_types::f32, format::bfyx, format::any)
    .add_smoke_3d(data_types::f32, data_types::f32, data_types::f32, format::bfzyx, format::any)
    .add_smoke_2d(data_types::f32, data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_smoke_3d(data_types::f32, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)

    .add_smoke_2d(data_types::f16, data_types::f16, data_types::f16, format::bfyx, format::any)
    .add_smoke_3d(data_types::f16, data_types::f16, data_types::f16, format::bfzyx, format::any)
    .add_smoke_2d(data_types::f16, data_types::f16, data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_smoke_3d(data_types::f16, data_types::f16, data_types::f16, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)

    .add_smoke_2d(data_types::i8, data_types::i8, data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_smoke_3d(data_types::i8, data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)
), deconvolution_random_test_params::print_params);

INSTANTIATE_TEST_SUITE_P(DISABLED_extended, deconvolution_random_test, testing::ValuesIn(
    deconvolution_random_test_params_generator()
    .add_extra_2d(data_types::f32, data_types::f32, data_types::f32, format::bfyx, format::any)
    .add_extra_3d(data_types::f32, data_types::f32, data_types::f32, format::bfzyx, format::any)
    .add_extra_2d(data_types::f32, data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_extra_3d(data_types::f32, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)

    .add_extra_2d(data_types::f16, data_types::f16, data_types::f16, format::bfyx, format::any)
    .add_extra_3d(data_types::f16, data_types::f16, data_types::f16, format::bfzyx, format::any)
    .add_extra_2d(data_types::f16, data_types::f16, data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_extra_3d(data_types::f16, data_types::f16, data_types::f16, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)

    .add_extra_2d(data_types::i8, data_types::i8, data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_all_2d(data_types::u8, data_types::i8, data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_extra_3d(data_types::i8, data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)
    .add_all_3d(data_types::u8, data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)

    .add_all_2d(data_types::i8, data_types::i8, data_types::f32, format::bs_fs_yx_bsv16_fsv16, format::bs_fs_yx_bsv16_fsv16)
    .add_all_2d(data_types::u8, data_types::i8, data_types::f32, format::bs_fs_yx_bsv16_fsv16, format::bs_fs_yx_bsv16_fsv16)
    .add_all_3d(data_types::i8, data_types::i8, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, format::bs_fs_zyx_bsv16_fsv16)
    .add_all_3d(data_types::u8, data_types::i8, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, format::bs_fs_zyx_bsv16_fsv16)
), deconvolution_random_test_params::print_params);
