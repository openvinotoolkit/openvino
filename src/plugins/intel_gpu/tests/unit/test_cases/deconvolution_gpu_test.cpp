// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/deconvolution.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

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
struct deconvolution_traits<ov::float16> {
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
    ov::Strides stride,
    ov::CoordinateDiff offset,
    size_t input_f_start
) {
    auto ifm = weights.size();
    int64_t filter_z = static_cast<int64_t>(weights[0].size());
    int64_t filter_y = static_cast<int64_t>(weights[0][0].size());
    int64_t filter_x = static_cast<int64_t>(weights[0][0][0].size());

    int64_t in_z = static_cast<int64_t>(input[0].size());
    int64_t in_y = static_cast<int64_t>(input[0][0].size());
    int64_t in_x = static_cast<int64_t>(input[0][0][0].size());

    int64_t offset_x = offset.size() >= 1 ? -offset[offset.size() - 1] : 0;
    int64_t offset_y = offset.size() >= 2 ? -offset[offset.size() - 2] : 0;
    int64_t offset_z = offset.size() >= 3 ? -offset[offset.size() - 3] : 0;

    int64_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;
    int64_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
    int64_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;

    int64_t out_x = 2 * offset_x + (in_x - 1) * stride_x + filter_x;
    int64_t out_y = 2 * offset_y + (in_y - 1) * stride_y + filter_y;
    int64_t out_z = 2 * offset_z + (in_z - 1) * stride_z + filter_z;
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

template <cldnn::format::type input>
struct deconvolution_input {
   static const cldnn::format::type input_layout_format = input;
};

template <typename TypeInput>
struct deconvolution_basic : public testing::Test {
protected:
    static const cldnn::format::type input_layout_format = TypeInput::input_layout_format;
};

using deconvolution_types = testing::Types<deconvolution_input<cldnn::format::bfyx>,
                                           deconvolution_input<cldnn::format::yxfb>,
                                           deconvolution_input<cldnn::format::b_fs_yx_fsv32>,
                                           deconvolution_input<cldnn::format::b_fs_yx_fsv16>,
                                           deconvolution_input<cldnn::format::bs_fs_yx_bsv32_fsv16>,
                                           deconvolution_input<cldnn::format::bs_fs_yx_bsv16_fsv16>,
                                           deconvolution_input<cldnn::format::bs_fs_yx_bsv32_fsv32>>;

TYPED_TEST_SUITE(deconvolution_basic, deconvolution_types);

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in2x2x1x1_nopad) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 1,1 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "plane_output");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, no_bias_basic_wsiz2x2_in2x2x1x1_nopad) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }),
        reorder("plane_output", input_info("deconv"), format::bfyx, data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "plane_output");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -16.f, 3.f, 0.25f,
        16.f, -1.25f, 5.25f,
        21.f, 40.5f, 13.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}


TYPED_TEST(deconvolution_basic, no_bias_basic_wsiz2x2_in2x2x1x1_nopad_exclude_fused_mem_dep) {
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
    // -16.f, 3.f, 0.25f,
    // 16.f, -1.25f, 5.25f,
    // 21.f, 40.5f, 13.5f

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx,{ 1, 1, 2, 2 } });
    auto elt_input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 9, 1, 1, 1 } });
    auto in_layout = layout(ov::PartialShape::dynamic(4), data_types::f32, format::yxfb);

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(elt_input, { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f });

    topology topology(
        input_layout("input", in_layout),
        input_layout("elt_input", elt_input->get_layout()),
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        reorder("reordered_elt_input", input_info("elt_input"), format::bfyx, data_types::f32),
        data("weights", weights),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }),
        eltwise("elt_scale", { input_info("deconv"), input_info("reordered_elt_input") }, eltwise_mode::prod),
        reorder("plane_output", input_info("elt_scale"), format::bfyx, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    network.set_input_data("elt_input", elt_input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "plane_output");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -16.f, 3.f, 0.25f,
        16.f, -1.25f, 5.25f,
        21.f, 40.5f, 13.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in2x2x1x1_nopad_bfyx) {    //  Filter : 2x2
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 1,1 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "plane_output");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in2x2x1x1_pad1) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 1, 1 }, { 1, 1}),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(0.75f, output_ptr[0]);
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in2x2x1x1_stride2_nopad) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 2,2 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -15.f, 5.f, 0.f, 1.25f,
        29.f, 13.f, 2.75f, 1.75,
        -11.f, 4.f, -17.f, 5.5f,
        22.f, 10.f, 32.5f, 14.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in2x2x1x1_stride4_pad2) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, {4, 4 }, { 2, 2 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        40.f, 0.f, 1.5f,
        0.f, 0.f, 0.f,
        6.f, 0.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in2x2x1x2_stride2_pad1) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 2, 2 }, { 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::yxfb, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 0.5f, 4.5f, 22.f,
        13.f, 5.f, -17.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2x2_in2x2x1x1_stride2_pad1) {
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
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::yxio, { 2, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.f, -2.f, 2.f, 2.f, 7.f, 7.f, -0.5f, -0.5f });
    set_values(biases, { 1.0f, 5.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 2, 2 }, { 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::yxfb, cldnn::data_types::f32)
    );

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 1.f, 4.5f, 8.5f,
        13.f, 17.f, -17.f, -13.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 2, 2 }, { 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
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
        reorder("reorder", input_info("input"), input->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reorder"), { "weights" }, { "biases" }, { 2, 2 }, { 1, 1 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
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
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::yxio,{ 2, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f });
    set_values(weights, { -2.f, -2.f, 2.f, 2.f, 7.f, 7.f, -0.5f, -0.5f });
    set_values(biases, { 1.0f, 5.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), input->get_layout().with_padding(padding{ { 0, 0, 1, 2 }, 0 })),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reorder"), { "weights" }, { "biases" }, { 2, 2 }, { 1, 1 })
    );

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 1.f, 4.5f, 8.5f,
        13.f, 17.f, -17.f, -13.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in2x2x1x2_bfyx_yxfb_stride2_pad1) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 2, 2 }, { 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();


    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_f16_wsiz2x2_in2x2x1x2_bfyx_yxfb_stride2_pad1) {
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

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    set_values(input, { ov::float16(8.f), ov::float16(0.5f),
                        ov::float16(6.f), ov::float16(9.f),

                        ov::float16(1.f), ov::float16(3.f),
                        ov::float16(2.f), ov::float16(4.f) });
    set_values(weights, { -2.f, 2.f,
                          7.f, -0.5f});
    set_values(biases, { 1.0f });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f16),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 2, 2 }, { 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f16)
    );

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<uint16_t> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], half_to_float(output_ptr[i]));
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_split2) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, 2, { 2, 2 }, { 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        -8.f, -28.f, 1.f, -17.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group2) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, 2, { 2, 2 }, { 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -3.f, 4.5f, 13.f, -17.f,
        -8.f, -28.f, 1.f, -17.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group16) {
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

    topology.add(
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "bias" }, 16, { 2, 2 }, { 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

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
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in1x2x2x2_bfyx_stride2_pad1_group16_ofm2) {
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

    topology.add(
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "bias" }, 16, { 2, 2 }, { 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

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
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic, basic_wsiz2x2_in1x6x1x1_bfyx_stride2_pad1_group2_ofm3) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, 2, { 1, 1 }, { 0, 0 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -1.5f, 8.0f, 7.75f, 11.0f, 6.0f, -2.0f
    };
    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

template<typename  DeconvolutionInput>
class deconvolution_basic_3d : public deconvolution_basic<DeconvolutionInput> {

};

using deconvolution_3d_types = testing::Types<deconvolution_input<cldnn::format::bfzyx>,
                                           deconvolution_input<cldnn::format::b_fs_zyx_fsv32>,
                                           deconvolution_input<cldnn::format::b_fs_zyx_fsv16>,
                                           deconvolution_input<cldnn::format::bs_fs_zyx_bsv32_fsv16>,
                                           deconvolution_input<cldnn::format::bs_fs_zyx_bsv16_fsv16>,
                                           deconvolution_input<cldnn::format::bs_fs_zyx_bsv16_fsv32>,
                                           deconvolution_input<cldnn::format::bs_fs_zyx_bsv32_fsv32>>;

TYPED_TEST_SUITE(deconvolution_basic_3d, deconvolution_3d_types);

TYPED_TEST(deconvolution_basic_3d, basic3D_wsiz2x2x1_in1x1x2x2x1_nopad) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 1,1,1 }, {0, 0, 0}),
        reorder("plane_output", input_info("deconv"), format::bfzyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();


    auto output_prim = outputs.at("plane_output").get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        -14.f, 5.f, 2.25f,
        18.f, 0.75f, 7.25f,
        23.f, 42.5f, 15.5f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic_3d, basic3D_wsiz3x3x3_in1x1x4x4x4_nopad) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, {1, 1, 1}, {0, 0, 0}),
        reorder("plane_output", input_info("deconv"), format::bfzyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();

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
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic_3d, basic3D_wsiz2x2x2_in1x1x2x2x2_stride2_nopad) {
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

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { 2,2,2 }, {0, 0, 0}),
        reorder("plane_output", input_info("deconv"), format::bfzyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();
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
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TYPED_TEST(deconvolution_basic_3d, basic3D_wsiz2x2x2_in1x1x2x2x2_stride2_pad1) {
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f32),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { 2,2,2 }, { 1, 1, 1 }),
        reorder("plane_output", input_info("deconv"), format::bfzyx, cldnn::data_types::f32)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_prim = outputs.at("plane_output").get_memory();
    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
        12.f, 1.75f, 3.f, -18.f,
        12.f, 1.75f, 3.f, -18.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }

}

TYPED_TEST(deconvolution_basic, basic_f16_k9x9_s2x2_pad4x4) {
    //  Filter : 1x32x9x9
    //  Input  : 1x32x16x16
    //  Stride : 2x2
    //  Pad    : 4x4
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    VVVVF<ov::float16> input_rnd = rg.generate_random_4d<ov::float16>(1, 32, 16, 16, -2, 2);
    VF<ov::float16> input_rnd_vec = flatten_4d<ov::float16>(format::bfyx, input_rnd);
    VVVVF<ov::float16> filter_rnd = rg.generate_random_4d<ov::float16>(1, 32, 9, 9, -1, 1);
    VF<ov::float16> filter_rnd_vec = flatten_4d<ov::float16>(format::bfyx, filter_rnd);
    VF<ov::float16> bias_rnd = rg.generate_random_1d<ov::float16>(1, -1, 1);
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
        reorder("reordered_input", input_info("input"), this->input_layout_format, data_types::f16),
        data("weights", weights),
        data("biases", biases),
        deconvolution("deconv", input_info("reordered_input"), { "weights" }, { "biases" }, { 2, 2 }, { 4, 4 }, { 1, 1 }, tensor{ 1, 1, 32, 32 }),
        reorder("plane_output", input_info("deconv"), format::bfyx, cldnn::data_types::f16)
    );

    network network_ref(engine, topology_ref, get_test_default_config(engine));
    network_ref.set_input_data("input", input);

    auto outputs_ref = network_ref.execute();
    auto output_ref_prim = outputs_ref.at("plane_output").get_memory();
    cldnn::mem_lock<ov::float16> output_ref_ptr(output_ref_prim, get_test_stream());

    std::vector<ov::float16> output_vec_ref;
    for (unsigned int i = 0; i < output_ref_prim->get_layout().count(); i++) {
        output_vec_ref.push_back(output_ref_ptr[i]);
    }

    topology topology_act(
        input_layout("input_act", input->get_layout()),
        data("weights_f32", weights_f32),
        data("biases_f32", biases_f32),
        deconvolution("deconv_act", input_info("input_act"), { "weights_f32" }, { "biases_f32" }, { 2, 2 }, { 4, 4 }),
        reorder("out", input_info("deconv_act"), format::bfyx, data_types::f16)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network_act(engine, topology_act, config);
    network_act.set_input_data("input_act", input);

    auto outputs_act = network_act.execute();
    ASSERT_EQ(outputs_act.size(), size_t(1));
    ASSERT_EQ(outputs_act.begin()->first, "out");
    auto output_act_prim = outputs_act.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16> output_act_ptr(output_act_prim, get_test_stream());

    std::vector<float> output_vec;
    for (unsigned int i = 0; i < output_act_prim->get_layout().count(); i++) {
        float x = float_round(output_act_ptr[i]), y = float_round(output_vec_ref[i]);
        ASSERT_NEAR(x, y, 1e-0f);
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
            deconvolution("deconv", input_info("input"), { "weights" }, { "biases" }, { 2, 2 }, { 1, 1 }),
            reorder("out", input_info("deconv"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -3.f, 4.5f, 13.f, -17.f,
            .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
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

    set_values(input, { ov::float16(8.f), ov::float16(0.5f),
                        ov::float16(6.f), ov::float16(9.f),

                        ov::float16(1.f), ov::float16(3.f),
                        ov::float16(2.f), ov::float16(4.f) });
    set_values(weights, { ov::float16(-2.f), ov::float16(2.f),
                          ov::float16(7.f), ov::float16(-0.5f)});
    set_values(biases, { ov::float16(1.0f) });

    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", input_info("input"), { "weights" }, { "biases" }, { 2, 2 }, { 1, 1 }),
            reorder("out", input_info("deconv"), format::bfyx, data_types::f16)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<uint16_t> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -3.f, 4.5f, 13.f, -17.f,
            .5f, 22.f, 5.f, -7.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        ASSERT_FLOAT_EQ(expected_output_vec[i], half_to_float(output_ptr[i]));
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
            deconvolution("deconv", input_info("input"), { "weights" }, { "biases" }, 2, { 2, 2 }, { 1, 1 }),
            reorder("out", input_info("deconv"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -3.f, 4.5f, 13.f, -17.f,
            -8.f, -28.f, 1.f, -17.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
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
            deconvolution("deconv", input_info("input"), { "weights" }, { "biases" }, 2, { 2, 2 }, { 1, 1 }),
            reorder("out", input_info("deconv"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    std::vector<float> expected_output_vec = {
            -4.f, 3.5f, 12.f, -18.f,
            -7.f, -27.f, 2.f, -16.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++) {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
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
            reorder("input_fsv16", input_info("input"), format::b_fs_yx_fsv16, data_types::f32),
            deconvolution("deconv", input_info("input_fsv16"), { "weights" }, { "biases" }, 2, { 1, 1 }, { 0, 0 }),
            reorder("out", input_info("deconv"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

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
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
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
            reorder("input_fsv16", input_info("input"), format::b_fs_yx_fsv16, data_types::f32),
            deconvolution("deconv", input_info("input_fsv16"), { "weights" }, { "biases" }, 2, { 1, 1 }, { 1, 1 }),
            reorder("out", input_info("deconv"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_FLOAT_EQ(0.75f, output_ptr[0]);
    ASSERT_FLOAT_EQ(0.75f, output_ptr[1]);
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
        deconvolution("deconv", input_info("input"), { "weights" }, { "biases" }, 2, { 2,2 }),
        reorder("out", input_info("deconv"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

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
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
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
            deconvolution("deconv", input_info("input"), { "weights" }, { "biases" }, 2, { 4, 4 }, { 2, 2 }),
            reorder("out", input_info("deconv"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

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
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
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
            deconvolution("deconv", input_info("input"), { "weights" }, { "biases" }, 2, { 4, 4 }, { 2, 2 }),
            reorder("out", input_info("deconv"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

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
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
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
            deconvolution("deconv", input_info("input"), { "weights" }, { 2,2,2 }, { 1, 1, 1 }),
            reorder("out", input_info("deconv"), format::bfzyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::bs_fs_zyx_bsv16_fsv16, "" };
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", impl} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

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

template <typename T>
void test_deconvolution_f16_fw_gpu_basic_wsiz2x2_in1x2x2x2_fs_b_yx_fsv32_stride1_pad1_replace_to_conv(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx,{ 2, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f16, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, { ov::float16(8.f), ov::float16(0.5f), ov::float16(6.f), ov::float16(9.f),
                        ov::float16(1.f), ov::float16(3.f), ov::float16(2.f), ov::float16(4.f)
                        });
    set_values(weights, {
            ov::float16(-2.f), ov::float16(2.f), ov::float16(7.f), ov::float16(-0.5f),
            ov::float16(-4.f), ov::float16(1.f), ov::float16(-9.f), ov::float16(-7.f)
    });
    set_values(biases, { ov::float16(1.0f), ov::float16(-1.0f) });

    topology topology(
            input_layout("input", input->get_layout()),
            reorder("reorder", input_info("input"), format::fs_b_yx_fsv32, data_types::f16),
            data("weights", weights),
            data("biases", biases),
            deconvolution("deconv", input_info("reorder"), { "weights" }, { "biases" }, 1, { 1, 1 }, { 0, 0 }),
            reorder("out", input_info("deconv"), format::bfyx, data_types::f32)
    );

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<T> output_ptr (output_prim, get_test_stream());

    std::vector<T> expected_output_vec = {
            -15.f, 16.f, 2.f, 45.f, -5.5f, 18.75f, 43.f, 61.f, -3.5f,
            -33.f, 5.f, -0.5f, -97.f, -91.5f, 4.5f, -55.f, -124.f, -64.f,
            -1.f, -3.f, 7.f, 4.f, 17.5f, 7.5f, 15.f, 28.f, -1.f,
            -5.f, -12.f, 2.f, -18.f, -49.f, -18.f, -19.f, -51.f, -29.f,
    };
    ASSERT_EQ(expected_output_vec.size(), output_prim->count());

    for (size_t i = 0; i < expected_output_vec.size(); i++) {
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]) << " index=" << i;
    }
}

TEST(deconvolution_f16_fw_gpu, basic_wsiz2x2_in1x2x2x2_fs_b_yx_fsv32_stride1_pad1_replace_to_conv) {
    test_deconvolution_f16_fw_gpu_basic_wsiz2x2_in1x2x2x2_fs_b_yx_fsv32_stride1_pad1_replace_to_conv<float>(false);
}

TEST(export_import_deconvolution_f16_fw_gpu, basic_wsiz2x2_in1x2x2x2_fs_b_yx_fsv32_stride1_pad1_replace_to_conv) {
    test_deconvolution_f16_fw_gpu_basic_wsiz2x2_in1x2x2x2_fs_b_yx_fsv32_stride1_pad1_replace_to_conv<float>(true);
}

struct deconvolution_random_test_params {
    data_types input_type;
    format::type input_format;
    tensor input_size;
    data_types weights_type;
    format::type weights_format;
    tensor weights_size;
    ov::Strides strides;
    ov::CoordinateDiff pad;
    bool with_bias;
    data_types output_type;
    ov::intel_gpu::ImplementationDesc deconv_desc;

    static std::string print_params(const testing::TestParamInfo<deconvolution_random_test_params>& param_info) {
        auto& param = param_info.param;
        auto to_string_neg = [](int64_t v) {
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

        auto print_strides = [&](const ov::Strides& s) {
            std::string res = to_string_neg(s[0]);
            for (size_t i = 1; i < s.size(); i++) {
                res += "x" + to_string_neg(s[i]);
            }
            return res;
        };

        auto print_coord = [&](const ov::CoordinateDiff& cd) {
            std::string res = to_string_neg(cd[0]);
            for (size_t i = 1; i < cd.size(); i++) {
                res += "x" + to_string_neg(cd[i]);
            }
            return res;
        };

        // construct a readable name
        return "in_" + dt_to_str(param.input_type) +
            "_" + fmt_to_str(param.input_format) +
            "_" + print_tensor(param.input_size) +
            "_wei_" + dt_to_str(param.weights_type) +
            "_" + fmt_to_str(param.weights_format) +
            "_" + print_tensor(param.weights_size) +
            (param.with_bias ? "_bias" : "") +
            "_s_" + print_strides(param.strides) +
            "_off_" + print_coord(param.pad) +
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
struct typed_comparator<ov::float16> {
    static ::testing::AssertionResult compare(const char* lhs_expr, const char* rhs_expr, ov::float16 ref, ov::float16 val) {
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

    void run(cldnn::engine& eng, const deconvolution_random_test_params& params, ExecutionConfig config, tests::random_generator& rg) {
        uint32_t groups = params.weights_size.group[0];
        size_t ifm = params.weights_size.feature[0];
        size_t ofm = params.weights_size.batch[0];

        auto input_data = rg.generate_random_5d<InputT>(params.input_size.batch[0],
                                                        params.input_size.feature[0],
                                                        params.input_size.spatial[2],
                                                        params.input_size.spatial[1],
                                                        params.input_size.spatial[0],
                                                        type_test_ranges<InputT>::min,
                                                        type_test_ranges<InputT>::max);
        auto weights_data = rg.generate_random_6d<WeightsT>(params.weights_size.group[0],
                                                            params.weights_size.batch[0],
                                                            params.weights_size.feature[0],
                                                            params.weights_size.spatial[2],
                                                            params.weights_size.spatial[1],
                                                            params.weights_size.spatial[0],
                                                            type_test_ranges<WeightsT>::min,
                                                            type_test_ranges<WeightsT>::max);

        auto in_layout = cldnn::layout(ov::element::from<InputT>(), params.input_format, params.input_size);
        auto wei_layout = cldnn::layout(ov::element::from<WeightsT>(), params.weights_format, params.weights_size);

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
            auto bias_lay = cldnn::layout(ov::element::from<OutputT>(), cldnn::format::bfyx, bias_size);
            auto bias_mem = eng.allocate_memory(bias_lay);
            bias_data = rg.generate_random_1d<OutputT>(bias_lay.feature(), -1, 1);
            set_values(bias_mem, bias_data);
            topo.add(cldnn::data("bias", bias_mem));
            topo.add(cldnn::deconvolution("deconv", input_info("input"), { "weights" }, { "bias" }, groups, params.strides, params.pad));
        } else {
            topo.add(cldnn::deconvolution("deconv", input_info("input"), { "weights" }, groups, params.strides, params.pad));
        }

        // turn off optimizer to check blocked format without reordering to plane format
        if (params.deconv_desc.output_format == cldnn::format::any && !format::is_simple_data_format(in_layout.format))  {
            config.set_property(ov::intel_gpu::optimize_data(false));
        }

        if (!params.deconv_desc.kernel_name.empty() || params.deconv_desc.output_format != cldnn::format::any) {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "deconv", params.deconv_desc } }));
        }

        cldnn::network net(eng, topo, config);
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
            cldnn::mem_lock<OutputT, mem_lock_type::read> ptr(out_mem, get_test_stream());

            auto b = static_cast<size_t>(out_mem->get_layout().batch());
            auto of = static_cast<size_t>(out_mem->get_layout().feature());

            for (size_t bi = 0; bi < b; ++bi) {
                for (size_t fi = 0; fi < of; ++fi) {
                    size_t group = fi / ofm;
                    auto reference = reference_deconvolution<InputT, WeightsT, OutputT>(
                        input_data[bi],
                        weights_data[group][fi % ofm],
                        bias_data.empty() ? 0.f : static_cast<float>(bias_data[fi]),
                        params.strides,
                        params.pad,
                        group * ifm);

                    ASSERT_EQ(reference.size(), out_mem->get_layout().spatial(2));
                    ASSERT_EQ(reference[0].size(), out_mem->get_layout().spatial(1));
                    ASSERT_EQ(reference[0][0].size(), out_mem->get_layout().spatial(0));
                    // check that reordering not happened
                    if (!format::is_simple_data_format(in_layout.format)) {
                        ASSERT_FALSE(format::is_simple_data_format(out_mem->get_layout().format));
                    }
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
        rg.set_seed(GET_SUITE_NAME);
        config.set_property(ov::intel_gpu::optimize_data(true));
    }

    void run() {
        auto params = GetParam();
        switch (params.input_type) {
        case data_types::f32:
            run_typed_in<float>();
            break;
        case data_types::f16:
            run_typed_in<ov::float16>();
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

    tests::random_generator rg;
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(get_test_engine());

private:
    template <typename InputT, typename WeightsT, typename OutputT>
    void run_typed() {
        auto& params = GetParam();
        deconvolution_random_test_base<InputT, WeightsT, OutputT> test;
        test.run(get_test_engine(), params, config, rg);
    }

    template <typename InputT, typename WeightsT>
    void run_typed_in_wei() {
        auto& params = GetParam();
        switch (params.output_type) {
        case data_types::f32:
            run_typed<InputT, WeightsT, float>();
            break;
        case data_types::f16:
            run_typed<InputT, WeightsT, ov::float16>();
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
            run_typed_in_wei<InputT, ov::float16>();
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
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 15, 7, 7}, wei_dt, format::oiyx, {15, 15, 1, 1}, {1, 1}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 15, 7, 7}, wei_dt, format::oiyx, {15, 15, 1, 1}, {2, 2}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            // 3x3
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 15, 7, 7}, wei_dt, format::oiyx, {15, 15, 3, 3}, {1, 1}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 15, 7, 7}, wei_dt, format::oiyx, {15, 15, 3, 3}, {2, 2}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            // Grouped
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 8, 7, 7}, wei_dt, format::goiyx, tensor(group(2), batch(16), feature(4), spatial(1, 1)), {1, 1}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 8, 7, 7}, wei_dt, format::goiyx, tensor(group(2), batch(16), feature(4), spatial(1, 1)), {2, 2}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 8, 7, 7}, wei_dt, format::goiyx, tensor(group(2), batch(16), feature(4), spatial(3, 3)), {1, 1}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 8, 7, 7}, wei_dt, format::goiyx, tensor(group(2), batch(16), feature(4), spatial(3, 3)), {2, 2}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            // Depthwise
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 16, 7, 7}, wei_dt, format::goiyx, tensor(group(16), spatial(1, 1)), {1, 1}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 16, 7, 7}, wei_dt, format::goiyx, tensor(group(16), spatial(1, 1)), {2, 2}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 16, 7, 7}, wei_dt, format::goiyx, tensor(group(16), spatial(3, 3)), {1, 1}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 16, 7, 7}, wei_dt, format::goiyx, tensor(group(16), spatial(3, 3)), {2, 2}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
        }
        return *this;
    }

    self& add_smoke_3d(data_types in_dt, data_types wei_dt, data_types out_dt, format::type in_fmt, format::type out_fmt) {
        std::vector<int> batches = { 1, 2, 16, 32 };
        for (auto b : batches) {
            // 1x1
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7, 7}, wei_dt, format::oizyx, {15, 15, 1, 1, 1}, {1, 1, 1}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 15, 7, 7, 7}, wei_dt, format::oizyx, {15, 15, 1, 1, 1}, {2, 2, 2}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            // 3x3
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 15, 7, 7, 7}, wei_dt, format::oizyx, {15, 15, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 15, 7, 7, 7}, wei_dt, format::oizyx, {15, 15, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            // Grouped
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 8, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(2), batch(16), feature(4), spatial(1, 1, 1)), {1, 1, 1}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 8, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(2), batch(16), feature(4), spatial(1, 1, 1)), {2, 2, 2}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 8, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(2), batch(16), feature(4), spatial(3, 3, 3)), {1, 1, 1}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 8, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(2), batch(16), feature(4), spatial(3, 3, 3)), {2, 2, 2}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            // Depthwise
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 16, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(16), spatial(1, 1, 1)), {1, 1, 1}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 16, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(16), spatial(1, 1, 1)), {2, 2, 2}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 16, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(16), spatial(3, 3, 3)), {1, 1, 1}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 16, 7, 7, 7}, wei_dt, format::goizyx, tensor(group(16), spatial(3, 3, 3)), {2, 2, 2}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
        }
        return *this;
    }

    self& add_extra_2d(data_types in_dt, data_types wei_dt, data_types out_dt, format::type in_fmt, format::type out_fmt) {
        std::vector<int> batches = { 1, 2, 16 };
        for (auto b : batches) {
            // 1x1
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 1, 1}, {1, 1}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 1, 1}, {2, 2}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            // 3x3
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 3, 3}, {1, 1}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 3, 3}, {2, 2}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            // Asymmetric weights
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 3, 2}, {1, 1}, {1, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 31, 19, 17}, wei_dt, format::oiyx, {41, 31, 3, 2}, {2, 2}, {1, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            // Uneven groups
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 27, 19, 17}, wei_dt, format::goiyx, tensor(group(3), batch(7), feature(9), spatial(1, 1)), {1, 1}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 27, 19, 17}, wei_dt, format::goiyx, tensor(group(3), batch(7), feature(9), spatial(1, 1)), {2, 2}, {0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 27, 19, 17}, wei_dt, format::goiyx, tensor(group(3), batch(7), feature(9), spatial(3, 3)), {1, 1}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
            push_back(deconvolution_random_test_params{in_dt, in_fmt, {b, 27, 19, 17}, wei_dt, format::goiyx, tensor(group(3), batch(7), feature(9), spatial(3, 3)), {2, 2}, {1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""}});
        }
        return *this;
    }

    self& add_extra_3d(data_types in_dt, data_types wei_dt, data_types out_dt, format::type in_fmt, format::type out_fmt) {
        std::vector<int> batches = { 1, 2, 16 };
        for (auto b : batches) {
            // 1x1
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 1, 1, 1}, {1, 1, 1}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 1, 1, 1}, {2, 2, 2}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            // 3x3
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            // Asymmetric weights
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 3, 2, 4}, {1, 1, 1}, {2, 1, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 31, 19, 17, 11}, wei_dt, format::oizyx, {41, 31, 3, 2, 4}, {2, 2, 2}, {2, 1, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            // Uneven groups
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17, 11}, wei_dt, format::goizyx, tensor(group(3), batch(7), feature(9), spatial(1, 1, 1)), {1, 1, 1}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17, 11}, wei_dt, format::goizyx, tensor(group(3), batch(7), feature(9), spatial(1, 1, 1)), {2, 2, 2}, {0, 0, 0}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17, 11}, wei_dt, format::goizyx, tensor(group(3), batch(7), feature(9), spatial(3, 3, 3)), {1, 1, 1}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
            push_back(deconvolution_random_test_params{ in_dt, in_fmt, {b, 27, 19, 17, 11}, wei_dt, format::goizyx, tensor(group(3), batch(7), feature(9), spatial(3, 3, 3)), {2, 2, 2}, {1, 1, 1}, true, out_dt, ov::intel_gpu::ImplementationDesc{out_fmt, ""} });
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
    .add_smoke_2d(data_types::f32, data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
    .add_smoke_2d(data_types::f32, data_types::f32, data_types::f32, format::bs_fs_yx_bsv16_fsv16, format::any)
    .add_smoke_2d(data_types::f32, data_types::f32, data_types::f32, format::bs_fs_yx_bsv32_fsv16, format::any)
    .add_smoke_2d(data_types::f32, data_types::f32, data_types::f32, format::bs_fs_yx_bsv32_fsv32, format::any)

    .add_smoke_3d(data_types::f32, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)
    .add_smoke_3d(data_types::f32, data_types::f32, data_types::f32, format::b_fs_zyx_fsv32, format::any)
    .add_smoke_3d(data_types::f32, data_types::f32, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, format::any)
    .add_smoke_3d(data_types::f32, data_types::f32, data_types::f32, format::bs_fs_zyx_bsv16_fsv32, format::any)
    .add_smoke_3d(data_types::f32, data_types::f32, data_types::f32, format::bs_fs_zyx_bsv32_fsv16, format::any)
    .add_smoke_3d(data_types::f32, data_types::f32, data_types::f32, format::bs_fs_zyx_bsv32_fsv32, format::any)

    .add_smoke_2d(data_types::f16, data_types::f16, data_types::f16, format::bfyx, format::any)
    .add_smoke_3d(data_types::f16, data_types::f16, data_types::f16, format::bfzyx, format::any)
    .add_smoke_2d(data_types::f16, data_types::f16, data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_smoke_2d(data_types::f16, data_types::f16, data_types::f16, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
    .add_smoke_2d(data_types::f16, data_types::f16, data_types::f16, format::bs_fs_yx_bsv16_fsv16, format::any)
    .add_smoke_2d(data_types::f16, data_types::f16, data_types::f16, format::bs_fs_yx_bsv32_fsv16, format::any)
    .add_smoke_2d(data_types::f16, data_types::f16, data_types::f16, format::bs_fs_yx_bsv32_fsv32, format::any)

    .add_smoke_3d(data_types::f16, data_types::f16, data_types::f16, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)
    .add_smoke_3d(data_types::f16, data_types::f16, data_types::f16, format::b_fs_zyx_fsv32, format::any)
    .add_smoke_3d(data_types::f16, data_types::f16, data_types::f16, format::bs_fs_zyx_bsv16_fsv16, format::any)
    .add_smoke_3d(data_types::f16, data_types::f16, data_types::f16, format::bs_fs_zyx_bsv16_fsv32, format::any)
    .add_smoke_3d(data_types::f16, data_types::f16, data_types::f16, format::bs_fs_zyx_bsv32_fsv16, format::any)
    .add_smoke_3d(data_types::f16, data_types::f16, data_types::f16, format::bs_fs_zyx_bsv32_fsv32, format::any)

    .add_smoke_2d(data_types::i8, data_types::i8, data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_smoke_2d(data_types::i8, data_types::i8, data_types::i8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
    .add_smoke_2d(data_types::i8, data_types::i8, data_types::i8, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32)
    .add_smoke_2d(data_types::i8, data_types::i8, data_types::i8, format::bs_fs_yx_bsv16_fsv16, format::any)
    .add_smoke_2d(data_types::i8, data_types::i8, data_types::i8, format::bs_fs_yx_bsv32_fsv16, format::any)
    .add_smoke_2d(data_types::i8, data_types::i8, data_types::i8, format::bs_fs_yx_bsv32_fsv32, format::any)

    .add_smoke_3d(data_types::i8, data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)
    .add_smoke_3d(data_types::i8, data_types::i8, data_types::i8, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16)
    .add_smoke_3d(data_types::i8, data_types::i8, data_types::i8, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32)
    .add_smoke_3d(data_types::i8, data_types::i8, data_types::i8, format::bs_fs_zyx_bsv16_fsv16, format::any)
    .add_smoke_3d(data_types::i8, data_types::i8, data_types::i8, format::bs_fs_zyx_bsv16_fsv32, format::any)
    .add_smoke_3d(data_types::i8, data_types::i8, data_types::i8, format::bs_fs_zyx_bsv32_fsv16, format::any)
    .add_smoke_3d(data_types::i8, data_types::i8, data_types::i8, format::bs_fs_zyx_bsv32_fsv32, format::any)
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

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(deconvolution_f32_fw_gpu_onednn, basic_wsiz2x2_in2x2x1x1_stride2_nopad) {
    //  Filter : 1x1
    //  Input  : 2x2
    //  Output : 4x4
    //  Stride : 2x2

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

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
        deconvolution("deconv", input_info("input"), { "weights" }, { "biases" }, { 2,2 })
    );

    ov::intel_gpu::ImplementationDesc conv_impl = { format::yxfb, "", impl_types::onednn };

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"deconv", conv_impl} }));
    network network(engine, topology, cfg);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "deconv");

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
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(deconvolution_gpu_onednn, spatial_1d) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    tests::random_generator rg(GET_SUITE_NAME);
    ov::PartialShape input_pshape = {1, 16, 6};
    ov::PartialShape weights_pshape = {16, 16, 3};
    layout in_layout{ ov::PartialShape::dynamic(input_pshape.size()), data_types::f16, format::bfyx };
    layout weights_layout{ weights_pshape, data_types::f16, format::bfyx };

    auto weights_data = rg.generate_random_1d<ov::float16>(weights_layout.count(), -1, 1);
    auto weights_mem = engine.allocate_memory(weights_layout);
    set_values(weights_mem, weights_data);

    auto create_topology =[&]() {
        topology topology;
        topology.add(input_layout("input", in_layout));
        topology.add(data("weights", weights_mem));
        topology.add(deconvolution("deconv",
                                   input_info("input"),
                                   { "weights" },
                                   1,
                                   ov::Strides{1},
                                   ov::CoordinateDiff{0},
                                   ov::Strides{1}));
        topology.add(reorder("reorder", input_info("deconv"), format::bfyx, data_types::f32));
        return topology;
    };

    auto topology_test = create_topology();
    auto topology_ref = create_topology();

    ExecutionConfig config_test = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc deconv_impl_test = { format::b_fs_yx_fsv16, "", impl_types::onednn };
    config_test.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "deconv", deconv_impl_test } }));
    config_test.set_property(ov::intel_gpu::optimize_data(true));
    config_test.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    ExecutionConfig config_ref = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc deconv_impl_ref = { format::bfyx, "", impl_types::ocl };
    config_ref.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{ "deconv", deconv_impl_ref } }));
    config_ref.set_property(ov::intel_gpu::optimize_data(true));
    config_ref.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network_test(engine, topology_test, config_test);
    network network_ref(engine, topology_ref, config_ref);

    auto input_data = rg.generate_random_1d<ov::float16>(input_pshape.size(), -1, 1);
    auto input_mem = engine.allocate_memory({ input_pshape, data_types::f16, format::bfyx });
    set_values(input_mem, input_data);

    network_test.set_input_data("input", input_mem);
    network_ref.set_input_data("input", input_mem);

    auto outputs_test = network_test.execute();
    auto outputs_ref = network_ref.execute();

    ASSERT_EQ(outputs_test.size(), size_t(1));
    ASSERT_EQ(outputs_test.begin()->first, "reorder");
    ASSERT_EQ(outputs_ref.size(), size_t(1));
    ASSERT_EQ(outputs_ref.begin()->first, "reorder");

    auto output_memory_test = outputs_test.at("reorder").get_memory();
    auto output_layout_test = output_memory_test->get_layout();
    cldnn::mem_lock<float> output_ptr_test(output_memory_test, get_test_stream());

    auto output_memory_ref = outputs_ref.at("reorder").get_memory();
    auto output_layout_ref = output_memory_ref->get_layout();
    cldnn::mem_lock<float> output_ptr_ref(output_memory_ref, get_test_stream());

    ov::PartialShape expected_shape = {1, 16, 8};
    ASSERT_EQ(output_layout_test.get_partial_shape(), expected_shape);
    ASSERT_EQ(output_layout_ref.get_partial_shape(), expected_shape);

    for (size_t i = 0; i < output_memory_ref->count(); i++) {
        ASSERT_EQ(output_ptr_ref.data()[i], output_ptr_test.data()[i]);
    }
}
#endif
