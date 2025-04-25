// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/resample.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

using namespace cldnn;
using namespace ::tests;

template <typename T>
void test_basic_in2x3x2x2_nearest(bool is_caching_test) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0     -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5   -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2   -13
    //  f1: b0:  7    8  -16   b1:   12   9     -17
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 3, 2 } });

    auto output_size = tensor(batch(2), feature(2), spatial(6, 4));
    uint32_t num_filter = 0u;

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", input_info("input"), output_size, num_filter, resample::InterpolateOp::InterpolateMode::NEAREST));

    set_values(input, {
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f,
        5.f, 6.f, -12.f,
        7.f, 8.f, -16.f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, -17.f,
    });

    cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    net->set_input_data("input", input);

    auto outputs = net->execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    T answers[96] = {
        1.f, 1.f, 2.f,   2.f,   -10.f,  -10.f,
        1.f, 1.f, 2.f,   2.f,   -10.f,  -10.f,
        3.f, 3.f, 4.f,   4.f,   -14.f,  -14.f,
        3.f, 3.f, 4.f,   4.f,   -14.f,  -14.f,
        5.f, 5.f, 6.f,   6.f,   -12.f,  -12.f,
        5.f, 5.f, 6.f,   6.f,   -12.f,  -12.f,
        7.f, 7.f, 8.f,   8.f,   -16.f,  -16.f,
        7.f, 7.f, 8.f,   8.f,   -16.f,  -16.f,
        0.f, 0.f, 0.f,   0.f,   -11.f,  -11.f,
        0.f, 0.f, 0.f,   0.f,   -11.f,  -11.f,
        0.5f,0.5f, -0.5f, -0.5f, -15.f,  -15.f,
        0.5f,0.5f, -0.5f, -0.5f, -15.f,  -15.f,
        1.5f,1.5f, 5.2f,  5.2f,  -13.f,  -13.f,
        1.5f,1.5f, 5.2f,  5.2f,  -13.f,  -13.f,
        12.f,12.f, 9.f,   9.f,  -17.f,  -17.f,
        12.f,12.f, 9.f,   9.f,  -17.f,  -17.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    ASSERT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id]));
                }
            }
        }
    }
}

TEST(resample_gpu, basic_in2x3x2x2_nearest) {
    test_basic_in2x3x2x2_nearest<float>(false);
}

TEST(resample_gpu, basic_in2x3x2x2_bilinear) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x4x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(4, 4));
    uint32_t num_filter = 1u;

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", input_info("input"), output_size, num_filter, resample::InterpolateOp::InterpolateMode::LINEAR));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    cldnn::network net{ engine, topology, get_test_default_config(engine) };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(output->get_layout().get_linear_size(), (size_t) 16);

    float answers[16] = {
        1.f, 1.25f, 1.75f, 2.f,
        1.5f, 1.75f, 2.25f, 2.5f,
        2.5f, 2.75f, 3.25f, 3.5f,
        3.f, 3.25f, 3.75f, 4.f,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 4; ++l) { // X
            auto linear_id = l + k * 4;
            ASSERT_NEAR(answers[linear_id], output_ptr[linear_id], 1e-05F);
        }
    }
}

TEST(resample_gpu, nearest_asymmetric) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x5x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(5, 4));

    topology topology;
    uint32_t num_filter = 1u;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", input_info("input"), output_size, num_filter, resample::InterpolateOp::InterpolateMode::NEAREST));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
    });

    cldnn::network net{ engine, topology, get_test_default_config(engine) };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(output->get_layout().get_linear_size(), (size_t)20);

    float answers[20] = {
        1.f, 1.f, 1.f, 2.f, 2.f,
        1.f, 1.f, 1.f, 2.f, 2.f,
        3.f, 3.f, 3.f, 4.f, 4.f,
        3.f, 3.f, 3.f, 4.f, 4.f,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 5; ++l) { // X
            auto linear_id = l + k * 5;
            ASSERT_NEAR(answers[linear_id], output_ptr[linear_id], 1e-05F);
        }
    }
}

TEST(resample_gpu, nearest_asymmetric_i8) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x5x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::i8, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(5, 4));

    topology topology;
    uint32_t num_filter = 1u;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", input_info("input"), output_size, num_filter, resample::InterpolateOp::InterpolateMode::NEAREST));

    set_values<int8_t>(input, {
            1, 2,
            3, 4,
    });

    cldnn::network net{ engine, topology, get_test_default_config(engine) };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    ASSERT_EQ(output->get_layout().get_linear_size(), (size_t)20);

    int8_t answers[20] = {
            1, 1, 1, 2, 2,
            1, 1, 1, 2, 2,
            3, 3, 3, 4, 4,
            3, 3, 3, 4, 4,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 5; ++l) { // X
            auto linear_id = l + k * 5;
            ASSERT_EQ(answers[linear_id], output_ptr[linear_id]);
        }
    }
}

TEST(resample_gpu, bilinear_asymmetric) {
    //  Input  : 1x1x2x2
    //  Output : 1x1x5x4
    //  Sample Type: Nearest

    //  Input:
    //  f0: b0:  1    2
    //  f0: b0:  3    4
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto output_size = tensor(batch(1), feature(1), spatial(6, 4));

    topology topology;
    uint32_t num_filter = 1u;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("upsampling", input_info("input"), output_size, num_filter, resample::InterpolateOp::InterpolateMode::LINEAR));

    set_values(input, {
        1.f, 2.f,
        3.f, 4.f,
               });

    cldnn::network net{ engine, topology, get_test_default_config(engine) };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("upsampling").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(output->get_layout().get_linear_size(), (size_t)24);

    float answers[24] = {
        1.f, 1.f, 1.33333f, 1.66667f, 2.f, 2.f,
        1.5f, 1.5f, 1.83333f, 2.16667f, 2.5f, 2.5f,
        2.5f, 2.5f, 2.83333f, 3.16667f, 3.5f, 3.5f,
        3.f, 3.f, 3.33333f, 3.66667f, 4.f, 4.f,
    };

    for (int k = 0; k < 4; ++k) { // Y
        for (int l = 0; l < 6; ++l) { // X
            auto linear_id = l + k * 6;
            ASSERT_NEAR(answers[linear_id], output_ptr[linear_id], 5e-03F) << l << " " << k;
        }
    }
}

struct resample_random_test_params {
    data_types input_type;
    tensor input_size;
    tensor output_size;
    uint32_t num_filter;
    resample::InterpolateOp::InterpolateMode operation_type;
    uint32_t align_corners;
    format::type in_format;
    format::type out_format;
};

struct resample_random_test : testing::TestWithParam<resample_random_test_params>{
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);

        auto data = rg.generate_random_4d<T>(b, f, y, x, min, max, k);
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<ov::float16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    template <typename T>
    void compare_nearest_typed(const memory::ptr input, const memory::ptr output, uint32_t align_corners) {
        auto output_lay = output->get_layout();
        size_t b = output_lay.batch();
        size_t f = output_lay.feature();
        size_t x = output_lay.spatial(0);
        size_t y = output_lay.spatial(1);
        size_t in_x = input->get_layout().spatial(0);
        size_t in_y = input->get_layout().spatial(1);
        float x_ratio = x > align_corners ? static_cast<float>(in_x - align_corners) / static_cast<float>(x - align_corners) : 0.f;
        float y_ratio = y > align_corners ? static_cast<float>(in_y - align_corners) / static_cast<float>(y - align_corners) : 0.f;

        cldnn::mem_lock<T> in_ptr(input, get_test_stream());
        cldnn::mem_lock<T> out_ptr(output, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto in_xi = static_cast<size_t>(floor(x_ratio * xi));
                        auto in_yi = static_cast<size_t>(floor(y_ratio * yi));
                        auto in_coords = tensor(batch(bi), feature(fi), spatial(in_xi, in_yi, 0, 0));
                        auto in_offset = input->get_layout().get_linear_offset(in_coords);
                        auto in_val = in_ptr[in_offset];
                        auto out_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto out_offset = output->get_layout().get_linear_offset(out_coords);
                        auto out_val = out_ptr[out_offset];
                        ASSERT_EQ(in_val, out_val) << " at bi=" << bi << ", fi=" << fi << ", xi=" << xi << ", yi=" << yi;
                    }
                }
            }
        }
    }

    template <typename InT, typename OutT>
    void compare_bilinear_typed(const memory::ptr input, const memory::ptr output, uint32_t align_corners) {
        auto output_lay = output->get_layout();
        size_t b = output_lay.batch();
        size_t f = output_lay.feature();
        size_t x = output_lay.spatial(0);
        size_t y = output_lay.spatial(1);
        auto input_lay = input->get_layout();
        size_t in_x = input_lay.spatial(0);
        size_t in_y = input_lay.spatial(1);
        float x_ratio = x > align_corners ? static_cast<float>(in_x - align_corners) / static_cast<float>(x - align_corners) : 0.f;
        float y_ratio = y > align_corners ? static_cast<float>(in_y - align_corners) / static_cast<float>(y - align_corners) : 0.f;

        cldnn::mem_lock<InT> in_ptr(input, get_test_stream());
        cldnn::mem_lock<OutT> out_ptr(output, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto low_in_xi = static_cast<size_t>(floor(x_ratio * xi));
                        auto low_in_yi = static_cast<size_t>(floor(y_ratio * yi));
                        auto high_in_xi = static_cast<size_t>(ceil(x_ratio * xi));
                        auto high_in_yi = static_cast<size_t>(ceil(y_ratio * yi));

                        high_in_xi = std::min(high_in_xi, static_cast<size_t>(in_x - 1));
                        high_in_yi = std::min(high_in_yi, static_cast<size_t>(in_y - 1));

                        auto dx = x_ratio * xi - static_cast<float>(low_in_xi);
                        auto dy = y_ratio * yi - static_cast<float>(low_in_yi);

                        auto top_left_coords = tensor(batch(bi), feature(fi), spatial(low_in_xi, low_in_yi, 0, 0));
                        auto top_right_coords = tensor(batch(bi), feature(fi), spatial(high_in_xi, low_in_yi, 0, 0));
                        auto bottom_left_coords = tensor(batch(bi), feature(fi), spatial(low_in_xi, high_in_yi, 0, 0));
                        auto bottom_right_coords = tensor(batch(bi), feature(fi), spatial(high_in_xi, high_in_yi, 0, 0));

                        auto top_left_val = in_ptr[input_lay.get_linear_offset(top_left_coords)];
                        auto top_right_val = in_ptr[input_lay.get_linear_offset(top_right_coords)];
                        auto bottom_left_val = in_ptr[input_lay.get_linear_offset(bottom_left_coords)];
                        auto bottom_right_val = in_ptr[input_lay.get_linear_offset(bottom_right_coords)];

                        auto top_val = static_cast<float>(top_left_val)
                            + (static_cast<float>(top_right_val) - static_cast<float>(top_left_val)) * dx;
                        auto bottom_val = static_cast<float>(bottom_left_val)
                            + (static_cast<float>(bottom_right_val) - static_cast<float>(bottom_left_val)) * dx;

                        auto final_val = top_val + (bottom_val - top_val) * dy;

                        auto output_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto output_val = out_ptr[output_lay.get_linear_offset(output_coords)];

                        ASSERT_NEAR(static_cast<float>(output_val), final_val, 1.e-1f)
                            << " at bi=" << bi << ", fi=" << fi << ", xi=" << xi << ", yi=" << yi;
                    }
                }
            }
        }
    }

    void compare(const memory::ptr input, const memory::ptr output, resample::InterpolateOp::InterpolateMode operation, uint32_t align_corners) {
        auto dt = input->get_layout().data_type;
        if (operation == resample::InterpolateOp::InterpolateMode::NEAREST) {
            // Nearest resampling implicitly ignores align_corners
            if (dt == data_types::f32) {
                compare_nearest_typed<float>(input, output, 0);
            } else if (dt == data_types::f16) {
                compare_nearest_typed<ov::float16>(input, output, 0);
            } else if (dt == data_types::i8) {
                compare_nearest_typed<int8_t>(input, output, 0);
            } else if (dt == data_types::u8) {
                compare_nearest_typed<uint8_t>(input, output, 0);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(dt);
            }
        } else {
            FAIL() << "Not supported resample_type: " << static_cast<int32_t>(operation);
        }
    }

    void execute(const resample_random_test_params& params, bool is_caching_test) {
        auto& engine = get_test_engine();

        auto in_layout = layout(params.input_type, params.in_format, params.input_size);

        cldnn::topology topo;
        topo.add(input_layout("in", in_layout));
        auto prim = resample("resample", input_info("in"), params.output_size, params.num_filter, params.operation_type);
        topo.add(prim);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"resample", {params.out_format, ""}} }));
        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        auto in_mem = engine.allocate_memory(in_layout);
        fill_random(in_mem);
        net->set_input_data("in", in_mem);

        auto result = net->execute();
        auto output = result.at("resample").get_memory();

        std::string kernel = "";
        if (!is_caching_test) {
            for (auto& info : net->get_primitives_info()) {
                if (info.original_id == "resample")
                    kernel = info.kernel_id;
            }
        }
    }
};

TEST_P(resample_random_test, random) {
    execute(GetParam(), false);
}

struct resample_random_test_param_generator : std::vector<resample_random_test_params> {
    resample_random_test_param_generator& add(resample_random_test_params params) {
        push_back(params);
        return *this;
    }

    resample_random_test_param_generator& smoke_params(data_types type, format::type input_format, format::type output_format) {
        push_back(resample_random_test_params{ type, {1, 17, 5, 9}, {1, 17, 15, 18}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, input_format, output_format });
        push_back(resample_random_test_params{ type, {2, 17, 5, 9}, {2, 17, 15, 18}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, input_format, output_format });
        push_back(resample_random_test_params{ type, {1, 7, 10, 17}, {1, 7, 21, 35}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, input_format, output_format });
        push_back(resample_random_test_params{ type, {2, 7, 10, 17}, {2, 7, 21, 35}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, input_format, output_format });
        return *this;
    }

};

INSTANTIATE_TEST_SUITE_P(smoke_resample,
                        resample_random_test,
                        testing::ValuesIn(
                            resample_random_test_param_generator()
                            .smoke_params(data_types::i8, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .smoke_params(data_types::u8, format::b_fs_yx_fsv4, format::b_fs_yx_fsv4)
                            .smoke_params(data_types::i8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::u8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)

                            .smoke_params(data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::i8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::u8, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                        ));


/////////////////////////////////////////////////////////////////////////

struct caffe_resample_random_test_params {
    data_types input_type;
    tensor input_size;
    tensor output_size;
    uint32_t num_filter;
    resample::InterpolateOp::InterpolateMode operation_type;
    uint32_t align_corners;
    format::type in_format;
    format::type out_format;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
};

struct caffe_resample_random_test : testing::TestWithParam<caffe_resample_random_test_params>
{
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);

        auto data = rg.generate_random_4d<T>(b, f, y, x, min, max, k);
        cldnn::mem_lock<T> ptr(mem, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<ov::float16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    template <typename T>
    void compare_outputs(const memory::ptr out_ref, const memory::ptr out_opt) {
        auto output_lay = out_ref->get_layout();
        auto opt_output_lay = out_opt->get_layout();

        size_t b = output_lay.batch();
        size_t f = output_lay.feature();
        size_t x = output_lay.spatial(0);
        size_t y = output_lay.spatial(1);
        cldnn::mem_lock<T> ref_ptr(out_ref, get_test_stream());
        cldnn::mem_lock<T> opt_ptr(out_opt, get_test_stream());
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto ref_out_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto ref_out_offset = output_lay.get_linear_offset(ref_out_coords);
                        auto ref_out_val = ref_ptr[ref_out_offset];

                        auto opt_out_offset = opt_output_lay.get_linear_offset(ref_out_coords);
                        auto opt_out_val = opt_ptr[opt_out_offset];

                        ASSERT_EQ(ref_out_offset, opt_out_offset);
                        ASSERT_EQ(opt_out_val, ref_out_val);
                    }
                }
            }
        }
    }

    void execute_compare(const caffe_resample_random_test_params& params, bool check_result, bool is_caching_test) {
        auto& engine = get_test_engine();

        auto in_layout = layout(params.input_type, params.in_format, params.input_size);
        auto in_mem = engine.allocate_memory(in_layout);
        fill_random(in_mem);

        cldnn::topology topo;
        topo.add(input_layout("in", in_layout));
        auto prim = resample("resample", input_info("in"), params.output_size, params.num_filter, params.operation_type);
        prim.pads_begin = params.pads_begin;
        prim.pads_end = params.pads_end;
        topo.add(prim);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"resample"}));
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"resample", {params.in_format, "resample_ref"}} }));

        cldnn::network net(engine, topo, config);
        net.set_input_data("in", in_mem);

        auto result = net.execute();
        auto output = result.at("resample").get_memory();

        // Execute resample_opt
        cldnn::topology topo_opt;
        topo_opt.add(input_layout("in", in_layout));
        auto prim_opt = resample("resample_opt", input_info("in"), params.output_size, params.num_filter, params.operation_type);
        prim_opt.pads_begin = params.pads_begin;
        prim_opt.pads_end = params.pads_end;
        topo_opt.add(prim_opt);

        ExecutionConfig config_opt = get_test_default_config(engine);
        config_opt.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"resample_opt"}));
        config_opt.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"resample_opt", {params.in_format, "resample_opt"}} }));

        cldnn::network::ptr net_opt = get_network(engine, topo_opt, config_opt, get_test_stream_ptr(), is_caching_test);

        // Use in_mem from ref network
        net_opt->set_input_data("in", in_mem);

        auto result_opt = net_opt->execute();
        auto output_opt = result_opt.at("resample_opt").get_memory();

        if (check_result == true) {
            // Check data_types
            if (params.input_type == data_types::f32) {
                compare_outputs<float>(output, output_opt);
            } else if (params.input_type == data_types::f16) {
                compare_outputs<ov::float16>(output, output_opt);
            } else if (params.input_type == data_types::i8) {
                compare_outputs<int8_t>(output, output_opt);
            } else if (params.input_type == data_types::u8) {
                compare_outputs<uint8_t>(output, output_opt);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(params.input_type);
            }
        }
    }
};

struct caffe_resample_random_test_param_generator : std::vector<caffe_resample_random_test_params> {
    caffe_resample_random_test_param_generator& add(caffe_resample_random_test_params params) {
        push_back(params);
        return *this;
    }

    caffe_resample_random_test_param_generator& smoke_params(data_types type, format::type input_format, format::type output_format) {
        push_back(caffe_resample_random_test_params{ type, {1, 512, 16, 16}, {1, 512, 32, 32}, 1, resample::InterpolateOp::InterpolateMode::LINEAR, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 512, 32, 32}, {1, 512, 16, 16}, 1, resample::InterpolateOp::InterpolateMode::LINEAR, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 24, 32, 32}, {1, 24, 64, 64}, 1,   resample::InterpolateOp::InterpolateMode::LINEAR, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 24, 96, 96}, {1, 24, 32, 32}, 1,   resample::InterpolateOp::InterpolateMode::LINEAR, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 8, 64, 64},  {1, 8, 32, 32},  1,   resample::InterpolateOp::InterpolateMode::LINEAR, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 20, 10, 10}, {1, 20, 20, 20}, 1,   resample::InterpolateOp::InterpolateMode::LINEAR, 1, input_format, output_format, {}, {}});
        push_back(caffe_resample_random_test_params{ type, {1, 20, 20, 20}, {1, 20, 10, 10}, 1,   resample::InterpolateOp::InterpolateMode::LINEAR, 1, input_format, output_format, {}, {}});
        // Padding applied
        push_back(caffe_resample_random_test_params{ type, {1, 96, 16, 16}, {1, 96, 32, 32}, 1, resample::InterpolateOp::InterpolateMode::LINEAR, 1, input_format, output_format, {0, 0, 1, 1}, {0, 0, 1, 1}});
        push_back(caffe_resample_random_test_params{ type, {1, 96, 32, 32}, {1, 96, 16, 16}, 1, resample::InterpolateOp::InterpolateMode::LINEAR, 1, input_format, output_format, {0, 0, 1, 1}, {0, 0, 1, 1}});
        return *this;
    }
};

TEST_P(caffe_resample_random_test, random) {
    auto param = GetParam();
    execute_compare(param, true, false);
}

INSTANTIATE_TEST_SUITE_P(caffe_smoke_caffe_fsv16,
                        caffe_resample_random_test,
                        testing::ValuesIn(
                            caffe_resample_random_test_param_generator()
                            .smoke_params(data_types::f32, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                            .smoke_params(data_types::f16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16)
                        ));

INSTANTIATE_TEST_SUITE_P(caffe_smoke_caffe_fsv32,
                        caffe_resample_random_test,
                        testing::ValuesIn(
                            caffe_resample_random_test_param_generator()
                            .smoke_params(data_types::f16, format::fs_b_yx_fsv32, format::fs_b_yx_fsv32)
                        ));

TEST(resample_gpu, interpolate_in2x2x3x2_nearest1) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    std::vector<int64_t> output_pattern {b, f, y*2, x*2};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
    auto nm = resample::InterpolateOp::NearestMode::CEIL;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  1.f,  1.f,  1.f,
         2.f,  3.f,  3.f,  3.f,
         2.f,  3.f,  3.f,  3.f,
         4.f,  5.f,  5.f,  5.f,
         4.f,  5.f,  5.f,  5.f,
         4.f,  5.f,  5.f,  5.f,

         6.f,  7.f,  7.f,  7.f,
         8.f,  9.f,  9.f,  9.f,
         8.f,  9.f,  9.f,  9.f,
        10.f, 11.f, 11.f, 11.f,
        10.f, 11.f, 11.f, 11.f,
        10.f, 11.f, 11.f, 11.f,

        12.f, 13.f, 13.f, 13.f,
        14.f, 15.f, 15.f, 15.f,
        14.f, 15.f, 15.f, 15.f,
        16.f, 17.f, 17.f, 17.f,
        16.f, 17.f, 17.f, 17.f,
        16.f, 17.f, 17.f, 17.f,

        18.f, 19.f, 19.f, 19.f,
        20.f, 21.f, 21.f, 21.f,
        20.f, 21.f, 21.f, 21.f,
        22.f, 23.f, 23.f, 23.f,
        22.f, 23.f, 23.f, 23.f,
        22.f, 23.f, 23.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    ASSERT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_nearest2) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    std::vector<int64_t> output_pattern {b, f, y*2, x*2};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
    auto nm = resample::InterpolateOp::NearestMode::ROUND_PREFER_FLOOR;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  0.f,  1.f,  1.f,
         0.f,  0.f,  1.f,  1.f,
         2.f,  2.f,  3.f,  3.f,
         2.f,  2.f,  3.f,  3.f,
         4.f,  4.f,  5.f,  5.f,
         4.f,  4.f,  5.f,  5.f,

         6.f,  6.f,  7.f,  7.f,
         6.f,  6.f,  7.f,  7.f,
         8.f,  8.f,  9.f,  9.f,
         8.f,  8.f,  9.f,  9.f,
        10.f, 10.f, 11.f, 11.f,
        10.f, 10.f, 11.f, 11.f,

        12.f, 12.f, 13.f, 13.f,
        12.f, 12.f, 13.f, 13.f,
        14.f, 14.f, 15.f, 15.f,
        14.f, 14.f, 15.f, 15.f,
        16.f, 16.f, 17.f, 17.f,
        16.f, 16.f, 17.f, 17.f,

        18.f, 18.f, 19.f, 19.f,
        18.f, 18.f, 19.f, 19.f,
        20.f, 20.f, 21.f, 21.f,
        20.f, 20.f, 21.f, 21.f,
        22.f, 22.f, 23.f, 23.f,
        22.f, 22.f, 23.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    ASSERT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_nearest3) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    std::vector<int64_t> output_pattern {b, f, y*2, x*2};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
    auto nm = resample::InterpolateOp::NearestMode::ROUND_PREFER_CEIL;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  0.f,  1.f,  1.f,
         0.f,  0.f,  1.f,  1.f,
         2.f,  2.f,  3.f,  3.f,
         2.f,  2.f,  3.f,  3.f,
         4.f,  4.f,  5.f,  5.f,
         4.f,  4.f,  5.f,  5.f,

         6.f,  6.f,  7.f,  7.f,
         6.f,  6.f,  7.f,  7.f,
         8.f,  8.f,  9.f,  9.f,
         8.f,  8.f,  9.f,  9.f,
        10.f, 10.f, 11.f, 11.f,
        10.f, 10.f, 11.f, 11.f,

        12.f, 12.f, 13.f, 13.f,
        12.f, 12.f, 13.f, 13.f,
        14.f, 14.f, 15.f, 15.f,
        14.f, 14.f, 15.f, 15.f,
        16.f, 16.f, 17.f, 17.f,
        16.f, 16.f, 17.f, 17.f,

        18.f, 18.f, 19.f, 19.f,
        18.f, 18.f, 19.f, 19.f,
        20.f, 20.f, 21.f, 21.f,
        20.f, 20.f, 21.f, 21.f,
        22.f, 22.f, 23.f, 23.f,
        22.f, 22.f, 23.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    ASSERT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_nearest4) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    std::vector<int64_t> output_pattern {b, f, y*2, x*2};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
    auto nm = resample::InterpolateOp::NearestMode::FLOOR;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  0.f,  0.f,  1.f,
         0.f,  0.f,  0.f,  1.f,
         0.f,  0.f,  0.f,  1.f,
         2.f,  2.f,  2.f,  3.f,
         2.f,  2.f,  2.f,  3.f,
         4.f,  4.f,  4.f,  5.f,

         6.f,  6.f,  6.f,  7.f,
         6.f,  6.f,  6.f,  7.f,
         6.f,  6.f,  6.f,  7.f,
         8.f,  8.f,  8.f,  9.f,
         8.f,  8.f,  8.f,  9.f,
        10.f, 10.f, 10.f, 11.f,

        12.f, 12.f, 12.f, 13.f,
        12.f, 12.f, 12.f, 13.f,
        12.f, 12.f, 12.f, 13.f,
        14.f, 14.f, 14.f, 15.f,
        14.f, 14.f, 14.f, 15.f,
        16.f, 16.f, 16.f, 17.f,

        18.f, 18.f, 18.f, 19.f,
        18.f, 18.f, 18.f, 19.f,
        18.f, 18.f, 18.f, 19.f,
        20.f, 20.f, 20.f, 21.f,
        20.f, 20.f, 20.f, 21.f,
        22.f, 22.f, 22.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    ASSERT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_nearest5) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    std::vector<int64_t> output_pattern {b, f, y*2, x*2};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
    auto nm = resample::InterpolateOp::NearestMode::SIMPLE;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    float answers[96] = {
         0.f,  0.f,  0.f,  1.f,
         0.f,  0.f,  0.f,  1.f,
         0.f,  0.f,  0.f,  1.f,
         2.f,  2.f,  2.f,  3.f,
         2.f,  2.f,  2.f,  3.f,
         4.f,  4.f,  4.f,  5.f,

         6.f,  6.f,  6.f,  7.f,
         6.f,  6.f,  6.f,  7.f,
         6.f,  6.f,  6.f,  7.f,
         8.f,  8.f,  8.f,  9.f,
         8.f,  8.f,  8.f,  9.f,
        10.f, 10.f, 10.f, 11.f,

        12.f, 12.f, 12.f, 13.f,
        12.f, 12.f, 12.f, 13.f,
        12.f, 12.f, 12.f, 13.f,
        14.f, 14.f, 14.f, 15.f,
        14.f, 14.f, 14.f, 15.f,
        16.f, 16.f, 16.f, 17.f,

        18.f, 18.f, 18.f, 19.f,
        18.f, 18.f, 18.f, 19.f,
        18.f, 18.f, 18.f, 19.f,
        20.f, 20.f, 20.f, 21.f,
        20.f, 20.f, 20.f, 21.f,
        22.f, 22.f, 22.f, 23.f,
    };

    for (int i = 0; i < 2; ++i) { // B
        for (int j = 0; j < 2; ++j) { // F
            for (int k = 0; k < 4; ++k) { // Y
                for (int l = 0; l < 6; ++l) { // X
                    auto linear_id = l + k * 6 + j * 4 * 6 + i * 2 * 4 * 6;
                    ASSERT_TRUE(are_equal(answers[linear_id], output_ptr[linear_id])) << linear_id;
                }
            }
        }
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode1) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    std::vector<int64_t> output_pattern {b, f, y, x};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
    auto nm = resample::InterpolateOp::NearestMode::ROUND_PREFER_FLOOR;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.f,  0.f,  1.f,
         4.f,  4.f,  5.f,

         6.f,  6.f,  7.f,
        10.f, 10.f, 11.f,

        12.f, 12.f, 13.f,
        16.f, 16.f, 17.f,

        18.f, 18.f, 19.f,
        22.f, 22.f, 23.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode2) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 1;
    x = 3;
    std::vector<int64_t> output_pattern {b, f, y, x};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    auto nm = resample::InterpolateOp::NearestMode::ROUND_PREFER_FLOOR;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.f,  0.f,  1.f,
         6.f,  6.f,  7.f,

        12.f, 12.f, 13.f,
        18.f, 18.f, 19.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode3) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    std::vector<int64_t> output_pattern {b, f, y, x};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::ASYMMETRIC;
    auto nm = resample::InterpolateOp::NearestMode::ROUND_PREFER_FLOOR;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.f,  1.f,  1.f,
         2.f,  3.f,  3.f,

         6.f,  7.f,  7.f,
         8.f,  9.f,  9.f,

        12.f, 13.f, 13.f,
        14.f, 15.f, 15.f,

        18.f, 19.f, 19.f,
        20.f, 21.f, 21.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode4) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    std::vector<int64_t> output_pattern {b, f, y, x};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN;
    auto nm = resample::InterpolateOp::NearestMode::ROUND_PREFER_FLOOR;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         2.f,  3.f,  3.f,
         4.f,  5.f,  5.f,

         8.f,  9.f,  9.f,
        10.f, 11.f, 11.f,

        14.f, 15.f, 15.f,
        16.f, 17.f, 17.f,

        20.f, 21.f, 21.f,
        22.f, 23.f, 23.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_coord_transform_mode5) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    std::vector<int64_t> output_pattern {b, f, y, x};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::NEAREST;
    auto ctm = resample::InterpolateOp::CoordinateTransformMode::ALIGN_CORNERS;
    auto nm = resample::InterpolateOp::NearestMode::ROUND_PREFER_FLOOR;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm, nm));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.f,  0.f,  1.f,
         4.f,  4.f,  5.f,

         6.f,  6.f,  7.f,
        10.f, 10.f, 11.f,

        12.f, 12.f, 13.f,
        16.f, 16.f, 17.f,

        18.f, 18.f, 19.f,
        22.f, 22.f, 23.f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_cubic) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    std::vector<int64_t> output_pattern {b, f, y, x};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::CUBIC;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.29600694f,  0.8828125f,  1.46961806f,
         3.53038194f,  4.1171875f,  4.70399306f,

         6.29600694f,  6.8828125f,  7.46961806f,
         9.53038194f, 10.1171875f, 10.70399306f,

        12.29600694f, 12.8828125f, 13.46961806f,
        15.53038194f, 16.1171875f, 16.70399306f,

        18.29600694f, 18.8828125f, 19.46961806f,
        21.53038194f, 22.1171875f, 22.70399306f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_cubic2) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 1;
    int f = 1;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    x = 3;
    std::vector<int64_t> output_pattern {b, f, y, x};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::CUBIC;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode));

    set_values(input, {
        5.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
          5.34722222f,  3.f, 0.65277778f,
          1.91319444f, 2.5f, 3.08680556f,
          3.91319444f, 4.5f, 5.08680556f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, interpolate_in2x2x3x2_linear) {
    //  Input  : 2x2x3x2
    //  Output : 2x2x6x4
    //  Sample Type: Nearest

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 2;
    int f = 2;
    int y = 3;
    int x = 2;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 2;
    x = 3;
    std::vector<int64_t> output_pattern {b, f, y, x};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::LINEAR;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SIZES;
    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{}, {}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode));

    set_values(input, {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f,
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        21.f, 22.f, 23.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0.5f,  1.f,  1.5f,
         3.5f,  4.f,  4.5f,

         6.5f,  7.f,  7.5f,
         9.5f, 10.f, 10.5f,

        12.5f, 13.f, 13.5f,
        15.5f, 16.f, 16.5f,

        18.5f, 19.f, 19.5f,
        21.5f, 22.f, 22.5f,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

static tensor create_tensor(const std::vector<int64_t>& shape) {
    switch (shape.size()) {
    case 4:
        return tensor{batch(shape[0]), feature(shape[1]), spatial(shape[2], shape[3])};
        break;
    case 5:
        return tensor{batch(shape[0]), feature(shape[1]), spatial(shape[4], shape[3], shape[2])};
        break;
    default:
        throw std::runtime_error("Only 4d or 5d formats are supported");
    }
}

template <cldnn::format::type FMT>
struct format_wrapper {
    static constexpr cldnn::format::type fmt = FMT;
};

template <typename T>
struct onnx_5d_format : public ::testing::Test {
    onnx_5d_format() : shapes_and_attrs {// resize_downsample_scales_linear
                            {{1, 1, 3, 2, 4},
                             {2, 3, 4},
                             {1, 1, 2, 1, 2},
                             {0.8f, 0.6f, 0.6f},
                             resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL,
                             resample::InterpolateOp::ShapeCalcMode::SCALES},
                            // resize_downsample_scales_linear_align_corners
                            {{1, 1, 3, 2, 4},
                             {2, 3, 4},
                             {1, 1, 2, 1, 2},
                             {0.8f, 0.6f, 0.6f},
                             resample::InterpolateOp::CoordinateTransformMode::ALIGN_CORNERS,
                             resample::InterpolateOp::ShapeCalcMode::SCALES},
                            // resize_upsample_scales_linear
                            {{1, 1, 2, 2, 2},
                             {2, 3, 4},
                             {1, 1, 4, 4, 4},
                             {2.0, 2.0, 2.0},
                             resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL,
                             resample::InterpolateOp::ShapeCalcMode::SCALES},
                            // resize_upsample_scales_linear_align_corners
                            {{1, 1, 2, 2, 2},
                             {2, 3, 4},
                             {1, 1, 4, 4, 4},
                             {2.0, 2.0, 2.0},
                             resample::InterpolateOp::CoordinateTransformMode::ALIGN_CORNERS,
                             resample::InterpolateOp::ShapeCalcMode::SCALES},
                            // resize_downsample_sizes_linear_pytorch_half_pixel
                            {{1, 1, 2, 4, 4},
                             {2, 3, 4},
                             {1, 1, 1, 3, 1},
                             {0.5, 0.75, 0.25},
                             resample::InterpolateOp::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
                             resample::InterpolateOp::ShapeCalcMode::SIZES}
        }
        , input_data_list {
            // resize_downsample_scales_linear
            {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
             13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
            // resize_downsample_scales_linear_align_corners
            {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
             13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
            // resize_upsample_scales_linear
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            // resize_upsample_scales_linear_align_corners
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            // resize_downsample_sizes_linear_pytorch_half_pixel
            {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
             12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f,
             23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f}}
        , expected_results {
            // resize_downsample_scales_linear
            {3.6666665, 5.333333, 13.666666, 15.333333},
            // resize_downsample_scales_linear_align_corners
            {1.0, 4.0, 17.0, 20.0},
            // resize_upsample_scales_linear
            {1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0,
             2.0, 2.25, 2.75, 3.0, 2.5, 2.75, 3.25, 3.5, 3.5, 3.75, 4.25, 4.5, 4.0, 4.25, 4.75, 5.0,
             4.0, 4.25, 4.75, 5.0, 4.5, 4.75, 5.25, 5.5, 5.5, 5.75, 6.25, 6.5, 6.0, 6.25, 6.75, 7.0,
             5.0, 5.25, 5.75, 6.0, 5.5, 5.75, 6.25, 6.5, 6.5, 6.75, 7.25, 7.5, 7.0, 7.25, 7.75, 8.0},
            // resize_upsample_scales_linear_align_corners
            {1.0,       1.3333333, 1.6666667, 2.0,       1.6666666, 2.0,       2.3333335, 2.6666667, 2.3333333, 2.6666665,
             3.0,       3.3333335, 3.0,       3.3333333, 3.6666665, 4.0,       2.3333335, 2.6666665, 3.0,       3.3333333,
             3.0,       3.333333,  3.6666665, 3.9999995, 3.6666665, 4.0,       4.3333335, 4.6666665, 4.333333,  4.6666665,
             4.9999995, 5.333333,  3.6666667, 4.0,       4.3333335, 4.6666665, 4.3333335, 4.6666665, 5.0,       5.333333,
             5.0,       5.3333335, 5.666667,  6.0,       5.666667,  5.9999995, 6.333333,  6.666667,  5.0,       5.333333,
             5.6666665, 6.0,       5.666667,  5.9999995, 6.333333,  6.666666,  6.3333335, 6.666666,  7.0,       7.3333335,
             7.0,       7.333333,  7.6666675, 8.0},
            // resize_downsample_sizes_linear_pytorch_half_pixel
            {1.6666667, 7.0, 12.333333}}
      , fmt{T::fmt}
    {}

    struct ShapesAndAttrs {
        std::vector<int64_t> input_data_shape;
        std::vector<int64_t> axes;
        std::vector<int64_t> out_shape;
        std::vector<float> scales_data;
        resample::InterpolateOp::CoordinateTransformMode transform_mode;
        resample::InterpolateOp::ShapeCalcMode calculation_mode;
    };
    std::vector<ShapesAndAttrs> shapes_and_attrs;
    std::vector<std::vector<float>> input_data_list;
    std::vector<std::vector<float>> expected_results;
    format fmt;
};

using cldnn_5d_formats = testing::Types<format_wrapper<format::bfzyx>,
                                        format_wrapper<format::bs_fs_zyx_bsv16_fsv32>,
                                        format_wrapper<format::bs_fs_zyx_bsv16_fsv16>,
                                        format_wrapper<format::bs_fs_zyx_bsv32_fsv32>,
                                        format_wrapper<format::bs_fs_zyx_bsv32_fsv16>>;
TYPED_TEST_SUITE(onnx_5d_format,  cldnn_5d_formats);

TYPED_TEST(onnx_5d_format, interpolate_linear_onnx5d)
{
    auto& engine = get_test_engine();

    std::size_t i = 0;
    for (const auto& s : this->shapes_and_attrs) {
        tensor input_tensor = create_tensor(s.input_data_shape);
        auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, input_tensor });;
        //auto output_tensor = create_tensor(s.out_shape);

        topology topology;

        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("input_reordered", input_info("input"), this->fmt, data_types::f32));
        int32_t antialias = 0;
        float cube_coeff = -0.75f;

        resample::InterpolateOp::InterpolateMode mode = resample::InterpolateOp::InterpolateMode::LINEAR_ONNX;
        resample::InterpolateOp::CoordinateTransformMode ctm = s.transform_mode;
        resample::InterpolateOp::ShapeCalcMode shapeCalcMode = s.calculation_mode;
        topology.add(resample("interpolate", input_info("input_reordered"), s.out_shape, s.scales_data, s.axes,
                   {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode, ctm));

        topology.add(reorder("output", input_info("interpolate"), format::bfzyx, data_types::f32));

        set_values(input, this->input_data_list[i]);

        cldnn::network net {engine, topology };
        net.set_input_data("input", input);
        auto outputs = net.execute();

        auto output = outputs.at("output").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(this->expected_results[i].size(), output_ptr.size());
        for (size_t j = 0; j < this->expected_results[i].size(); ++j) {
            //ASSERT_TRUE(are_equal(expected_results[i][j], output_ptr[i])) << i;
            ASSERT_NEAR(this->expected_results[i][j], output_ptr[j], 0.001) << j;
        }

        ++i;
    }
}

TEST(resample_gpu, interpolate_in1x1x2x4_linear_scale) {
    //  Input  : 1x1x2x4
    //  Output : 1x1x1x2
    //  Sample Type: Linear

    auto& engine = get_test_engine();
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    int b = 1;
    int f = 1;
    int y = 2;
    int x = 4;
    tensor shape = tensor{batch(b), feature(f), spatial(x, y)};
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, shape });

    y = 1;
    x = 2;
    std::vector<int64_t> output_pattern {b, f, y, x};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    auto mode = resample::InterpolateOp::InterpolateMode::LINEAR;
    auto shapeCalcMode = resample::InterpolateOp::ShapeCalcMode::SCALES;

    topology.add(resample("interpolate", input_info("input"), output_pattern, std::vector<float>{0.6f, 0.6f}, {2, 3}, {0, 0, 0, 0}, {0, 0, 0, 0}, antialias, cube_coeff, mode, shapeCalcMode));

    set_values(input, {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
    });

    cldnn::network net{ engine, topology, config };

    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("interpolate").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         2.6666665f,  4.3333331f
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(resample_gpu, downsampling_u8) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ { 1, 1, 9, 16 }, data_types::u8, format::bfyx });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(resample("resample", input_info("input"), {1, 1, 3, 5}, {}, {0, 1, 2, 3}));

    set_values<uint8_t>(input, {
        124, 126, 128, 130, 131, 131, 129, 128, 119, 119, 119, 119, 119, 119, 119, 119,
        125, 124, 120, 118, 115, 111, 109, 106, 103, 109, 95, 120, 155, 150, 142, 139,
        129, 128, 122, 110, 98, 85, 81, 78, 79, 78, 76, 75, 74, 75, 76, 77,
        78, 77, 77, 76, 76, 77, 77, 78, 80, 77, 75, 75, 78, 81, 83, 83,
        79, 79, 78, 78, 78, 77, 75, 75, 70, 71, 73, 74, 76, 76, 75, 74,
        74, 72, 71, 70, 71, 72, 75, 76, 74, 73, 74, 72, 73, 73, 72, 72,
        71, 71, 71, 72, 72, 72, 73, 73, 73, 71, 69, 71, 75, 78, 78, 75,
        72, 72, 75, 77, 79, 79, 79, 78, 76, 75, 74, 72, 72, 73, 74, 75,
        74, 74, 75, 75, 75, 75, 75, 75, 77, 74, 70, 67, 67, 69, 73, 76
    });

    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network net{ engine, topology, config };
    net.set_input_data("input", input);

    auto outputs = net.execute();

    auto output = outputs.at("resample").get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output, get_test_stream());

    std::vector<uint8_t> ref_out = { 124, 114, 104, 112, 143, 79, 78, 72, 74, 75, 72, 79, 77, 73, 74 };

    for (size_t i = 0; i < ref_out.size(); ++i) {
        ASSERT_EQ(ref_out[i], output_ptr[i]);
    }
}

struct resample_opt_random_test_params {
    data_types input_type;
    tensor input_size;
    tensor output_size;
    uint32_t num_filter;
    resample::InterpolateOp::InterpolateMode operation_type;
    uint32_t align_corners;
    format::type in_format;
    format::type out_format;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
};

struct resample_opt_random_test : testing::TestWithParam<resample_opt_random_test_params>
{
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);
        size_t z = l.spatial(2);

        auto data = rg.generate_random_5d<T>(b, f, z, y, x, min, max, k);
        mem_lock<T> ptr{mem, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t zi = 0; zi < z; ++zi) {
                    for (size_t yi = 0; yi < y; ++yi) {
                        for (size_t xi = 0; xi < x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, 0));
                            auto offset = mem->get_layout().get_linear_offset(coords);
                            ptr[offset] = data[bi][fi][zi][yi][xi];
                        }
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<ov::float16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    template <typename T>
    void compare_outputs(const memory::ptr out_ref, const memory::ptr out_opt) {
        auto output_lay = out_ref->get_layout();
        auto opt_output_lay = out_opt->get_layout();

        size_t b = output_lay.batch();
        size_t f = output_lay.feature();
        size_t x = output_lay.spatial(0);
        size_t y = output_lay.spatial(1);
        size_t z = output_lay.spatial(2);
        mem_lock<T> ref_ptr{out_ref, get_test_stream()};
        mem_lock<T> opt_ptr{out_opt, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t zi = 0; zi < z; ++zi) {
                    for (size_t yi = 0; yi < y; ++yi) {
                        for (size_t xi = 0; xi < x; ++xi) {
                            auto ref_out_coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, 0));
                            auto ref_out_offset = output_lay.get_linear_offset(ref_out_coords);
                            auto ref_out_val = ref_ptr[ref_out_offset];
                            auto opt_out_offset = opt_output_lay.get_linear_offset(ref_out_coords);
                            auto opt_out_val = opt_ptr[opt_out_offset];
                            ASSERT_EQ(ref_out_offset, opt_out_offset);
                            if (std::is_same<T, ov::float16>::value) {
                                ASSERT_NEAR(static_cast<float>(opt_out_val), static_cast<float>(ref_out_val), 1.e-1f);
                            } else {
                                ASSERT_EQ(opt_out_val, ref_out_val);
                            }
                        }
                    }
                }
            }
        }
    }

    void execute_compare(const resample_opt_random_test_params& params, bool check_result,
                         bool is_caching_test, const std::string& kernel = "resample_opt") {
        auto& engine = get_test_engine();

        const format origin_format = format::dimension(params.in_format) == 4 ? format::bfyx : format::bfzyx;
        auto in_layout = layout(params.input_type, origin_format, params.input_size);
        auto in_mem = engine.allocate_memory(in_layout);
        fill_random(in_mem);

        /// bfyx or bfzyx
        cldnn::topology topo;
        topo.add(input_layout("in", in_layout));
        auto prim = resample("resample", input_info("in"), params.output_size, params.num_filter, params.operation_type);
        prim.pads_begin = params.pads_begin;
        prim.pads_end = params.pads_end;
        topo.add(prim);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"resample"}));

        network net(engine, topo, config);
        net.set_input_data("in", in_mem);

        // first execution of ref
        auto result = net.execute();
        auto output = result.at("resample").get_memory();

        cldnn::topology topo_opt;
        topo_opt.add(input_layout("in", in_layout));
        topo_opt.add(reorder("in_to_input_type", input_info("in"), params.in_format, params.input_type));
        auto prim_opt = resample("resample_opt", input_info("in_to_input_type"), params.output_size, params.num_filter, params.operation_type);
        prim_opt.pads_begin = params.pads_begin;
        prim_opt.pads_end = params.pads_end;
        topo_opt.add(prim_opt);
        topo_opt.add(reorder("res_to_bfyx", input_info("resample_opt"), origin_format, params.input_type));

        ExecutionConfig config_opt = get_test_default_config(engine);
        config_opt.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"resample_opt", "res_to_bfyx"}));

        cldnn::network::ptr net_opt = get_network(engine, topo_opt, config_opt, get_test_stream_ptr(), is_caching_test);

        // Use in_mem from ref network
        net_opt->set_input_data("in", in_mem);

        // first execution of opt
        auto result_opt = net_opt->execute();
        auto output_opt = result_opt.at("res_to_bfyx").get_memory();
        if (!format::is_simple_data_format(params.in_format)) {
            ASSERT_FALSE(format::is_simple_data_format(result_opt.at("resample_opt").get_memory()->get_layout().format));
        }
        if (check_result == true) {
            // Check data_types
            if (params.input_type == data_types::f32) {
                compare_outputs<float>(output, output_opt);
            } else if (params.input_type == data_types::f16) {
                compare_outputs<ov::float16>(output, output_opt);
            } else if (params.input_type == data_types::i8) {
                compare_outputs<int8_t>(output, output_opt);
            } else if (params.input_type == data_types::u8) {
                compare_outputs<uint8_t>(output, output_opt);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(params.input_type);
            }
        }
    }
};

struct resample_opt_random_test_ext : resample_opt_random_test
{
    void execute_perf_test(const resample_opt_random_test_params& params, const std::string& kernel, const bool do_planar = false) {
        auto& engine = get_test_engine();

        const format origin_format = format::dimension(params.in_format) == 4 ? format::bfyx : format::bfzyx;
        auto in_layout = layout(params.input_type, origin_format, params.input_size);
        auto in_mem = engine.allocate_memory(in_layout);
        fill_random(in_mem);

        format working_format = do_planar == true ? origin_format : format(params.in_format);

        cldnn::topology topo_opt;
        topo_opt.add(input_layout("in", in_layout));
        topo_opt.add(reorder("in_to_input_type", input_info("in"), working_format, params.input_type));
        auto prim_opt = resample("resample_opt", input_info("in_to_input_type"), params.output_size, params.num_filter, params.operation_type);
        prim_opt.pads_begin = params.pads_begin;
        prim_opt.pads_end = params.pads_end;
        topo_opt.add(prim_opt);
        topo_opt.add(reorder("res_to_bfyx", input_info("resample_opt"), origin_format, params.input_type));

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::enable_profiling(true));
        cfg.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"res_to_bfyx"}));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"resample_opt", {working_format, kernel}} }));

        network net_opt(engine, topo_opt, cfg);

        // Use in_mem from ref network
        net_opt.set_input_data("in", in_mem);

        // first execution of opt
        std::map<primitive_id, network_output> result_opt;
        auto r = 100;
        double exectime = 0.f;
        for (int i = 0; i < r; ++i) {
            result_opt = net_opt.execute();
            exectime += get_profiling_exectime(result_opt, "resample_opt");
        }
        exectime /= r;
        std::string frm_str = format(working_format).to_string();
        std::string input_type = ov::element::Type(params.input_type).get_type_name();
        std::string is_opt = (do_planar == true) ? " not optimazed " : " optimized ";
        std::string mode;
        switch (params.operation_type) {
        case resample::InterpolateOp::InterpolateMode::NEAREST:
            mode = "nearest";
            break;
        case resample::InterpolateOp::InterpolateMode::LINEAR_ONNX:
            mode = "onnx";
            break;
        default:
            mode = "unknown";
        }

        std::cout << "Exectued time " << "" << mode << " " << is_opt << " " << kernel << " " << " input(" << params.input_size.to_string()
                  << ") output(" << params.output_size.to_string() << ") "
                  << frm_str << " " << input_type << " " << exectime << std::endl;

        // Uncomment line below if you like to see the latencies of all operations from last iteration
        //print_profiling_all_exectimes(result_opt);
    }
};

struct resample_onnx_random_test : resample_opt_random_test {
    template <typename T>
    void compare(const memory::ptr out_ref, const memory::ptr out_opt) {
        auto ref_count = out_ref->count();
        auto opt_count = out_opt->count();

        ASSERT_EQ(ref_count, opt_count);

        mem_lock<T> ref_ptr{out_ref, get_test_stream()};
        mem_lock<T> opt_ptr{out_opt, get_test_stream()};
        for (size_t i = 0; i < ref_count; ++i) {
            ASSERT_NEAR(static_cast<float>(opt_ptr[i]), static_cast<float>(ref_ptr[i]), 1.e-1f);
        }
    }

    void execute_compare(const resample_opt_random_test_params& params, bool check_result) {
        auto& engine = get_test_engine();

        const format origin_format = format::dimension(params.in_format) == 4 ? format::bfyx : format::bfzyx;
        auto in_layout = layout(params.input_type, origin_format, params.input_size);
        auto in_mem = engine.allocate_memory(in_layout);
        fill_random(in_mem);

        /// bfyx or bfzyx
        cldnn::topology topo;
        topo.add(input_layout("in", in_layout));
        topo.add(reorder("in_to_input_type", input_info("in"), params.in_format, params.input_type));
        auto prim = resample("resample", input_info("in_to_input_type"), params.output_size, params.num_filter, params.operation_type);
        topo.add(prim);
        topo.add(reorder("res_to_bfyx", input_info("resample"), origin_format, params.input_type));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"resample", "res_to_bfyx"}));
        ov::intel_gpu::ImplementationDesc resample_impl = { params.in_format, "resample_ref", impl_types::ocl };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "resample", resample_impl } }));

        network net(engine, topo, config);
        net.set_input_data("in", in_mem);

        // first execution of ref
        auto result = net.execute();
        auto output = result.at("resample").get_memory();

        cldnn::topology topo_opt;
        topo_opt.add(input_layout("in", in_layout));
        topo_opt.add(reorder("in_to_input_type", input_info("in"), params.in_format, params.input_type));
        auto prim_opt = resample("resample_opt", input_info("in_to_input_type"), params.output_size, params.num_filter, params.operation_type);
        topo_opt.add(prim_opt);
        topo_opt.add(reorder("res_to_bfyx", input_info("resample_opt"), origin_format, params.input_type));

        ExecutionConfig config_opt = get_test_default_config(engine);
        config_opt.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"resample_opt", "res_to_bfyx"}));
        ov::intel_gpu::ImplementationDesc resample_opt_impl = { params.in_format, "resample_onnx", impl_types::ocl };
        config_opt.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "resample_opt", resample_opt_impl } }));

        cldnn::network::ptr net_opt = get_network(engine, topo_opt, config_opt, get_test_stream_ptr(), false);

        // Use in_mem from ref network
        net_opt->set_input_data("in", in_mem);

        // first execution of opt
        auto result_opt = net_opt->execute();
        auto output_opt = result_opt.at("resample_opt").get_memory();
        if (!format::is_simple_data_format(params.in_format)) {
            ASSERT_FALSE(format::is_simple_data_format(result_opt.at("resample_opt").get_memory()->get_layout().format));
        }

        // Compares not only outputs but also padding values for block format.
        if (check_result == true) {
            // Check data_types
            if (params.input_type == data_types::f32) {
                compare<float>(output, output_opt);
            } else if (params.input_type == data_types::f16) {
                compare<ov::float16>(output, output_opt);
            } else if (params.input_type == data_types::i8) {
                compare<int8_t>(output, output_opt);
            } else if (params.input_type == data_types::u8) {
                compare<uint8_t>(output, output_opt);
            } else {
                FAIL() << "Not supported data type: " << static_cast<size_t>(params.input_type);
            }
        }
    }
};

TEST_P(resample_opt_random_test, random) {
    auto param = GetParam();
    execute_compare(param, true, false);
}

TEST_P(resample_opt_random_test_ext, DISABLED_random) {
    auto param = GetParam();
//   Comparison tests (2 lines below) are disabled because they took too much time on big shapes
//    execute_compare(param, true, "resample_opt");
//    execute_compare(param, true, "resample_ref");
    execute_perf_test(param, "resample_opt");
    execute_perf_test(param, "resample_ref", false);
    execute_perf_test(param, "resample_ref", true);
}

TEST_P(resample_onnx_random_test, random) {
    auto param = GetParam();
    execute_compare(param, true);
}

INSTANTIATE_TEST_SUITE_P(resample_onnx_smoke_not_aligned,
                         resample_onnx_random_test,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                { data_types::f16, {1, 24, 13, 13},  {1, 24, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, {}, {}},
                                { data_types::f16, {1, 24, 13, 13},  {1, 24, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, {}, {}},
                                { data_types::f16, {1, 24, 13, 13},  {1, 24, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, {}, {}},
                                { data_types::f16, {1, 24, 13, 13},  {1, 24, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv16_fsv16, format::bs_fs_yx_bsv16_fsv16, {}, {}},
                                { data_types::f16, {1, 24, 13, 13},  {1, 24, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32, {}, {}},

                                { data_types::f16, {1,  9, 13, 13, 5}, { 1, 9, 26, 26, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::f32, {1,  9, 13, 13, 5}, { 1, 9, 26, 26, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::f16, {16, 9,  7,  7, 5}, {16, 9, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv16, format::bs_fs_zyx_bsv16_fsv16, {}, {}},
                                { data_types::f32, {16, 9,  7,  7, 5}, {16, 9, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv16, format::bs_fs_zyx_bsv16_fsv16, {}, {}},
                                { data_types::f16, {32, 9,  7,  7, 5}, {32, 9, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16, {}, {}},
                                { data_types::f32, {32, 9,  7,  7, 5}, {32, 9, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16, {}, {}},

                                { data_types::i8, {1,  9, 13, 13, 5}, {1,  9, 26, 26, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32, {}, {}},
                                { data_types::u8, {1,  9, 13, 13, 5}, {1,  9, 26, 26, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32, {}, {}},
                                { data_types::i8, {16, 9,  7,  7, 5}, {16, 9, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32, {}, {}},
                                { data_types::u8, {16, 9,  7,  7, 5}, {16, 9, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32, {}, {}},
                                { data_types::i8, {32, 9,  7,  7, 5}, {32, 9, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32, {}, {}},
                                { data_types::u8, {32, 9,  7,  7, 5}, {32, 9, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32, {}, {}},
                            }
                        ));

INSTANTIATE_TEST_SUITE_P(resample_opt_smoke_nearest,
                         resample_opt_random_test,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                { data_types::i8,  {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, {}, {}},
                                { data_types::i8,  {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, {}, {}},
                                { data_types::i8,  {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, {}, {}},
                                { data_types::i8,  {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, {}, {}},
                                { data_types::i8,  {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_yx_bsv16_fsv16, format::bs_fs_yx_bsv16_fsv16, {}, {}},

                                { data_types::u8,  {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, {}, {}},
                                { data_types::u8,  {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, {}, {}},
                                { data_types::u8,  {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, {}, {}},
                                { data_types::u8,  {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, {}, {}},

                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, {}, {}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, {}, {}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, {}, {}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, {}, {}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_yx_bsv16_fsv16, format::bs_fs_yx_bsv16_fsv16, {}, {}},
                            }
                        ));

INSTANTIATE_TEST_SUITE_P(resample_opt_smoke_linear_onnx_4d_padding,
                         resample_opt_random_test,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, {0, 0, 1, 1}, {0, 0, 1, 1}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, {0, 0, 0, 0}, {0, 0, 1, 1}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, {0, 0, 1, 1}, {0, 0, 0, 0}},
                            }
                        ));

INSTANTIATE_TEST_SUITE_P(resample_opt_smoke_linear_onnx_4d_simple,
                         resample_opt_random_test,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, {}, {}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, {}, {}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, {}, {}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv16_fsv16, format::bs_fs_yx_bsv16_fsv16, {}, {}},
                                { data_types::f16, {1, 128, 13, 13},  {1, 128, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32, {}, {}},
                                { data_types::f16, {2, 32, 14, 14},  {2, 32, 28, 28},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::fs_b_yx_fsv32, format::fs_b_yx_fsv32, {}, {}},
                            }
                        ));

INSTANTIATE_TEST_SUITE_P(resample_opt_smoke_5d_nearest,
                         resample_opt_random_test,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                { data_types::i8, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::i8, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32, {}, {}},
                                { data_types::i8, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_zyx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32, {}, {}},
                                { data_types::i8, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32, {}, {}},

                                { data_types::u8, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::u8, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32, {}, {}},
                                { data_types::u8, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_zyx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32, {}, {}},
                                { data_types::u8, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32, {}, {}},

                                { data_types::f16, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::f16, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32, {}, {}},
                                { data_types::f16, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_zyx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32, {}, {}},
                                { data_types::f16, {1, 16, 13, 13, 13}, {1, 16, 26, 26, 26}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32, {}, {}},
                            }
                        ));

INSTANTIATE_TEST_SUITE_P(resample_opt_smoke_5d_onnx,
                         resample_opt_random_test,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                 { data_types::f16, {1, 16, 13, 13, 5}, {1, 16, 26, 26, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                 { data_types::f32, {1, 16, 13, 13, 5}, {1, 16, 26, 26, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                 { data_types::f16, {16, 16, 7, 7, 5}, {16, 16, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv16, format::bs_fs_zyx_bsv16_fsv16, {}, {}},
                                 { data_types::f32, {16, 16, 7, 7, 5}, {16, 16, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv16, format::bs_fs_zyx_bsv16_fsv16, {}, {}},
                                 { data_types::f16, {32, 16, 7, 7, 5}, {32, 16, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16, {}, {}},
                                 { data_types::f32, {32, 16, 7, 7, 5}, {32, 16, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16, {}, {}},

                                 { data_types::i8, {1, 16, 13, 13, 5}, {1, 16, 26, 26, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32, {}, {}},
                                 { data_types::u8, {1, 16, 13, 13, 5}, {1, 16, 26, 26, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32, {}, {}},
                                 { data_types::i8, {16, 16, 7, 7, 5}, {16, 16, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32, {}, {}},
                                 { data_types::u8, {16, 16, 7, 7, 5}, {16, 16, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32, {}, {}},
                                 { data_types::i8, {32, 16, 7, 7, 5}, {32, 16, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32, {}, {}},
                                 { data_types::u8, {32, 16, 7, 7, 5}, {32, 16, 14, 14, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32, {}, {}},
                            }
                        ));

// those tests should be disabled or deleted
INSTANTIATE_TEST_SUITE_P(resample_opt_perf_linear_5_onnx,
                         resample_opt_random_test_ext,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                { data_types::f16, {1, 32, 64, 64, 5}, {1, 32, 128, 128, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::f32, {1, 32, 64, 64, 5}, {1, 32, 128, 128, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::f16, {16, 32, 64, 64, 5}, {16, 32, 128, 128, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv16, format::bs_fs_zyx_bsv16_fsv16, {}, {}},
                                { data_types::f32, {16, 32, 64, 64, 5}, {16, 32, 128, 128, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv16, format::bs_fs_zyx_bsv16_fsv16, {}, {}},
                                { data_types::f16, {32, 32, 64, 64, 5}, {32, 32, 128, 128, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16, {}, {}},
                                { data_types::f32, {32, 32, 64, 64, 5}, {32, 32, 128, 128, 5}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16, {}, {}},

                                { data_types::i8, {1, 32, 64, 64, 5}, {1, 32, 128, 128, 5},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32, {}, {}},
                                { data_types::u8, {1, 32, 64, 64, 5}, {1, 32, 128, 128, 5},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv32, {}, {}},
                                { data_types::i8, {16, 32, 64, 64, 5}, {16, 32, 128, 128, 5},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32, {}, {}},
                                { data_types::u8, {16, 32, 64, 64, 5}, {16, 32, 128, 128, 5},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32, {}, {}},
                                { data_types::i8, {32, 32, 64, 64, 5}, {32, 32, 128, 128, 5},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32, {}, {}},
                                { data_types::u8, {32, 32, 64, 64, 5}, {32, 32, 128, 128, 5},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32, {}, {}},

                                { data_types::f16, {32, 32, 256, 256, 1}, {32, 32, 512, 512, 1}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_zyx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16, {}, {}},
                                { data_types::f16, {1, 32, 64, 64, 32}, {1, 32, 128, 128, 32}, 1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                            }
                        ));

INSTANTIATE_TEST_SUITE_P(resample_opt_perf_linear_5_nearest,
                         resample_opt_random_test_ext,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                { data_types::f16, {1, 128, 16, 16, 16}, {1, 128, 32, 32, 32}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::f16, {1, 128, 32, 32, 32}, {1, 128, 64, 64, 64}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::f16, {1, 128, 64, 64, 64}, {1, 128, 128, 128, 128}, 1, resample::InterpolateOp::InterpolateMode::NEAREST, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                            }
                        ));
INSTANTIATE_TEST_SUITE_P(resample_opt_smoke_linear_onnx_5d_3axes_padding,
                         resample_opt_random_test,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                { data_types::f16, {1, 16, 13, 13, 13},  {1, 16, 26, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}},
                                { data_types::f16, {1, 16, 13, 13, 13},  {1, 16, 26, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, {0, 0, 0, 0, 0}, {0, 0, 1, 1, 1}},
                                { data_types::f16, {1, 16, 13, 13, 13},  {1, 16, 26, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, {0, 0, 1, 1, 1}, {0, 0, 0, 0, 0}},
                            }
                        ));

INSTANTIATE_TEST_SUITE_P(resample_opt_smoke_linear_onnx_5d_3axes_simple,
                         resample_opt_random_test,
                         testing::ValuesIn(
                            std::vector<resample_opt_random_test_params>{
                                { data_types::f16, {1, 16, 13, 13, 13},  {1, 16, 26, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, {}, {}},
                                { data_types::f16, {1, 16, 13, 13, 13},  {1, 16, 26, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, {}, {}},
                                { data_types::f16, {1, 16, 13, 13, 13},  {1, 16, 26, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, {}, {}},
                                { data_types::f16, {1, 16, 13, 13, 13},  {1, 16, 26, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, {}, {}},
                                { data_types::f16, {1, 16, 13, 13, 13},  {1, 16, 26, 26, 26},  1, resample::InterpolateOp::InterpolateMode::LINEAR_ONNX, 1, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32, {}, {}},
                            }
                        ));

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(resample_random_test, random_cached) {
    execute(GetParam(), true);
}

TEST_P(caffe_resample_random_test, random_cached) {
    auto param = GetParam();
    execute_compare(param, true, true);
}

TEST_P(resample_opt_random_test, random_cached) {
    auto param = GetParam();
    execute_compare(param, true, true);
}
#endif
TEST(resample_gpu, basic_in2x3x2x2_nearest_cached) {
    test_basic_in2x3x2x2_nearest<float>(true);
}
