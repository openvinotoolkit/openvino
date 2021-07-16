// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/softmax.hpp>

using namespace cldnn;
using namespace std;
using namespace ::tests;

class softmax_gpu_xb_f32_test_fixture: public ::testing::Test {
public:
    static const int32_t
        output_x  = 10, output_b  = 2,  // size of whole output buffer
        input_x   = 10, input_b   = 2,  // size of whole input buffer
        in_size   = input_x*input_b,
        out_size  = output_x*output_b;

    float in_buffer[in_size];
    float out_buffer[out_size];
    float expected_buffer[out_size];

    cldnn::engine& engine;
    cldnn::memory::ptr input;

    //neural::primitive output = memory::allocate({ memory::format::xb_f32, {output_b, {{output_x}}, 1}});

    softmax_gpu_xb_f32_test_fixture()
        : engine(get_test_engine())
        , input(engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, 1, input_x, 1}}))
    {}

    void compare_out_buffer_with_expected() {
        for(size_t i = 0; i < out_size; ++i) {
            // does output have expected values
            EXPECT_TRUE(are_equal(out_buffer[i], expected_buffer[i]))
                << "At ["<< i <<  "] Expected : " << expected_buffer[i] << " actual : " << out_buffer[i];
        }
    }

    void compare_out_buffer_with_expected_batch_wise() {
        for(size_t b = 0; b < output_b; ++b) {
            float batch_wise_sum = 0;
            for(size_t x = 0; x < output_x; ++x) {
                auto idx = b+x*output_b;
                batch_wise_sum += out_buffer[idx];
                // does output have expected values
                EXPECT_TRUE(are_equal(out_buffer[idx], expected_buffer[idx]))
                    << "At ["<< idx <<  "] Expected : " << expected_buffer[idx] << " actual : " << out_buffer[idx];
            }
            // does it sum to 1 batch wise
            EXPECT_TRUE(are_equal(batch_wise_sum, 1.0f))
                << "Expected : " << 1.0f << " actual : " << batch_wise_sum;
        }
    }
};

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values) {
// in_buffer filled with same value == 1.0f
    for(uint32_t i = 0; i < out_size; ++i) {
              in_buffer[i] = 1.0f;
        expected_buffer[i] = 0.1f;
    }
    std::vector<float> in_b(std::begin(in_buffer), std::end(in_buffer));

    set_values(input, in_b);

    network network(engine, topology(input_layout("input", input->get_layout()), softmax("softmax", "input")));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = output_ptr[i];
    }
    compare_out_buffer_with_expected();
}

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values_batch_wise) {
// in_buffer filled with same value == 1..2 each batch accordingly (softmax can only xb_f32 )
    for(size_t i = 0; i < output_x; ++i) {
        for(size_t j = 0; j < output_b; ++j)
            in_buffer[j+i*output_b] = (j+i*output_b) % 2 +1.0f;
    }

    std::vector<float> in_b(std::begin(in_buffer), std::end(in_buffer));
    set_values(input, in_b);
    // fill buffer with the expected 0.1f value
    for(size_t i = 0; i < out_size; ++i)
        expected_buffer[i] = 0.1f;

    network network(engine, topology(input_layout("input", input->get_layout()), softmax("softmax", "input")));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = output_ptr[i];
    }
    compare_out_buffer_with_expected_batch_wise();
}

TEST_F(softmax_gpu_xb_f32_test_fixture, values_batch_wise) {

    float in_buf[in_size] = {
       //b0  b1
        2.0f, 2.0f, //x0
        2.0f, 2.0f, //x1
        2.0f, 2.0f, //x2
        3.0f, 3.0f, //x3
        5.0f, 5.0f, //x4
        4.0f, 4.0f, //x5
        3.0f, 3.0f, //x6
        2.0f, 2.0f, //x7
        2.0f, 2.0f, //x8
        2.0f, 2.0f  //x9
    };

    float exp_buf[out_size] = {
        0.02569957f,     0.02569957f,
        0.02569957f,     0.02569957f,
        0.02569957f,     0.02569957f,
        0.069858674f,    0.069858674f,
        0.516189665f,    0.516189665f,
        0.189895565f,    0.189895565f,
        0.069858674f,    0.069858674f,
        0.02569957f,     0.02569957f,
        0.02569957f,     0.02569957f,
        0.02569957f,     0.02569957f

    };

    std::vector<float> in_b(std::begin(in_buf), std::end(in_buf));
    set_values(input, in_b);
    std::copy(exp_buf, exp_buf+in_size, expected_buffer);

    // out_buffer filled with non-signaling NaN
    for(size_t i = 0; i < out_size; ++i)
        out_buffer[i] = NAN;

    network network(engine, topology(input_layout("input", input->get_layout()), softmax("softmax", "input")));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = output_ptr[i];
    }
    compare_out_buffer_with_expected_batch_wise();
}

TEST(softmax_gpu_bfyx_f32, normalize_fyx) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input"));

    set_values(input, {  //bfyx
             //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    });

    float expected_max_values[2] = {
        0.481618381f, 0.953259517f
    };

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++)
    {
        out_buffer[i] = output_ptr[i];
    }

    float sum = 0;
    float expected_sum = 1.0f;

    float temp_max = 0;
    int max_value_buffer_index = 0;

    for (uint32_t i = 0; i < batch_num; i++) //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
    {
        for (uint32_t j = 0; j < y_size; j++)
        {
            for (uint32_t k = 0; k < x_size; k++)
            {
                for (uint32_t l = 0; l < feature_num; l++)
                {
                    int index = i * feature_num * x_size * y_size + j * x_size + k + l * x_size * y_size;
                    sum += out_buffer[index];
                    if (out_buffer[index] >= temp_max)
                    {
                        temp_max = out_buffer[index];
                    }
                }
            }
        }

        EXPECT_EQ(true, are_equal(sum, expected_sum));
        sum = 0.0f;
        EXPECT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
        temp_max = 0;
        max_value_buffer_index++;
    }
}

TEST(softmax_gpu_bfyx_f32, normalize_y) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_y));

    vector<float> input_vec = {
              //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[12] = {
        0.689974481f,   //b=0, f=0, x=0
        0.832018385f,   //b=0, f=0, x=1

        0.999962831f,   //b=0, f=1, x=0
        0.993307149f,   //b=0, f=1, x=1

        0.999962831f,   //b=0, f=2, x=0
        0.993307149f,   //b=0, f=2, x=1

        0.98201379f,    //b=1, f=0, x=0
        0.99998987f,    //b=1, f=0, x=1

        0.98201379f,    //b=1, f=1, x=0
        0.999547378f,   //b=1, f=1, x=1

        0.999962831f,   //b=1, f=2, x=0
        0.993307149f    //b=1, f=2, x=1
    };

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++)
    {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++) //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
    {
        for (uint32_t l = 0; l < feature_num; l++)
        {
            for (uint32_t k = 0; k < x_size; k++)
            {
                float sum = 0.0f;
                for (uint32_t j = 0; j < y_size; j++)
                {
                    int index = i * feature_num * x_size * y_size +
                        l * x_size * y_size +
                        j * x_size +
                        k;

                    if (out_buffer[index] >= temp_max)
                    {
                        temp_max = out_buffer[index];
                    }

                    sum += out_buffer[index];
                }
                EXPECT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                EXPECT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

TEST(softmax_gpu_bfyx_f32, normalize_f) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_f));

    vector<float> input_vec = {
        //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[8] = {
        0.344253346f, //b=0, y=0, x=0
        0.364854551f, //b=0, y=0, x=1

        0.999963085f, //b=0, y=1, x=0
        0.493894592f, //b=0, y=1, x=1

        0.719294981f, //b=1, y=0, x=0
        0.364854551f, //b=1, y=0, x=1

        0.73105857f, //b=1, y=1, x=0
        0.977054322f //b=1, y=1, x=1
    };

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++)
    {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++) //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
    {
        for (uint32_t j = 0; j < y_size; j++)
        {
            for (uint32_t k = 0; k < x_size; k++)
            {
                float sum = 0.0f;
                for (uint32_t l = 0; l < feature_num; l++)
                {
                    int index = i * feature_num * x_size * y_size +
                        l * x_size * y_size +
                        j * x_size +
                        k;

                    if (out_buffer[index] >= temp_max)
                    {
                        temp_max = out_buffer[index];
                    }

                    sum += out_buffer[index];
                }
                EXPECT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                EXPECT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

TEST(softmax_gpu_yxfb_f32, normalize_f) {

    static const int32_t x_size = 1, y_size = 2, feature_num = 1,
        batch_num = 12, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, y_size , x_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_fyx));

    set_values(input, {  //yxfb
                //f0b0  f0b1  f0b2  f0b3  f0b4    f0b5    f0b6   f0b7   f0b8    f0b9   f0b10  f0b11
        /*y0x0*/ 0.1f, -0.1f, 0.9f, 1.5f, 0.15f, -0.01f, 0.19f,  0.45f, 0.41f, -0.12f, 0.39f, 0.65f,
        /*y1x0*/ 0.2f, 0.2f, -10.f, 5.2f, 0.01f, 0.015f, 0.29f,  0.05f, 0.41f, -0.31f, 0.29f, 1.35f
    });

    float expected_max_values[batch_num * feature_num * x_size] = {
        0.524979174f,
        0.574442506f,
        0.999981523f,
        0.975872993f,
        0.534942925f,
        0.506249666f,
        0.524979174f,
        0.598687649f,
        0.500000000f,
        0.547357619f,
        0.524979174f,
        0.668187797f
    };

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++)
    {
        out_buffer[i] = output_ptr[i];
    }

    float expected_sum = 1.0f;

    float temp_max = 0;

    for (uint32_t b = 0; b < batch_num; b++)
    {
        for (uint32_t f = 0; f < feature_num; f++)
        {
            for (uint32_t x = 0; x < x_size; x++)
            {
                float sum = 0.0f;
                for (uint32_t y = 0; y < y_size; y++)
                {
                    int index = b + y * batch_num + f * feature_num + x * x_size;
                    if (out_buffer[index] >= temp_max)
                    {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                EXPECT_EQ(true, are_equal(temp_max, expected_max_values[b * feature_num * x_size + f * x_size + x]));
                temp_max = 0;
                EXPECT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

TEST(softmax_gpu_bfzyx_f32, normalize_z) {
    //  Input  : 2x3x2x2x2
    static const int32_t x_size = 2, y_size = 2, z_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size  *y_size * z_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ batch_num, feature_num, x_size , y_size, z_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_z));

    vector<float> input_vec = {
        //    z0y0x0 z0y0x1 z0y1x0 z0y1x1 z1y0x0 z1y0x1 z1y1x0 z1y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f, 0.2f, -0.2f, 0.9f,  2.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f, 0.3f, 0.1f,  -11.f, 6.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f, 0.1f, 0.3f,  -9.f,  4.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f, 5.f,  0.1f,  6.f,   22.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f, 2.2f,  0.3f,  6.f,  5.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f, 1.2f, 0.3f,  -12.f,  2.2f
    };
    set_values(input, input_vec);

    float expected_max_values[24] = {
        0.524979f, 0.524979f,
        0.5f,      0.731059f,
        0.524979f, 0.524979f,
        0.731059f, 0.731059f,
        0.524979f, 0.524979f,
        0.731059f, 0.731059f,
        0.880797f, 0.598688f,
        0.731059f, 0.999955f,
        0.858149f, 0.549834f,
        0.880797f, 0.952574f,
        0.731059f, 0.524979f,
        0.880797f, 0.952574f,
    };

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++)
    {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++)
    {
        for (uint32_t l = 0; l < feature_num; l++)
        {
            for (uint32_t j = 0; j < y_size; j++)
            {
                for (uint32_t k = 0; k < x_size; k++)
                {
                    float sum = 0.0f;
                    for (uint32_t m = 0; m < z_size; m++)
                    {
                        int index = i * feature_num * x_size * y_size * z_size +
                            l * x_size * y_size * z_size +
                            m * x_size * y_size +
                            j * x_size +
                            k;

                        if (out_buffer[index] >= temp_max)
                        {
                            temp_max = out_buffer[index];
                        }

                        sum += out_buffer[index];
                    }
                    EXPECT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                    temp_max = 0;
                    max_value_buffer_index++;
                    EXPECT_EQ(true, are_equal(sum, expected_sum));
                    sum = 0.0f;
                }
            }
        }
    }
}

TEST(softmax_gpu_bfyx_f32, normalize_all) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
                         batch_num = 2, buf_size = x_size * y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_all));

    set_values(input, {//bfyx
                       //       y0x0  y0x1   y1x0    y1x1
                       /*b0f0*/ 0.1f, -0.1f, 0.9f, 1.5f,
                       /*b0f1*/ 0.2f, 0.2f, -10.f, 5.2f,
                       /*b0f2*/ 0.2f, 0.2f, -10.f, 5.2f,
                       /*b1f0*/ 3.f, 0.5f, 7.f, 12.f,
                       /*b1f1*/ 4.f, 0.5f, 8.f, 8.2f,
                       /*b1f2*/ 0.2f, 0.2f, -10.f, 5.2f});

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float sum = 0.0f;
    float expected_sum = 1.0f;
    for (uint32_t i = 0; i < buf_size; i++) {
        sum += output_ptr[i];
    }
    EXPECT_EQ(true, are_equal(sum, expected_sum));
}

TEST(softmax_gpu_yxfb_f32, normalize_all) {
    //  Input  : 2x2x3x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
                         batch_num = 2, buf_size = x_size * y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {y_size, x_size, feature_num, batch_num}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_all));

    set_values(input, {//yxfb
                       //       f0b0  f0b1   f1b0    f1b1
                       /*y0x0*/ 0.1f, -0.1f, 0.9f, 1.5f,
                       /*y0x1*/ 0.2f, 0.2f, -10.f, 5.2f,
                       /*y0x2*/ 0.2f, 0.2f, -10.f, 5.2f,
                       /*y1x0*/ 3.f, 0.5f, 7.f, 12.f,
                       /*y1x1*/ 4.f, 0.5f, 8.f, 8.2f,
                       /*y1x2*/ 0.2f, 0.2f, -10.f, 5.2f});

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float sum = 0.0f;
    float expected_sum = 1.0f;
    for (uint32_t i = 0; i < buf_size; i++) {
        sum += output_ptr[i];
    }
    EXPECT_EQ(true, are_equal(sum, expected_sum));
}

TEST(softmax_gpu_bfzyx_f32, normalize_all) {
    //  Input  : 2x3x2x2x2
    static const int32_t x_size = 2, y_size = 2, z_size = 2, feature_num = 3,
                         batch_num = 2, buf_size = x_size * y_size * z_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfzyx, {batch_num, feature_num, x_size, y_size, z_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_all));

    set_values(input, {//    z0y0x0 z0y0x1 z0y1x0 z0y1x1 z1y0x0 z1y0x1 z1y1x0 z1y1x1
                       /*b0f0*/ 0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                       /*b0f1*/ 0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                       /*b0f2*/ 0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,

                       /*b1f0*/ 3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                       /*b1f1*/ 4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                       /*b1f2*/ 0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f});
    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float sum = 0.0f;
    float expected_sum = 1.0f;
    for (uint32_t i = 0; i < buf_size; i++) {
        sum += output_ptr[i];
    }
    EXPECT_EQ(true, are_equal(sum, expected_sum));
}

TEST(softmax_gpu_bfyx_f16, normalize_all) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
                         batch_num = 2, buf_size = x_size * y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfyx, {batch_num, feature_num, x_size, y_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_all));

    set_values(input, {//bfyx
                       //           y0x0            y0x1            y1x0            y1x1
                       /*b0f0*/ FLOAT16(0.1f), FLOAT16(-0.1f), FLOAT16(0.9f), FLOAT16(1.5f),
                       /*b0f1*/ FLOAT16(0.2f), FLOAT16(0.2f), FLOAT16(-10.f), FLOAT16(5.2f),
                       /*b0f2*/ FLOAT16(0.2f), FLOAT16(0.2f), FLOAT16(-10.f), FLOAT16(5.2f),
                       /*b1f0*/ FLOAT16(3.f), FLOAT16(0.5f), FLOAT16(7.f), FLOAT16(12.f),
                       /*b1f1*/ FLOAT16(4.f), FLOAT16(0.5f), FLOAT16(8.f), FLOAT16(8.2f),
                       /*b1f2*/ FLOAT16(0.2f), FLOAT16(0.2f), FLOAT16(-10.f), FLOAT16(5.2f)});

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());
    float sum = 0.0f;
    float expected_sum = 1.0f;
    for (uint32_t i = 0; i < buf_size; i++) {
        sum += float16_to_float32(output_ptr[i]);
    }
    ASSERT_NEAR(sum, expected_sum, 0.001);
}

TEST(softmax_gpu_yxfb_f16, normalize_all) {
    //  Input  : 2x2x3x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
                         batch_num = 2, buf_size = x_size * y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::yxfb, {y_size, x_size, feature_num, batch_num}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_all));

    set_values(input, {//yxfb
                       //           f0b0            f0b1            f1b0            f1b1
                       /*y0x0*/ FLOAT16(0.1f), FLOAT16(-0.1f), FLOAT16(0.9f), FLOAT16(1.5f),
                       /*y0x1*/ FLOAT16(0.2f), FLOAT16(0.2f), FLOAT16(-10.f), FLOAT16(5.2f),
                       /*y0x2*/ FLOAT16(0.2f), FLOAT16(0.2f), FLOAT16(-10.f), FLOAT16(5.2f),
                       /*y1x0*/ FLOAT16(3.f), FLOAT16(0.5f), FLOAT16(7.f), FLOAT16(12.f),
                       /*y1x1*/ FLOAT16(4.f), FLOAT16(0.5f), FLOAT16(8.f), FLOAT16(8.2f),
                       /*y1x2*/ FLOAT16(0.2f), FLOAT16(0.2f), FLOAT16(-10.f), FLOAT16(5.2f)});

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());
    float sum = 0.0f;
    float expected_sum = 1.0f;
    for (uint32_t i = 0; i < buf_size; i++) {
        sum += float16_to_float32(output_ptr[i]);
    }
    ASSERT_NEAR(sum, expected_sum, 0.001);
}

TEST(softmax_gpu_bfzyx_f16, normalize_all) {
    //  Input  : 2x3x2x2x2
    static const int32_t x_size = 2, y_size = 2, z_size = 2, feature_num = 3,
                         batch_num = 2, buf_size = x_size * y_size * z_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f16, format::bfzyx, {batch_num, feature_num, x_size, y_size, z_size}});
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", "input", softmax::normalize_all));

    set_values(input, {//           z0y0x0          z0y0x1          z0y1x0        z0y1x1        z1y0x0          z1y0x1          z1y1x0          z1y1x1
                       /*b0f0*/ FLOAT16(0.1f), FLOAT16(-0.1f), FLOAT16(0.9f), FLOAT16(1.5f), FLOAT16(0.2f), FLOAT16(-0.2f), FLOAT16(0.9f), FLOAT16(2.5f),
                       /*b0f1*/ FLOAT16(0.2f), FLOAT16(0.2f), FLOAT16(-10.f), FLOAT16(5.2f), FLOAT16(0.3f), FLOAT16(0.1f), FLOAT16(-11.f), FLOAT16(6.2f),
                       /*b0f2*/ FLOAT16(0.2f), FLOAT16(0.2f), FLOAT16(-10.f), FLOAT16(5.2f), FLOAT16(0.1f), FLOAT16(0.3f), FLOAT16(-9.f), FLOAT16(4.2f),

                       /*b1f0*/ FLOAT16(3.f), FLOAT16(0.5f), FLOAT16(7.f), FLOAT16(12.f), FLOAT16(5.f), FLOAT16(0.1f), FLOAT16(6.f), FLOAT16(22.f),
                       /*b1f1*/ FLOAT16(4.f), FLOAT16(0.5f), FLOAT16(8.f), FLOAT16(8.2f), FLOAT16(2.2f), FLOAT16(0.3f), FLOAT16(6.f), FLOAT16(5.2f),
                       /*b1f2*/ FLOAT16(0.2f), FLOAT16(0.2f), FLOAT16(-10.f), FLOAT16(5.2f), FLOAT16(1.2f), FLOAT16(0.3f), FLOAT16(-12.f), FLOAT16(2.2f)});
    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());
    float sum = 0.0f;
    float expected_sum = 1.0f;
    for (uint32_t i = 0; i < buf_size; i++) {
        sum += float16_to_float32(output_ptr[i]);
    }
    ASSERT_NEAR(sum, expected_sum, 0.001);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Negative Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

//TODO:
//TEST(NegativeSoftmaxTest, DISABLED_TestAll) {
//}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Positive Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

using namespace cldnn;

class softmax_test : public tests::generic_test
{

public:
    softmax_test() : tests::generic_test()
    {
    }

    void SetUp() override
    {
        max_ulps_diff_allowed = 6;
    }

    static void TearDownTestCase()
    {
        all_layer_params.clear();
        all_generic_params.clear();
    }

    static std::vector<std::shared_ptr<cldnn::primitive>> generate_specific_test_params()
    {
        all_layer_params.emplace_back(new softmax("softmax", "input0", softmax::normalize_f));

        //The test checks only valid combinations.
        //TODO: add more combinations.

        return all_layer_params;
    }

    static std::vector<std::shared_ptr<tests::test_params>> generate_generic_test_params()
    {
        return generic_test::generate_generic_test_params(all_generic_params);
    }

    bool is_format_supported(cldnn::format format) override
    {
        return
            format == cldnn::format::yxfb ||
            format == cldnn::format::bfyx;
    }

    template<typename Type>
    memory::ptr generate_reference_typed(const std::vector<memory::ptr> & inputs)
    {
        assert(inputs.size() == 1);
        const memory::ptr input = inputs[0];

        //Output is bfyx
        auto output = engine.allocate_memory(cldnn::layout(input->get_layout().data_type, input->get_layout().format, input->get_layout().size));

        cldnn::mem_lock<Type> in0_mem(input, get_test_stream());
        cldnn::mem_lock<Type> out_mem(output, get_test_stream());

        const int in0_b = input->get_layout().size.sizes()[0];
        const int in0_f = input->get_layout().size.sizes()[1];
        const int in0_h = input->get_layout().size.sizes()[3];
        const int in0_w = input->get_layout().size.sizes()[2];

//        const int out_b = output->get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[0];
//        const int out_f = output->get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[1];
//        const int out_h = output->get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[2];
//        const int out_w = output->get_layout().size.transform(cldnn::format::bfyx, 0).sizes()[3];

//        assert(in0_b == out_b);
//        assert(in0_f == out_f);
//        assert(in0_h == out_h);
//        assert(in0_w == out_w);

        std::vector<float> cached_exp_vals;
        cached_exp_vals.resize(in0_f);

        const auto input_desc = get_linear_memory_desc(input->get_layout());

        for (int n = 0; n < in0_b; ++n)
        for (int y = 0; y < in0_h; ++y)
        for (int x = 0; x < in0_w; ++x)
        {
            float max_val = -std::numeric_limits<float>::infinity();

            for (int c = 0; c < in0_f; ++c)
            {
                const size_t in0_idx = get_linear_index(input->get_layout(), n, c, y, x, input_desc);

                max_val = std::max(max_val, static_cast<float>(in0_mem[in0_idx]));
            }

            float Z = 0;

            for (int c = 0; c < in0_f; ++c)
            {
                const size_t in0_idx = get_linear_index(input->get_layout(), n, c, y, x, input_desc);

                float tmp = static_cast<float>((Type)std::exp(static_cast<float>(in0_mem[in0_idx]) - max_val));
                Z += tmp;
                cached_exp_vals[c] = tmp;
            }

            for (int c = 0; c < in0_f; ++c)
            {
                const size_t out_idx = get_linear_index(output->get_layout(), n, c, y, x, input_desc);
                out_mem[out_idx] = (Type)(cached_exp_vals[c] / Z);
            }
        }

        return output;
    }

    virtual memory::ptr generate_reference(const std::vector<memory::ptr> & inputs) override
    {
        if (generic_params->data_type == data_types::f32)
        {
            return generate_reference_typed<float>(inputs);
        }
        else
        {
            return generate_reference_typed<FLOAT16>(inputs);
        }
    }

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>>& info)
    {
        std::stringstream res;

        const auto & p = std::get<0>(info.param);

        assert (p->data_type == data_types::f32 ||
                p->data_type == data_types::f16);

        res << info.index
            << "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < p->input_layouts.size(); ++i)
        {
            const auto chans = format::traits(p->fmt).order;

            res << "_" << "Input" << i;
            for (unsigned int j = 0; j < p->input_layouts[i].size.sizes(p->fmt).size(); ++j)
            {
                res << chans[j] << p->input_layouts[i].size.sizes(p->fmt)[j];
            }
        }

        return res.str();
    }

private:

    static std::vector<std::shared_ptr<tests::test_params>> all_generic_params;
    static std::vector<std::shared_ptr<cldnn::primitive>> all_layer_params;

};

std::vector<std::shared_ptr<cldnn::primitive>> softmax_test::all_layer_params = {};
std::vector<std::shared_ptr<tests::test_params>> softmax_test::all_generic_params = {};

TEST_P(softmax_test, SOFTMAX)
{
    run_single_test();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_SOFTMAX,
    softmax_test,
    ::testing::Combine(::testing::ValuesIn(softmax_test::generate_generic_test_params()), ::testing::ValuesIn(softmax_test::generate_specific_test_params())),
    softmax_test::custom_param_name);
