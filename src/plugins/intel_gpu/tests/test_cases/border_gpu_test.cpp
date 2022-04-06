// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/border.hpp>
#include "ngraph/runtime/reference/pad.hpp"

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

template<typename T>
static std::vector<T> generate_rnd_real_input(
    const std::vector<size_t> sizes,
    const T min = static_cast<T>(0), const T max = static_cast<T>(1), const unsigned rnd_bits = 9)
{
    static std::default_random_engine rnd_gen(random_seed);
    tests::distributions::uniform_quantized_real_distribution<T> rnd_dist(min, max, rnd_bits);

    auto acum = std::accumulate(sizes.begin(), sizes.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>());

    std::vector<T> data;
    data.reserve(acum);
    for (size_t i = 0; i < static_cast<size_t>(acum); ++i)
        data.push_back(rnd_dist(rnd_gen));

    return data;
}

template<class T>
tensor to_tensor(std::vector<T> a){
    reverse(a.begin()+2,a.end());
    return tensor(std::vector<tensor::value_type>(a.begin(),a.end()));
}
size_t total_size(const std::vector<size_t>& a){
    size_t ret=1;
    for(auto i:a)
        ret*=i;
    return ret;
}

TEST(border_gpu, mytest1) {
    ov::Shape sh_in{2,3,3,2};
    ov::Shape sh_out{4,5,5,4};
    ov::CoordinateDiff cd_lt{1,1,1,1};
    ov::CoordinateDiff cd_rb{1,1,1,1};
    float pad_val=0.3f;

    auto& engine = get_test_engine();
    std::vector<float> input_data = generate_random_1d<float>(total_size(sh_in), -9, 9);
    
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {sh_in[0],sh_in[1],sh_in[3],sh_in[2]}});
    set_values(input, input_data);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output", "input", to_tensor(cd_lt), to_tensor(cd_rb), border_type::constant, pad_val));

    cldnn::network network(engine, topology);
    network.set_input_data("input", input);

    auto output = network.execute().at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> ans(total_size(sh_out));
    auto to_vec_size_t=[](const std::vector<int>& vec){return std::vector<size_t>(vec.begin(),vec.end());};
    ngraph::runtime::reference::pad(
        (const char *)(input_data.data()),
        (const char *)(&pad_val),
        (char *)ans.data(),
        sizeof(float),
        sh_in, sh_out, cd_lt, cd_rb,
        ngraph::op::PadMode::CONSTANT);

    ASSERT_EQ(ans.size(), total_size(sh_out));
    EXPECT_TRUE(!memcmp(output_ptr.data(),ans.data(),sizeof(float)*ans.size()));
    // for(int i=0;i<sh_out[0];i++,std::cout<<std::endl)
    //     for(int j=0;j<sh_out[1];j++,std::cout<<std::endl)
    //         for(int k=0;k<sh_out[2];k++,std::cout<<std::endl)
    //             for(int l=0;l<sh_out[3];l++,std::cout<<' ')
    //                 std::cout<<std::setw(4)<<ans[i*sh_out[1]*sh_out[2]*sh_out[3]+j*sh_out[2]*sh_out[3]+k*sh_out[3]+l];
    // for(int i=0;i<sh_out[0];i++,std::cout<<std::endl)
    //     for(int j=0;j<sh_out[1];j++,std::cout<<std::endl)
    //         for(int k=0;k<sh_out[2];k++,std::cout<<std::endl)
    //             for(int l=0;l<sh_out[3];l++,std::cout<<' ')
    //                 std::cout<<std::setw(4)<<output_ptr[i*sh_out[1]*sh_out[2]*sh_out[3]+j*sh_out[2]*sh_out[3]+k*sh_out[3]+l];
}

TEST(border_gpu, mytest0) {
    //  Input (XY) : 4x3
    //  Output (XY): 10x7

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 3;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    std::vector<float> input_data = {
          1, -2,  3,  -4,
          5,  6,  7,   8,
        -10, 12, 13, -13,
    };
    float pad_val=0.3f;
    
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {in_size_b, in_size_f, in_size_x, in_size_y}});
    set_values(input, input_data);

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::constant, pad_val)
    );

    cldnn::network network(engine, topology);
    network.set_input_data("input", input);

    auto output = network.execute().at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> ans(out_size_b * out_size_f * out_size_y * out_size_x);
    auto to_vec_size_t=[](const std::vector<int>& vec){return std::vector<size_t>(vec.begin(),vec.end());};
    ngraph::runtime::reference::pad(
        (const char *)(input_data.data()),
        (const char *)(&pad_val),
        (char *)ans.data(),
        sizeof(float),
        ov::Shape(to_vec_size_t(input->get_layout().get_dims())),
        ov::Shape(to_vec_size_t(output->get_layout().get_dims())),
        ov::CoordinateDiff({ blt_size_b, blt_size_f, blt_size_y, blt_size_x }),
        ov::CoordinateDiff({ brb_size_b, brb_size_f, brb_size_y, brb_size_x }),
        ngraph::op::PadMode::CONSTANT);

    ASSERT_EQ(ans.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));
    EXPECT_TRUE(!memcmp(output_ptr.data(),ans.data(),sizeof(float)*ans.size()));
    // for(int i=0;i<out_size_y;i++,std::cout<<std::endl)
    //     for(int j=0;j<out_size_x;j++)
    //         std::cout<<std::setw(4)<<ans[i*out_size_x+j];
    // std::cout<<std::endl;
    // for(int i=0;i<out_size_y;i++,std::cout<<std::endl)
    //     for(int j=0;j<out_size_x;j++)
    //         std::cout<<std::setw(4)<<output_ptr[i*out_size_x+j];
}

TEST(border_gpu, basic_yxfb_0x0x1x2_0x0x3x4_border_constant) {
    //  Input (XY) : 4x3
    //  Output (XY): 10x7

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 3;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::constant, 0.0f)
    );

    std::vector<float> input_data = {
          1, -2,  3,  -4,
          5,  6,  7,   8,
        -10, 12, 13, -13,
    };
    std::vector<float> out_data = {
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
        0, 0,   1, -2,  3,  -4, 0, 0, 0, 0,
        0, 0,   5,  6,  7,   8, 0, 0, 0, 0,
        0, 0, -10, 12, 13, -13, 0, 0, 0, 0,
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
    };
    set_values(input, input_data);

    cldnn::network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((y * out_size_x + x) * out_size_f + f) * out_size_b + b; // YXFB

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_fsv16_0x0x1x2_0x0x3x4_border_constant) {
    //  Input (XY) : 4x3
    //  Output (XY): 10x7

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 3;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        reorder("border_input", "input", cldnn::format::b_fs_yx_fsv16, cldnn::data_types::f32),
        border("border", "border_input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::constant, 0.0f),
        reorder("output", "border", cldnn::format::yxfb, cldnn::data_types::f32)
    );

    std::vector<float> input_data = {
          1, -2,  3,  -4,
          5,  6,  7,   8,
        -10, 12, 13, -13,
    };
    std::vector<float> out_data = {
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
        0, 0,   1, -2,  3,  -4, 0, 0, 0, 0,
        0, 0,   5,  6,  7,   8, 0, 0, 0, 0,
        0, 0, -10, 12, 13, -13, 0, 0, 0, 0,
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
        0, 0,   0,  0,  0,   0, 0, 0, 0, 0,
    };
    set_values(input, input_data);

    cldnn::network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((y * out_size_x + x) * out_size_f + f) * out_size_b + b; // YXFB

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfzyx_0x0x1x01_0x0x0x0x3_border_constant) {

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 2;
    constexpr auto in_size_x = 2;
    constexpr auto in_size_z = 3;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 0;
    constexpr auto blt_size_z = 1;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 0;
    constexpr auto brb_size_x = 0;
    constexpr auto brb_size_z = 3;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;
    constexpr auto out_size_z = in_size_z + blt_size_z + brb_size_z;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ in_size_b, in_size_f, in_size_x, in_size_y, in_size_z } });

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
            { blt_size_b, blt_size_f, blt_size_x, blt_size_y, blt_size_z },
            { brb_size_b, brb_size_f, brb_size_x, brb_size_y, brb_size_z },
            border_type::constant, 0.0f)
    );

    std::vector<float> input_data = {
        1, -2,
        3, -4,

        5, 6,
        7, 8,

        -10, 12,
        13, -13,
    };
    std::vector<float> out_data = {
        0, 0,
        0, 0,
        0, 0,

        0, 0,
        1, -2,
        3,  -4,

        0, 0,
        5,  6,
        7,   8,

        0, 0,
        -10, 12,
        13, -13,

        0, 0,
        0, 0,
        0, 0,

        0, 0,
        0, 0,
        0, 0,

        0, 0,
        0, 0,
        0, 0,
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x * out_size_z));

    uint32_t idx = 0;
    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto z = 0; z < out_size_z; ++z) {     // z
                for (auto y = 0; y < out_size_y; ++y) {     // Y
                    for (auto x = 0; x < out_size_x; ++x) { // X
                        EXPECT_EQ(output_ptr[idx], out_data[idx]);
                        idx++;
                    }
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfwzyx_0x0x0x1x0x1_0x0x0x1x0x1_border_constant) {

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 2;
    constexpr auto in_size_x = 2;
    constexpr auto in_size_z = 3;
    constexpr auto in_size_w = 1;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 0;
    constexpr auto blt_size_x = 1;
    constexpr auto blt_size_z = 0;
    constexpr auto blt_size_w = 1;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 0;
    constexpr auto brb_size_x = 1;
    constexpr auto brb_size_z = 0;
    constexpr auto brb_size_w = 1;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;
    constexpr auto out_size_z = in_size_z + blt_size_z + brb_size_z;
    constexpr auto out_size_w = in_size_w + blt_size_w + brb_size_w;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx, tensor{ batch(in_size_b), feature(in_size_f), spatial(in_size_x, in_size_y, in_size_z, in_size_w) } });

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
            tensor{ batch(blt_size_b), feature(blt_size_f), spatial(blt_size_x, blt_size_y, blt_size_z, blt_size_w) },
            tensor{ batch(brb_size_b), feature(brb_size_f), spatial(brb_size_x, brb_size_y, brb_size_z, brb_size_w) },
            border_type::constant, 0.0f)
    );

    std::vector<float> input_data = {
        1, -2,
        3, -4,

        5, 6,
        7, 8,

        -10, 12,
        13, -13,
    };
    std::vector<float> out_data = {
        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 1, -2, 0,
        0, 3, -4, 0,

        0, 5, 6, 0,
        0, 7, 8, 0,

        0, -10, 12, 0,
        0, 13, -13, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,

        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    set_values(input, input_data);

    cldnn::network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x * out_size_z * out_size_w));

    uint32_t idx = 0;
    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto w = 0; w < out_size_w; ++w) {     // z
                for (auto z = 0; z < out_size_z; ++z) {     // z
                    for (auto y = 0; y < out_size_y; ++y) {     // Y
                        for (auto x = 0; x < out_size_x; ++x) { // X
                            EXPECT_EQ(output_ptr[idx], out_data[idx]);
                            idx++;
                        }
                    }
                }
            }
        }
    }
}

TEST(border_gpu, basic_yxfb_0x0x1x2_0x0x3x4_border_constant_non_constant) {
    //  Input (XY) : 4x3
    //  Output (XY): 10x7

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 3;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               tensor{blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               tensor{brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::constant, 1.0f)
    );

    std::vector<float> input_data = {
          1, -2,  3,  -4,
          5,  6,  7,   8,
        -10, 12, 13, -13,
    };
    std::vector<float> out_data = {
        1, 1,   1,  1,  1,   1, 1, 1, 1, 1,
        1, 1,   1, -2,  3,  -4, 1, 1, 1, 1,
        1, 1,   5,  6,  7,   8, 1, 1, 1, 1,
        1, 1, -10, 12, 13, -13, 1, 1, 1, 1,
        1, 1,   1,  1,  1,   1, 1, 1, 1, 1,
        1, 1,   1,  1,  1,   1, 1, 1, 1, 1,
        1, 1,   1,  1,  1,   1, 1, 1, 1, 1,
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((y * out_size_x + x) * out_size_f + f) * out_size_b + b; // YXFB

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_yxfb_0x0x1x2_0x0x3x4_border_mirror) {
    //  Input (XY) : 4x3
    //  Output (XY): 10x7

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 3;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, {in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               {blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               {brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::mirror)
    );

    std::vector<float> input_data = {
          1, -2,  3,  -4,
          5,  6,  7,   8,
        -10, 12, 13, -13,
    };
    std::vector<float> out_data = {
        -2,   1,   1, -2,  3,  -4,  -4,  3, -2,   1,
        -2,   1,   1, -2,  3,  -4,  -4,  3, -2,   1,
         6,   5,   5,  6,  7,   8,   8,  7,  6,   5,
        12, -10, -10, 12, 13, -13, -13, 13, 12, -10,
        12, -10, -10, 12, 13, -13, -13, 13, 12, -10,
         6,   5,   5,  6,  7,   8,   8,  7,  6,   5,
        -2,   1,   1, -2,  3,  -4,  -4,  3, -2,   1,
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((y * out_size_x + x) * out_size_f + f) * out_size_b + b; // YXFB

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfzyx_0x0x0x0x1_0x0x0x0x1_border_mirror) {

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 2;
    constexpr auto in_size_x = 4;
    constexpr auto in_size_z = 2;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 0;
    constexpr auto blt_size_x = 0;
    constexpr auto blt_size_z = 1;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 0;
    constexpr auto brb_size_x = 0;
    constexpr auto brb_size_z = 1;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;
    constexpr auto out_size_z = in_size_z + blt_size_z + brb_size_z;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ in_size_b, in_size_f, in_size_x, in_size_y, in_size_z } });

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
            { blt_size_b, blt_size_f, blt_size_x, blt_size_y, blt_size_z },
            { brb_size_b, brb_size_f, brb_size_x, brb_size_y, brb_size_z },
            border_type::mirror)
    );

    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                     static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x),
                                     static_cast<std::size_t>(in_size_z) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto z = 0; z < out_size_z; ++z) {         // F
                for (auto y = 0; y < out_size_y; ++y) {     // Y
                    for (auto x = 0; x < out_size_x; ++x) { // X
                        auto output_off = (((b * out_size_f + f) * out_size_z + z) * out_size_y + y) * out_size_x + x; // BFZYX

                        auto in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? blt_size_b - 1 - b : in_size_b + out_size_b - brb_size_b - 1 - b);
                        auto in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? blt_size_f - 1 - f : in_size_f + out_size_f - brb_size_f - 1 - f);
                        auto in_z = (z >= blt_size_z && z < out_size_z - brb_size_z) ? z - blt_size_z : (z < blt_size_z ? blt_size_z - 1 - z : in_size_z + out_size_z - brb_size_z - 1 - z);
                        auto in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? blt_size_y - 1 - y : in_size_y + out_size_y - brb_size_y - 1 - y);
                        auto in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? blt_size_x - 1 - x : in_size_x + out_size_x - brb_size_x - 1 - x);

                        auto input_off = (((in_b * in_size_f + in_f) * in_size_z + in_z) * in_size_y + in_y) * in_size_x + in_x; // BFZYX

                        EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                    }
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfzyxw_0x0x0x0x1_0x0x0x0x1_border_mirror) {

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 2;
    constexpr auto in_size_x = 4;
    constexpr auto in_size_z = 2;
    constexpr auto in_size_w = 2;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 0;
    constexpr auto blt_size_x = 0;
    constexpr auto blt_size_z = 1;
    constexpr auto blt_size_w = 1;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 0;
    constexpr auto brb_size_x = 0;
    constexpr auto brb_size_z = 1;
    constexpr auto brb_size_w = 1;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;
    constexpr auto out_size_z = in_size_z + blt_size_z + brb_size_z;
    constexpr auto out_size_w = in_size_w + blt_size_w + brb_size_w;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx, tensor{ batch(in_size_b), feature(in_size_f), spatial(in_size_x, in_size_y, in_size_z, in_size_w) } });

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
            tensor{ batch(blt_size_b), feature(blt_size_f), spatial(blt_size_x, blt_size_y, blt_size_z, blt_size_w) },
            tensor{ batch(brb_size_b), feature(brb_size_f), spatial(brb_size_x, brb_size_y, brb_size_z, brb_size_w) },
            border_type::mirror)
    );

    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                     static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x),
                                     static_cast<std::size_t>(in_size_z), static_cast<std::size_t>(in_size_w) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto w = 0; w < out_size_w; ++w) {         // F
                for (auto z = 0; z < out_size_z; ++z) {         // F
                    for (auto y = 0; y < out_size_y; ++y) {     // Y
                        for (auto x = 0; x < out_size_x; ++x) { // X
                            auto output_off = ((((b * out_size_f + f) * out_size_w + w)* out_size_z + z) * out_size_y + y) * out_size_x + x; // BFZYX

                            auto in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? blt_size_b - 1 - b : in_size_b + out_size_b - brb_size_b - 1 - b);
                            auto in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? blt_size_f - 1 - f : in_size_f + out_size_f - brb_size_f - 1 - f);
                            auto in_w = (w >= blt_size_w && w < out_size_w - brb_size_w) ? w - blt_size_w : (w < blt_size_w ? blt_size_w - 1 - w : in_size_w + out_size_w - brb_size_w - 1 - w);
                            auto in_z = (z >= blt_size_z && z < out_size_z - brb_size_z) ? z - blt_size_z : (z < blt_size_z ? blt_size_z - 1 - z : in_size_z + out_size_z - brb_size_z - 1 - z);
                            auto in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? blt_size_y - 1 - y : in_size_y + out_size_y - brb_size_y - 1 - y);
                            auto in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? blt_size_x - 1 - x : in_size_x + out_size_x - brb_size_x - 1 - x);

                            auto input_off = ((((in_b * in_size_f + in_f) * in_size_w + in_w)* in_size_z + in_z) * in_size_y + in_y) * in_size_x + in_x; // BFZYX

                            EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                        }
                    }
                }
            }
        }
    }
}

TEST(border_gpu, basic_yxfb_0x0x1x2_0x0x3x4_border_mirror_101) {
    //  Input (XY) : 5x4
    //  Output (XY): 11x8

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 4;
    constexpr auto in_size_x = 5;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, tensor{in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               tensor{blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               tensor{brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::mirror_101)
    );

    std::vector<float> input_data = {
          1, -2,  3,  -4,  4,
          5,  6,  7,   8, -8,
        -10, 12, 13, -13, 10,
        -20, 22, 23, -23, 20,
    };
    std::vector<float> out_data = {
         7,  6,   5,  6,  7,   8, -8,   8,  7,  6,   5,
         3, -2,   1, -2,  3,  -4,  4,  -4,  3, -2,   1,
         7,  6,   5,  6,  7,   8, -8,   8,  7,  6,   5,
        13, 12, -10, 12, 13, -13, 10, -13, 13, 12, -10,
        23, 22, -20, 22, 23, -23, 20, -23, 23, 22, -20,
        13, 12, -10, 12, 13, -13, 10, -13, 13, 12, -10,
         7,  6,   5,  6,  7,   8, -8,   8,  7,  6,   5,
         3, -2,   1, -2,  3,  -4,  4,  -4,  3, -2,   1,
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((y * out_size_x + x) * out_size_f + f) * out_size_b + b; // YXFB

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfzyx_0x0x0x0x1_0x0x0x0x1_border_mirror_101) {
    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 2;
    constexpr auto in_size_x = 5;
    constexpr auto in_size_z = 2;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 0;
    constexpr auto blt_size_x = 0;
    constexpr auto blt_size_z = 1;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 0;
    constexpr auto brb_size_x = 0;
    constexpr auto brb_size_z = 1;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;
    constexpr auto out_size_z = in_size_z + blt_size_z + brb_size_z;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, tensor{ in_size_b, in_size_f, in_size_x, in_size_y, in_size_z } });

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
            tensor{ blt_size_b, blt_size_f, blt_size_x, blt_size_y, blt_size_z },
            tensor{ brb_size_b, brb_size_f, brb_size_x, brb_size_y, brb_size_z },
            border_type::mirror_101)
    );

    std::vector<float> input_data = {
        1, -2,  3,  -4,  4,
        5,  6,  7,   8, -8,

        -10, 12, 13, -13, 10,
        -20, 22, 23, -23, 20,
    };
    std::vector<float> out_data = {
        -10, 12, 13, -13, 10,
        -20, 22, 23, -23, 20,
        1, -2,  3,  -4,  4,
        5,  6,  7,   8, -8,
        -10, 12, 13, -13, 10,
        -20, 22, 23, -23, 20,
        1, -2,  3,  -4,  4,
        5,  6,  7,   8, -8,
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x * out_size_z));

    uint32_t idx = 0;
    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto z = 0; z < out_size_z; ++z) {         // Z
                for (auto y = 0; y < out_size_y; ++y) {     // Y
                    for (auto x = 0; x < out_size_x; ++x) { // X
                        EXPECT_EQ(output_ptr[idx], out_data[idx]);
                        idx++;
                    }
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfwzyx_0x0x0x0x1x1_0x0x0x0x1x1_border_mirror_101) {
    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 4;
    constexpr auto in_size_x = 2;
    constexpr auto in_size_z = 1;
    constexpr auto in_size_w = 3;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 0;
    constexpr auto blt_size_x = 0;
    constexpr auto blt_size_z = 0;
    constexpr auto blt_size_w = 1;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 0;
    constexpr auto brb_size_x = 0;
    constexpr auto brb_size_z = 0;
    constexpr auto brb_size_w = 1;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;
    constexpr auto out_size_z = in_size_z + blt_size_z + brb_size_z;
    constexpr auto out_size_w = in_size_w + blt_size_w + brb_size_w;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx, tensor{ batch(in_size_b), feature(in_size_f), spatial(in_size_x, in_size_y, in_size_z, in_size_w) } });

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
            tensor{ batch(blt_size_b), feature(blt_size_f), spatial(blt_size_x, blt_size_y, blt_size_z, blt_size_w) },
            tensor{ batch(brb_size_b), feature(brb_size_f), spatial(brb_size_x, brb_size_y, brb_size_z, brb_size_w) },
            border_type::mirror_101)
    );

    std::vector<float> input_data = {
        1, -2,  3,  -4,
        5,  6,  7,   8,

        2, -3,  4,  -5,
        15,  4,  4,   4,

        2, -6,  13,  -14,
        3,  7,  7,   7,
    };
    std::vector<float> out_data = {
        2, -3,  4,  -5,
        15,  4,  4,   4,

        1, -2,  3,  -4,
        5,  6,  7,   8,

        2, -3,  4,  -5,
        15,  4,  4,   4,

        2, -6,  13,  -14,
        3,  7,  7,   7,

        2, -3,  4,  -5,
        15,  4,  4,   4,
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x * out_size_z * out_size_w));

    uint32_t idx = 0;
    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto w = 0; w < out_size_w; ++w) {         // F
                for (auto z = 0; z < out_size_z; ++z) {         // Z
                    for (auto y = 0; y < out_size_y; ++y) {     // Y
                        for (auto x = 0; x < out_size_x; ++x) { // X
                            EXPECT_EQ(output_ptr[idx], out_data[idx]);
                            idx++;
                        }
                    }
                }
            }
        }
    }
}

TEST(border_gpu, basic_yxfb_0x0x1x2_0x0x3x4_border_edge) {
    //  Input (XY) : 5x4
    //  Output (XY): 11x8

    constexpr auto in_size_b = 1;
    constexpr auto in_size_f = 1;
    constexpr auto in_size_y = 4;
    constexpr auto in_size_x = 5;

    constexpr auto blt_size_b = 0;
    constexpr auto blt_size_f = 0;
    constexpr auto blt_size_y = 1;
    constexpr auto blt_size_x = 2;

    constexpr auto brb_size_b = 0;
    constexpr auto brb_size_f = 0;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::yxfb, tensor{in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               tensor{blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               tensor{brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::edge)
    );

    std::vector<float> input_data = {
          1, -2,  3,  -4,  4,
          5,  6,  7,   8, -8,
        -10, 12, 13, -13, 10,
        -20, 22, 23, -23, 20,
    };
    std::vector<float> out_data = {
          1,   1,   1, -2,  3,  -4,  4,  4,  4,  4,  4,
          1,   1,   1, -2,  3,  -4,  4,  4,  4,  4,  4,
          5,   5,   5,  6,  7,   8, -8, -8, -8, -8, -8,
        -10, -10, -10, 12, 13, -13, 10, 10, 10, 10, 10,
        -20, -20, -20, 22, 23, -23, 20, 20, 20, 20, 20,
        -20, -20, -20, 22, 23, -23, 20, 20, 20, 20, 20,
        -20, -20, -20, 22, 23, -23, 20, 20, 20, 20, 20,
        -20, -20, -20, 22, 23, -23, 20, 20, 20, 20, 20
    };
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(out_data.size(), static_cast<std::size_t>(out_size_b * out_size_f * out_size_y * out_size_x));

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((y * out_size_x + x) * out_size_f + f) * out_size_b + b; // YXFB

                    EXPECT_EQ(output_ptr[output_off], out_data[output_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfyx_2x1x2x3_1x2x3x4_border_constant) {
    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 3;
    constexpr auto in_size_y = 5;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 2;
    constexpr auto blt_size_f = 1;
    constexpr auto blt_size_y = 2;
    constexpr auto blt_size_x = 3;

    constexpr auto brb_size_b = 1;
    constexpr auto brb_size_f = 2;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, tensor{in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               tensor{blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               tensor{brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::constant,
               0.0f)
    );

    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                     static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x)};
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x; // BFYX

                    if (b < blt_size_b || b >= out_size_b - brb_size_b ||
                        f < blt_size_f || f >= out_size_f - brb_size_f ||
                        y < blt_size_y || y >= out_size_y - brb_size_y ||
                        x < blt_size_x || x >= out_size_x - brb_size_x)
                    {
                        EXPECT_EQ(output_ptr[output_off], 0.0f);
                    }
                    else
                    {
                        auto input_off  = (((b - blt_size_b) * in_size_f + f - blt_size_f) * in_size_y + y - blt_size_y) * in_size_x + x - blt_size_x; // BFYX
                        EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                    }
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfyx_2x1x2x3_1x2x3x4_border_mirror) {
    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 3;
    constexpr auto in_size_y = 5;
    constexpr auto in_size_x = 4;

    constexpr auto blt_size_b = 2;
    constexpr auto blt_size_f = 1;
    constexpr auto blt_size_y = 2;
    constexpr auto blt_size_x = 3;

    constexpr auto brb_size_b = 1;
    constexpr auto brb_size_f = 2;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, tensor{in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               tensor{blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               tensor{brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::mirror)
    );

    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                     static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x; // BFYX

                    auto in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? blt_size_b - 1 - b : in_size_b + out_size_b - brb_size_b - 1 - b);
                    auto in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? blt_size_f - 1 - f : in_size_f + out_size_f - brb_size_f - 1 - f);
                    auto in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? blt_size_y - 1 - y : in_size_y + out_size_y - brb_size_y - 1 - y);
                    auto in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? blt_size_x - 1 - x : in_size_x + out_size_x - brb_size_x - 1 - x);

                    auto input_off  = ((in_b * in_size_f + in_f) * in_size_y + in_y) * in_size_x + in_x; // BFYX

                    EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfyx_2x1x2x3_1x2x3x4_border_mirror_101) {
    constexpr auto in_size_b = 3;
    constexpr auto in_size_f = 4;
    constexpr auto in_size_y = 6;
    constexpr auto in_size_x = 5;

    constexpr auto blt_size_b = 2;
    constexpr auto blt_size_f = 1;
    constexpr auto blt_size_y = 2;
    constexpr auto blt_size_x = 3;

    constexpr auto brb_size_b = 1;
    constexpr auto brb_size_f = 2;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, tensor{in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               tensor{blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               tensor{brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::mirror_101)
    );
    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                    static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x; // BFYX

                    auto in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? blt_size_b - b : in_size_b + out_size_b - brb_size_b - 2 - b);
                    auto in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? blt_size_f - f : in_size_f + out_size_f - brb_size_f - 2 - f);
                    auto in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? blt_size_y - y : in_size_y + out_size_y - brb_size_y - 2 - y);
                    auto in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? blt_size_x - x : in_size_x + out_size_x - brb_size_x - 2 - x);

                    auto input_off  = ((in_b * in_size_f + in_f) * in_size_y + in_y) * in_size_x + in_x; // BFYX

                    EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfyx_2x1x2x3_1x2x3x4_border_edge) {
    constexpr auto in_size_b = 3;
    constexpr auto in_size_f = 4;
    constexpr auto in_size_y = 6;
    constexpr auto in_size_x = 5;

    constexpr auto blt_size_b = 2;
    constexpr auto blt_size_f = 1;
    constexpr auto blt_size_y = 2;
    constexpr auto blt_size_x = 3;

    constexpr auto brb_size_b = 1;
    constexpr auto brb_size_f = 2;
    constexpr auto brb_size_y = 3;
    constexpr auto brb_size_x = 4;

    constexpr auto out_size_b = in_size_b + blt_size_b + brb_size_b;
    constexpr auto out_size_f = in_size_f + blt_size_f + brb_size_f;
    constexpr auto out_size_y = in_size_y + blt_size_y + brb_size_y;
    constexpr auto out_size_x = in_size_x + blt_size_x + brb_size_x;

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, tensor{in_size_b, in_size_f, in_size_x, in_size_y}});

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        border("output", "input",
               tensor{blt_size_b, blt_size_f, blt_size_x, blt_size_y},
               tensor{brb_size_b, brb_size_f, brb_size_x, brb_size_y},
               border_type::edge)
    );
    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                    static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x; // BFYX

                    auto in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? 0 : in_size_b - 1);
                    auto in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? 0 : in_size_f - 1);
                    auto in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? 0 : in_size_y - 1);
                    auto in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? 0 : in_size_x - 1);

                    auto input_off  = ((in_b * in_size_f + in_f) * in_size_y + in_y) * in_size_x + in_x; // BFYX

                    EXPECT_EQ(output_ptr[output_off], input_data[input_off]);
                }
            }
        }
    }
}
