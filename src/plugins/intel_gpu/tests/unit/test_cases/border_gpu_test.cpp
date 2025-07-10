// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/border.hpp>

#include <border_inst.h>

#include <cstddef>
#include <array>

using namespace cldnn;
using namespace ::tests;

namespace {
template<typename T>
static std::vector<T> generate_rnd_real_input(
    const std::vector<size_t> sizes,
    const T min = static_cast<T>(0), const T max = static_cast<T>(1), const unsigned rnd_bits = 9) {
    static std::default_random_engine rnd_gen(random_seed);
    tests::distributions::uniform_quantized_real_distribution<T> rnd_dist(min, max, rnd_bits);

    auto acum = std::accumulate(sizes.begin(), sizes.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>());

    std::vector<T> data;
    data.reserve(acum);
    for (size_t i = 0; i < static_cast<size_t>(acum); ++i)
        data.push_back(rnd_dist(rnd_gen));

    return data;
}

template <class T>
static int mult(T arr) {
    return std::accumulate(arr.begin(), arr.end(), 1, [](int x, int y) {
        return x * y;
    });
}

#define PAD_MODES \
    ov::op::PadMode::CONSTANT, ov::op::PadMode::EDGE, ov::op::PadMode::SYMMETRIC, ov::op::PadMode::REFLECT
#define FORMATS                                                                                                   \
    format::type::bfyx, format::type::yxfb, format::type::b_fs_yx_fsv4, format::type::b_fs_yx_fsv16,              \
        format::type::b_fs_yx_fsv32, format::type::bs_fs_yx_bsv4_fsv2, format::type::bs_fs_yx_bsv16_fsv16,        \
        format::type::bs_fs_yx_bsv32_fsv16, format::type::bs_fs_yx_bsv32_fsv32, format::type::bs_fs_yx_bsv4_fsv4, \
        format::type::bs_fs_yx_bsv8_fsv2, format::type::bs_fs_yx_bsv8_fsv4

template <class T>
using border_test_param = std::tuple<ov::op::PadMode,      // pad mode
                                     T,                    // pad value
                                     format::type,         // format
                                     std::array<int, 4>,   // shape in
                                     std::array<int, 4>,   // coord diff lt
                                     std::array<int, 4>,   // coord diff rb
                                     bool,                 // allow negative pads
                                     bool>;                // is_caching_test

template <class T, data_types T_dt>
class border_test : public ::testing::TestWithParam<border_test_param<T>> {
public:
    tests::random_generator rg;
    ov::op::PadMode pad_mode;
    T pad_value;
    format::type fmt;
    std::array<int, 4> sh_in, cd_lt, cd_rb, sh_out;
    bool allow_negative_pads;
    bool is_caching_test;
    void SetUp() override {
        ::testing::TestWithParam<border_test_param<T>>::SetUp();
        rg.set_seed(GET_SUITE_NAME);
        std::tie(pad_mode, pad_value, fmt, sh_in, cd_lt, cd_rb, allow_negative_pads, is_caching_test) = this->GetParam();
        sh_out = {sh_in[0] + cd_lt[0] + cd_rb[0],
                  sh_in[1] + cd_lt[1] + cd_rb[1],
                  sh_in[2] + cd_lt[2] + cd_rb[2],
                  sh_in[3] + cd_lt[3] + cd_rb[3]};
        auto& engine = get_test_engine();
        auto input_data = rg.generate_random_1d<T>(mult(sh_in), -9, 9, 1);
        auto input = engine.allocate_memory({T_dt, format::bfyx, {sh_in[0], sh_in[1], sh_in[3], sh_in[2]}});
        set_values(input, input_data);

        topology target_topology;
        target_topology.add(input_layout("input", input->get_layout()));
        target_topology.add(reorder("border_input", input_info("input"), fmt, T_dt),
                            border("border",
                                   {input_info("border_input")},
                                   0,
                                   ov::CoordinateDiff(cd_lt.begin(), cd_lt.end()),
                                   ov::CoordinateDiff(cd_rb.begin(), cd_rb.end()),
                                   pad_mode,
                                   pad_value,
                                   allow_negative_pads),
                            reorder("output", input_info("border"), cldnn::format::bfyx, T_dt));
        cldnn::network::ptr target_network = get_network(engine, target_topology,  get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        target_network->set_input_data("input", input);
        auto target_output = target_network->execute().at("output").get_memory();
        cldnn::mem_lock<T> target_output_ptr(target_output, get_test_stream());

        topology base_topology;
        base_topology.add(input_layout("input", input->get_layout()));
        base_topology.add(border("border",
                                 {input_info("input")},
                                 0,
                                 ov::CoordinateDiff(cd_lt.begin(), cd_lt.end()),
                                 ov::CoordinateDiff(cd_rb.begin(), cd_rb.end()),
                                 pad_mode,
                                 pad_value,
                                 allow_negative_pads));

        cldnn::network base_network(engine, base_topology, get_test_default_config(engine));
        base_network.set_input_data("input", input);
        auto base_output = base_network.execute().at("border").get_memory();
        cldnn::mem_lock<T> base_output_ptr(base_output, get_test_stream());

        ASSERT_TRUE(!memcmp(target_output_ptr.data(), base_output_ptr.data(), sizeof(T) * mult(sh_out)));
    }
};
using border_test_i8 = border_test<char, data_types::i8>;
TEST_P(border_test_i8, border_test_i8) {}
INSTANTIATE_TEST_SUITE_P(border_test_i8,
                         border_test_i8,
                         testing::Combine(testing::Values(PAD_MODES),
                                          testing::Values(99),
                                          testing::Values(FORMATS),
                                          testing::Values(std::array<int, 4>{2, 3, 4, 5}),
                                          testing::Values(std::array<int, 4>{1, 2, 3, 4}),
                                          testing::Values(std::array<int, 4>{1, 1, 1, 1}),
                                          testing::Values(false),
                                          testing::Values(false)));
using border_test_u8 = border_test<char, data_types::u8>;
TEST_P(border_test_u8, border_test_u8) {}
INSTANTIATE_TEST_SUITE_P(border_test_u8,
                         border_test_u8,
                         testing::Combine(testing::Values(ov::op::PadMode::EDGE),
                                          testing::Values(99),
                                          testing::Values(format::type::bs_fs_yx_bsv16_fsv16),
                                          testing::Values(std::array<int, 4>{2, 3, 4, 5}),
                                          testing::Values(std::array<int, 4>{1, 2, 3, 4}),
                                          testing::Values(std::array<int, 4>{1, 1, 1, 1}),
                                          testing::Values(false),
                                          testing::Values(false)));
using border_test_i32 = border_test<int, data_types::i32>;
TEST_P(border_test_i32, border_test_i32) {}
INSTANTIATE_TEST_SUITE_P(border_test_i32,
                         border_test_i32,
                         testing::Combine(testing::Values(ov::op::PadMode::SYMMETRIC),
                                          testing::Values(11),
                                          testing::Values(format::type::b_fs_yx_fsv16),
                                          testing::Values(std::array<int, 4>{2, 3, 4, 5}),
                                          testing::Values(std::array<int, 4>{1, 2, 3, 4}),
                                          testing::Values(std::array<int, 4>{1, 1, 1, 1}),
                                          testing::Values(false),
                                          testing::Values(false)));
INSTANTIATE_TEST_SUITE_P(negative_pads,
                         border_test_i32,
                         testing::Combine(testing::Values(PAD_MODES),
                                          testing::Values(-333),
                                          testing::Values(format::type::b_fs_yx_fsv16),
                                          testing::Values(std::array<int, 4>{6, 8, 7, 11}),
                                          testing::ValuesIn({std::array<int, 4>{-1, -2, -2, -3}, std::array<int, 4>{-1, 3, 4, -3}}),
                                          testing::ValuesIn({std::array<int, 4>{-1, -2, -2, -1}, std::array<int, 4>{2, -3, 3, -2}}),
                                          testing::Values(true),
                                          testing::Values(false)));

using border_test_f16 = border_test<ov::float16, data_types::f16>;
TEST_P(border_test_f16, border_test_f16) {}
INSTANTIATE_TEST_SUITE_P(border_test_f16,
                         border_test_f16,
                         testing::Combine(testing::Values(ov::op::PadMode::REFLECT),
                                          testing::Values(ov::float16(123)),
                                          testing::Values(format::type::bs_fs_yx_bsv32_fsv16),
                                          testing::Values(std::array<int, 4>{2, 3, 4, 5}),
                                          testing::Values(std::array<int, 4>{1, 2, 3, 4}),
                                          testing::Values(std::array<int, 4>{1, 1, 1, 1}),
                                          testing::Values(false),
                                          testing::Values(false)));
INSTANTIATE_TEST_SUITE_P(export_import,
                         border_test_f16,
                         testing::Combine(testing::Values(ov::op::PadMode::REFLECT),
                                          testing::Values(ov::float16(123)),
                                          testing::Values(format::type::bs_fs_yx_bsv32_fsv16),
                                          testing::Values(std::array<int, 4>{2, 3, 4, 5}),
                                          testing::Values(std::array<int, 4>{1, 2, 3, 4}),
                                          testing::Values(std::array<int, 4>{1, 1, 1, 1}),
                                          testing::Values(false),
                                          testing::Values(true)));
using border_test_f32 = border_test<float, data_types::f32>;
TEST_P(border_test_f32, border_test_f32) {}
INSTANTIATE_TEST_SUITE_P(border_test_f32,
                         border_test_f32,
                         testing::Combine(testing::Values(ov::op::PadMode::EDGE),
                                          testing::Values(12.34),
                                          testing::Values(format::type::bs_fs_yx_bsv4_fsv2),
                                          testing::Values(std::array<int, 4>{2, 3, 4, 5}),
                                          testing::Values(std::array<int, 4>{1, 2, 3, 4}),
                                          testing::Values(std::array<int, 4>{1, 1, 1, 1}),
                                          testing::Values(false),
                                          testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(bsv16fsv16_reorder,
                         border_test_i32,
                         testing::Combine(testing::Values(ov::op::PadMode::SYMMETRIC),
                                          testing::Values(99),
                                          testing::Values(format::type::bs_fs_yx_bsv16_fsv16),
                                          testing::Values(std::array<int, 4>{2, 3, 4, 5}),
                                          testing::Values(std::array<int, 4>{1, 2, 3, 4}),
                                          testing::Values(std::array<int, 4>{1, 1, 1, 1}),
                                          testing::Values(false),
                                          testing::Values(false)));

TEST(border_gpu, bsv16fsv16_without_reorder) {
    using T = int;
    data_types T_dt = data_types::i32;
    ov::op::PadMode pad_mode = ov::op::PadMode::CONSTANT;
    T pad_value = 0;
    std::array<int, 4> sh_in = {16, 16, 2, 3}, cd_lt = {0, 0, 1, 1}, cd_rb = {0, 0, 1, 1}, sh_out;
    sh_out = {sh_in[0] + cd_lt[0] + cd_rb[0],
              sh_in[1] + cd_lt[1] + cd_rb[1],
              sh_in[2] + cd_lt[2] + cd_rb[2],
              sh_in[3] + cd_lt[3] + cd_rb[3]};
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto input_data = rg.generate_random_1d<T>(mult(sh_in), -9, 9, 1);
    auto input = engine.allocate_memory({T_dt, format::bfyx, {sh_in[0], sh_in[1], sh_in[3], sh_in[2]}});
    set_values(input, input_data);

    auto index_bfyx = [=](std::array<int, 4> s, int b, int f, int y, int x) {
        return b * s[1] * s[2] * s[3] + f * s[2] * s[3] + y * s[3] + x;
    };
    auto index_bsv16fsv16 = [=](std::array<int, 4> s, int b, int f, int y, int x) {
        int b0 = b / 16, b1 = b % 16, f0 = f / 16, f1 = f % 16;
        return b0 * s[1] / 16 * s[2] * s[3] * 16 * 16 +
               f0 * s[2] * s[3] * 16 * 16 +
               y * s[3] * 16 * 16 +
               x * 16 * 16 +
               b1 * 16 +
               f1;
    };

    auto input_data_b16f16 = input_data;
    for (int b = 0; b < sh_in[0]; b++)
        for (int f = 0; f < sh_in[1]; f++)
            for (int y = 0; y < sh_in[2]; y++)
                for (int x = 0; x < sh_in[3]; x++)
                    input_data_b16f16[index_bsv16fsv16(sh_in, b, f, y, x)] = input_data[index_bfyx(sh_in, b, f, y, x)];

    auto input_b16f16 = engine.allocate_memory({T_dt, format::bs_fs_yx_bsv16_fsv16, {sh_in[0], sh_in[1], sh_in[3], sh_in[2]}});
    set_values(input_b16f16, input_data_b16f16);

    topology target_topology;
    target_topology.add(input_layout("input", input_b16f16->get_layout()));
    target_topology.add(border("border",
                               {input_info("input")},
                               0,
                               ov::CoordinateDiff(cd_lt.begin(), cd_lt.end()),
                               ov::CoordinateDiff(cd_rb.begin(), cd_rb.end()),
                               pad_mode,
                               pad_value));
    cldnn::network target_network(engine, target_topology, get_test_default_config(engine));
    target_network.set_input_data("input", input_b16f16);
    auto target_output = target_network.execute().at("border").get_memory();
    cldnn::mem_lock<T> target_output_ptr(target_output, get_test_stream());

    topology base_topology;
    base_topology.add(input_layout("input", input->get_layout()));
    base_topology.add(border("border",
                             {input_info("input")},
                             0,
                             ov::CoordinateDiff(cd_lt.begin(), cd_lt.end()),
                             ov::CoordinateDiff(cd_rb.begin(), cd_rb.end()),
                             pad_mode,
                             pad_value));
    cldnn::network base_network(engine, base_topology, get_test_default_config(engine));
    base_network.set_input_data("input", input);
    auto base_output = base_network.execute().at("border").get_memory();
    cldnn::mem_lock<T> base_output_ptr(base_output, get_test_stream());

    std::vector<T> b16f16_to_bfyx(mult(sh_out));
    for (int b = 0; b < sh_out[0]; b++)
        for (int f = 0; f < sh_out[1]; f++)
            for (int y = 0; y < sh_out[2]; y++)
                for (int x = 0; x < sh_out[3]; x++)
                    b16f16_to_bfyx[index_bfyx(sh_out, b, f, y, x)] =
                        target_output_ptr.data()[index_bsv16fsv16(sh_out, b, f, y, x)];

    ASSERT_TRUE(!memcmp(b16f16_to_bfyx.data(), base_output_ptr.data(), sizeof(T) * mult(sh_out)));
}

TEST(border_gpu, zyx_bsv16fsv16) {
    using T = int;
    data_types T_dt = data_types::i32;
    ov::op::PadMode pad_mode = ov::op::PadMode::REFLECT;
    T pad_value = 0;
    std::array<int, 5> sh_in = {16, 16, 4, 5, 6}, cd_lt = {0, 0, 1, 1, 1}, cd_rb = {0, 0, 2, 3, 4}, sh_out;
    sh_out = {sh_in[0] + cd_lt[0] + cd_rb[0],
              sh_in[1] + cd_lt[1] + cd_rb[1],
              sh_in[2] + cd_lt[2] + cd_rb[2],
              sh_in[3] + cd_lt[3] + cd_rb[3],
              sh_in[4] + cd_lt[4] + cd_rb[4]};
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto input_data = rg.generate_random_1d<T>(mult(sh_in), -9, 9, 1);
    auto input = engine.allocate_memory({T_dt, format::bfzyx, {sh_in[0], sh_in[1], sh_in[4], sh_in[3], sh_in[2]}});
    set_values(input, input_data);

    topology target_topology;
    target_topology.add(input_layout("input", input->get_layout()));
    target_topology.add(reorder("border_input", input_info("input"), format::bs_fs_zyx_bsv16_fsv16, T_dt),
                        border("border",
                               {input_info("border_input")},
                               0,
                               ov::CoordinateDiff(cd_lt.begin(), cd_lt.end()),
                               ov::CoordinateDiff(cd_rb.begin(), cd_rb.end()),
                               pad_mode,
                               pad_value),
                        reorder("output", input_info("border"), cldnn::format::bfzyx, T_dt));
    cldnn::network target_network(engine, target_topology, get_test_default_config(engine));
    target_network.set_input_data("input", input);
    auto target_output = target_network.execute().at("output").get_memory();
    cldnn::mem_lock<T> target_output_ptr(target_output, get_test_stream());

    topology base_topology;
    base_topology.add(input_layout("input", input->get_layout()));
    base_topology.add(border("border",
                             {input_info("input")},
                             0,
                             ov::CoordinateDiff(cd_lt.begin(), cd_lt.end()),
                             ov::CoordinateDiff(cd_rb.begin(), cd_rb.end()),
                             pad_mode,
                             pad_value));
    cldnn::network base_network(engine, base_topology, get_test_default_config(engine));
    base_network.set_input_data("input", input);
    auto base_output = base_network.execute().at("border").get_memory();
    cldnn::mem_lock<T> base_output_ptr(base_output, get_test_stream());

    ASSERT_TRUE(!memcmp(target_output_ptr.data(), base_output_ptr.data(), sizeof(T) * mult(sh_out)));
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::CONSTANT,
                        0.0f));

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

    cldnn::network network(engine, topology, get_test_default_config(engine));
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

                    ASSERT_EQ(output_ptr[output_off], out_data[output_off]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("border_input", input_info("input"), cldnn::format::b_fs_yx_fsv16, cldnn::data_types::f32),
                 border("border",
                        {input_info("border_input")},
                        0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::CONSTANT,
                        0.0f),
                 reorder("output", input_info("border"), cldnn::format::yxfb, cldnn::data_types::f32));

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

    cldnn::network network(engine, topology, get_test_default_config(engine));
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

                    ASSERT_EQ(output_ptr[output_off], out_data[output_off]);
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
    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { in_size_b, in_size_f, in_size_x, in_size_y, in_size_z } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_z, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_z, brb_size_y, brb_size_x},
                        ov::op::PadMode::CONSTANT,
                        0.0f));

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

    network network(engine, topology, get_test_default_config(engine));
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
                        ASSERT_EQ(output_ptr[idx], out_data[idx]);
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
    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx,
                                          tensor{ batch(in_size_b), feature(in_size_f), spatial(in_size_x, in_size_y, in_size_z, in_size_w) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_w, blt_size_z, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_w, brb_size_z, brb_size_y, brb_size_x},
                        ov::op::PadMode::CONSTANT,
                        0.0f));

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

    cldnn::network network(engine, topology, get_test_default_config(engine));
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
                            ASSERT_EQ(output_ptr[idx], out_data[idx]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::CONSTANT,
                        1.0f));

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

    network network(engine, topology, get_test_default_config(engine));
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

                    ASSERT_EQ(output_ptr[output_off], out_data[output_off]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::SYMMETRIC));

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

    network network(engine, topology, get_test_default_config(engine));
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

                    ASSERT_EQ(output_ptr[output_off], out_data[output_off]);
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
    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { in_size_b, in_size_f, in_size_x, in_size_y, in_size_z } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_z, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_z, brb_size_y, brb_size_x},
                        ov::op::PadMode::SYMMETRIC));

    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                     static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x),
                                     static_cast<std::size_t>(in_size_z) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology, get_test_default_config(engine));
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

                        ASSERT_EQ(output_ptr[output_off], input_data[input_off]);
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
    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx,
                                          tensor{ batch(in_size_b), feature(in_size_f), spatial(in_size_x, in_size_y, in_size_z, in_size_w) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(
        border("output",
               {input_info("input")}, 0,
               ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_w, blt_size_z, blt_size_y, blt_size_x},
               ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_w, brb_size_z, brb_size_y, brb_size_x},
               ov::op::PadMode::SYMMETRIC));

    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                     static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x),
                                     static_cast<std::size_t>(in_size_z), static_cast<std::size_t>(in_size_w) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology, get_test_default_config(engine));
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

                            ASSERT_EQ(output_ptr[output_off], input_data[input_off]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::REFLECT));

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

    network network(engine, topology, get_test_default_config(engine));
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

                    ASSERT_EQ(output_ptr[output_off], out_data[output_off]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_z, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_z, brb_size_y, brb_size_x},
                        ov::op::PadMode::REFLECT));

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

    network network(engine, topology, get_test_default_config(engine));
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
                        ASSERT_EQ(output_ptr[idx], out_data[idx]);
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
    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx,
                                          tensor{ batch(in_size_b), feature(in_size_f), spatial(in_size_x, in_size_y, in_size_z, in_size_w) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(
        border("output",
               {input_info("input")}, 0,
               ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_w, blt_size_z, blt_size_y, blt_size_x},
               ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_w, brb_size_z, brb_size_y, brb_size_x},
               ov::op::PadMode::REFLECT));

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

    network network(engine, topology, get_test_default_config(engine));
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
                            ASSERT_EQ(output_ptr[idx], out_data[idx]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::EDGE));

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

    network network(engine, topology, get_test_default_config(engine));
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

                    ASSERT_EQ(output_ptr[output_off], out_data[output_off]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::CONSTANT,
                        0.0f));

    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                     static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x)};
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology, get_test_default_config(engine));
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
                        x < blt_size_x || x >= out_size_x - brb_size_x) {
                        ASSERT_EQ(output_ptr[output_off], 0.0f);
                    } else {
                        auto input_off  = (((b - blt_size_b) * in_size_f + f - blt_size_f) * in_size_y + y - blt_size_y) * in_size_x + x - blt_size_x; // BFYX
                        ASSERT_EQ(output_ptr[output_off], input_data[input_off]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::SYMMETRIC));

    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                     static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology, get_test_default_config(engine));
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

                    ASSERT_EQ(output_ptr[output_off], input_data[input_off]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::REFLECT));
    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                    static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology, get_test_default_config(engine));
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

                    ASSERT_EQ(output_ptr[output_off], input_data[input_off]);
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
    topology.add(input_layout("input", input->get_layout()));
    topology.add(border("output",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::EDGE));
    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                    static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x) };
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    network network(engine, topology, get_test_default_config(engine));
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

                    ASSERT_EQ(output_ptr[output_off], input_data[input_off]);
                }
            }
        }
    }
}

TEST(border_gpu, basic_bfyx_2x1x2x3_1x2x3x4_border_constant_dynamic) {
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
    auto input_layout_dynamic = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{in_size_b, in_size_f, in_size_y, in_size_x}, data_types::f32, format::bfyx};
    auto input = engine.allocate_memory(input_layout_static);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(border("border",
                        {input_info("input")}, 0,
                        ov::CoordinateDiff{blt_size_b, blt_size_f, blt_size_y, blt_size_x},
                        ov::CoordinateDiff{brb_size_b, brb_size_f, brb_size_y, brb_size_x},
                        ov::op::PadMode::CONSTANT,
                        0.0f));

    const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                     static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x)};
    std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
    set_values(input, input_data);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("border");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "border");

    auto output = outputs.at("border").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (auto b = 0; b < out_size_b; ++b) {             // B
        for (auto f = 0; f < out_size_f; ++f) {         // F
            for (auto y = 0; y < out_size_y; ++y) {     // Y
                for (auto x = 0; x < out_size_x; ++x) { // X
                    auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x; // BFYX

                    if (b < blt_size_b || b >= out_size_b - brb_size_b ||
                        f < blt_size_f || f >= out_size_f - brb_size_f ||
                        y < blt_size_y || y >= out_size_y - brb_size_y ||
                        x < blt_size_x || x >= out_size_x - brb_size_x) {
                        ASSERT_EQ(output_ptr[output_off], 0.0f);
                    } else {
                        auto input_off  = (((b - blt_size_b) * in_size_f + f - blt_size_f) * in_size_y + y - blt_size_y) * in_size_x + x - blt_size_x; // BFYX
                        ASSERT_EQ(output_ptr[output_off], input_data[input_off]);
                    }
                }
            }
        }
    }
}

struct border_dynamic_test_param {
    ov::op::PadMode mode;
    std::array<int, 4> in_shape;
    std::array<int, 4> lt;
    std::array<int, 4> rb;
};

class border_dynamic_test : public ::testing::TestWithParam<border_dynamic_test_param> {
public:
    void SetUp() override {
        ::testing::TestWithParam<border_dynamic_test_param>::SetUp();

        const  border_dynamic_test_param p = this->GetParam();

        mode = p.mode;
        in_size_b = p.in_shape[0];
        in_size_f = p.in_shape[1];
        in_size_y = p.in_shape[2];
        in_size_x = p.in_shape[3];

        blt_size_b = p.lt[0];
        blt_size_f = p.lt[1];
        blt_size_y = p.lt[2];
        blt_size_x = p.lt[3];

        brb_size_b = p.rb[0];
        brb_size_f = p.rb[1];
        brb_size_y = p.rb[2];
        brb_size_x = p.rb[3];

        out_size_b = in_size_b + blt_size_b + brb_size_b;
        out_size_f = in_size_f + blt_size_f + brb_size_f;
        out_size_y = in_size_y + blt_size_y + brb_size_y;
        out_size_x = in_size_x + blt_size_x + brb_size_x;

        auto& engine = get_test_engine();

        const auto input_layout_dynamic = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
        const auto input_layout_static = layout{ov::PartialShape{in_size_b, in_size_f, in_size_y, in_size_x}, data_types::f32, format::bfyx};
        const auto input = engine.allocate_memory(input_layout_static);
        const auto pads_begin = engine.allocate_memory({{4}, data_types::i32, format::bfyx});
        const auto pads_end = engine.allocate_memory({{4}, data_types::i32, format::bfyx});

        set_values(pads_begin, {blt_size_b, blt_size_f, blt_size_y, blt_size_x});
        set_values(pads_end, {brb_size_b, brb_size_f, brb_size_y, brb_size_x});

        constexpr auto pad_value = -333.0f;

        topology topology;
        topology.add(input_layout("input", input_layout_dynamic));
        topology.add(data("pads_begin", pads_begin));
        topology.add(data("pads_end", pads_end));
        topology.add(border("output",
                            {input_info("input"), input_info("pads_begin"), input_info("pads_end")},
                            cldnn::border::PAD_NON_CONST_INPUT::BEGIN |
                            cldnn::border::PAD_NON_CONST_INPUT::END,
                            std::vector<int64_t>{},
                            std::vector<int64_t>{},
                            mode,
                            pad_value,
                            true));

        const std::vector<size_t> sizes{ static_cast<std::size_t>(in_size_b), static_cast<std::size_t>(in_size_f),
                                         static_cast<std::size_t>(in_size_y), static_cast<std::size_t>(in_size_x) };
        const std::vector<float> input_data = generate_rnd_real_input<float>(sizes, -8.0f, 8.0f);
        set_values(input, input_data);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network(engine, topology, config);
        network.set_input_data("input", input);

        const auto inst = network.get_primitive("output");
        const auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        const auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "output");

        const auto output = outputs.at("output").get_memory();
        const cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        const auto expected_size = out_size_b * out_size_f * out_size_y * out_size_x;
        ASSERT_EQ(output_ptr.size(), expected_size);

        for (auto b = 0; b < out_size_b; ++b) {
            for (auto f = 0; f < out_size_f; ++f) {
                for (auto y = 0; y < out_size_y; ++y) {
                    for (auto x = 0; x < out_size_x; ++x) {
                        const auto output_off = ((b * out_size_f + f) * out_size_y + y) * out_size_x + x;
                        ASSERT_GE(output_off, 0);

                        if (mode == ov::op::PadMode::CONSTANT) {
                            if (b < blt_size_b || b >= out_size_b - brb_size_b ||
                                f < blt_size_f || f >= out_size_f - brb_size_f ||
                                y < blt_size_y || y >= out_size_y - brb_size_y ||
                                x < blt_size_x || x >= out_size_x - brb_size_x) {
                                ASSERT_EQ(output_ptr[output_off], pad_value);
                            } else {
                                const auto input_off  = (((b - blt_size_b) * in_size_f + f - blt_size_f) * in_size_y + y - blt_size_y) * in_size_x + x - blt_size_x; // BFYX
                                ASSERT_GE(input_off, 0);
                                ASSERT_EQ(output_ptr[output_off], input_data[input_off]);
                            }
                        } else {
                            int in_b, in_f, in_y, in_x;
                            CalcInIndices(b, f, y, x, in_b, in_f, in_y, in_x);
                            const auto input_off  = ((in_b * in_size_f + in_f) * in_size_y + in_y) * in_size_x + in_x;
                            ASSERT_GE(input_off, 0);
                            ASSERT_EQ(output_ptr[output_off], input_data[input_off]);
                        }
                    }
                }
            }
        }
   }

private:
    void CalcInIndices(const int b, const int f, const int y, const int x, int& in_b, int& in_f, int& in_y, int& in_x) {
        switch (mode) {
            case ov::op::PadMode::REFLECT: {
                in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? blt_size_b - b : in_size_b + out_size_b - brb_size_b - 2 - b);
                in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? blt_size_f - f : in_size_f + out_size_f - brb_size_f - 2 - f);
                in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? blt_size_y - y : in_size_y + out_size_y - brb_size_y - 2 - y);
                in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? blt_size_x - x : in_size_x + out_size_x - brb_size_x - 2 - x);
                break;
            }
            case ov::op::PadMode::SYMMETRIC: {
                in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? blt_size_b - 1 - b : in_size_b + out_size_b - brb_size_b - 1 - b);
                in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? blt_size_f - 1 - f : in_size_f + out_size_f - brb_size_f - 1 - f);
                in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? blt_size_y - 1 - y : in_size_y + out_size_y - brb_size_y - 1 - y);
                in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? blt_size_x - 1 - x : in_size_x + out_size_x - brb_size_x - 1 - x);
                break;
            }
            case ov::op::PadMode::EDGE: {
                in_b = (b >= blt_size_b && b < out_size_b - brb_size_b) ? b - blt_size_b : (b < blt_size_b ? 0 : in_size_b - 1);
                in_f = (f >= blt_size_f && f < out_size_f - brb_size_f) ? f - blt_size_f : (f < blt_size_f ? 0 : in_size_f - 1);
                in_y = (y >= blt_size_y && y < out_size_y - brb_size_y) ? y - blt_size_y : (y < blt_size_y ? 0 : in_size_y - 1);
                in_x = (x >= blt_size_x && x < out_size_x - brb_size_x) ? x - blt_size_x : (x < blt_size_x ? 0 : in_size_x - 1);
                break;
            }
            default: {
                throw std::runtime_error("Invalid PadMode");
            }
        }
    }

    ov::op::PadMode mode;
    int in_size_b, in_size_f, in_size_y, in_size_x;
    int blt_size_b, blt_size_f, blt_size_y, blt_size_x;
    int brb_size_b, brb_size_f, brb_size_y, brb_size_x;
    int out_size_b, out_size_f, out_size_y, out_size_x;
};

const std::vector<border_dynamic_test_param> dynamic_params {
    {ov::op::PadMode::CONSTANT, {2, 3, 5, 4}, {-1, 2, -2, 3}, {2, -1, 3, -2}},
    {ov::op::PadMode::EDGE, {3, 4, 6, 5}, {-1, 1, -3, 2}, {3, -1, 1, -3}},
    {ov::op::PadMode::REFLECT, {3, 4, 6, 5}, {-1, 1, -3, 2}, {2, -1, 2, -3}},
    {ov::op::PadMode::SYMMETRIC, {2, 3, 5, 4}, {-1, 2, -2, 3}, {2, -1, 3, -2}}
    };
TEST_P(border_dynamic_test, border_dynamic_test) {}
INSTANTIATE_TEST_SUITE_P(border_dynamic_test,
                         border_dynamic_test,
                         ::testing::ValuesIn(dynamic_params));

TEST(border_gpu, basic_zero_input_dynamic) {
    auto& engine = get_test_engine();

    // WA to avoid crash due to attempt to allocate 0 bytes for USM memory
    layout fake_input_layout = {{1}, data_types::dynamic, format::bfyx};
    auto input = engine.allocate_memory(fake_input_layout);

    layout zero_input_layout = {{0, 1}, data_types::f32, format::bfyx};
    input = engine.reinterpret_buffer(*input, zero_input_layout);

    layout input_layout_dynamic = {ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};

    ov::CoordinateDiff pads_begin = {4, 0};
    ov::CoordinateDiff pads_end = {0, 0};

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(border("border", {input_info("input")}, 0, pads_begin, pads_end, ov::op::PadMode::CONSTANT, 1.0f));

    std::vector<float> ref_output = {
        1, 1, 1, 1
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("border");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "border");

    auto output = outputs.at("border").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(ref_output.size(), output_ptr.size());

    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_EQ(ref_output[i], output_ptr[i]);
    }
}

TEST(border_gpu, basic_zero_input) {
    auto& engine = get_test_engine();

    // WA to avoid crash due to attempt to allocate 0 bytes for USM memory
    layout fake_input_layout = {{1}, data_types::u8, format::bfyx};
    auto input = engine.allocate_memory(fake_input_layout);

    layout zero_input_layout = {{0, 1}, data_types::f32, format::bfyx};
    input = engine.reinterpret_buffer(*input, zero_input_layout);

    std::vector<int> pads_begin = {4, 0};
    ov::PartialShape pads_begin_shape = { ov::Dimension(pads_begin.size()) };
    auto pads_begin_input = engine.allocate_memory({pads_begin_shape, data_types::i32, format::bfyx});
    set_values(pads_begin_input, pads_begin);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("pads_begin", pads_begin_input->get_layout()));
    topology.add(border("border", {input_info("input"), input_info("pads_begin")},
                        border::PAD_NON_CONST_INPUT::BEGIN,
                        /*pads_begin*/{}, /*pads_end*/{0, 0},
                        ov::op::PadMode::CONSTANT,
                        2.0f));

    std::vector<float> ref_output = {
        2, 2, 2, 2
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);
    network.set_input_data("pads_begin", pads_begin_input);
    auto outputs = network.execute();

    auto output = outputs.at("border").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(ref_output.size(), output_ptr.size());

    for (size_t i = 0; i < output_ptr.size(); ++i) {
        ASSERT_EQ(ref_output[i], output_ptr[i]);
    }
}

TEST(border_gpu, 3d_input) {
    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    ov::op::PadMode pad_mode = ov::op::PadMode::CONSTANT;
    ov::float16 pad_value = 0;
    format::type fmt = format::type::bfyx;
    std::array<int, 3> sh_in = {2, 3, 4};
    std::vector<int> cd_lt = {5, 6, 7};
    std::vector<int> cd_rb = {1, 8, 9};
    std::array<int, 3> sh_out = {sh_in[0] + cd_lt[0] + cd_rb[0],
                                 sh_in[1] + cd_lt[1] + cd_rb[1],
                                 sh_in[2] + cd_lt[2] + cd_rb[2]};
    bool allow_negative_pads = false;
    auto& engine = get_test_engine();

    auto input_data = rg.generate_random_1d<ov::float16>(mult(sh_in), -9, 9, 1);
    auto input = engine.allocate_memory({{sh_in[0], sh_in[1], sh_in[2]}, data_types::f16, format::bfyx});
    set_values(input, input_data);

    auto begin = engine.allocate_memory({{3}, data_types::i32, format::bfyx});
    set_values(begin, cd_lt);

    auto end = engine.allocate_memory({{3}, data_types::i32, format::bfyx});
    set_values(end, cd_rb);

    topology target_topology;
    const auto input_layout_dynamic = layout{ov::PartialShape::dynamic(3), data_types::f16, format::bfyx};

    target_topology.add(input_layout("input", input_layout_dynamic));
    target_topology.add(data("begin", begin));
    target_topology.add(data("end", end));
    target_topology.add(reorder("border_input", input_info("input"), fmt, data_types::f16),
                        border("border",
                               {input_info("border_input"), input_info("begin"), input_info("end")},
                               cldnn::border::PAD_NON_CONST_INPUT::BEGIN | cldnn::border::PAD_NON_CONST_INPUT::END,
                               std::vector<int64_t>{},
                               std::vector<int64_t>{},
                               pad_mode,
                               pad_value,
                               allow_negative_pads),
                        reorder("output", input_info("border"), cldnn::format::bfyx, data_types::f16));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network target_network(engine, target_topology, config);
    target_network.set_input_data("input", input);
    auto target_output = target_network.execute().at("output").get_memory();
    cldnn::mem_lock<ov::float16> target_output_ptr(target_output, get_test_stream());

    topology base_topology;
    base_topology.add(input_layout("input", input_layout_dynamic));
    base_topology.add(data("begin", begin));
    base_topology.add(data("end", end));
    base_topology.add(border("border",
                             {input_info("input"), input_info("begin"), input_info("end")},
                             cldnn::border::PAD_NON_CONST_INPUT::BEGIN | cldnn::border::PAD_NON_CONST_INPUT::END,
                             std::vector<int64_t>{},
                             std::vector<int64_t>{},
                             pad_mode,
                             pad_value,
                             allow_negative_pads));
    network base_network(engine, base_topology, config);
    base_network.set_input_data("input", input);
    auto base_output = base_network.execute().at("border").get_memory();
    cldnn::mem_lock<ov::float16> base_output_ptr(base_output, get_test_stream());

    ASSERT_TRUE(!memcmp(target_output_ptr.data(), base_output_ptr.data(), sizeof(ov::float16) * mult(sh_out)));

    for (auto b = 0; b < sh_out[0]; ++b) {
        for (auto f = 0; f < sh_out[1]; ++f) {
            for (auto y = 0; y < sh_out[2]; ++y) {
                const auto output_off = ((b * sh_out[1] + f) * sh_out[2] + y);
                ASSERT_GE(output_off, 0);

                if (b < cd_lt[0] || b >= sh_out[0] - cd_rb[0] ||
                    f < cd_lt[1] || f >= sh_out[1] - cd_rb[1] ||
                    y < cd_lt[2] || y >= sh_out[2] - cd_rb[2]) {
                    ASSERT_EQ(target_output_ptr[output_off], pad_value);
                } else {
                    const auto input_off  = (((b - cd_lt[0]) * sh_in[1] + f - cd_lt[1]) * sh_in[2] + y - cd_lt[2]);
                    ASSERT_GE(input_off, 0);
                    ASSERT_EQ(target_output_ptr[output_off], input_data[input_off]);
                }
            }
        }
    }
}
};  // namespace
