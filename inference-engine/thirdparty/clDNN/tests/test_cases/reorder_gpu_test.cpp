// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/reshape.hpp>
#include "cldnn/primitives/reorder.hpp"
#include "cldnn/primitives/crop.hpp"
#include <cldnn/primitives/data.hpp>

#include <cmath>
#include <limits>

using namespace cldnn;
using namespace ::tests;
using namespace testing;

static void compare_bfyx2blocked_with_ref(const std::string& kernel_name,
    const data_types input_data_type, const data_types output_data_type,
    cldnn::format input_format, cldnn::format output_format,
    int32_t b_in, int32_t f_in, int32_t x_in, int32_t y_in, int32_t z_in = 0, int32_t w_in = 0) {
    auto& engine = get_test_engine();

    tensor ts;
    if (input_format.dimension() == 4) {
        ts = { b_in, f_in, x_in, y_in };
    }
    else if (input_format.dimension() == 5) {
        ts = { b_in, f_in, x_in, y_in, z_in };
    }
    else {
        ts = { b_in, f_in, x_in, y_in, z_in, w_in };
    }

    auto input = engine.allocate_memory({ input_data_type, input_format, ts });
    layout output_layout(output_data_type, output_format, ts);

    if (input_data_type == data_types::i8) {
        mem_lock<uint8_t> input_ptr{input, get_test_stream()};
        unsigned char i = 1;
        for (auto it = input_ptr.begin(); it != input_ptr.end(); ++it)
        {
            *it = (i++);
            if (i > 100) {
                i = 1;
            }
        }
    } else {
        mem_lock<float> input_ptr{input, get_test_stream()};
        float i = 1.f;
        for (auto it = input_ptr.begin(); it != input_ptr.end(); ++it)
        {
            *it = (i);
            i += 1.f;
        }
    }

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", output_layout));

    // run on reference(reorder_data) kernel
    cldnn::build_options options_ref;
    cldnn::implementation_desc reorder_ref = { output_format, "reorder_data" };
    options_ref.set_option(cldnn::build_option::force_implementations({ {"reorder", reorder_ref} }));

    network network_ref(engine, topology, options_ref);
    network_ref.set_input_data("input", input);

    std::map<cldnn::primitive_id, cldnn::network_output> outputs_ref;

    outputs_ref = network_ref.execute();
    cldnn::event::ptr e1 = outputs_ref.at("reorder").get_event();
    e1->wait();

    auto output_ref = outputs_ref.begin()->second.get_memory();
    mem_lock<uint8_t> output_ref_ptr{output_ref, get_test_stream()};

    // run on optimized kernel
    cldnn::build_options options;
    cldnn::implementation_desc reorder_optimized = { output_format, kernel_name };
    options.set_option(cldnn::build_option::force_implementations({ {"reorder", reorder_optimized} }));

    network network(engine, topology, options);
    network.set_input_data("input", input);

    std::map<cldnn::primitive_id, cldnn::network_output> outputs;

    outputs = network.execute();
    cldnn::event::ptr e2 = outputs.at("reorder").get_event();
    e2->wait();

    auto output = outputs.begin()->second.get_memory();
    mem_lock<uint8_t> output_ptr{output, get_test_stream()};

    // compare results
    const size_t output_size = output_ref_ptr.size();
    for (size_t i = 0; i < output_size; i++)
    {
        EXPECT_EQ(output_ref_ptr[i], output_ptr[i]);
    }
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv32_to_bfyx_f32) {
    // b_fs_yx_fsv32 -> bfyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfyx, 3, 64 + 5, 16 + 11, 3);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfyx, 3, 96 - 12, 16 + 4, 3);
    // b_fs_zyx_fsv32 -> bfzyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfzyx, 3, 64 + 9, 16 - 1, 2, 8);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfzyx, 2, 64 + 30, 16 + 1, 3, 4);
    // incremental dims
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfzyx, 2, 64 + 4, 24 - 1, 3);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfwzyx, 2, 64 + 2, 32 - 3, 4);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv32, format::bfwzyx, 1, 96 + 10, 32 - 3, 4, 3);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv32_to_bfyx_different_datatype) {
    // f32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::u8, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 8 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i64, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 16 + 2, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f16, format::b_fs_yx_fsv32, format::bfyx, 1, 64, 16 + 1, 2);
    // i32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i8, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 8 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i64, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 16 + 2, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::f16, format::b_fs_yx_fsv32, format::bfyx, 1, 64, 16 + 1, 2);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv16_to_bfyx_f32) {
    // u-net
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 1, 64, 388, 388);
    // b_fs_yx_fsv16 -> bfyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 3, 48 + 1, 16, 3);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 2, 32 - 1, 24 - 1, 3);
    // b_fs_zyx_fsv16 -> bfzyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx, 5, 48 - 1, 16, 3, 8);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx, 2, 32 + 1, 24 - 1, 3, 17);
    // incremental dims
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfzyx, 3, 32 - 1, 24 - 1, 3);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfwzyx, 4, 16 + 1, 32 - 3, 4);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx, 3, 16 + 2, 32 - 3, 4, 9);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv16_to_bfyx_different_datatype) {
    // f32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::u8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i32, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i64, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f16, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
    // i32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::u8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i64, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::f16, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_blocked_f32) {
    // bfyx_to_b_fs_yx_fsv4
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv4, 4, 32, 16, 4);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv4, 3, 32 + 2, 32 + 3, 4);
    // bfyx_to_b_fs_yx_fsv16
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 2, 48, 8, 4);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2);
    // bfyx to b_fs_yx_fsv32
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv32, 2, 64, 64, 4);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv32, 4, 32 + 6, 96 - 4, 2);
    // bfyx to fs_b_yx_fsv32
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::fs_b_yx_fsv32, 2, 64, 8, 4);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::fs_b_yx_fsv32, 3, 64 + 5, 8 + 7, 2);
    // bfzyx to b_fs_zyx_fsv16
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv16, 2, 48, 8, 4, 4);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv16, 3, 32 + 5, 16 + 7, 2, 2);
    // bfzyx to b_fs_zyx_fsv32
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv32, 2, 64, 8, 4, 4);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv32, 3, 64 + 5, 8 + 7, 2, 2);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f32) {
    // bfyx to double blocked format (bs_fs_yx_bsv16_fsv16)
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32, 48, 8, 4);                    // no
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32 + 2, 48, 16, 4);               // b
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32, 48 + 5, 16, 4);               // f
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32, 48, 48 + 3, 4);               // x
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32 + 2, 48 + 3, 16 + 1, 4);       // b-f-x
    // bfzyx to double blocked format (bs_fs_zyx_bsv16_fsv16)
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32, 48, 8, 4, 16);              // no
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32 + 2, 48, 16, 4, 2);          // b
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32, 48 + 5, 16, 4, 3);          // f
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32, 48, 48 + 3, 4, 4);          // x
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32 + 2, 48 + 3, 16 + 1, 4, 2);  // b-f-x
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_blocked_format_different_datatype) {
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f16, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::i8, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::i64, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2);
}

TEST(reorder_gpu_optimization, bfyx_to_fsv16_without_f_remainder) {
    auto& engine = get_test_engine();
    const int32_t b_in = 1;
    const int32_t f_in = 8 * 4;
    const int32_t y_in = 4;
    const int32_t x_in = 8 * 2;

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { b_in,f_in,x_in,y_in } });
    layout output_layout(data_types::f32, format::b_fs_yx_fsv16, { b_in,f_in,x_in,y_in });

    // Set incremental input value
    mem_lock<float> input_ptr{input, get_test_stream()};
    float i = 0.f;
    for (auto it = input_ptr.begin(); it != input_ptr.end(); ++it)
    {
        *it = (i++);
    }

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    mem_lock<float> output_ptr{output, get_test_stream()};

    auto get_fsv16_index = [](int32_t /* b_size */, int32_t f_size, int32_t y_size, int32_t x_size,
        int32_t b, int32_t f, int32_t y, int32_t x) {
            const int32_t alignment = 16;
            const int32_t fs = f / alignment;
            const int32_t fsv = f % alignment;

            const int32_t x_pitch = alignment;
            const int32_t y_pitch = x_pitch * (x_size);
            const int32_t fs_pitch = y_pitch * (y_size);
            const int32_t b_pitch = fs_pitch * ((f_size)/alignment);

            const int32_t output_offset = (b * b_pitch) + (fs * fs_pitch) + (y * y_pitch) + (x * x_pitch) + (fsv);

            return output_offset;
    };

    int32_t linear_index = 0;
    for (int32_t b = 0; b < b_in; b++) {
        for (int32_t f = 0; f < f_in; f++) {
            for (int32_t y = 0; y < y_in; y++) {
                for (int32_t x = 0; x < x_in; x++) {
                    int32_t b_fs_yx_fsv16_index = get_fsv16_index(b_in, f_in, y_in, x_in, b, f, y, x);
                    EXPECT_FLOAT_EQ(input_ptr[linear_index++], output_ptr[b_fs_yx_fsv16_index]);
                }
            }
        }
    }

}

TEST(reorder_gpu_f32, basic) {
    //  Input               : yxfb:2x2x2x2
    //  Output              : bfyx:2x2x2x2
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Output:
    //  b0 f0:  1    2
    //  b0 f0:  3    4
    //
    //  b0 f1:  5    6
    //  b0 f1:  7    8
    //
    //  b1 f0:  0    0
    //  b1 f0: 0.5 -0.5
    //
    //  b1 f1: 1.5  5.2
    //  b1 f1: 12    8
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx,{ 2,2,2,2 });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = {
        1.0f,  2.0f,
        3.0f,  4.0f,

        5.0f,  6.0f,
        7.0f,  8.0f,

        0.0f,  0.0f,
        0.5f, -0.5f,

        1.5f,  5.2f,
        12.0f, 8.0f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(reorder_gpu_f32, basic_subtract) {
    //  Input               : 2x2x2x2
    //  Output              : 2x2x2x2
    //  Subtract            : 1x2x2x2 (only first batch is taken into consideration)
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Subtract:
    //  f0: b0:  1    1.5
    //  f0: b0:  2    2.5
    //  f1: b0:  4    3
    //  f1: b0:  2    1
    //
    //
    //  Output:
    //  b0 f0:  0    0.5
    //  b0 f0:  1    1.5
    //
    //  b0 f1:  1    3
    //  b0 f1:  5    7
    //
    //  b1 f0: -1   -1.5
    //  b1 f0: -1.5 -3
    //
    //  b1 f1: -2.5  2.2
    //  b1 f1: 10    7
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32,  format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout( data_types::f32, format::bfyx, {2,2,2,2} );
    auto subtract = engine.allocate_memory({ data_types::f32, format::byxf, { 1, 2, 2, 2 } });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    set_values(subtract, {
        1.0f,  4.0f,      1.5f,  3.0f,
        2.0f,  2.0f,      2.5f,  1.0f,
    });

    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("subtract", subtract->get_layout()),
        reorder("reorder", "input", output_layout, "subtract"));

    network network(engine, topology);
    network.set_input_data("input", input);
    network.set_input_data("subtract", subtract);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = { 0.0f,  0.5f,
                          1.0f,  1.5f,

                          1.0f,  3.0f,
                          5.0f,  7.0f,

                         -1.0f, -1.5f,
                         -1.5f, -3.0f,

                         -2.5f,  2.2f,
                         10.0f,  7.0f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(reorder_gpu_f32, basic_subtract_value) {
    //  Values_to_subtract  : 2
    //  Input               : 2x2x2x2
    //  Output              : 2x2x2x2
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  subtract values
    //  f0: 0.5
    //  f1: 2.5
    //
    //  Output:
    //  b0 f0:  0.5  1.5
    //  b0 f0:  2.5  3.5
    //
    //  b0 f1:  2.5  3.5
    //  b0 f1:  4.5  5.5
    //
    //  b1 f0: -0.5 -0.5
    //  b1 f0:  0.0 -1.0
    //
    //  b1 f1: -1.0  2.7
    //  b1 f1:  9.5  5.5
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx,{ 2,2,2,2 });
    std::vector<float> subtract_val = { 0.5, 2.5 };

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()), reorder("reorder", "input", output_layout, subtract_val));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = { 0.5f, 1.5f,
                          2.5f, 3.5f,

                          2.5f, 3.5f,
                          4.5f, 5.5f,

                         -0.5f, -0.5f,
                          0.0f, -1.0f,

                         -1.0f,  2.7f,
                          9.5f,  5.5f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reorder_gpu_f16, basic_subtract_f32_output_f32) {
    //  Input               : 2x2x2x2 (FP16)
    //  Output              : 2x2x2x2 (FP32)
    //  Subtract            : 1x2x2x2 (FP32, only first batch is taken into consideration)
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Subtract (FP32 - converted internally to FP16 before subtraction):
    //  f0: b0:  1    1.5
    //  f0: b0:  2    2.5
    //  f1: b0:  4    3
    //  f1: b0:  2    1
    //
    //
    //  Output:
    //  b0 f0:  0    0.5
    //  b0 f0:  1    1.5
    //
    //  b0 f1:  1    3
    //  b0 f1:  5    7
    //
    //  b1 f0: -1   -1.5
    //  b1 f0: -1.5 -3
    //
    //  b1 f1: -2.5  2.2
    //  b1 f1: 10    7
    //

    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx,{ 2,2,2,2 });
    auto subtract = engine.allocate_memory({ data_types::f32, format::byxf, { 1, 2, 2, 2 } });

    set_values(input, {
        half_t(1.f), half_t(0.f),
        half_t(5.f), half_t(1.5f),

        half_t(2.f), half_t(0.f),
        half_t(6.f), half_t(5.2f),

        half_t(3.f), half_t(0.5f),
        half_t(7.f), half_t(12.f),

        half_t(4.f), half_t(-0.5f),
        half_t(8.f), half_t(8.f)
    });

    set_values(subtract, {
        1.0f,  4.0f,      1.5f,  3.0f,
        2.0f,  2.0f,      2.5f,  1.0f,
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("subtract", subtract));
    topology.add(reorder("reorder", "input", output_layout, "subtract"));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = { 0.0f,  0.5f,
                          1.0f,  1.5f,

                          1.0f,  3.0f,
                          5.0f,  7.0f,

                         -1.0f, -1.5f,
                         -1.5f, -3.0f,

                         -2.5f,  2.2f,
                         10.0f,  7.0f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reorder_gpu_f16, basic_subtract_value) {
    //  Values_to_subtract  : 2
    //  Input               : 2x2x2x2 (FP16)
    //  Output              : 2x2x2x2 (FP16)
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  subtract values (FP32 - converted internally to FP16 before subtraction)
    //  f0: 0.5
    //  f1: 2.5
    //
    //  Output:
    //  b0 f0:  0.5  1.5
    //  b0 f0:  2.5  3.5
    //
    //  b0 f1:  2.5  3.5
    //  b0 f1:  4.5  5.5
    //
    //  b1 f0: -0.5 -0.5
    //  b1 f0:  0.0 -1.0
    //
    //  b1 f1: -1.0  2.7
    //  b1 f1:  9.5  5.5
    //

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f16, format::bfyx,{ 2,2,2,2 });
    std::vector<float> subtract_val = { 0.5, 2.5 };

    set_values(input, {
        half_t(1.f), half_t(0.f),
        half_t(5.f), half_t(1.5f),

        half_t(2.f), half_t(0.f),
        half_t(6.f), half_t(5.2f),

        half_t(3.f), half_t(0.5f),
        half_t(7.f), half_t(12.f),

        half_t(4.f), half_t(-0.5f),
        half_t(8.f), half_t(8.f)
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", "input", output_layout, subtract_val));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    half_t answers[16] = { half_t(0.5f), half_t(1.5f),
                           half_t(2.5f), half_t(3.5f),

                           half_t(2.5f), half_t(3.5f),
                           half_t(4.5f), half_t(5.5f),

                           half_t(-0.5f), half_t(-0.5f),
                           half_t(0.f), half_t(-1.f),

                           half_t(-1.f), half_t(2.7f),
                           half_t(9.5f), half_t(5.5f)
    };

    cldnn::mem_lock<half_t> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_TRUE(are_equal(static_cast<uint16_t>(answers[i]), static_cast<uint16_t>(output_ptr[i])));
    }
}

TEST(reorder_gpu, basic_convert_f16_f32_f16) {
    //  Converts entire unambiguous range of FP16 numbers to FP32 and back.
    //
    //  Input               : 2x2x15873x1 (FP16)
    //  Intermediate        : 1x2x2x15873 (FP32) {different mem format but the same ordering because batch is 1}
    //  Output              : 2x2x15673x1 (FP16)
    //
    //  Output is expected to contain the same value as input in range of indices from 0x0000 to 0xF801.
    //

    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    std::vector<half_t> expected_values;
    expected_values.resize(0xF804);
    for (int i = 0; i < 0x7C00; ++i)
        expected_values[i] = half_t(i, 0);          // norms/denorms/zero (positive).
    for (int i = 0x7C00; i < 0xF800; ++i)
        expected_values[i] = half_t(i + 0x0400, 0); // norms/denorms (negative).
    expected_values[0x7C00] = half_t(0x0000, 0);    // NOTE: do not do final test for negative 0 (-0).
    // Special values.
    expected_values[0xF800] = half_t(0x7C00, 0);    // +infinity
    expected_values[0xF801] = half_t(0xFC00, 0);    // -infinity
    // Special values (ambiguous ones).
    expected_values[0xF802] = half_t(0x8000, 0);    // -0
    expected_values[0xF803] = half_t(0xFC12, 0);    // A NaN (sample: -NaN.0x12).

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb, { 1, static_cast<int32_t>(expected_values.size()) / 4, 2, 2 } });
    layout interm_layout( data_types::f32, format::byxf, { 1, static_cast<int32_t>(expected_values.size()) / 4, 2, 2 });
    auto output_layout = input->get_layout();

    set_values(input, expected_values);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder_f16_f32", "input", interm_layout));
    topology.add(reorder("reorder_f32_f16", "reorder_f16_f32", output_layout));

    network network(
        engine,
        topology,
        build_options{
            build_option::outputs({"reorder_f16_f32", "reorder_f32_f16"})
        });

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(2));
    EXPECT_TRUE(outputs.find("reorder_f16_f32") != outputs.end());
    EXPECT_TRUE(outputs.find("reorder_f32_f16") != outputs.end());

    auto interm = outputs.at("reorder_f16_f32").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());

    // Sample positive.
    EXPECT_TRUE(are_equal(interm_ptr[0x3400], 0.25f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3800], 0.5f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3C00], 1.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4000], 2.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4400], 4.0f));
    // Sample negative.
    EXPECT_TRUE(are_equal(interm_ptr[0x3400 + 0x7C00], -0.25f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3800 + 0x7C00], -0.5f));
    EXPECT_TRUE(are_equal(interm_ptr[0x3C00 + 0x7C00], -1.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4000 + 0x7C00], -2.0f));
    EXPECT_TRUE(are_equal(interm_ptr[0x4400 + 0x7C00], -4.0f));
    // Special values.
    EXPECT_TRUE(are_equal(interm_ptr[0xF800], std::numeric_limits<float>::infinity()));
    EXPECT_TRUE(are_equal(interm_ptr[0xF801], -std::numeric_limits<float>::infinity()));
    EXPECT_TRUE(are_equal(interm_ptr[0xF802], -0.0f));
    EXPECT_TRUE(std::isnan(interm_ptr[0xF803]));

    auto output = outputs.at("reorder_f32_f16").get_memory();
    cldnn::mem_lock<half_t> output_ptr(output, get_test_stream());
    for (int i = 0; i < 0xF802; ++i) // NOTE: do not test for possibly ambiguous values of floating point (-0, NaNs).
    {
        EXPECT_TRUE(are_equal(static_cast<uint16_t>(expected_values[i]), static_cast<uint16_t>(output_ptr[i])));
    }
}

TEST(reorder_gpu, basic_convert_int8) {

    auto& engine = get_test_engine();
    layout in_layout = { type_to_data_type<float>::value,format::byxf,{ 1,1,3,3 } };
    layout byte_layout = { type_to_data_type<int8_t>::value, format::bfyx,{ 1,1,3,3 } };
    std::initializer_list<float> input_f = { 1.0f, -2.5f, 3.1f, -4.0f, 5.03f, -6.99f, 7.0f, -8.0f, 9.0f };
    std::list<float> final_results = { 1.0f, -3.0f, 3.0f, -4.0f, 5.0f, -7.0f, 7.0f, -8.0f, 9.0f };

    // Allocate memory for input image.
    auto input_memory = engine.allocate_memory(in_layout);
    set_values(input_memory, input_f);

    // Create input_layout description
    // "input" - is the primitive id inside topology
    input_layout input("input", in_layout);

    topology topology(
        // 1. input layout primitive.
        input,
        // 2. reorder primitive with id "reorder_input"
        reorder("reorder_input",
            // input primitive for reorder (implicitly converted to primitive_id)
            input,
            // output layout for reorder
            byte_layout),
        reorder("reorder2", "reorder_input", in_layout)
    );

    network network(
        engine,
        topology,
        build_options{
            build_option::outputs({ "reorder_input", "reorder2"})
        });

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    auto interm = outputs.at("reorder2").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());
    unsigned int cntr = 0;
    for (const auto& exp : final_results)
    {
        EXPECT_EQ(exp, interm_ptr[cntr++]);
    }
}

TEST(reorder_gpu, basic_convert_uint8rgbabyxf_to_fp32_bfyx) {
	//  Converts an ARGB(uint8) image to common clDNN input of bfyx FP32
	//
	//  Input               : 1x5x5x4 (UINT8)
	//  Intermediate        : 1x4x5x5 (FP32) {different mem format and ordering}
	//  Output              : 1x3x5x5 (FP32) {using crop layer to reduce feature dimention and drop A from RGBA}
	//
	//  Output is expected to contain the same value as input
	//
	const int kernel_size = 5;
	const int feature_size = 4;
	auto& engine = get_test_engine();

	if (!engine.get_device_info().supports_fp16)
	{
		std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
		EXPECT_EQ(1, 1);
		return;
	}

	std::initializer_list<uint8_t> input_i8 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
		55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36,
		101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
		155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136,
		255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 237, 236
	};

	layout in_layout = { type_to_data_type<uint8_t>::value,format::byxf,{ 1,4,kernel_size,kernel_size } };
	layout output_layout = { type_to_data_type<float>::value, format::bfyx, {1,4,kernel_size,kernel_size } };

	// Allocate memory for input image.
	auto input_memory = engine.allocate_memory(in_layout);
	set_values(input_memory, input_i8);

    // Create input_layout description
	// "input" - is the primitive id inside topology
	input_layout input("input", in_layout);

	// Create topology object with 2 primitives
	topology topology(
		// 1. input layout primitive.
		input,
		// 2. reorder primitive with id "reorder_input"
		reorder("reorder_input",
			// input primitive for reorder (implicitly converted to primitive_id)
			input,
			// output layout for reorder
			output_layout)
	);

	tensor crop_reference_input_tensor(spatial(kernel_size, kernel_size), batch(1), feature(4 - 1));
	tensor crop_offset_tensor(spatial(0, 0), batch(0), feature(0));
	padding output_padding = padding({ 0,0,0,0 }, { 0,0,0,0 }, 0);
	topology.add(
		// cropping primitive with id "crop1"
		crop("crop",
			"reorder_input",    // primitive id of the cropping input
			crop_reference_input_tensor,  // input tensor
			crop_offset_tensor,    // bias primitive id
			output_padding
		)
	);

	network network(
		engine,
		topology,
        build_options{
			build_option::outputs({ "reorder_input", "crop" })
		});

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(2));
    EXPECT_TRUE(outputs.find("reorder_input") != outputs.end());
    EXPECT_TRUE(outputs.find("crop") != outputs.end());

    auto interm = outputs.at("reorder_input").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());
    auto interm_size = outputs.at("reorder_input").get_memory()->count();
    EXPECT_EQ(interm_size,(size_t) (1*feature_size*kernel_size*kernel_size));

    // Sample positive.
    EXPECT_TRUE(are_equal(interm_ptr[0], 1.0f));
    size_t source_index = 0;
    size_t target_index = 0;
    std::vector<uint8_t> testinput;// This will be used to direct access elements of test input in the next test
    for (auto it = input_i8.begin(); it < input_i8.end(); it++)
    {

        uint8_t val = *it;
        testinput.push_back(val); // This will be used to direct access elements of test input in the next test
        size_t current_feature = source_index % feature_size;
        size_t current_x = (source_index / feature_size) % kernel_size;
        size_t current_y = (source_index / (feature_size * kernel_size));
        target_index = current_x + current_y*kernel_size + current_feature*(kernel_size*kernel_size);
        EXPECT_TRUE(are_equal(val, interm_ptr[target_index]));
        source_index++;
    }

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_size = output->count();
    EXPECT_EQ(output_size,(size_t) (1 * (feature_size-1)*kernel_size*kernel_size));

    for (target_index = 0; target_index < output_size; target_index++)
    {
        float output_val = output_ptr[target_index];
        int current_x = target_index % kernel_size;
        int current_y = (target_index / kernel_size) % kernel_size;
        size_t current_feature = target_index / (kernel_size*kernel_size);

        source_index = current_x*feature_size + current_y*(kernel_size*feature_size) + current_feature;
        EXPECT_TRUE(are_equal(output_val, testinput[source_index]));
    }

}

TEST(reorder_gpu_f32, basic_yxfb_to_bfyx_input_padding)
{
    //  Input               : yxfb:2x2x2x2
    //  Output              : bfyx:2x2x2x2
    //
    //  Input:
    //  b0 f0:  1    2
    //  b0 f0:  3    4
    //
    //  b0 f1:  5    6
    //  b0 f1:  7    8
    //
    //  b1 f0:  0    0
    //  b1 f0: 0.5 -0.5
    //
    //  b1 f1: 1.5  5.2
    //  b1 f1: 12    8
    //
    //  Output:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx, { 2,2,2,2 });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", input->get_layout().format, input->get_layout().data_type, "", reorder_mean_mode::subtract, padding{ { 0, 0, 1, 2 }, 0 }),
        reorder("reorder2", "reorder", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder2");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = {
        1.0f,  2.0f,
        3.0f,  4.0f,

        5.0f,  6.0f,
        7.0f,  8.0f,

        0.0f,  0.0f,
        0.5f, -0.5f,

        1.5f,  5.2f,
        12.0f, 8.0f
    };
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(reorder_gpu_f32, basic_bfyx_to_yxfb_input_padding)
{
    //  Input               : bfyx:2x2x2x2
    //  Output              : yxfb:2x2x2x2
    //
    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Output:
    //  b0 f0:  1    2
    //  b0 f0:  3    4
    //
    //  b0 f1:  5    6
    //  b0 f1:  7    8
    //
    //  b1 f0:  0    0
    //  b1 f0: 0.5 -0.5
    //
    //  b1 f1: 1.5  5.2
    //  b1 f1: 12    8
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::yxfb, { 2,2,2,2 });

    set_values(input, {
        1.0f,  2.0f,
        3.0f,  4.0f,

        5.0f,  6.0f,
        7.0f,  8.0f,

        0.0f,  0.0f,
        0.5f, -0.5f,

        1.5f,  5.2f,
        12.0f, 8.0f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", input->get_layout().format, input->get_layout().data_type, "", reorder_mean_mode::subtract, padding{ { 0, 0, 2, 1 }, 0 }),
        reorder("reorder2", "reorder", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder2");

    auto output = outputs.begin()->second.get_memory();

    float answers[16] = {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    };
    std::vector<float> out;
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        out.push_back(output_ptr[i]);
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(reorder_gpu_f32, basic_bfyx_to_bfzyx)
{
    //  Input               : bfyx:2x2x2x2
    //  Output              : bfzyx:2x2x1X2x2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", format::bfzyx, data_types::f32));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    EXPECT_TRUE(output->get_layout().format == format::bfzyx);
    auto sizes = output->get_layout().size;
    EXPECT_TRUE(sizes.batch[0] == 2);
    EXPECT_TRUE(sizes.feature[0] == 2);
    EXPECT_TRUE(sizes.spatial[0] == 2);
    EXPECT_TRUE(sizes.spatial[1] == 2);
    EXPECT_TRUE(sizes.spatial[2] == 1);

    float answers[16] = {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(reorder_gpu_f32, basic_yxfb_to_bfzyx)
{
    //  Input               : yxfb:2x2x2x2
    //  Output              : bfzyx:2x2x1X2x2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 2, 2, 2, 2 } });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", format::bfzyx, data_types::f32));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    EXPECT_TRUE(output->get_layout().format == format::bfzyx);
    auto sizes = output->get_layout().size;
    EXPECT_TRUE(sizes.batch[0] == 2);
    EXPECT_TRUE(sizes.feature[0] == 2);
    EXPECT_TRUE(sizes.spatial[0] == 2);
    EXPECT_TRUE(sizes.spatial[1] == 2);
    EXPECT_TRUE(sizes.spatial[2] == 1);

    float answers[16] = {
        1.0f,  2.0f,
        3.0f,  4.0f,

        5.0f,  6.0f,
        7.0f,  8.0f,

        0.0f,  0.0f,
        0.5f, -0.5f,

        1.5f,  5.2f,
        12.0f, 8.0f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(reorder_gpu_f32, basic_bfzyx_to_bfyx)
{
    //  Input               : bfzyx:2x2x2x2x2
    //  Output              : bfyx:2x2x4x2

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 2, 2, 2, 2 } });

    set_values(input, {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f,

        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", format::bfyx, data_types::f32));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    EXPECT_TRUE(output->get_layout().format == format::bfyx);
    auto sizes = output->get_layout().size;
    EXPECT_TRUE(sizes.batch[0] == 2);
    EXPECT_TRUE(sizes.feature[0] == 2);
    EXPECT_TRUE(sizes.spatial[0] == 2);
    EXPECT_TRUE(sizes.spatial[1] == 4);
    EXPECT_TRUE(sizes.spatial[2] == 1);

    float answers[32] = {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f,

        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, -0.5f,
        8.f, 8.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(reorder_gpu_opt, basic_remove_redundant)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 2, 2, 1 } });
    topology tpl{
        input_layout("in", in->get_layout()),
        reorder("r1", "in", format::bfyx, data_types::f32),
        reorder("r2", "r1", format::yxfb, data_types::f32)
    };

    build_options opts;
    opts.set_option(build_option::optimize_data(true));

    network net(engine, tpl, opts);
    net.set_input_data("in", in);
    auto outputs = net.execute();
    auto executed_primitives = net.get_executed_primitives();

    EXPECT_TRUE(executed_primitives.count("r1") == 0);
    ASSERT_TRUE(outputs.count("r2") == 1);
    EXPECT_TRUE(outputs.at("r2").get_memory()->get_layout().format == format::yxfb);
}

TEST(reorder_gpu_opt, remove_redundant_activation_fuse)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 1, 2, 1 } });
    set_values(in, { -1.0f, -1.0f });
    memory::ptr scale_mem = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{1, 1, 1, 1 } });
    set_values(scale_mem, { 2.0f });
    topology tpl{
        input_layout("in", in->get_layout()),
        reorder("r1", "in", format::bfyx, data_types::f32),
        activation("relu", "r1", activation_func::relu_negative_slope, {0.01f, 0.0f}),
        data("scale_data", scale_mem),
        scale("output", "relu", "scale_data")
    };

    build_options opts;
    opts.set_option(build_option::optimize_data(true));

    network net(engine, tpl, opts);
    net.set_input_data("in", in);
    auto outputs = net.execute();
    cldnn::mem_lock<float> out_ptr(outputs.begin()->second.get_memory(), get_test_stream());
    EXPECT_FLOAT_EQ(out_ptr[0], -0.02f);
    EXPECT_FLOAT_EQ(out_ptr[1], -0.02f);
}

TEST(reorder_gpu_opt, basic_remove_redundant_output_due_to_implicit_reorders)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::yxfb, tensor{ 1, 2, 2, 1 } });
    memory::ptr weights = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 2, 2, 1 } });
    topology tpl{
        input_layout("in", in->get_layout()),
        convolution("conv", "in",{ "weights" }),
        data("weights", weights),
        reorder("r1", "conv", format::bfyx, data_types::f32) //optimize data should add conversion from yxfb to bfyx and 'conv' should output data in bfyx as well (IE case)
    };

    build_options opts;

    //we need to check if r1 will be successfully opimized and still we should be able to query for r1's output which should point to conv's output (note conv cannot be marked as output in this case)
    opts.set_option(build_option::outputs({ "r1" }));
    opts.set_option(build_option::optimize_data(true));

    network net(engine, tpl, opts);
    net.set_input_data("in", in);
    auto outputs = net.execute();

    EXPECT_TRUE(outputs.count("conv") == 0);
    ASSERT_TRUE(outputs.count("r1") == 1);
    EXPECT_TRUE(outputs.at("r1").get_memory()->get_layout().format == format::bfyx);
}

TEST(reorder_gpu_opt, basic_remove_redundant_due_to_implicit_reorders)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::yxfb, tensor{ 1, 2, 2, 1 } });
    memory::ptr weights = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 2, 2, 1 } });
    topology tpl{
        input_layout("in", in->get_layout()),
        convolution("conv", "in",{ "weights" }),
        data("weights", weights),
        reorder("r1", "conv", format::bfyx, data_types::f32), //optimize data should add conversion from yxfb to bfyx and 'conv' should output data in bfyx as well (IE case)
        softmax("output", "r1")
    };

    build_options opts;
    opts.set_option(build_option::optimize_data(true));

    network net(engine, tpl, opts);
    net.set_input_data("in", in);
    auto outputs = net.execute();
    auto executed_primitives = net.get_executed_primitives();

    //remove redundant reorder optimization should remove r1 node
    EXPECT_TRUE(executed_primitives.count("r1") == 0);
    //all pirmitives in this test needs to be executed
    ASSERT_TRUE(outputs.count("output") == 1);
    EXPECT_TRUE(outputs.at("output").get_memory()->get_layout().format == format::bfyx);
}

TEST(reorder_gpu_opt, non_trivial_remove_redundant)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::yxfb, tensor{ 1, 1, 5, 2 } });
    topology tpl{
        input_layout("in", in->get_layout()),
        reorder("r1", "in", format::bfyx, data_types::f32)
    };

    build_options opts;

    opts.set_option(build_option::optimize_data(true));

    network net(engine, tpl, opts);
    net.set_input_data("in", in);
    auto outputs = net.execute();
    auto executed_primitives = net.get_executed_primitives();
    auto all_primitives = net.get_all_primitives();

    ASSERT_TRUE(executed_primitives.count("in") == 1);
    //ASSERT_TRUE(all_primitives.at("r1") == "_optimized_");
    EXPECT_TRUE(executed_primitives.at("in") != outputs.at("r1").get_event());
    ASSERT_TRUE(outputs.count("r1") == 1);
    EXPECT_TRUE(outputs.at("r1").get_memory()->get_layout().format == format::bfyx);
}

TEST(reorder_gpu_opt, mean_mul)
{
    auto& engine = get_test_engine();

    memory::ptr in  = engine.allocate_memory({ data_types::i8, format::bfyx, tensor{ 1, 3, 1, 2 } });
    memory::ptr mul = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{1, 3, 1, 2 } });

    set_values<char>(in,
    { 1, 2,
      3, 4,
      5, 6 });
    set_values<float>(mul,
    { 0.5f, 2.5f, -5.0f, 4.3f, 1.2f, -3.5f });

    topology tpl{
        input_layout("in", in->get_layout()),
        data("mul",mul),
        reorder("r1", "in", format::bfyx, data_types::f32,"mul", reorder_mean_mode::mul)
    };

    float answers[] = { 0.5f, 5.0f, -15.0f, 17.2f, 6.0f, -21.0f };
    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network net(engine, tpl, opts);
    net.set_input_data("in", in);

    auto outputs = net.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> ptr(output, get_test_stream());
    float* a_ptr = answers;
    for (auto& val : ptr)
        EXPECT_FLOAT_EQ(*(a_ptr++), val);;

}

TEST(reorder_gpu_opt, mean_div)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::i8, format::bfyx, tensor{ 1, 3, 1, 2 } });
    memory::ptr mul = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 3, 1, 2 } });

    set_values<char>(in,
    { 1, 2,
      3, 4,
      5, 6 });
    set_values<float>(mul,
    { 0.5f, 2.0f, -3.0f, 8.0f, 1.25f, -3.0f });

    topology tpl{
        input_layout("in", in->get_layout()),
        data("mul",mul),
        reorder("r1", "in", format::bfyx, data_types::f32,"mul", reorder_mean_mode::div)
    };

    float answers[] = { 2.0f, 1.0f, -1.0f, 0.5f, 4.0f, -2.0f };
    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network net(engine, tpl, opts);
    net.set_input_data("in", in);

    auto outputs = net.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> ptr(output, get_test_stream());
    float* a_ptr = answers;
    for (auto& val : ptr)
        EXPECT_FLOAT_EQ(*(a_ptr++), val);;

}

TEST(reorder_gpu_opt, mean_mul_val)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::i8, format::bfyx, tensor{ 1, 3, 1, 2 } });

    set_values<char>(in,
    { 1, 2,
      3, 4,
      5, 60 });
    std::vector<float> mul_val = { 2.0f, 0.5f, 10.0f };
    topology tpl{
        input_layout("in", in->get_layout()),
        reorder("r1", "in", format::bfyx, data_types::f32, mul_val, reorder_mean_mode::mul)
    };

    float answers[] = { 2.0f, 4.0f, 1.5f, 2.0f, 50.0f, 600.0f };
    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network net(engine, tpl, opts);
    net.set_input_data("in", in);

    auto outputs = net.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> ptr(output, get_test_stream());
    float* a_ptr = answers;
    for (auto& val : ptr)
        EXPECT_FLOAT_EQ(*(a_ptr++), val);;
}

TEST(reorder_gpu_opt, mean_mul_val_float_to_int)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 3, 1, 2 } });

    set_values<float>(in,
    { 0.6f, 1.5f,
      3.0f, 4.2f,
      5.0f, 60.0f });
    std::vector<float> mul_val = { 1.4f, 0.5f, 5.0f };
    topology tpl{
        input_layout("in", in->get_layout()),
        reorder("r1", "in", format::bfyx, data_types::i8, mul_val, reorder_mean_mode::mul)
    };

    char answers[] = { 1, 2, 2, 2, 25, 127 };
    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network net(engine, tpl, opts);
    net.set_input_data("in", in);

    auto outputs = net.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<char> ptr(output, get_test_stream());
    char* a_ptr = answers;
    for (auto& val : ptr)
        EXPECT_EQ(*(a_ptr++), val);
}

TEST(reorder_gpu_i32, basic)
{
    //  Test for converting data types f32->i32
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::i32, format::bfyx, { 2,2,2,2 });

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    int32_t answers[16] = {
        1, 0, 5, 2,
        2, 0, 6, 5,
        3, 1, 7, 12,
        4, -1, 8, 8
    };

    int32_t* a_ptr = answers;
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    for (auto& val : output_ptr)
        EXPECT_EQ(*(a_ptr++), val);
}

TEST(reorder_gpu_i64, basic)
{
    //  Test for converting data types f32->i64
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::i64, format::bfyx, { 2,2,2,2 });

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    int64_t answers[16] = {
        1, 0, 5, 2,
        2, 0, 6, 5,
        3, 1, 7, 12,
        4, -1, 8, 8
    };

    int64_t* a_ptr = answers;
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());
    for (auto& val : output_ptr)
        EXPECT_EQ(*(a_ptr++), val);
}

TEST(reorder_gpu_binary, binary_output)
{
    auto& engine = get_test_engine();

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::bin, format::b_fs_yx_32fp, { 2, 2, 2, 2 });

    // Data is supposed to be quantized to {0,1} values
    set_values(input, {
        1.f, 0.f, 1.f, 1.f,
        0.f, 1.f, 1.f, 0.f,

        1.f, 1.f, 0.f, 1.f,
        0.f, 0.f, 0.f, 1.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<uint32_t> output_ptr(output, get_test_stream());

    std::vector<uint32_t > answers = { 1, 2, 3, 1,
                                       1, 1, 0, 3 };

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), input->get_layout().count());
    ASSERT_EQ(output->get_layout().count(), input->get_layout().count());

    // Check that memory physical size consider binary pack
    ASSERT_EQ(output->size(), answers.size() * sizeof(uint32_t));

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]) << "index: " << i;
    }
}

TEST(reorder_gpu_binary, binary_input)
{
    auto& engine = get_test_engine();

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::bin, format::b_fs_yx_32fp,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx, { 2, 2, 2, 2 });

    // Data is supposed to be quantized to {0,1} values
    std::vector<float> answers = {
            1.f, -1.f, 1.f, 1.f,
            -1.f, 1.f, 1.f, -1.f,

            1.f, 1.f, -1.f, 1.f,
            -1.f, -1.f, -1.f, 1.f
    };

    set_values<int32_t>(input, { 1, 2, 3, 1,
                                 1, 1, 0, 3 });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // Check that layout and memory contains logical size of tensor
    ASSERT_EQ(output->count(), input->get_layout().count());
    ASSERT_EQ(output->get_layout().count(), input->get_layout().count());

    ASSERT_EQ(output->size(), answers.size() * sizeof(float));

    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_EQ(answers[i], output_ptr[i]) << "index: " << i;
    }
}

TEST(reorder_gpu_f32, bfwzyx_bfyx_chain)
{
    // Topology:
    // input               : bfyx 1x4x2x2
    // reorder1            : bfwzyx
    // reshape1            : 2x2x1x1x2x2
    // reorder2            : bfwzyx -- subtract [1, 5]
    // reshape2            : 4x2x1x1x1x2
    // reshape3            : 1x4x2x2x1x1
    // reorder3            : bfyx -- subtract [0, 1, 0, 1]
    // out_reorder         : bfwzyx  1x4x2x2x1x1

    // Input:
    // 1 2   5 6   9  10   13 14
    // 3 4   7 8   11 12   15 16
    // Expected output:
    // 0 1  -1 0    8  9   7  8
    // 2 3   1 2   10 11   9 10
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout{ data_types::f32, format::bfyx, tensor{ batch(1), feature(4), spatial(2, 2) } });

    std::vector<float> data = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f
    };
    set_values(input, data);

    std::vector<float> sub_bfwzyx = { 1.f, 5.f };
    std::vector<float> sub_bfyx = { 0.f, 1.f, 0.f, 1.f };
    std::vector<float> expected = {
        0.f, 1.f, 2.f, 3.f,
        -1.f, 0.f, 1.f, 2.f,
        8.f, 9.f, 10.f, 11.f,
        7.f, 8.f, 9.f, 10.f
    };

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder1", "input", format::bfwzyx, data_types::f32),
        reshape("reshape1", "reorder1", tensor(batch(2), feature(2), spatial(1, 1, 2, 2) )),
        reorder("reorder2", "reshape1", format::bfwzyx, data_types::f32, sub_bfwzyx),
        reshape("reshape2", "reorder2", tensor(batch(4), feature(2), spatial(1, 1, 1, 2))),
        reshape("reshape3", "reshape2", tensor(batch(1), feature(4), spatial(2, 2))),
        reorder("reorder3", "reshape3", format::bfyx, data_types::f32, sub_bfyx),
        reorder("out_reorder", "reorder3", format::bfwzyx, data_types::f32)
        );
    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    EXPECT_TRUE(output->get_layout().format == format::bfwzyx);
    auto sizes = output->get_layout().size;
    EXPECT_EQ(sizes.batch[0], 1);
    EXPECT_EQ(sizes.feature[0], 4);
    EXPECT_EQ(sizes.spatial[0], 2);
    EXPECT_EQ(sizes.spatial[1], 2);
    EXPECT_EQ(sizes.spatial[2], 1);
    EXPECT_EQ(sizes.spatial[2], 1);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), expected.size());

    for (size_t i = 0; i < expected.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected[i], output_ptr[i]);
    }
}

TEST(reorder_gpu_f32, bfzyx_to_bsv16_fsv16)
{
    auto& engine = get_test_engine();
    const int32_t b_in = 2;
    const int32_t f_in = 2;
    const int32_t x_in = 2;
    const int32_t y_in = 2;
    const int32_t z_in = 2;

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { b_in,f_in,x_in,y_in,z_in } });
    layout output_layout(data_types::f32, format::bs_fs_zyx_bsv16_fsv16,{ b_in,f_in,x_in,y_in,z_in });

    tests::set_random_values<float>(input);

    topology topology(
            input_layout("input", input->get_layout()),
            reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    auto get_bsv16_fsv16_index = [] (int32_t /* b_size */, int32_t /* f_size */, int32_t z_size, int32_t y_size, int32_t x_size, int32_t b,
                                     int32_t f_pad_before, int32_t f, int32_t f_pad_after,
                                     int32_t z_pad_before, int32_t z, int32_t z_pad_after,
                                     int32_t y_pad_before, int32_t y, int32_t y_pad_after,
                                     int32_t x_pad_before, int32_t x, int32_t x_pad_after) {
        const int32_t alignment = 16;
        const int32_t fs = f / alignment;
        const int32_t fsv = f % alignment;
        const int32_t bs = b / alignment;
        const int32_t bsv = b % alignment;
        const int32_t x_pitch = alignment * alignment;
        const int32_t y_pitch = x_pitch * (x_pad_before +  x_size + x_pad_after);
        const int32_t z_pitch = y_pitch * (y_pad_before +  y_size + y_pad_after);
        const int32_t total_f_size = f_pad_before + f + f_pad_after;
        const int32_t fs_pitch = z_pitch * (z_pad_before +  z_size + z_pad_after);
        const int32_t b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);

        const int32_t fs_pad_before = f_pad_before / alignment;

        const int32_t output_offset = (bs * b_pitch) + (bsv * alignment) +
                                      (fs_pad_before + fs) * fs_pitch +
                                      (z_pad_before + z) * z_pitch +
                                      (y_pad_before + y) * y_pitch +
                                      (x_pad_before + x) * x_pitch
                                      + fsv;

        return output_offset;
    };

    cldnn::mem_lock<float> input_ptr(input, get_test_stream());
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    int32_t linear_index = 0;
    for (int32_t b = 0; b < b_in; b++) {
        for (int32_t f = 0; f < f_in; f++) {
            for (int32_t z = 0; z < z_in; z++) {
                for (int32_t y = 0; y < y_in; y++) {
                    for (int32_t x = 0; x < x_in; x++) {
                        int32_t bsv16_fsv16_index = get_bsv16_fsv16_index(b_in,f_in,z_in,y_in,x_in,b,
                                                                          0,f,0,
                                                                          0,z,0,
                                                                          0,y,0,
                                                                          0,x,0);
                        EXPECT_FLOAT_EQ(input_ptr[linear_index++], output_ptr[bsv16_fsv16_index]);
                    }
                }
            }
        }
    }
}


TEST(reorder_gpu_f32, bfzyx_to_bsv16_fsv16_padded)
{
    auto& engine = get_test_engine();
    const int32_t b_in = 2;
    const int32_t f_in = 2;
    const int32_t x_in = 2;
    const int32_t y_in = 2;
    const int32_t z_in = 2;
    const int32_t f_pad = 0;
    const int32_t z_pad= 0;
    const int32_t y_pad= 2;
    const int32_t x_pad= 1;

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { b_in,f_in,x_in,y_in,z_in } });
    layout output_layout(data_types::f32, format::bs_fs_zyx_bsv16_fsv16,{ b_in,f_in,x_in,y_in,z_in });

    tests::set_random_values<float>(input);

    topology topology(
            input_layout("input", input->get_layout()),
            reorder("reorder", "input", output_layout.with_padding(padding({0, 0, x_pad, y_pad, 0}, 0.f))));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    auto get_bsv16_fsv16_index = [] (int32_t /* b_size */, int32_t /* f_size */, int32_t z_size, int32_t y_size, int32_t x_size, int32_t b,
                                     int32_t f_pad_before, int32_t f, int32_t f_pad_after,
                                     int32_t z_pad_before, int32_t z, int32_t z_pad_after,
                                     int32_t y_pad_before, int32_t y, int32_t y_pad_after,
                                     int32_t x_pad_before, int32_t x, int32_t x_pad_after) {
        const int32_t alignment = 16;
        const int32_t fs = f / alignment;
        const int32_t fsv = f % alignment;
        const int32_t bs = b / alignment;
        const int32_t bsv = b % alignment;
        const int32_t x_pitch = alignment * alignment;
        const int32_t y_pitch = x_pitch * (x_pad_before +  x_size + x_pad_after);
        const int32_t z_pitch = y_pitch * (y_pad_before +  y_size + y_pad_after);
        const int32_t total_f_size = f_pad_before + f + f_pad_after;
        const int32_t fs_pitch = z_pitch * (z_pad_before +  z_size + z_pad_after);
        const int32_t b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);

        const int32_t fs_pad_before = f_pad_before / alignment;

        const int32_t output_offset = (bs * b_pitch) + (bsv * alignment) +
                                      (fs_pad_before + fs) * fs_pitch +
                                      (z_pad_before + z) * z_pitch +
                                      (y_pad_before + y) * y_pitch +
                                      (x_pad_before + x) * x_pitch
                                      + fsv;

        return output_offset;
    };

    cldnn::mem_lock<float> input_ptr(input, get_test_stream());
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    int32_t linear_index = 0;
    for (int32_t b = 0; b < b_in; b++) {
        for (int32_t f = 0; f < f_in; f++) {
            for (int32_t z = 0; z < z_in; z++) {
                for (int32_t y = 0; y < y_in; y++) {
                    for (int32_t x = 0; x < x_in; x++) {
                        int32_t bsv16_fsv16_index = get_bsv16_fsv16_index(b_in,f_in,z_in,y_in,x_in,b,
                                                                          f_pad,f,f_pad,
                                                                          z_pad,z,z_pad,
                                                                          y_pad,y,y_pad,
                                                                          x_pad,x,x_pad);
                        EXPECT_FLOAT_EQ(input_ptr[linear_index++], output_ptr[bsv16_fsv16_index]);
                    }
                }
            }
        }
    }
}

TEST(reorder_gpu_f32, b_fs_yx_fsv16_to_bfyx_opt_allowed)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::b_fs_yx_fsv16, { 2, 12, 1, 1 } });

    set_values(input, { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f,
                        16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f });

    const std::string reorder_name = "reorder_prim";
    topology topology(
            input_layout("input", input->get_layout()),
            activation("first_activation", "input", activation_func::abs),
            reorder(reorder_name, "first_activation", format::bfyx, data_types::f32),
            activation("second_activation", reorder_name, activation_func::abs));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto executed_prims = network.get_executed_primitives();

    EXPECT_TRUE(executed_prims.find(reorder_name) == executed_prims.end());
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "second_activation");

    auto output = outputs.begin()->second.get_memory();

    float answers[24] = {
            0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f,
            16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f,
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), 24);
    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]) << "i=" << i;
    }
}

TEST(reorder_gpu_f32, b_fs_yx_fsv16_to_bfyx_opt_not_allowed)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::b_fs_yx_fsv16, { 1, 8, 1, 1 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::oiyx, { 1, 8, 3, 3 } });

    set_values(input, { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f });

    set_values(weights, std::vector<float>(weights->count(), 1));

    const std::string reorder_name = "reorder";
    const std::string reorder_primitive_name = "reorder:" + reorder_name;
    topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            reorder(reorder_name, "input", format::bfyx, data_types::f32),
            convolution("convolution", reorder_name, {"weights"}, {1,1,1,1}, {0,0,-1,-1}, {1,1,1,1}));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto executed_prims = network.get_executed_primitive_ids();

    EXPECT_FALSE(std::find(executed_prims.begin(), executed_prims.end(), reorder_primitive_name) != executed_prims.end());
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "convolution");

    auto output = outputs.begin()->second.get_memory();

    float answers[1] = { 28.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 1; i++)
    {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]) << "i=" << i;
    }
}

TEST(reorder_gpu_f32, b_fs_yx_fsv16_to_bfyx_opt_padded)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32,
                                            format::b_fs_yx_fsv16,
                                            { 2, 4, 1, 1 },
                                            padding({1, 16, 0, 0}, {1, 0, 0, 0}) });

    std::vector<float> in_data = {
        // b -1 (lower pad)
        -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
        -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
        // b 0
        -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f,
        // b 1
        -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
        16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f,
        // b +1 (upper pad)
        -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
        -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
    };

    set_values(input, in_data);

    const std::string reorder_name = "reorder_prim";
    topology topology(
        input_layout("input", input->get_layout()),
        reorder(reorder_name, "input", format::bfyx, data_types::f32),
        activation("activation", reorder_name, activation_func::abs));

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto executed_prims = network.get_executed_primitives();

    EXPECT_TRUE(executed_prims.find(reorder_name) == executed_prims.end());
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "activation");

    auto output = outputs.begin()->second.get_memory();

    float answers[8] = {
            0.f, 1.f, 2.f, 3.f,
            16.f, 17.f, 18.f, 19.f,
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), 8);
    for (size_t i = 0; i < output_ptr.size(); i++) {
        EXPECT_FLOAT_EQ(answers[i], output_ptr[i]) << "i=" << i;
    }
}

TEST(reorder_gpu, any_format) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout(data_types::f32, format::yxfb, tensor(5, 7, 13, 9)));

    topology topo;
    topo.add(input_layout("in", input->get_layout()));
    topo.add(reorder("out", "in", format::any, data_types::f32));

    network net(engine, topo);

    auto data = generate_random_1d<float>(input->count(), -1, 1);
    set_values(input, data);
    net.set_input_data("in", input);

    auto outputs = net.execute();
    auto out_mem = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output(out_mem, get_test_stream());

    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(output[i], data[i]) << "i = " << i;
    }
}

TEST(reorder_image2d_rgba_to_bfyx_gpu, basic)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::u8, format::image_2d_rgba, { 1, 3, 2, 2 } });
    layout output_layout(data_types::f16, format::bfyx, { 1, 3, 2, 2 });

    set_values<unsigned char>(input, {
        1, 0, 5, 7,
        2, 111, 123, 8,
        124, 125, 50, 9,
        251, 252, 253, 210
        });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[12] = {
        1.0f,  2.0f,
        124.0f,  251.0f,

        0.0f,  111.0f,
        125.0f,  252.0f,

        5.0f,  123.0f,
        50.0f, 253.0f,
    };

    cldnn::mem_lock<FLOAT16> output_ptr (output, get_test_stream());
    for (int i = 0; i < 12; i++)
    {
        EXPECT_NEAR(FLOAT16(answers[i] / 255.f), output_ptr[i], 1e-3f);
    }

}

TEST(reorder_bfyx_to_image2d_rgba_gpu, basic)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 3, 2, 2 } });
    layout output_layout(data_types::u8, format::image_2d_rgba, { 1, 3, 2, 2 });

    set_values<FLOAT16>(input, {
        FLOAT16(1.0f / 255.f),  FLOAT16(2.0f / 255.f),
        FLOAT16(124.0f / 255.f),  FLOAT16(251.0f / 255.f),

        FLOAT16(0.0f / 255.f),  FLOAT16(111.0f / 255.f),
        FLOAT16(125.0f / 255.f),  FLOAT16(252.0f / 255.f),

        FLOAT16(5.0f / 255.f),  FLOAT16(123.0f / 255.f),
        FLOAT16(50.0f / 255.f), FLOAT16(253.0f / 255.f),
        });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    unsigned char answers[16] = {
        1, 0, 5, 0,
        2, 111, 123, 0,
        124, 125, 50, 0,
        251, 252, 253, 0
    };

    cldnn::mem_lock<unsigned char> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        EXPECT_EQ(answers[i], output_ptr[i]);
    }

}

using namespace cldnn;

class reorder_test : public tests::generic_test
{

public:

    static void TearDownTestCase()
    {
        all_generic_params.clear();
        all_test_params.clear();
    }

    static std::vector<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>> generate_specific_test_params()
    {
        generic_test::generate_generic_test_params(all_generic_params);

        const auto data_types = test_data_types();

        for (const auto& test_param : all_generic_params)
        {
            cldnn::tensor input_tensor = test_param->input_layouts[0].size;

            std::vector<cldnn::layout> output_layouts = {};

            for (const auto& dt : data_types)
            {
                for (const auto& fmt : generic_test::test_input_formats)
                {
                    output_layouts.push_back({ dt, fmt, input_tensor });
                }
            }
            // TODO: check unsupported formats.

            //TODO: check subtract.
            std::vector<float> subtract = {};

            for (const auto& output_layout : output_layouts)
            {
                //TODO: check input + output padding.
                all_test_params.emplace_back(std::make_tuple(test_param, std::make_shared<reorder>("reorder", "input0", output_layout, subtract)));

            }
        }

        return all_test_params;
    }

    bool is_format_supported(cldnn::format format) override
    {
        return (    (format == cldnn::format::yxfb) ||
                    (format == cldnn::format::byxf) ||
                    (format == cldnn::format::bfyx) ||
                    (format == cldnn::format::fyxb)
                );
    }

    template<typename InputType, typename OutputType>
    memory::ptr generate_reference_typed(const std::vector<cldnn::memory::ptr>& inputs)
    {
        auto reorder = std::static_pointer_cast<cldnn::reorder>(layer_params);
        primitive_id mean = reorder->mean;
        std::vector<float> subtract_per_feature = reorder->subtract_per_feature;
        assert(mean == "");
        assert(subtract_per_feature.size() == 0);

        auto output = engine.allocate_memory(cldnn::layout(*reorder->output_data_type, inputs[0]->get_layout().format, inputs[0]->get_layout().size));

        cldnn::mem_lock<InputType> input_mem(inputs[0], get_test_stream());
        cldnn::mem_lock<OutputType> output_mem(output, get_test_stream());

        for (size_t i = 0; i < inputs[0]->get_layout().get_linear_size(); i++)
        {
            // Write the output in the same order as the input with type conversion as needed.
            // The correct order will be checked in generic_test::compare_buffers.
            output_mem[i] = (OutputType)input_mem[i];
        }

        return output;
    }

    memory::ptr generate_reference(const std::vector<cldnn::memory::ptr>& inputs) override
    {
        if (generic_params->data_type == data_types::f32)
        {
            if (*layer_params->output_data_type == data_types::f32)
            {
                return generate_reference_typed<float, float>(inputs);
            }
            else
            {
                return generate_reference_typed<float, FLOAT16>(inputs);
            }
        }
        else
        {
            if (*layer_params->output_data_type == data_types::f32)
            {
                return generate_reference_typed<FLOAT16, float>(inputs);
            }
            else
            {
                return generate_reference_typed<FLOAT16, FLOAT16>(inputs);
            }
        }
    }

private:

    static std::vector<std::shared_ptr<tests::test_params>> all_generic_params;
    static std::vector<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>> all_test_params;

};

std::vector<std::shared_ptr<tests::test_params>> reorder_test::all_generic_params = {};
std::vector<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>> reorder_test::all_test_params = {};

TEST_P(reorder_test, REORDER)
{
    run_single_test();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_REORDER,
                        reorder_test,
                        ::testing::ValuesIn(reorder_test::generate_specific_test_params()),
                        tests::generic_test::custom_param_name_functor());
