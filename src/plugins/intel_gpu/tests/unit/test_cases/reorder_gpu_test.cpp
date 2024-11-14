// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include <intel_gpu/primitives/data.hpp>

#include "reorder_inst.h"

#include <cmath>
#include <limits>

using namespace cldnn;
using namespace ::tests;
using namespace testing;

template <typename T>
static void compare_result(std::map<cldnn::primitive_id, cldnn::network_output> ref_result,
                           std::map<cldnn::primitive_id, cldnn::network_output> opt_result) {
    auto output_ref = ref_result.begin()->second.get_memory();
    mem_lock<T> output_ref_ptr{output_ref, get_test_stream()};

    auto output_opt = opt_result.begin()->second.get_memory();
    mem_lock<T> output_opt_ptr{output_opt, get_test_stream()};

    // compare results
    const size_t output_size = output_ref_ptr.size();
    for (size_t i = 0; i < output_size; i++)
    {
        ASSERT_EQ(output_ref_ptr[i], output_opt_ptr[i]);
    }
}

static void compare_bfyx2blocked_with_ref(const std::string& kernel_name,
    const data_types input_data_type, const data_types output_data_type,
    cldnn::format input_format, cldnn::format output_format,
    int32_t b_in, int32_t f_in, int32_t x_in, int32_t y_in, int32_t z_in, int32_t w_in,
    bool is_caching_test) {
    auto& engine = get_test_engine();
    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order));
    if (engine.get_device_info().supports_immad) {
        // Onednn currently does NOT support out_of_order : skip this test
        return;
    }

    auto stream = std::shared_ptr<cldnn::stream>(engine.create_stream(cfg));

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
        mem_lock<uint8_t> input_ptr{input, *stream};
        unsigned char i = 1;
        for (auto it = input_ptr.begin(); it != input_ptr.end(); ++it)
        {
            *it = (i++);
            if (i > 100) {
                i = 1;
            }
        }
    } else {
        mem_lock<float> input_ptr{input, *stream};
        float i = 1.f;
        for (auto it = input_ptr.begin(); it != input_ptr.end(); ++it)
        {
            *it = (i);
            i += 1.f;
        }
    }

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), output_layout));

    // run on reference(reorder_data) kernel
    ov::intel_gpu::ExecutionConfig config_ref = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc reorder_ref = { output_format, "reorder_data" };
    config_ref.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"reorder", reorder_ref} }));

    cldnn::network::ptr network_ref = get_network(engine, topology, config_ref, stream, is_caching_test);

    network_ref->set_input_data("input", input);

    auto outputs_ref = network_ref->execute();
    cldnn::event::ptr e1 = outputs_ref.at("reorder").get_event();
    e1->wait();

    // run on optimized kernel
    ov::intel_gpu::ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc reorder_optimized = { output_format, kernel_name };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"reorder", reorder_optimized} }));

    cldnn::network::ptr network = get_network(engine, topology, config, stream, is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();
    cldnn::event::ptr e2 = outputs.at("reorder").get_event();
    e2->wait();

    // compare output_ref and output_opt.
    if (output_data_type == data_types::i8)
        compare_result<int8_t>(outputs_ref, outputs);
    else if (output_data_type == data_types::u8)
        compare_result<uint8_t>(outputs_ref, outputs);
    else if (output_data_type == data_types::i32)
        compare_result<int32_t>(outputs_ref, outputs);
    else if (output_data_type == data_types::i64)
        compare_result<int64_t>(outputs_ref, outputs);
    else if (output_data_type == data_types::f16)
        compare_result<int16_t>(outputs_ref, outputs);
    else if (output_data_type == data_types::f32)
        compare_result<float>(outputs_ref, outputs);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv32_to_bfyx_f32) {
    // b_fs_yx_fsv32 -> bfyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfyx, 3, 64 + 5, 16 + 11, 3, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfyx, 3, 96 - 12, 16 + 4, 3, 0, 0, false);
    // b_fs_zyx_fsv32 -> bfzyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfzyx, 3, 64 + 9, 16 - 1, 2, 8, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfzyx, 2, 64 + 30, 16 + 1, 3, 4, 0, false);
    // incremental dims
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfzyx, 2, 64 + 4, 24 - 1, 3, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfwzyx, 2, 64 + 2, 32 - 3, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv32, format::bfwzyx, 1, 96 + 10, 32 - 3, 4, 3, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv32_to_bfyx_different_datatype) {
    // f32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::u8, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 8 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i64, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 16 + 2, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f16, format::b_fs_yx_fsv32, format::bfyx, 1, 64, 16 + 1, 2, 0, 0, false);
    // i32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i8, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 8 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i64, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 16 + 2, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::f16, format::b_fs_yx_fsv32, format::bfyx, 1, 64, 16 + 1, 2, 0, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv16_to_bfyx_f32) {
    // u-net
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 1, 64, 388, 388, 0, 0, false);
    // b_fs_yx_fsv16 -> bfyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 3, 48 + 1, 16, 3, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 2, 32 - 1, 24 - 1, 3, 0, 0, false);
    // b_fs_zyx_fsv16 -> bfzyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx, 5, 48 - 1, 16, 3, 8, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx, 2, 32 + 1, 24 - 1, 3, 17, 0, false);
    // incremental dims
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfzyx, 3, 32 - 1, 24 - 1, 3, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfwzyx, 4, 16 + 1, 32 - 3, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx, 3, 16 + 2, 32 - 3, 4, 9, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv16_to_bfyx_different_datatype) {
    // f32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::u8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i32, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i64, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f16, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
    // i32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::u8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i64, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::f16, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_blocked_f32) {
    // bfyx_to_b_fs_yx_fsv4
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv4, 4, 32, 16, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv4, 3, 32 + 2, 32 + 3, 4, 0, 0, false);
    // bfyx_to_b_fs_yx_fsv16
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 2, 48, 8, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2, 0, 0, false);
    // bfyx to b_fs_yx_fsv32
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv32, 2, 64, 64, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv32, 4, 32 + 6, 96 - 4, 2, 0, 0, false);
    // bfyx to fs_b_yx_fsv32
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::fs_b_yx_fsv32, 2, 64, 8, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::fs_b_yx_fsv32, 3, 64 + 5, 8 + 7, 2, 0, 0, false);
    // bfzyx to b_fs_zyx_fsv16
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv16, 2, 48, 8, 4, 4, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv16, 3, 32 + 5, 16 + 7, 2, 2, 0, false);
    // bfzyx to b_fs_zyx_fsv32
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv32, 2, 64, 8, 4, 4, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv32, 3, 64 + 5, 8 + 7, 2, 2, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f32) {
    // bfyx to double blocked format (bs_fs_yx_bsv16_fsv16)
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32, 48, 8, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32 + 2, 48, 16, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32, 48 + 5, 16, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32, 48, 48 + 3, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32 + 2, 48 + 3, 16 + 1, 4, 0, 0, false);
    // bfyx to double blocked format (bs_fs_yx_bsv16_fsv32)
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32, 48, 8, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32 + 2, 48, 16, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32, 48 + 5, 16, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32, 48, 48 + 3, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32 + 2, 48 + 3, 16 + 1, 4, 0, 0, false);

    // bfzyx to double blocked format (bs_fs_zyx_bsv16_fsv16)
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32, 48, 8, 4, 16, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32 + 2, 48, 16, 4, 2, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32, 48 + 5, 16, 4, 3, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32, 48, 48 + 3, 4, 4, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32 + 2, 48 + 3, 16 + 1, 4, 2, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f16) {
    // bfyx to double blocked format (bs_fs_yx_bsv16_fsv32)
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f16, data_types::f16, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32, 48, 8, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f16, data_types::f16, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32 + 2, 48, 16, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f16, data_types::f16, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32, 48 + 5, 16, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f16, data_types::f16, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32, 48, 48 + 3, 4, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f16, data_types::f16, format::bfyx, format::bs_fs_yx_bsv16_fsv32, 32 + 2, 48 + 3, 16 + 1, 4, 0, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f32_bsv16_fsv32) {
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 3, 16, 4, 5, 7, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 1, 1, 1, 1, 1, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32 + 2, 48, 16, 4, 2, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32 + 1, 1, 1, 1, 1, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32, 48 + 5, 16, 4, 3, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32, 48, 48 + 3, 4, 4, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32 + 2, 48 + 3, 16 + 1, 4, 2, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f32_bsv32_fsv16) {
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 1, 1, 1, 1, 1, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 32 + 2, 48, 16, 4, 2, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 32, 48 + 5, 16, 4, 3, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 32, 48, 48 + 3, 4, 4, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 32 + 2, 48 + 3, 16 + 1, 4, 2, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f32_bsv32_fsv32) {
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 1, 1, 1, 1, 1, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 32 + 2, 48, 16, 4, 2, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 32, 48 + 5, 16, 4, 3, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 32, 48, 48 + 3, 4, 4, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 32 + 2, 48 + 3, 16 + 1, 4, 2, 0, false);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_blocked_format_different_datatype) {
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f16, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::i8, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2, 0, 0, false);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::i64, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2, 0, 0, false);
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
        reorder("reorder", input_info("input"), output_layout));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

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
                    ASSERT_FLOAT_EQ(input_ptr[linear_index++], output_ptr[b_fs_yx_fsv16_index]);
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
    layout output_layout(data_types::f32, format::bfyx,{ 2, 2, 2, 2 });

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
        reorder("reorder", input_info("input"), output_layout));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

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
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
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
    layout output_layout( data_types::f32, format::bfyx, { 2, 2, 2, 2 } );
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
        reorder("reorder", input_info("input"), output_layout, "subtract"));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    network.set_input_data("subtract", subtract);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

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
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
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
    layout output_layout(data_types::f32, format::bfyx,{ 2, 2, 2, 2 });
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
    topology.add(input_layout("input", input->get_layout()), reorder("reorder", input_info("input"), output_layout, subtract_val));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

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
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reorder_gpu_f32, fusing_double_activations) {
    // reorder_data                      reorder_data
    //      |                                 |
    //     sqrt                               |
    //       |               fuse             |
    //     power data        ---->            | data
    //       \   /                            |  /
    //       divide                         divide
    //         |                              |
    //       result                         result
    //
    // This test case is limited to the case of reorder_data using ReorderKernelRef.
    // Because other kernels for reorder_data don't support fusing double activations e.g. reorder_data_fast_b1
    //
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({{1}, data_types::f32, format::bfyx});
    auto input2 = engine.allocate_memory({{1, 1, 1, 2, 2}, data_types::f32, format::bfzyx});

    topology topology {
        input_layout("input1", input1->get_layout()),
        reorder("reorder", input_info("input1"), format::bfyx, data_types::f32),
        activation("sqrt", input_info("reorder"), activation_func::sqrt),
        activation("power", input_info("sqrt"), activation_func::pow),
        input_layout("input2", input2->get_layout()),
        eltwise("divide", {input_info("power"), input_info("input2")}, eltwise_mode::div),
        reorder("result", input_info("divide"), format::bfyx, data_types::f32)
    };

    set_values(input1, {25000});
    set_values(input2, {0.1f, 0.2f, 0.5f, 1.0f});

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc reorder_impl = {format::bfyx, "reorder_data"};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"reorder", reorder_impl}}));

    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto output = network.execute();

    mem_lock<float> output_mem(output.at("result").get_memory(), network.get_stream());
    std::vector<int32_t> output_ref = {10, 5, 2, 1};
    for (size_t i = 0; i < output_mem.size(); ++i) {
        ASSERT_EQ(output_mem[i], output_ref[i]);
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
        ASSERT_EQ(1, 1);
        return;
    }

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f32, format::bfyx,{ 2, 2, 2, 2 });
    auto subtract = engine.allocate_memory({ data_types::f32, format::byxf, { 1, 2, 2, 2 } });

    set_values(input, {
        ov::float16(1.f), ov::float16(0.f),
        ov::float16(5.f), ov::float16(1.5f),

        ov::float16(2.f), ov::float16(0.f),
        ov::float16(6.f), ov::float16(5.2f),

        ov::float16(3.f), ov::float16(0.5f),
        ov::float16(7.f), ov::float16(12.f),

        ov::float16(4.f), ov::float16(-0.5f),
        ov::float16(8.f), ov::float16(8.f)
    });

    set_values(subtract, {
        1.0f,  4.0f,      1.5f,  3.0f,
        2.0f,  2.0f,      2.5f,  1.0f,
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("subtract", subtract));
    topology.add(reorder("reorder", input_info("input"), output_layout, "subtract"));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

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
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
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
        ASSERT_EQ(1, 1);
        return;
    }

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb, { 2, 2, 2, 2 } });
    layout output_layout(data_types::f16, format::bfyx,{ 2, 2, 2, 2 });
    std::vector<float> subtract_val = { 0.5, 2.5 };

    set_values(input, {
        ov::float16(1.f), ov::float16(0.f),
        ov::float16(5.f), ov::float16(1.5f),

        ov::float16(2.f), ov::float16(0.f),
        ov::float16(6.f), ov::float16(5.2f),

        ov::float16(3.f), ov::float16(0.5f),
        ov::float16(7.f), ov::float16(12.f),

        ov::float16(4.f), ov::float16(-0.5f),
        ov::float16(8.f), ov::float16(8.f)
    });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", input_info("input"), output_layout, subtract_val));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    ov::float16 answers[16] = { ov::float16(0.5f), ov::float16(1.5f),
                           ov::float16(2.5f), ov::float16(3.5f),

                           ov::float16(2.5f), ov::float16(3.5f),
                           ov::float16(4.5f), ov::float16(5.5f),

                           ov::float16(-0.5f), ov::float16(-0.5f),
                           ov::float16(0.f), ov::float16(-1.f),

                           ov::float16(-1.f), ov::float16(2.7f),
                           ov::float16(9.5f), ov::float16(5.5f)
    };

    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        ASSERT_TRUE(are_equal(static_cast<uint16_t>(answers[i]), static_cast<uint16_t>(output_ptr[i])));
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
        ASSERT_EQ(1, 1);
        return;
    }

    std::vector<ov::float16> expected_values;
    expected_values.resize(0xF804);
    for (int i = 0; i < 0x7C00; ++i)
        expected_values[i] = ov::float16::from_bits(i);          // norms/denorms/zero (positive).
    for (int i = 0x7C00; i < 0xF800; ++i)
        expected_values[i] = ov::float16::from_bits(i + 0x0400); // norms/denorms (negative).
    expected_values[0x7C00] = ov::float16::from_bits(0x0000);    // NOTE: do not do final test for negative 0 (-0).
    // Special values.
    expected_values[0xF800] = ov::float16::from_bits(0x7C00);    // +infinity
    expected_values[0xF801] = ov::float16::from_bits(0xFC00);    // -infinity
    // Special values (ambiguous ones).
    expected_values[0xF802] = ov::float16::from_bits(0x8000);    // -0
    expected_values[0xF803] = ov::float16::from_bits(0xFC12);    // A NaN (sample: -NaN.0x12).

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb, { 1, static_cast<int32_t>(expected_values.size()) / 4, 2, 2 } });
    layout interm_layout( data_types::f32, format::byxf, { 1, static_cast<int32_t>(expected_values.size()) / 4, 2, 2 });
    auto output_layout = input->get_layout();

    set_values(input, expected_values);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder_f16_f32", input_info("input"), interm_layout));
    topology.add(reorder("reorder_f32_f16", input_info("reorder_f16_f32"), output_layout));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"reorder_f16_f32", "reorder_f32_f16"}));
    network network(engine, topology, cfg);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(2));
    ASSERT_TRUE(outputs.find("reorder_f16_f32") != outputs.end());
    ASSERT_TRUE(outputs.find("reorder_f32_f16") != outputs.end());

    auto interm = outputs.at("reorder_f16_f32").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());

    // Sample positive.
    ASSERT_TRUE(are_equal(interm_ptr[0x3400], 0.25f));
    ASSERT_TRUE(are_equal(interm_ptr[0x3800], 0.5f));
    ASSERT_TRUE(are_equal(interm_ptr[0x3C00], 1.0f));
    ASSERT_TRUE(are_equal(interm_ptr[0x4000], 2.0f));
    ASSERT_TRUE(are_equal(interm_ptr[0x4400], 4.0f));
    // Sample negative.
    ASSERT_TRUE(are_equal(interm_ptr[0x3400 + 0x7C00], -0.25f));
    ASSERT_TRUE(are_equal(interm_ptr[0x3800 + 0x7C00], -0.5f));
    ASSERT_TRUE(are_equal(interm_ptr[0x3C00 + 0x7C00], -1.0f));
    ASSERT_TRUE(are_equal(interm_ptr[0x4000 + 0x7C00], -2.0f));
    ASSERT_TRUE(are_equal(interm_ptr[0x4400 + 0x7C00], -4.0f));
    // Special values.
    ASSERT_TRUE(are_equal(interm_ptr[0xF800], std::numeric_limits<float>::infinity()));
    ASSERT_TRUE(are_equal(interm_ptr[0xF801], -std::numeric_limits<float>::infinity()));
    ASSERT_TRUE(are_equal(interm_ptr[0xF802], -0.0f));
    ASSERT_TRUE(std::isnan(interm_ptr[0xF803]));

    auto output = outputs.at("reorder_f32_f16").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());
    for (int i = 0; i < 0xF802; ++i) // NOTE: do not test for possibly ambiguous values of floating point (-0, NaNs).
    {
        ASSERT_TRUE(are_equal(static_cast<uint16_t>(expected_values[i]), static_cast<uint16_t>(output_ptr[i])));
    }
}

TEST(reorder_gpu, basic_convert_int8) {

    auto& engine = get_test_engine();
    layout in_layout = { ov::element::from<float>(),format::byxf,{ 1, 1, 3, 3 } };
    layout byte_layout = { ov::element::from<int8_t>(), format::bfyx,{ 1, 1, 3, 3 } };
    std::initializer_list<float> input_f = { 1.0f, -2.5f, 3.1f, -4.0f, 5.03f, -6.99f, 7.0f, -8.0f, 9.0f };
    std::list<float> final_results = { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f };

    if (engine.get_device_info().supports_immad) {
        // Use onednn when reordering byxf format.
        final_results = { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -7.0f, 7.0f, -8.0f, 9.0f };
    }

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
            input_info(input),
            // output layout for reorder
            byte_layout),
        reorder("reorder2", input_info("reorder_input"), in_layout)
    );

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "reorder_input", "reorder2"}));
    cfg.set_property(ov::intel_gpu::optimize_data(true)); // to enable onednn
    network network(engine, topology, cfg);

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    auto interm = outputs.at("reorder2").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());
    unsigned int cntr = 0;
    for (const auto& exp : final_results)
    {
        ASSERT_EQ(exp, interm_ptr[cntr++]);
    }
}

TEST(reorder_gpu, basic_convert_uint8) {
    auto& engine = get_test_engine();
    layout in_layout = { ov::element::from<float>(),format::byxf,{ 1, 1, 3, 3 } };
    layout byte_layout = { ov::element::from<uint8_t>(), format::bfyx,{ 1, 1, 3, 3 } };
    std::initializer_list<float> input_f = { 1.0f, -2.5f, 3.1f, -4.0f, 5.03f, -6.99f, 7.0f, -8.0f, 9.0f };
    std::list<float> final_results = { 1.0f, 254.0f, 3.0f, 252.0f, 5.0f, 250.0f, 7.0f, 248.0f, 9.0f };

    if (engine.get_device_info().supports_immad) {
        // Use onednn when reordering byxf format.
        final_results = { 1.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f, 7.0f, 0.0f, 9.0f };
    }

    // Allocate memory for input image.
    auto input_memory = engine.allocate_memory(in_layout);
    set_values(input_memory, input_f);

    // Create input_layout description
    input_layout input("input", in_layout);

    topology topology(
        input,
        reorder("reorder_input",
            input_info(input),
            cldnn::format::any,
            cldnn::data_types::u8,
            std::vector<float>(),
            cldnn::reorder_mean_mode::subtract,
            cldnn::padding(),
            true),
        reorder("reorder2", input_info("reorder_input"), in_layout)
    );

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "reorder_input", "reorder2" }));
    cfg.set_property(ov::intel_gpu::optimize_data(true)); // to enable onednn
    network network(engine, topology, cfg);

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    auto interm = outputs.at("reorder2").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());
    unsigned int cntr = 0;
    for (const auto& exp : final_results) {
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
        ASSERT_EQ(1, 1);
        return;
    }

    std::initializer_list<uint8_t> input_i8 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136,
        255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 237, 236
    };

    layout in_layout = { ov::element::from<uint8_t>(),format::byxf,{ 1, 4, kernel_size,kernel_size } };
    layout output_layout = { ov::element::from<float>(), format::bfyx, { 1, 4, kernel_size,kernel_size } };

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
            input_info(input),
            // output layout for reorder
            output_layout)
    );

    tensor crop_reference_input_tensor(spatial(kernel_size, kernel_size), batch(1), feature(4 - 1));
    tensor crop_offset_tensor(spatial(0, 0), batch(0), feature(0));
    topology.add(
        // cropping primitive with id "crop1"
        crop("crop",
             input_info("reorder_input"),               // primitive id of the cropping input
             crop_reference_input_tensor,   // input tensor
             crop_offset_tensor            // bias primitive id
            )
    );

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "reorder_input", "crop" }));
    network network(engine, topology, cfg);

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(2));
    ASSERT_TRUE(outputs.find("reorder_input") != outputs.end());
    ASSERT_TRUE(outputs.find("crop") != outputs.end());

    auto interm = outputs.at("reorder_input").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());
    auto interm_size = outputs.at("reorder_input").get_memory()->count();
    ASSERT_EQ(interm_size,(size_t) (1*feature_size*kernel_size*kernel_size));

    // Sample positive.
    ASSERT_TRUE(are_equal(interm_ptr[0], 1.0f));
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
        ASSERT_TRUE(are_equal(val, interm_ptr[target_index]));
        source_index++;
    }

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    auto output_size = output->count();
    ASSERT_EQ(output_size,(size_t) (1 * (feature_size-1)*kernel_size*kernel_size));

    for (target_index = 0; target_index < output_size; target_index++)
    {
        float output_val = output_ptr[target_index];
        int current_x = target_index % kernel_size;
        int current_y = (target_index / kernel_size) % kernel_size;
        size_t current_feature = target_index / (kernel_size*kernel_size);

        source_index = current_x*feature_size + current_y*(kernel_size*feature_size) + current_feature;
        ASSERT_TRUE(are_equal(output_val, testinput[source_index]));
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
    layout output_layout(data_types::f32, format::bfyx, { 2, 2, 2, 2 });

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
        reorder("reorder", input_info("input"), input->get_layout().format, input->get_layout().data_type, "", reorder_mean_mode::subtract, padding{ { 0, 0, 1, 2 }, 0 }),
        reorder("reorder2", input_info("reorder"), output_layout));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder2");

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
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
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
    layout output_layout(data_types::f32, format::yxfb, { 2, 2, 2, 2 });

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
        reorder("reorder", input_info("input"), input->get_layout().format, input->get_layout().data_type, "", reorder_mean_mode::subtract, padding{ { 0, 0, 2, 1 }, 0 }),
        reorder("reorder2", input_info("reorder"), output_layout));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder2");

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
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
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
        reorder("reorder", input_info("input"), format::bfzyx, data_types::f32));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    ASSERT_TRUE(output->get_layout().format == format::bfzyx);
    auto l = output->get_layout();
    ASSERT_TRUE(l.batch() == 2);
    ASSERT_TRUE(l.feature() == 2);
    ASSERT_TRUE(l.spatial(0) == 2);
    ASSERT_TRUE(l.spatial(1) == 2);
    ASSERT_TRUE(l.spatial(2) == 1);

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
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}
TEST(reorder_gpu_f32, dynamic_bfyx_to_bfyx_dynamic_padding_x) {
    auto& engine = get_test_engine();

    ov::Shape in_shape{1, 1, 4, 2};
    padding::DynamicDimsMask dyn_pad_dims("1000"); // {0, 0, 0, 1}
    layout in_dynamic_layout{ov::PartialShape::dynamic(in_shape.size()),
                             data_types::f16,
                             format::bfyx,
                             padding({0, 0, 0, 0}, {0, 0, 0, 0}, dyn_pad_dims /*dynamic_pad_dim : x*/)};

    std::vector<float> subtract_val = {};
    topology topology(input_layout("input", in_dynamic_layout),
                      reorder("reorder",
                              input_info("input"),
                              format::bfyx,
                              data_types::f32,
                              subtract_val,
                              cldnn::reorder_mean_mode::subtract,
                              padding({0, 0, 0, 0}, {0, 0, 0, 0}, 0.0f)));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    auto input_mem = engine.allocate_memory({ov::PartialShape(in_shape),
                                             data_types::f16,
                                             format::bfyx,
                                             padding({0, 0, 0, 2}, {0, 0, 0, 1}, dyn_pad_dims)});
    set_values<ov::float16>(input_mem, {
        ov::float16(0.f), ov::float16(0.f), // padding
        ov::float16(1.f), ov::float16(2.f), // data
        ov::float16(0.f),               // padding

        ov::float16(0.f), ov::float16(0.f), // padding
        ov::float16(3.f), ov::float16(4.f), // data
        ov::float16(0.f),               // padding

        ov::float16(0.f), ov::float16(0.f), // padding
        ov::float16(5.f), ov::float16(6.f), // data
        ov::float16(0.f),               // padding

        ov::float16(0.f), ov::float16(0.f), // padding
        ov::float16(7.f), ov::float16(8.f), // data
        ov::float16(0.f),               // padding
    });

    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    auto output = outputs.begin()->second.get_memory();

    float answer[8] = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(answer[i], output_ptr[i], 1e-2f);
    }

}

TEST(reorder_gpu_f32, dynamic_bfyx_to_bfyx_dynamic_padding_f) {
    auto& engine = get_test_engine();

    ov::Shape in_shape{2, 3, 2, 1};
    padding::DynamicDimsMask dyn_pad_dims("10");
    layout in_dynamic_layout{ov::PartialShape::dynamic(in_shape.size()),
                             data_types::f16,
                             format::bfyx,
                             padding({0, 0, 0, 0}, {0, 0, 0, 0}, dyn_pad_dims)};

    std::vector<float> subtract_val = {};
    topology topology(input_layout("input", in_dynamic_layout),
                      reorder("reorder",
                              input_info("input"),
                              format::bfyx,
                              data_types::f32,
                              subtract_val,
                              cldnn::reorder_mean_mode::subtract,
                              padding({0, 0, 0, 0}, {0, 0, 0, 0}, 0.0f)));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    auto input_mem = engine.allocate_memory({ov::PartialShape(in_shape),
                                             data_types::f16,
                                             format::bfyx,
                                             padding({0, 2, 0, 0}, {0, 1, 0, 0}, dyn_pad_dims)});
    set_values<ov::float16>(input_mem, {
        ov::float16(0.f), ov::float16(0.f), // f before
        ov::float16(0.f), ov::float16(0.f), // f before
        ov::float16(1.f), ov::float16(2.f), // b0 f0
        ov::float16(3.f), ov::float16(4.f), // b0 f1
        ov::float16(5.f), ov::float16(6.f), // b0 f2
        ov::float16(0.f), ov::float16(0.f), // f after

        ov::float16(0.f), ov::float16(0.f),   // f before
        ov::float16(0.f), ov::float16(0.f),   // f before
        ov::float16(11.f), ov::float16(22.f), // b1 f0
        ov::float16(33.f), ov::float16(44.f), // b1 f1
        ov::float16(55.f), ov::float16(66.f), // b1 f2
        ov::float16(0.f), ov::float16(0.f),   // f after
    });

    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    auto output = outputs.begin()->second.get_memory();

    float answer[12] = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
        11.f, 22.f, 33.f, 44.f, 55.f, 66.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 12; i++) {
        ASSERT_NEAR(answer[i], output_ptr[i], 1e-2f);
    }
}

TEST(reorder_gpu_f32, dynamic_bfyx_to_bfzyx) {
    auto& engine = get_test_engine();

    ov::Shape in_shape{ 1, 2, 4, 2 };
    layout in_layout{ov::PartialShape::dynamic(in_shape.size()), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape(in_shape), data_types::f16, format::bfyx});

    set_values<ov::float16>(input, {
        ov::float16(1.f), ov::float16(0.f),
        ov::float16(5.f), ov::float16(1.5f),

        ov::float16(2.f), ov::float16(0.f),
        ov::float16(6.f), ov::float16(5.2f),

        ov::float16(3.f), ov::float16(0.5f),
        ov::float16(7.f), ov::float16(12.f),

        ov::float16(4.f), ov::float16(-0.5f),
        ov::float16(8.f), ov::float16(8.f)
    });

    topology topology(
        input_layout("input", in_layout),
        reorder("reorder", input_info("input"), format::bfzyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    auto inst = network.get_primitive("reorder");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    ASSERT_TRUE(output->get_layout().format == format::bfzyx);
    auto l = output->get_layout();
    auto expected_shape = ov::PartialShape(in_shape);
    ASSERT_EQ(l.get_partial_shape(), expected_shape);

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
    for (int i = 0; i < 16; i++) {
        ASSERT_NEAR(answers[i], output_ptr[i], 1e-2f);
    }
}

TEST(reorder_gpu_f32, dynamic_bfyx_to_fsv16) {
    auto& engine = get_test_engine();

    ov::Shape in_shape{ 1, 2, 4, 2 };
    layout in_layout{ov::PartialShape::dynamic(in_shape.size()), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape(in_shape), data_types::f16, format::bfyx});

    set_values<ov::float16>(input, {
        ov::float16(1.f), ov::float16(0.f),
        ov::float16(5.f), ov::float16(1.5f),

        ov::float16(2.f), ov::float16(0.f),
        ov::float16(6.f), ov::float16(5.2f),

        ov::float16(3.f), ov::float16(0.5f),
        ov::float16(7.f), ov::float16(12.f),

        ov::float16(4.f), ov::float16(-0.5f),
        ov::float16(8.f), ov::float16(8.f)
    });

    topology topology(
        input_layout("input", in_layout),
        reorder("reorder", input_info("input"), format::b_fs_yx_fsv16, data_types::f16),
        activation("relu", input_info("reorder"), activation_func::relu),
        reorder("output_reorder", input_info("relu"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    auto fsv16_reorder_inst = network.get_primitive("reorder");
    auto fsv16_reorder_impl = fsv16_reorder_inst->get_impl();
    ASSERT_TRUE(fsv16_reorder_impl != nullptr);
    ASSERT_TRUE(fsv16_reorder_impl->is_dynamic());

    auto output_reorder_inst = network.get_primitive("output_reorder");
    auto output_reorder_impl = output_reorder_inst->get_impl();
    ASSERT_TRUE(output_reorder_impl != nullptr);
    ASSERT_TRUE(output_reorder_impl->is_dynamic());

    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output_reorder");

    auto output = outputs.begin()->second.get_memory();
    ASSERT_TRUE(output->get_layout().format == format::bfyx);
    auto l = output->get_layout();
    auto expected_shape = ov::PartialShape(in_shape);
    ASSERT_EQ(l.get_partial_shape(), expected_shape);

    float answers[16] = {
        1.f, 0.f,
        5.f, 1.5f,

        2.f, 0.f,
        6.f, 5.2f,

        3.f, 0.5f,
        7.f, 12.f,

        4.f, 0.f,
        8.f, 8.f
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++) {
        ASSERT_NEAR(answers[i], output_ptr[i], 1e-2f);
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
        reorder("reorder", input_info("input"), format::bfzyx, data_types::f32));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    ASSERT_TRUE(output->get_layout().format == format::bfzyx);
    auto l = output->get_layout();
    ASSERT_TRUE(l.batch() == 2);
    ASSERT_TRUE(l.feature() == 2);
    ASSERT_TRUE(l.spatial(0) == 2);
    ASSERT_TRUE(l.spatial(1) == 2);
    ASSERT_TRUE(l.spatial(2) == 1);

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
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
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
        reorder("reorder", input_info("input"), format::bfyx, data_types::f32));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();
    ASSERT_TRUE(output->get_layout().format == format::bfyx);
    auto l = output->get_layout();
    ASSERT_TRUE(l.batch() == 2);
    ASSERT_TRUE(l.feature() == 2);
    ASSERT_TRUE(l.spatial(0) == 2);
    ASSERT_TRUE(l.spatial(1) == 4);
    ASSERT_TRUE(l.spatial(2) == 1);

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
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(reorder_gpu_opt, basic_remove_redundant)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 2, 2, 1 } });
    topology tpl{
        input_layout("in", in->get_layout()),
        reorder("r1", input_info("in"), format::bfyx, data_types::f32),
        reorder("r2", input_info("r1"), format::yxfb, data_types::f32)
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network net(engine, tpl, config);
    net.set_input_data("in", in);
    auto outputs = net.execute();
    auto executed_primitives = net.get_executed_primitives();

    ASSERT_TRUE(executed_primitives.count("r1") == 0);
    ASSERT_TRUE(outputs.count("r2") == 1);
    ASSERT_TRUE(outputs.at("r2").get_memory()->get_layout().format == format::yxfb);
}

TEST(reorder_gpu_opt, remove_redundant_reorder_reorder_with_padding)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 4, 4, 1 } });
    layout r2_output(data_types::f32, format::b_fs_yx_fsv16, { 1, 4, 4, 1 });
    memory::ptr scale_mem = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 1, 1, 1 } });
    set_values(scale_mem, { 2.0f });

    topology tpl{
        input_layout("in", in->get_layout()),
        data("scale_data", scale_mem),
        eltwise("eltwise", { input_info("in"), input_info("scale_data") }, eltwise_mode::prod),
        reorder("r1", input_info("eltwise"), format::bfyx, data_types::f32, std::vector<float>{0, 0, 0, 1}),
        reorder("r2", input_info("r1"), r2_output.with_padding(padding({ 0, 0, 1, 1 }, 0.f))),
        eltwise("output", { input_info("r2"), input_info("scale_data") }, eltwise_mode::prod)
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network net(engine, tpl, config);
    net.set_input_data("in", in);
    auto outputs = net.execute();
    auto executed_primitives = net.get_executed_primitives();

    ASSERT_EQ(executed_primitives.count("r1"), 1);

    // r2 would be removed, but the padding value should be remained at the input primitive of r2.
    std::vector<int32_t> gt = {0, 0, 1, 1};
    auto r1_output_data_padding = net.get_primitive("r1")->get_output_layout().data_padding;
    const auto& upper_padding = r1_output_data_padding._upper_size;
    const auto& lower_padding = r1_output_data_padding._lower_size;
    for (int32_t i = 0 ; i < 4; i++) {
        ASSERT_EQ(upper_padding[i], gt[i]);
        ASSERT_EQ(lower_padding[i], gt[i]);
    }
}

TEST(reorder_gpu_opt, remove_redundant_activation_fuse)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 1, 2, 1 } });
    set_values(in, { -1.0f, -1.0f });
    memory::ptr scale_mem = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 1, 1, 1 } });
    set_values(scale_mem, { 2.0f });
    topology tpl{
        input_layout("in", in->get_layout()),
        reorder("r1", input_info("in"), format::bfyx, data_types::f32),
        activation("relu", input_info("r1"), activation_func::relu_negative_slope, { 0.01f, 0.0f }),
        data("scale_data", scale_mem),
        eltwise("output", { input_info("relu"), input_info("scale_data") }, eltwise_mode::prod)
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network net(engine, tpl, config);
    net.set_input_data("in", in);
    auto outputs = net.execute();
    cldnn::mem_lock<float> out_ptr(outputs.begin()->second.get_memory(), get_test_stream());
    ASSERT_FLOAT_EQ(out_ptr[0], -0.02f);
    ASSERT_FLOAT_EQ(out_ptr[1], -0.02f);
}

TEST(reorder_gpu_opt, basic_remove_redundant_output_due_to_implicit_reorders)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::yxfb, tensor{ 1, 2, 2, 1 } });
    memory::ptr weights = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 2, 2, 1 } });
    topology tpl{
        input_layout("in", in->get_layout()),
        convolution("conv", input_info("in"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
        data("weights", weights),
        reorder("r1", input_info("conv"), format::bfyx, data_types::f32) //optimize data should add conversion from yxfb to bfyx and 'conv' should output data in bfyx as well (OV case)
    };

    ExecutionConfig config = get_test_default_config(engine);

    //we need to check if r1 will be successfully opimized and still we should be able to query for r1's output which should point to conv's output (note conv cannot be marked as output in this case)
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "r1" }));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network net(engine, tpl, config);
    net.set_input_data("in", in);
    auto outputs = net.execute();

    ASSERT_TRUE(outputs.count("conv") == 0);
    ASSERT_TRUE(outputs.count("r1") == 1);
    ASSERT_TRUE(outputs.at("r1").get_memory()->get_layout().format == format::bfyx);
}

TEST(reorder_gpu_opt, basic_remove_redundant_due_to_implicit_reorders)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::yxfb, tensor{ 1, 2, 2, 1 } });
    memory::ptr weights = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 2, 2, 1 } });
    topology tpl{
        input_layout("in", in->get_layout()),
        convolution("conv", input_info("in"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false),
        data("weights", weights),
        reorder("r1", input_info("conv"), format::bfyx, data_types::f32), //optimize data should add conversion from yxfb to bfyx and 'conv' should output data in bfyx as well (OV case)
        softmax("output", input_info("r1"))
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network net(engine, tpl, config);
    net.set_input_data("in", in);
    auto outputs = net.execute();
    auto executed_primitives = net.get_executed_primitives();

    //remove redundant reorder optimization should remove r1 node
    ASSERT_TRUE(executed_primitives.count("r1") == 0);
    //all pirmitives in this test needs to be executed
    ASSERT_TRUE(outputs.count("output") == 1);
    ASSERT_TRUE(outputs.at("output").get_memory()->get_layout().format == format::bfyx);
}

TEST(reorder_gpu_opt, non_trivial_remove_redundant)
{
    auto& engine = get_test_engine();

    memory::ptr in = engine.allocate_memory({ data_types::f32, format::yxfb, tensor{ 1, 1, 5, 2 } });
    topology tpl{
        input_layout("in", in->get_layout()),
        reorder("r1", input_info("in"), format::bfyx, data_types::f32)
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network net(engine, tpl, config);
    net.set_input_data("in", in);
    auto outputs = net.execute();
    auto executed_primitives = net.get_executed_primitives();

    if (engine.get_device_info().supports_immad) {
        // Currently, oneDNN only supports in_order_queue
        return;
    }

    ASSERT_TRUE(executed_primitives.count("in") == 1);
    ASSERT_TRUE(executed_primitives.at("in") != outputs.at("r1").get_event());
    ASSERT_TRUE(outputs.count("r1") == 1);
    ASSERT_TRUE(outputs.at("r1").get_memory()->get_layout().format == format::bfyx);
}

TEST(reorder_gpu_opt, mean_mul)
{
    auto& engine = get_test_engine();

    memory::ptr in  = engine.allocate_memory({ data_types::i8, format::bfyx, tensor{ 1, 3, 1, 2 } });
    memory::ptr mul = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 3, 1, 2 } });

    set_values<char>(in,
    { 1, 2,
      3, 4,
      5, 6 });
    set_values<float>(mul,
    { 0.5f, 2.5f, -5.0f, 4.3f, 1.2f, -3.5f });

    topology tpl{
        input_layout("in", in->get_layout()),
        data("mul",mul),
        reorder("r1", input_info("in"), format::bfyx, data_types::f32,"mul", reorder_mean_mode::mul)
    };

    float answers[] = { 0.5f, 5.0f, -15.0f, 17.2f, 6.0f, -21.0f };
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network net(engine, tpl, config);
    net.set_input_data("in", in);

    auto outputs = net.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> ptr(output, get_test_stream());
    float* a_ptr = answers;
    for (auto& val : ptr)
        ASSERT_FLOAT_EQ(*(a_ptr++), val);

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
        reorder("r1", input_info("in"), format::bfyx, data_types::f32,"mul", reorder_mean_mode::div)
    };

    float answers[] = { 2.0f, 1.0f, -1.0f, 0.5f, 4.0f, -2.0f };
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network net(engine, tpl, config);
    net.set_input_data("in", in);

    auto outputs = net.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> ptr(output, get_test_stream());
    float* a_ptr = answers;
    for (auto& val : ptr)
        ASSERT_FLOAT_EQ(*(a_ptr++), val);

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
        reorder("r1", input_info("in"), format::bfyx, data_types::f32, mul_val, reorder_mean_mode::mul)
    };

    float answers[] = { 2.0f, 4.0f, 1.5f, 2.0f, 50.0f, 600.0f };
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network net(engine, tpl, config);
    net.set_input_data("in", in);

    auto outputs = net.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> ptr(output, get_test_stream());
    float* a_ptr = answers;
    for (auto& val : ptr)
        ASSERT_FLOAT_EQ(*(a_ptr++), val);
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
        reorder("r1", input_info("in"), format::bfyx, data_types::i8, mul_val, reorder_mean_mode::mul)
    };

    char answers[] = { 0, 2, 1, 2, 25, 127 };
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network net(engine, tpl, config);
    net.set_input_data("in", in);

    auto outputs = net.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<char> ptr(output, get_test_stream());
    char* a_ptr = answers;
    for (auto& val : ptr)
        ASSERT_EQ(*(a_ptr++), val);
}

TEST(reorder_gpu_i32, basic)
{
    //  Test for converting data types f32->i32
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::i32, format::bfyx, { 2, 2, 2, 2 });

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), output_layout));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    int32_t answers[16] = {
        1, 0, 5, 1,
        2, 0, 6, 5,
        3, 0, 7, 12,
        4, 0, 8, 8
    };

    int32_t* a_ptr = answers;
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    for (auto& val : output_ptr)
        ASSERT_EQ(*(a_ptr++), val);
}

TEST(reorder_weights_gpu_i32, reorder_weights)
{
    auto& engine = get_test_engine();

    layout in_layout(data_types::f32, format::bfyx, { 2, 2, 2, 2 });
    layout out_layout(data_types::i32, format::oiyx, { 2, 2, 2, 2 });
    auto weights_reorder_params = std::make_shared<WeightsReorderParams>(in_layout, out_layout);

    auto input = engine.allocate_memory(in_layout);

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f
    });

    topology topology {
        input_layout("input", in_layout),
        reorder("reorder", input_info("input"), weights_reorder_params)
    };

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc wr_impl_desc = { format::oiyx, "reorder_weights", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"reorder", wr_impl_desc} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    std::vector<int32_t> ref_output = {
        1, 0, 5, 1,
        2, 0, 6, 5,
        3, 0, 7, 12,
        4, 0, 8, 8
    };

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), ref_output.size());
    for (size_t i = 0; i < ref_output.size(); ++i) {
        ASSERT_EQ(output_ptr[i], ref_output[i]);
    }
}

TEST(reorder_weights_gpu_i32, reorder_weights_opt)
{
    auto& engine = get_test_engine();

    layout in_layout(data_types::f32, format::bfyx, { 16, 1, 2, 1 });
    layout out_layout(data_types::i32, format::os_iyx_osv16, { 16, 1, 2, 1 });
    auto weights_reorder_params = std::make_shared<WeightsReorderParams>(in_layout, out_layout);

    auto input = engine.allocate_memory(in_layout);

    set_values(input, {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        8.f, 9.f, 10.f, 0.5f, 12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f, 20.f, -1.6f, 22.f, 23.f,
        -1.0f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f
    });

    topology topology {
        input_layout("input", in_layout),
        reorder("reorder", input_info("input"), weights_reorder_params)
    };

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc wr_impl_desc = { format::os_iyx_osv16, "reorder_weights_opt", impl_types::ocl };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"reorder", wr_impl_desc} }));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    std::vector<int32_t> ref_output = {
        0, 2, 4, 6, 8, 10, 12, 14,
        16, 18, 20, 22, -1, 26, 28, 30,
        1, 3, 5, 7, 9, 0, 13, 15,
        17, 19, -1, 23, 25, 27, 29, 31
    };

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), ref_output.size());
    for (size_t i = 0; i < ref_output.size(); ++i) {
        ASSERT_EQ(output_ptr[i], ref_output[i]);
    }
}

TEST(reorder_gpu_i64, basic)
{
    //  Test for converting data types f32->i64
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 2, 2 } });
    layout output_layout(data_types::i64, format::bfyx, { 2, 2, 2, 2 });

    set_values(input, {
        1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), output_layout));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    int64_t answers[16] = {
        1, 0, 5, 1,
        2, 0, 6, 5,
        3, 0, 7, 12,
        4, 0, 8, 8
    };

    int64_t* a_ptr = answers;
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());
    for (auto& val : output_ptr)
        ASSERT_EQ(*(a_ptr++), val);
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
        reorder("reorder1", input_info("input"), format::bfwzyx, data_types::f32),
        reshape("reshape1", input_info("reorder1"), tensor(batch(2), feature(2), spatial(1, 1, 2, 2) )),
        reorder("reorder2", input_info("reshape1"), format::bfwzyx, data_types::f32, sub_bfwzyx),
        reshape("reshape2", input_info("reorder2"), tensor(batch(4), feature(2), spatial(1, 1, 1, 2))),
        reshape("reshape3", input_info("reshape2"), tensor(batch(1), feature(4), spatial(2, 2))),
        reorder("reorder3", input_info("reshape3"), format::bfyx, data_types::f32, sub_bfyx),
        reorder("out_reorder", input_info("reorder3"), format::bfwzyx, data_types::f32)
        );
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    ASSERT_TRUE(output->get_layout().format == format::bfwzyx);
    auto l = output->get_layout();
    ASSERT_EQ(l.batch(), 1);
    ASSERT_EQ(l.feature(), 4);
    ASSERT_EQ(l.spatial(0), 2);
    ASSERT_EQ(l.spatial(1), 2);
    ASSERT_EQ(l.spatial(2), 1);

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), expected.size());

    for (size_t i = 0; i < expected.size(); i++)
    {
        ASSERT_FLOAT_EQ(expected[i], output_ptr[i]);
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
            reorder("reorder", input_info("input"), output_layout));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

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
                        int32_t bsv16_fsv16_index = get_bsv16_fsv16_index(b_in, f_in, z_in, y_in, x_in, b,
                                                                          0, f, 0,
                                                                          0, z, 0,
                                                                          0, y, 0,
                                                                          0, x, 0);
                        ASSERT_FLOAT_EQ(input_ptr[linear_index++], output_ptr[bsv16_fsv16_index]);
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

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { b_in, f_in, x_in, y_in, z_in } });
    layout output_layout(data_types::f32, format::bs_fs_zyx_bsv16_fsv16, { b_in, f_in, x_in, y_in, z_in });

    tests::set_random_values<float>(input);

    topology topology(
            input_layout("input", input->get_layout()),
            reorder("reorder", input_info("input"), output_layout.with_padding(padding({ 0, 0, z_pad, y_pad, x_pad }, 0.f))));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

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
                        int32_t bsv16_fsv16_index = get_bsv16_fsv16_index(b_in, f_in, z_in, y_in, x_in, b,
                                                                          f_pad, f, f_pad,
                                                                          z_pad, z, z_pad,
                                                                          y_pad, y, y_pad,
                                                                          x_pad, x, x_pad);
                        ASSERT_FLOAT_EQ(input_ptr[linear_index++], output_ptr[bsv16_fsv16_index]);
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
            activation("first_activation", input_info("input"), activation_func::abs),
            reorder(reorder_name, input_info("first_activation"), format::bfyx, data_types::f32),
            activation("second_activation", input_info(reorder_name), activation_func::abs));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto executed_prims = network.get_executed_primitives();

    ASSERT_TRUE(executed_prims.find(reorder_name) == executed_prims.end());
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "second_activation");

    auto output = outputs.begin()->second.get_memory();

    float answers[24] = {
            0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f,
            16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f,
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), 24);
    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]) << "i=" << i;
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
            reorder(reorder_name, input_info("input"), format::bfyx, data_types::f32),
            convolution("convolution", input_info(reorder_name), "weights", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto executed_prims = network.get_executed_primitive_ids();

    EXPECT_FALSE(std::find(executed_prims.begin(), executed_prims.end(), reorder_primitive_name) != executed_prims.end());
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "convolution");

    auto output = outputs.begin()->second.get_memory();

    float answers[1] = { 28.f };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int i = 0; i < 1; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]) << "i=" << i;
    }
}

TEST(reorder_gpu_f32, b_fs_yx_fsv16_to_bfyx_opt_padded)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32,
                                            format::b_fs_yx_fsv16,
                                            { 2, 4, 1, 1 },
                                            padding({ 1, 16, 0, 0 }, { 1, 0, 0, 0 }) });

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
        reorder(reorder_name, input_info("input"), format::bfyx, data_types::f32),
        activation("activation", input_info(reorder_name), activation_func::abs));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto executed_prims = network.get_executed_primitives();

    ASSERT_TRUE(executed_prims.find(reorder_name) == executed_prims.end());
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "activation");

    auto output = outputs.begin()->second.get_memory();

    float answers[8] = {
            0.f, 1.f, 2.f, 3.f,
            16.f, 17.f, 18.f, 19.f,
    };

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), 8);
    for (size_t i = 0; i < output_ptr.size(); i++) {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]) << "i=" << i;
    }
}

TEST(reorder_gpu, any_format) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory(layout(data_types::f32, format::yxfb, tensor(5, 7, 13, 9)));

    topology topo;
    topo.add(input_layout("in", input->get_layout()));
    topo.add(reorder("out", input_info("in"), format::any, data_types::f32));

    network net(engine, topo, get_test_default_config(engine));

    auto data = rg.generate_random_1d<float>(input->count(), -1, 1);
    set_values(input, data);
    net.set_input_data("in", input);

    auto outputs = net.execute();
    auto out_mem = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output(out_mem, get_test_stream());

    for (size_t i = 0; i < data.size(); ++i) {
        ASSERT_EQ(output[i], data[i]) << "i = " << i;
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
        reorder("reorder", input_info("input"), output_layout));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

    auto output = outputs.begin()->second.get_memory();

    float answers[12] = {
        1.0f,  2.0f,
        124.0f,  251.0f,

        0.0f,  111.0f,
        125.0f,  252.0f,

        5.0f,  123.0f,
        50.0f, 253.0f,
    };

    cldnn::mem_lock<ov::float16> output_ptr (output, get_test_stream());
    for (int i = 0; i < 12; i++)
    {
        ASSERT_NEAR(ov::float16(answers[i] / 255.f), output_ptr[i], 1e-3f);
    }

}

TEST(reorder_bfyx_to_image2d_rgba_gpu, basic)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 3, 2, 2 } });
    layout output_layout(data_types::u8, format::image_2d_rgba, { 1, 3, 2, 2 });

    set_values<ov::float16>(input, {
        ov::float16(1.0f / 255.f),  ov::float16(2.0f / 255.f),
        ov::float16(124.0f / 255.f),  ov::float16(251.0f / 255.f),

        ov::float16(0.0f / 255.f),  ov::float16(111.0f / 255.f),
        ov::float16(125.0f / 255.f),  ov::float16(252.0f / 255.f),

        ov::float16(5.0f / 255.f),  ov::float16(123.0f / 255.f),
        ov::float16(50.0f / 255.f), ov::float16(253.0f / 255.f),
        });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), output_layout));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder");

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
        ASSERT_EQ(answers[i], output_ptr[i]);
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
            cldnn::tensor input_tensor = test_param->input_layouts[0].get_tensor();

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
                all_test_params.emplace_back(std::make_tuple(test_param, std::make_shared<reorder>("reorder", input_info("input0"), output_layout, subtract)));

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

        auto output = engine.allocate_memory(cldnn::layout(*reorder->output_data_types[0], inputs[0]->get_layout().format, inputs[0]->get_layout().get_tensor()));

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
            if (*layer_params->output_data_types[0] == data_types::f32)
            {
                return generate_reference_typed<float, float>(inputs);
            }
            else
            {
                return generate_reference_typed<float, ov::float16>(inputs);
            }
        }
        else
        {
            if (*layer_params->output_data_types[0] == data_types::f32)
            {
                return generate_reference_typed<ov::float16, float>(inputs);
            }
            else
            {
                return generate_reference_typed<ov::float16, ov::float16>(inputs);
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


struct reorder_test_param {
    tensor in_shape;
    tensor out_shape;
    tensor kernel;
    ov::Strides stride;
    ov::CoordinateDiff pad;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
};

template<typename T>
class ReorderTest : public ::testing::TestWithParam<T> {
public:
    tests::random_generator rg;
    cldnn::engine& engine = get_test_engine();
    cldnn::topology topology_test;
    ExecutionConfig config = get_test_default_config(engine);
    static const int min_random = -200;
    static const int max_random = 200;
    std::vector<primitive_id> executed_prims;

    void execute(T& p, bool is_caching_test) {
        auto input_prim = this->get_mem(get_input_layout(p));

        cldnn::network::ptr network_test = get_network(this->engine, this->topology_test, this->config, get_test_stream_ptr(), is_caching_test);

        network_test->set_input_data("input", input_prim);

        executed_prims = network_test->get_executed_primitive_ids();
    }

    std::shared_ptr<primitive_inst> execute_and_query(T& p, primitive_id prim_id) {
        auto input_prim = this->get_mem(get_input_layout(p));

        cldnn::network::ptr network_test = get_network(this->engine, this->topology_test, this->config, get_test_stream_ptr(), false);

        network_test->set_input_data("input", input_prim);

        executed_prims = network_test->get_executed_primitive_ids();

        return network_test->get_primitive(prim_id);
    }

    bool check_optimized_out(T& p, primitive_id target_id) {
        for (auto& prim : executed_prims)
            if (prim == target_id)
                return false;

        return true;
    }

    bool check_supports_immad() {
        return this->engine.get_device_info().supports_immad;
    }

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
        config.set_property(ov::intel_gpu::optimize_data(true));
    }

    void setup_with_build_ops(const ExecutionConfig& c) {
        config = c;
    }

    cldnn::memory::ptr get_mem(cldnn::layout l) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.get_tensor();
        if (l.data_type == data_types::i8 || l.data_type == data_types::u8) {
            VF<uint8_t> rnd_vec = rg.generate_random_1d<uint8_t>(s.count(), min_random, max_random);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<uint16_t> rnd_vec = rg.generate_random_1d<uint16_t>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        } else {
            VF<float> rnd_vec = rg.generate_random_1d<float>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    layout get_input_layout(T& p) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, static_cast<int>(pad[1]), static_cast<int>(pad[0]) };
    return layout{ p.data_type, p.input_format, p.in_shape, padding{pad_} };
    }

    layout get_weights_layout(T& p) {
        cldnn::tensor weights_tensor;
        weights_tensor = cldnn::tensor(batch(p.out_shape.feature[0]), feature(p.in_shape.feature[0]), spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        return layout{p.weights_type, p.weights_format, weights_tensor};
    }

    layout get_bias_layout(T& p) {
        return layout{ p.default_type, format::bfyx, tensor{1, p.out_shape.feature[0], 1, 1} };
    }

    template <class... Args>
    void create_topologies(Args const&... args) {
        topology_test.add(args...);
    }
};

class testing_removal_reorder : public ReorderTest<reorder_test_param> {};
// Testing bugfix not to remove reorder in front of conv has deep depth input
TEST_P(testing_removal_reorder, only_remove_reorder_shallow_depth_input) {
    auto p = GetParam();
    layout reorder_layout(data_types::u8, format::b_fs_yx_fsv32, p.in_shape, padding({0, }, 0));

    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("weights_sec", get_mem(get_weights_layout(p))),
        reorder("reorder_fp32", input_info("input"), format::bfyx, data_types::f32),
        convolution("conv_prim", input_info("reorder_fp32"), "weights", "bias", 1, p.stride, {1, 1}, p.pad, p.pad, false),
        reorder("reorder_conv", input_info("conv_prim"), reorder_layout),
        convolution("conv_output", input_info("reorder_conv"), "weights_sec", "", 1, p.stride, {1, 1}, p.pad, p.pad, false),
        reorder("reorder_bfyx", input_info("conv_output"), format::b_fs_yx_fsv32, data_types::f32),
        resample("resample", input_info("reorder_bfyx"), p.out_shape, 1),
        reorder("reorder_output", input_info("resample"), p.default_format, data_types::f32)
    );

    execute(p, false);

    ASSERT_EQ(check_optimized_out(p, "reorder_conv"), false);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
// Check to remove reorder between onednn and cldnn conv if the reorder has no padded output
TEST_P(testing_removal_reorder, removal_no_padded_reorder) {
    if (!engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    layout reorder_layout(data_types::f16, format::b_fs_yx_fsv16, p.in_shape, padding({0, }, 0));

    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        reorder("reorder_fp32", input_info("input"), format::bfyx, data_types::f16),
        convolution("conv_prim", input_info("reorder_fp32"), "weights", "", 1, p.stride, {1, 1}, p.pad, p.pad, false),
        reorder("reorder_conv", input_info("conv_prim"), reorder_layout),
        convolution("conv_output", input_info("reorder_conv"), "weights", "", 1, p.stride, {1, 1}, p.pad, p.pad, false),
        reorder("reorder_output", input_info("conv_output"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, std::string(""), impl_types::ocl };
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv_output", impl} }));

    setup_with_build_ops(config);

    execute(p, false);

    ASSERT_EQ(check_optimized_out(p, "reorder_conv"), true);
}

// Check not to remove reorder between onednn and cldnn conv if the reorder has padded output
TEST_P(testing_removal_reorder, removal_padded_reorder) {
    if (!engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    layout reorder_layout(data_types::f16, format::b_fs_yx_fsv16, p.in_shape, padding({0, 0, 1, 1}, 0));

    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        reorder("reorder_fp32", input_info("input"), format::bfyx, data_types::f16),
        convolution("conv_prim", input_info("reorder_fp32"), "weights", "", 1, p.stride, {1, 1}, p.pad, p.pad, false),
        reorder("reorder_conv", input_info("conv_prim"), reorder_layout),
        convolution("conv_output", input_info("reorder_conv"), "weights", "", 1, p.stride, {1, 1}, p.pad, p.pad, false),
        reorder("reorder_output", input_info("conv_output"), p.default_format, data_types::f32)
    );

    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, std::string(""), impl_types::ocl };
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv_output", impl} }));

    setup_with_build_ops(config);

    execute(p, false);

    ASSERT_EQ(check_optimized_out(p, "reorder_conv"), false);
}
#endif // ENABLE_ONEDNN_FOR_GPU

INSTANTIATE_TEST_SUITE_P(reorder_gpu_testing, testing_removal_reorder,
                        ::testing::ValuesIn(std::vector<reorder_test_param>{
                                            reorder_test_param{{1, 32, 4, 4}, {1, 32, 8, 8}, {1, 1, 1, 1}, {1, 1}, {0, 0},
                                                                data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f16, format::b_fs_yx_fsv16},
                                            }));

#ifdef ENABLE_ONEDNN_FOR_GPU
class testing_onednn_reorder : public ReorderTest<reorder_test_param> {};
// Check that onednn reorder is chosen at target scenario
TEST_P(testing_onednn_reorder, basic_selection) {
    if (!engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    layout reorder_layout(data_types::f16, format::bfyx, p.in_shape);

    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", input_info("input"), "weights", "", 1, p.stride, {1, 1}, p.pad, p.pad, false),
        reorder("reorder_conv", input_info("conv_prim"), reorder_layout)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    setup_with_build_ops(config);

    const auto& target_reorder = execute_and_query(p, "reorder_conv");
    ASSERT_EQ(target_reorder->get_node().get_preferred_impl_type(), impl_types::onednn);
}

INSTANTIATE_TEST_SUITE_P(basic_onednn_reorder, testing_onednn_reorder,
                        ::testing::ValuesIn(std::vector<reorder_test_param>{
                                            reorder_test_param{{1, 32, 16, 16}, {1, 32, 16, 16}, {1, 1, 1, 1}, {1, 1}, {0, 0},
                                                                data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::oiyx, data_types::f16, format::b_fs_yx_fsv16},
                                            }));

#endif // ENABLE_ONEDNN_FOR_GPU

struct redundant_reorder_test_param {
    tensor in_shape;
    tensor out_shape;
    tensor kernel;
    ov::Strides stride;
    ov::CoordinateDiff pad;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    bool opt_out;
};

class testing_removal_1d_reorder : public ReorderTest<redundant_reorder_test_param> {};
TEST_P(testing_removal_1d_reorder, removal_reorder_1d_along_f_mixed_format) {
    auto p = GetParam();

    std::vector<int> pad = { 0, 0, static_cast<int>(p.pad[1]), static_cast<int>(p.pad[0]) };
    layout in_layout{ p.data_type, p.input_format, p.in_shape, padding{pad} };

    create_topologies(input_layout("input", in_layout),
                data("weights", get_mem(get_weights_layout(p))),
                data("bias1", get_mem(get_bias_layout(p))),
                reorder("reorder_bias1", input_info("bias1"), format::b_fs_yx_fsv32, data_types::f16),
                convolution("conv_prim", input_info("input"), "weights", "", 1, p.stride, {1, 1}, p.pad, p.pad, false),
                reorder("reorder_conv", input_info("conv_prim"), format::b_fs_yx_fsv32, data_types::f16),
                eltwise("add_bias1", { input_info("reorder_conv"), input_info("reorder_bias1") }, eltwise_mode::sum),
                reorder("reorder_bfyx", input_info("add_bias1"), p.default_format, data_types::f16)
    );

    const auto& target_reorder = execute_and_query(p, "reorder_conv");
    ASSERT_EQ(check_optimized_out(p, "reorder_conv"), false);
    ASSERT_EQ(target_reorder->can_be_optimized(), true);
}

// Negative : reorder is padded
TEST_P(testing_removal_1d_reorder, padded_reorder_1d_along_f_mixed_format) {
    auto p = GetParam();

    std::vector<int> pad = { 0, 0, static_cast<int>(p.pad[1]), static_cast<int>(p.pad[0]) };
    layout in_layout{ p.data_type, p.input_format, p.in_shape, padding{pad} };

    layout reorder_layout(data_types::f16, format::b_fs_yx_fsv32, p.out_shape, padding({0, 0, 1, 1}, 0));

    create_topologies(input_layout("input", in_layout),
                data("weights", get_mem(get_weights_layout(p))),
                data("bias1", get_mem(get_bias_layout(p))),
                reorder("reorder_bias1", input_info("bias1"), format::b_fs_yx_fsv32, data_types::f16),
                convolution("conv_prim", input_info("input"), "weights", "", 1, p.stride, {1, 1}, p.pad, p.pad, false),
                reorder("reorder_conv", input_info("conv_prim"), reorder_layout),
                eltwise("add_bias1", { input_info("reorder_conv"), input_info("reorder_bias1") }, eltwise_mode::sum),
                reorder("reorder_bfyx", input_info("add_bias1"), p.default_format, data_types::f16)
    );

    const auto& target_reorder = execute_and_query(p, "reorder_conv");
    ASSERT_EQ(check_optimized_out(p, "reorder_conv"), false);
    ASSERT_EQ(target_reorder->can_be_optimized(), false);
}

INSTANTIATE_TEST_SUITE_P(reorder_gpu_testing_1d_removal, testing_removal_1d_reorder,
                        ::testing::ValuesIn(std::vector<redundant_reorder_test_param>{
                                            redundant_reorder_test_param{{1, 32, 1, 1}, {1, 32, 1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0},
                                                                data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::oiyx, data_types::f16, format::b_fs_yx_fsv16, false},
                                            }));

class testing_removal_feature_aligned_reorder : public ReorderTest<redundant_reorder_test_param> {};
TEST_P(testing_removal_feature_aligned_reorder, removal_reorder_aligned_mixed_format) {
    auto p = GetParam();

    std::vector<int> pad = { 0, 0, static_cast<int>(p.pad[1]), static_cast<int>(p.pad[0]) };
    layout in_layout{ p.data_type, p.input_format, p.in_shape, padding{pad} };

    create_topologies(input_layout("input", in_layout),
                data("bias1", get_mem(get_bias_layout(p))),
                reorder("reorder_bias1", input_info("bias1"), format::b_fs_yx_fsv32, data_types::f16),
                reorder("reorder_input", input_info("input"), format::b_fs_yx_fsv32, data_types::f16),
                eltwise("add_bias1", { input_info("reorder_input"), input_info("reorder_bias1") }, eltwise_mode::sum),
                reorder("reorder_bfyx", input_info("add_bias1"), p.default_format, data_types::f16)
    );

    const auto& target_reorder = execute_and_query(p, "reorder_input");
    ASSERT_EQ(check_optimized_out(p, "reorder_input"), false);
    ASSERT_EQ(target_reorder->can_be_optimized(), p.opt_out);
}

// Negative : reorder is padded
TEST_P(testing_removal_feature_aligned_reorder, padded_reorder_aligned_mixed_format) {
    auto p = GetParam();

    std::vector<int> pad = { 0, 0, static_cast<int>(p.pad[1]), static_cast<int>(p.pad[0]) };
    layout in_layout{ p.data_type, p.input_format, p.in_shape, padding{pad} };

    layout reorder_layout(data_types::f16, format::b_fs_yx_fsv32, p.out_shape, padding({0, 0, 1, 1}, 0));

    create_topologies(input_layout("input", in_layout),
                data("bias1", get_mem(get_bias_layout(p))),
                reorder("reorder_bias1", input_info("bias1"), format::b_fs_yx_fsv32, data_types::f16),
                reorder("reorder_input", input_info("input"), reorder_layout),
                eltwise("add_bias1", { input_info("reorder_input"), input_info("reorder_bias1") }, eltwise_mode::sum),
                reorder("reorder_bfyx", input_info("add_bias1"), p.default_format, data_types::f16)
    );

    const auto& target_reorder = execute_and_query(p, "reorder_input");
    ASSERT_EQ(check_optimized_out(p, "reorder_input"), false);
    ASSERT_EQ(target_reorder->can_be_optimized(), false);
}

INSTANTIATE_TEST_SUITE_P(reorder_gpu_testing_1d_removal, testing_removal_feature_aligned_reorder,
                        ::testing::ValuesIn(std::vector<redundant_reorder_test_param>{
                                            redundant_reorder_test_param{{1, 32, 8, 8}, {1, 32, 8, 8}, {1, 1, 1, 1}, {1, 1}, {0, 0},
                                                                data_types::f16, format::byxf, data_types::f16, format::goiyx, data_types::f16, format::b_fs_yx_fsv16, true},
                                            redundant_reorder_test_param{{1, 32, 1, 1}, {1, 32, 1, 1}, {1, 1, 1, 1}, {1, 1}, {0, 0},
                                                                data_types::f16, format::byxf, data_types::f16, format::goiyx, data_types::f16, format::b_fs_yx_fsv16, true},
                                            redundant_reorder_test_param{{1, 64, 8, 8}, {1, 64, 8, 8}, {1, 1, 1, 1}, {1, 1}, {0, 0},
                                                                data_types::f16, format::byxf, data_types::f16, format::goiyx, data_types::f16, format::b_fs_yx_fsv16, false},
                                            redundant_reorder_test_param{{1, 1, 1, 32}, {1, 1, 1, 32}, {1, 1, 1, 1}, {1, 1}, {0, 0},
                                                                data_types::f16, format::byxf, data_types::f16, format::goiyx, data_types::f16, format::b_fs_yx_fsv16, false},
                                            redundant_reorder_test_param{{1, 1, 1, 32}, {1, 1, 1, 32}, {1, 1, 1, 1}, {1, 1}, {0, 0},
                                                                data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::goiyx, data_types::f16, format::b_fs_yx_fsv16, false},
                                            }));

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(reorder_onednn_gpu, basic_convert_int8) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;
    layout in_layout = { ov::element::from<float>(), format::byxf, { 1, 1, 3, 3 } };
    layout byte_layout = { ov::element::from<int8_t>(), format::bfyx, { 1, 1, 3, 3 } };
    std::initializer_list<float> input_f = { 1.0f, -2.6f, 3.1f, -4.0f, 5.03f, -6.99f, 7.0f, -8.0f, 9.0f };
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
            input_info(input),
            // output layout for reorder
            byte_layout),
        reorder("reorder2", input_info("reorder_input"), in_layout)
    );

    ov::intel_gpu::ImplementationDesc impl = { format::bfyx, std::string(""), impl_types::onednn };
    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{ "reorder_input", "reorder2"}));
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{ "reorder_input", impl }}));

    network network(
        engine,
        topology,
        cfg);

    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    auto interm = outputs.at("reorder2").get_memory();
    cldnn::mem_lock<float> interm_ptr(interm, get_test_stream());
    unsigned int cntr = 0;
    for (const auto& exp : final_results)
    {
        ASSERT_EQ(exp, interm_ptr[cntr++]);
    }
}
#endif // ENABLE_ONEDNN_FOR_GPU

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv32_to_bfyx_f32_cached) {
    // b_fs_yx_fsv32 -> bfyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfyx, 3, 64 + 5, 16 + 11, 3, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfyx, 3, 96 - 12, 16 + 4, 3, 0, 0, true);
    // b_fs_zyx_fsv32 -> bfzyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfzyx, 3, 64 + 9, 16 - 1, 2, 8, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfzyx, 2, 64 + 30, 16 + 1, 3, 4, 0, true);
    // incremental dims
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfzyx, 2, 64 + 4, 24 - 1, 3, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv32, format::bfwzyx, 2, 64 + 2, 32 - 3, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv32, format::bfwzyx, 1, 96 + 10, 32 - 3, 4, 3, 0, true);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv32_to_bfyx_different_datatype_cached) {
    // f32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::u8, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 8 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i64, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 16 + 2, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f16, format::b_fs_yx_fsv32, format::bfyx, 1, 64, 16 + 1, 2, 0, 0, true);
    // i32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i8, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 8 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i64, format::b_fs_yx_fsv32, format::bfyx, 2, 64, 16 + 2, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::f16, format::b_fs_yx_fsv32, format::bfyx, 1, 64, 16 + 1, 2, 0, 0, true);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv16_to_bfyx_f32_cached) {
    // u-net
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 1, 64, 388, 388, 0, 0, true);
    // b_fs_yx_fsv16 -> bfyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 3, 48 + 1, 16, 3, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 2, 32 - 1, 24 - 1, 3, 0, 0, true);
    // b_fs_zyx_fsv16 -> bfzyx
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx, 5, 48 - 1, 16, 3, 8, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx, 2, 32 + 1, 24 - 1, 3, 17, 0, true);
    // incremental dims
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfzyx, 3, 32 - 1, 24 - 1, 3, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfwzyx, 4, 16 + 1, 32 - 3, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx, 3, 16 + 2, 32 - 3, 4, 9, 0, true);
}

TEST(reorder_gpu_optimization, compare_with_ref__b_fs_yx_fsv16_to_bfyx_different_datatype_cached) {
    // f32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::u8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i32, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::i64, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::f32, data_types::f16, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
    // i32 -> other types
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::u8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i8, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::i64, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::f16, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_b_fs_yx_fsv16_fsv32_to_bfyx", data_types::i32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx, 2, 32, 16 + 7, 2, 0, 0, true);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_blocked_f32_cached) {
    // bfyx_to_b_fs_yx_fsv4
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv4, 4, 32, 16, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv4, 3, 32 + 2, 32 + 3, 4, 0, 0, true);
    // bfyx_to_b_fs_yx_fsv16
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 2, 48, 8, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2, 0, 0, true);
    // bfyx to b_fs_yx_fsv32
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv32, 2, 64, 64, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::b_fs_yx_fsv32, 4, 32 + 6, 96 - 4, 2, 0, 0, true);
    // bfyx to fs_b_yx_fsv32
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::fs_b_yx_fsv32, 2, 64, 8, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::fs_b_yx_fsv32, 3, 64 + 5, 8 + 7, 2, 0, 0, true);
    // bfzyx to b_fs_zyx_fsv16
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv16, 2, 48, 8, 4, 4, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv16, 3, 32 + 5, 16 + 7, 2, 2, 0, true);
    // bfzyx to b_fs_zyx_fsv32
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv32, 2, 64, 8, 4, 4, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::b_fs_zyx_fsv32, 3, 64 + 5, 8 + 7, 2, 2, 0, true);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f32_cached) {
    // bfyx to double blocked format (bs_fs_yx_bsv16_fsv16)
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32, 48, 8, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32 + 2, 48, 16, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32, 48 + 5, 16, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32, 48, 48 + 3, 4, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfyx, format::bs_fs_yx_bsv16_fsv16, 32 + 2, 48 + 3, 16 + 1, 4, 0, 0, true);
    // bfzyx to double blocked format (bs_fs_zyx_bsv16_fsv16)
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32, 48, 8, 4, 16, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32 + 2, 48, 16, 4, 2, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32, 48 + 5, 16, 4, 3, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32, 48, 48 + 3, 4, 4, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv16, 32 + 2, 48 + 3, 16 + 1, 4, 2, 0, true);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f32_bsv16_fsv32_cached) {
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 3, 16, 4, 5, 7, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 1, 1, 1, 1, 1, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32 + 2, 48, 16, 4, 2, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32 + 1, 1, 1, 1, 1, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32, 48 + 5, 16, 4, 3, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32, 48, 48 + 3, 4, 4, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv16_fsv32, 32 + 2, 48 + 3, 16 + 1, 4, 2, 0, true);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f32_bsv32_fsv16_cached) {
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 1, 1, 1, 1, 1, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 32 + 2, 48, 16, 4, 2, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 32, 48 + 5, 16, 4, 3, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 32, 48, 48 + 3, 4, 4, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv16, 32 + 2, 48 + 3, 16 + 1, 4, 2, 0, true);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_double_blocked_f32_bsv32_fsv32_cached) {
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 1, 1, 1, 1, 1, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 32 + 2, 48, 16, 4, 2, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 32, 48 + 5, 16, 4, 3, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 32, 48, 48 + 3, 4, 4, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f32, format::bfzyx, format::bs_fs_zyx_bsv32_fsv32, 32 + 2, 48 + 3, 16 + 1, 4, 2, 0, true);
}

TEST(reorder_gpu_optimization, compare_with_ref__bfyx_to_blocked_format_different_datatype_cached) {
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::f32, data_types::f16, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::i8, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2, 0, 0, true);
    compare_bfyx2blocked_with_ref("reorder_data_bfyx_to_blocked_format", data_types::i64, data_types::f32, format::bfyx, format::b_fs_yx_fsv16, 3, 32 + 4, 16 + 7, 2, 0, 0, true);
}

#endif

TEST_P(testing_removal_reorder, only_remove_reorder_shallow_depth_input_cached) {
    auto p = GetParam();
    layout reorder_layout(data_types::u8, format::b_fs_yx_fsv32, p.in_shape, padding({0, }, 0));

    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("weights_sec", get_mem(get_weights_layout(p))),
        reorder("reorder_fp32", input_info("input"), format::bfyx, data_types::f32),
        convolution("conv_prim", input_info("reorder_fp32"), "weights", "bias", 1, p.stride, {1, 1}, p.pad, p.pad, false),
        reorder("reorder_conv", input_info("conv_prim"), reorder_layout),
        convolution("conv_output", input_info("reorder_conv"), "weights_sec", "", 1, p.stride, {1, 1}, p.pad, p.pad, false),
        reorder("reorder_bfyx", input_info("conv_output"), format::b_fs_yx_fsv32, data_types::f32),
        resample("resample", input_info("reorder_bfyx"), p.out_shape, 1),
        reorder("reorder_output", input_info("resample"), p.default_format, data_types::f32)
    );

    execute(p, true);

    ASSERT_EQ(check_optimized_out(p, "reorder_conv"), false);
}

TEST(reorder_gpu_fp32, test_needs_completion_events) {
    // input1  input2
    //    |      |
    //    |      |
    //    \     /
    //      mul
    //       |
    //  permute(skippable)
    //       |
    //  reorder1(skippable)
    //       |
    //  reorder2(not skippable)
    //

    auto& engine = get_test_engine();

    auto input_dynamic_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto input_static_layout = layout{ { 2, 1, 2, 8 }, data_types::f32, format::bfyx };

    auto input1 = engine.allocate_memory(input_static_layout);
    auto input2 = engine.allocate_memory(input_static_layout);

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f,
        16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f,
        42.f, 42.f, 42.f, 42.f, 42.f, 42.f, 42.f, 42.f
    };

    set_values(input1, expected_results);

    set_values(input2, {
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f
    });

    topology topology;
    topology.add(input_layout("input1", input_dynamic_layout));
    topology.add(input_layout("input2", input_dynamic_layout));
    topology.add(eltwise("mul", { input_info("input1"), input_info("input2")}, eltwise_mode::prod));
    topology.add(permute("permute", input_info("mul"), { 0, 1, 2, 3 }));
    topology.add(reorder("reorder1", input_info("permute"), format::bfyx, data_types::f32));
    topology.add(reorder("reorder2", input_info("reorder1"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    auto force_impl = ov::intel_gpu::ImplementationDesc{ format::bfyx, "", impl_types::cpu };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {primitive_id("reorder2"), force_impl} }));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("reorder2");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("reorder2").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}
