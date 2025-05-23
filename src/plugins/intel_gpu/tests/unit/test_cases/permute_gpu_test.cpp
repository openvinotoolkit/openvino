// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type_traits.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/crop.hpp>

#include <pass_manager.h>
#include <program_wrapper.h>
#include <test_utils.h>

#include "permute_inst.h"

#include <cmath>
#include <limits>
#include <type_traits>

using namespace cldnn;
using namespace ::tests;
using namespace testing;

namespace {

// TODO: Move somewhere
template <class vecElementType>
std::string vec2str(const std::vector<vecElementType>& vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<vecElementType>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return "()";
}

}  // namespace

TEST(permute_gpu_f32, output_ordering_test)
{
    auto& engine = get_test_engine();

    std::vector<std::vector<int32_t>> input_tensors = {
        { 10, 5, 15, 2 },
        { 2, 4, 6, 8 },
        { 2, 2, 3, 2 },
        { 9, 8, 7, 4 }
    };
    std::vector<std::vector<uint16_t>> permutations = {
        { 0, 1, 2, 3 }, //do nothing
        { 0, 1, 3, 2 }, //replace x with y
        { 1, 0, 3, 2 }, //replace b with f
        { 0, 2, 3, 1 }  //big permutation
    };
    std::vector<format> input_formats = { format::bfyx, format::yxfb };

    auto get_permutation = [&](const std::vector<int32_t>& inp1, const std::vector<uint16_t>& order) -> std::vector<int32_t> {
        EXPECT_EQ(inp1.size(), order.size());
        std::vector<int32_t> output;
        for (auto const& o : order) {
            output.push_back(inp1.at(o));
        }
        return output;
    };

    for (auto const& fr : input_formats) {
        for (auto const& inp_t : input_tensors) {
            for (auto const& perm : permutations) {
                auto input = engine.allocate_memory({ data_types::f32, fr, tensor(format::bfyx, inp_t) });
                topology topology(
                    input_layout("input", input->get_layout()),
                    permute("permute", input_info("input"), perm));

                network network(engine, topology, get_test_default_config(engine));
                network.set_input_data("input", input);
                auto outputs = network.execute();
                auto output = outputs.at("permute");
                auto output_mem = output.get_memory();
                ASSERT_EQ(outputs.size(), size_t(1));
                auto ref_tensor = get_permutation(inp_t, perm);
                auto dims = output_mem->get_layout().get_dims();
                ASSERT_EQ(dims[0], ref_tensor[0]);
                ASSERT_EQ(dims[1], ref_tensor[1]);
                ASSERT_EQ(dims[2], ref_tensor[2]);
                ASSERT_EQ(dims[3], ref_tensor[3]);
            }
        }
    }
}

TEST(permute_gpu_f32, basic_bfyx_permute_0_1_2_3)
{
    //  Input               : bfyx:2x2x3x2
    //  Permute order       : { 0,1,3,2 }
    //
    //  Input:
    //  f0: b0:  1    2   -15  b1:   0    0     -15
    //  f0: b0:  3    4   -15  b1:   0.5 -0.5   -15
    //  f1: b0:  5    6   -15  b1:   1.5  5.2   -15
    //  f1: b0:  7    8   -15  b1:   12   8     -15
    //
    //  Output = input

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });

    std::vector<float> values =
    {
        1.0f,  2.0f, -15.f,
        3.0f,  4.0f, -15.f,

        5.0f,  6.0f, -15.f,
        7.0f,  8.0f, -15.f,

        0.0f,  0.0f, -15.f,
        0.5f, -0.5f, -15.f,

        1.5f,  5.2f, -15.f,
        12.0f, 8.0f, -15.f
    };

    set_values(input, values);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 1, 2, 3 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < 24; i++)
    {
        ASSERT_FLOAT_EQ(values[i], output_ptr[i]);
    }

}

TEST(permute_gpu_f32, basic_bfyx_permute_0_1_3_2)
{
    //  Input               : bfyx:2x2x3x2
    //  Permute order       : { 0,1,3,2 }
    //
    //  Input:
    //  f0: b0:  1    2   -15  b1:   0    0     -15
    //  f0: b0:  3    4   -15  b1:   0.5 -0.5   -15
    //  f1: b0:  5    6   -15  b1:   1.5  5.2   -15
    //  f1: b0:  7    8   -15  b1:   12   8     -15
    //
    //  Output
    //  f0: b0:  1    3  b1:   0    0.5
    //  f0: b0:  2    4  b1:   0    -0.5
    //  f0: b0:  -15 -15 b1:   -15  -15
    //  f1: b0:  5    7  b1:   1.5  12
    //  f1: b0:  6    8  b1:   5.2   8
    //  f1: b0:  -15 -15 b1:   -15   -15
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,
        3.0f,  4.0f, -15.f,

        5.0f,  6.0f, -15.f,
        7.0f,  8.0f, -15.f,

        0.0f,  0.0f, -15.f,
        0.5f, -0.5f, -15.f,

        1.5f,  5.2f, -15.f,
        12.0f, 8.0f, -15.f,
        });

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 1, 3, 2 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[24] = {
        1.0f,  3.0f,
        2.0f,  4.0f,
        -15.0f,  -15.0f,

        5.0f,  7.0f,
        6.0f,  8.0f,
        -15.0f,  -15.0f,

        0.0f,  0.5f,
        0.0f, -0.5f,
        -15.0f,  -15.0f,

        1.5f,  12.0f,
        5.2f, 8.0f,
        -15.0f,  -15.0f,
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < 24; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(permute_gpu_f32, basic_yxfb_permute_1_0_2_3)
{
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 100, 64, 1 } });

    tests::set_random_values<float>(input_mem);

    topology topology(
        input_layout("input", input_mem->get_layout()),
        permute("permute", input_info("input"), { 1, 0, 2, 3 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> input_ptr(input_mem, get_test_stream());
    for (int i = 0; i < 6400; i++)
    {
        ASSERT_FLOAT_EQ(input_ptr[i], output_ptr[i]);
    }

}

TEST(permute_gpu_f32, basic_bfyx_permute_0_1_3_2_input_padding)
{
    //  Input               : bfyx:2x2x3x2
    //  Permute order       : { 0,1,3,2 }
    //  Input padding       : 2x1
    //
    //  Input:
    //  f0: b0:  1    2   -15  b1:   0    0     -15
    //  f0: b0:  3    4   -15  b1:   0.5 -0.5   -15
    //  f1: b0:  5    6   -15  b1:   1.5  5.2   -15
    //  f1: b0:  7    8   -15  b1:   12   8     -15
    //
    //  Input:
    //  f0: b0:  1    3  b1:   0    0.5
    //  f0: b0:  2    4  b1:   0    -0.5
    //  f0: b0:  -15 -15 b1:   -15  -15
    //  f1: b0:  5    7  b1:   1.5  12
    //  f1: b0:  6    8  b1:   5.2   8
    //  f1: b0:  -15 -15 b1:   -15   -15
    //

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,
        3.0f,  4.0f, -15.f,

        5.0f,  6.0f, -15.f,
        7.0f,  8.0f, -15.f,

        0.0f,  0.0f, -15.f,
        0.5f, -0.5f, -15.f,

        1.5f,  5.2f, -15.f,
        12.0f, 8.0f, -15.f,
        });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), input->get_layout().with_padding(padding{ { 0, 0, 2, 1 }, 0 })),
        permute("permute", input_info("reorder"), { 0, 1, 3, 2 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[24] = {
        1.0f,  3.0f,
        2.0f,  4.0f,
        -15.0f,  -15.0f,

        5.0f,  7.0f,
        6.0f,  8.0f,
        -15.0f,  -15.0f,

        0.0f,  0.5f,
        0.0f, -0.5f,
        -15.0f,  -15.0f,

        1.5f,  12.0f,
        5.2f, 8.0f,
        -15.0f,  -15.0f,
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < 24; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(permute_gpu_f32, basic_yxfb_permute_batch_with_feature)
{
    //  Input               : yxfb:8x2x1x1
    //  Permute order       : { 1, 0, 2, 3 }
    //  Output              : yxfb:2x8x1x1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 8, 2, 1, 1 } });

    set_values(input, {
        //b0 - b7 for f=0
        1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f,

        //b0 - b7 for f=1
        3.f, 0.5f, 7.f, 12.f, 4.f, -0.5f, 8.f, 8.f
        });

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 1, 0, 2, 3 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();
    auto l = output->get_layout();
    ASSERT_EQ(l.batch(), 2);
    ASSERT_EQ(l.feature(), 8);
    ASSERT_EQ(l.spatial(0), 1);
    ASSERT_EQ(l.spatial(1), 1);

    float answers[16] = {
        1.0f, 3.0f,
        0.0f, 0.5f,
        5.f, 7.f,
        1.5f, 12.f,
        2.f, 4.f,
        0.f, -0.5f,
        6.f, 8.f,
        5.2f, 8.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

TEST(permute_gpu_f32, basic_bfyx_permute_batch_with_feature)
{
    //  Input               : yxfb:8x2x1x1
    //  Permute order       : { 1, 0, 2, 3 }
    //  Output              : yxfb:2x8x1x1

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 8, 1, 1 } });

    set_values(input, {
        //f0 - f7 for b=0
        1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f,

        //f0 - f7 for b=1
        3.f, 0.5f, 7.f, 12.f, 4.f, -0.5f, 8.f, 8.f
        });

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 1, 0, 2, 3 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();
    auto l = output->get_layout();
    ASSERT_EQ(l.batch(), 8);
    ASSERT_EQ(l.feature(), 2);
    ASSERT_EQ(l.spatial(0), 1);
    ASSERT_EQ(l.spatial(1), 1);

    float answers[16] = {
        1.0f, 3.0f,
        0.0f, 0.5f,
        5.f, 7.f,
        1.5f, 12.f,
        2.f, 4.f,
        0.f, -0.5f,
        6.f, 8.f,
        5.2f, 8.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < 16; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

template<data_types DType>
void permute_test_with_reorder()
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,
        3.0f,  4.0f, -15.f,

        5.0f,  6.0f, -15.f,
        7.0f,  8.0f, -15.f,

        0.0f,  0.0f, -15.f,
        0.0f,  0.0f, -15.f,

        1.0f,  5.0f, -15.f,
        12.0f, 8.0f, -15.f,
        });

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), { DType, format::bfyx,{ 2, 2, 3, 2 } }),
        permute("permute", input_info("reorder"), { 0, 1, 3, 2 }),
        reorder("reorder_out", input_info("permute"), { data_types::f32, format::bfyx,{ 2, 2, 3, 2 } }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder_out");

    auto output = outputs.begin()->second.get_memory();

    float answers[24] = {
        1.0f,  3.0f,
        2.0f,  4.0f,
        -15.0f,  -15.0f,

        5.0f,  7.0f,
        6.0f,  8.0f,
        -15.0f,  -15.0f,

        0.0f,  0.0f,
        0.0f,  0.0f,
        -15.0f,  -15.0f,

        1.0f,  12.0f,
        5.0f, 8.0f,
        -15.0f,  -15.0f,
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < 24; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_i8, basic_bfyx_permute_0_1_3_2) {
    permute_test_with_reorder<data_types::i8>();
}

TEST(permute_gpu_i32, basic_bfyx_permute_0_1_3_2) {
    permute_test_with_reorder<data_types::i32>();
}

TEST(permute_gpu_i64, basic_bfyx_permute_0_1_3_2) {
    permute_test_with_reorder<data_types::i64>();
}

TEST(permute_fuse_reorder_gpu_f32, basic_b_fs_yx_fsv4_permute_1_8_16_1)
{
    //  Input               : bfyx:1x32x1x2
    //  Permute1 order       : {0, 3, 1, 2}
    //  Permute2 order       : {0, 2, 3, 1}

    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, {1, 8, 1, 16}});

    std::vector<float> values = {
            0.0f, 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f,
            12.0f, 13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f, 19.0f,
            20.0f, 21.0f, 22.0f, 23.0f,
            24.0f, 25.0f, 26.0f, 27.0f,
            28.0f, 29.0f, 30.0f, 31.0f,
            32.0f, 33.0f, 34.0f, 35.0f,
            36.0f, 37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f, 43.0f,
            44.0f, 45.0f, 46.0f, 47.0f,
            48.0f, 49.0f, 50.0f, 51.0f,
            52.0f, 53.0f, 54.0f, 55.0f,
            56.0f, 57.0f, 58.0f, 59.0f,
            60.0f, 61.0f, 62.0f, 63.0f,
            64.0f, 65.0f, 66.0f, 67.0f,
            68.0f, 69.0f, 70.0f, 71.0f,
            72.0f, 73.0f, 74.0f, 75.0f,
            76.0f, 77.0f, 78.0f, 79.0f,
            80.0f, 81.0f, 82.0f, 83.0f,
            84.0f, 85.0f, 86.0f, 87.0f,
            88.0f, 89.0f, 90.0f, 91.0f,
            92.0f, 93.0f, 94.0f, 95.0f,
            96.0f, 97.0f, 98.0f, 99.0f,
            100.0f, 101.0f, 102.0f, 103.0f,
            104.0f, 105.0f, 106.0f, 107.0f,
            108.0f, 109.0f, 110.0f, 111.0f,
            112.0f, 113.0f, 114.0f, 115.0f,
            116.0f, 117.0f, 118.0f, 119.0f,
            120.0f, 121.0f, 122.0f, 123.0f,
            124.0f, 125.0f, 126.0f, 127.0f
    };

    set_values(input, values);
    // unfused
    topology topology_unfused(
        input_layout("input", input->get_layout()),
        reorder("reorder1", input_info("input"), format::b_fs_yx_fsv4, data_types::f32),
        permute("permute", input_info("reorder1"), { 0, 2, 3, 1}),
        reorder("reorder2", input_info("permute"), format::bfyx, data_types::f32),
        permute("out", input_info("reorder2"), { 0, 3, 1, 2}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));
    config.set_property(ov::intel_gpu::allow_static_input_reorder(true));

    network unfused(engine, topology_unfused, config);
    unfused.set_input_data("input", input);

    // fused network
    topology topology_fused(
        input_layout("input", input->get_layout()),
        reorder("reorder1", input_info("input"), format::b_fs_yx_fsv4, data_types::f32),
        permute("permute", input_info("reorder1"), { 0, 2, 3, 1}),
        reorder("reorder2", input_info("permute"), format::bfyx, data_types::f32), // to be fused to previous permute
        permute("out", input_info("reorder2"), { 0, 3, 1, 2})); // return to original value

    ExecutionConfig config_fused = get_test_default_config(engine);
    config_fused.set_property(ov::intel_gpu::optimize_data(true));
    network fused(engine, topology_fused, config_fused);
    fused.set_input_data("input", input);

    auto outputs_fused = fused.execute();
    auto outputs_unfused = unfused.execute();
    auto output_fused = outputs_fused.begin()->second.get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_fused_ptr(output_fused, get_test_stream());
    auto output_unfused = outputs_unfused.begin()->second.get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_unfused_ptr(output_unfused, get_test_stream());
    ASSERT_EQ(output_fused->get_layout().format, cldnn::format::bfyx);
    ASSERT_EQ(output_unfused->get_layout().format, cldnn::format::bfyx);
    ASSERT_EQ(fused.get_executed_primitives().size(), 4);
    ASSERT_EQ(unfused.get_executed_primitives().size(), 5);

    for (size_t i = 0; i < values.size(); i++)
    {
        ASSERT_FLOAT_EQ(output_unfused_ptr[i], output_fused_ptr[i]);
        ASSERT_FLOAT_EQ(output_unfused_ptr[i], values[i]);
    }
}

TEST(fc_permute_crop_gpu, basic_permute_yxfb)
{
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 5, 1, 512 } });

    //Topolgy creates permute which "repalces" the batch with the feature.
    topology topology(
        input_layout("input", input_mem->get_layout()),         // yxfb {1, 5, 1, 512 }}
        permute("permute", input_info("input"), { 1, 0, 2, 3 }) // yxfb {5, 1, 1, 512}  --- without permute fix yxfb {1, 5, 512, 1}
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();
    auto l = output->get_layout();
    ASSERT_EQ(l.batch(), 5);
    ASSERT_EQ(l.feature(), 1);
    ASSERT_EQ(l.spatial(0), 1);
    ASSERT_EQ(l.spatial(1), 512);
    ASSERT_EQ(output->get_layout().format, cldnn::format::yxfb);
}

TEST(fc_permute_crop_gpu, basic_0)
{

    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory({ data_types::f32, format::yxfb,{ 5, 11264, 1, 1 } });
    auto weights_mem = engine.allocate_memory({ data_types::f32, format::yxio,{ 512, 11264, 1, 1 } });
    auto bias_mem = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 512, 1 } });

    topology topology(
        input_layout("input", input_mem->get_layout()),                             // bfyx {5, 11264, 1, 1}}
        data("weights", weights_mem),
        data("bias", bias_mem),
        fully_connected("fully_connected", input_info("input"), "weights", "bias"), // yxfb {5, 512, 1, 1}
        reshape("reshape", input_info("fully_connected"), { 1, 5, 1, 512 }),        // yxfb {1, 5, 1, 512}
        permute("permute", input_info("reshape"), { 1, 0, 2, 3 }),                  // yxfb {5, 1, 1, 512}        --- without permute fix yxfb {1, 5, 512, 1}
        crop("crop", input_info("permute"), { 1, 1, 1, 512 }, { 4, 0, 0 ,0 })       // without permute fix it will fail "Tensor pitches didn't set correctly"
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "crop");

    auto output = outputs.begin()->second.get_memory();
    auto l = output->get_layout();
    ASSERT_EQ(l.batch(), 1);
    ASSERT_EQ(l.feature(), 1);
    ASSERT_EQ(l.spatial(0), 1);
    ASSERT_EQ(l.spatial(1), 512);
    ASSERT_EQ(output->get_layout().format, cldnn::format::yxfb);
}

TEST(fc_permute_gpu, basic_permute_bfyx)
{
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 5, 1, 256 } });

    tests::set_random_values<float>(input_mem);

    //Topolgy creates permute which "repalces" the batch with the feature.
    topology topology(
        input_layout("input", input_mem->get_layout()),
        permute("permute", input_info("input"), { 1, 0, 2, 3 })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();
    auto l = output->get_layout();
    ASSERT_EQ(l.batch(), 5);
    ASSERT_EQ(l.feature(), 1);
    ASSERT_EQ(l.spatial(0), 1);
    ASSERT_EQ(l.spatial(1), 256);
    ASSERT_EQ(output->get_layout().format, cldnn::format::bfyx);

    cldnn::mem_lock<float> input_ptr(input_mem, get_test_stream());
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < 5 * 256; i++)
        ASSERT_NEAR(input_ptr[i], output_ptr[i], 1e-3f);

}

TEST(permute_gpu_f32, permute_bfwzyx)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    const int b = 1;
    const int f = 2;
    const int x = 3;
    const int y = 4;
    const int z = 5;
    const int w = 6;
    std::vector<uint16_t> permute_order = { 1, 0, 5, 4, 3, 2 };

    auto input_size = cldnn::tensor(batch(b), feature(f), spatial(x, y, z, w));
    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfwzyx, input_size });
    auto input_data = rg.generate_random_1d<float>(input_mem->get_layout().count(), -1, 1);

    set_values(input_mem, input_data);

    auto expected_size = cldnn::tensor(batch(f), feature(b), spatial(w, z, y, x));
    auto expected_layout = cldnn::layout(data_types::f32, format::bfwzyx, expected_size);
    auto expected_output = std::vector<float>(expected_layout.count());
    for (int bi = 0; bi < b; ++bi)
    for (int fi = 0; fi < f; ++fi)
    for (int wi = 0; wi < w; ++wi)
    for (int zi = 0; zi < z; ++zi)
    for (int yi = 0; yi < y; ++yi)
    for (int xi = 0; xi < x; ++xi)
    {
        auto in_index = cldnn::tensor(batch(bi), feature(fi), spatial(xi, yi, zi, wi));
        auto out_index = cldnn::tensor(batch(fi), feature(bi), spatial(wi, zi, yi, xi));
        expected_output[expected_layout.get_linear_offset(out_index)] =
            input_data[input_mem->get_layout().get_linear_offset(in_index)];
    }

    topology topology(
        input_layout("input", input_mem->get_layout()),
        permute("permute", input_info("input"), permute_order)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();
    auto l = output->get_layout();
    ASSERT_EQ(l.batch(), 2);
    ASSERT_EQ(l.feature(), 1);
    ASSERT_EQ(l.spatial(0), 6);
    ASSERT_EQ(l.spatial(1), 5);
    ASSERT_EQ(l.spatial(2), 4);
    ASSERT_EQ(l.spatial(3), 3);
    ASSERT_EQ(output->get_layout().format, cldnn::format::bfwzyx);

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < output_ptr.size(); ++i)
    {
        ASSERT_EQ(expected_output[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32, 6D_reshape_permute_reshape)
{
    // Input - 1x4x2x2:
    //
    // 0 0   1 1   2 2   3 3
    // 0 0   1 1   2 2   3 3
    //
    // Reshape 4 to 6 - 1x1x2x2x2x2:
    //
    // 0 0   1 1
    // 0 0   1 1
    //
    // 2 2   3 3
    // 2 2   3 3
    //
    // Permute 0, 1, 5, 4, 2, 3
    //
    // 0 2   0 2
    // 1 3   1 3
    //
    // 0 2   0 2
    // 1 3   1 3
    //
    // Reshape 6 to 4 - 1x4x2x2
    //
    // 0 2   0 2   0 2   0 2
    // 1 3   1 3   1 3   1 3

    auto& engine = get_test_engine();
    const int b = 1;
    const int f = 4;
    const int x = 2;
    const int y = 2;

    const int f_reshape = 1;
    const int w_reshape = 2;
    const int z_reshape = 2;

    std::vector<uint16_t> permute_order = { 0, 1, 4, 5, 3, 2 };

    auto input_size = cldnn::tensor(batch(b), feature(f), spatial(x, y));
    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx, input_size });
    std::vector<float> input_data = {
        0.f, 0.f, 0.f, 0.f,
        1.f, 1.f, 1.f, 1.f,
        2.f, 2.f, 2.f, 2.f,
        3.f, 3.f, 3.f, 3.f
    };

    std::vector<float> expected_out = {
        0.f, 2.f, 1.f, 3.f,
        0.f, 2.f, 1.f, 3.f,
        0.f, 2.f, 1.f, 3.f,
        0.f, 2.f, 1.f, 3.f
    };

    set_values(input_mem, input_data);

    topology topology(
        input_layout("input", input_mem->get_layout()),
        reorder("input_6d", input_info("input"), { data_types::f32, format::bfwzyx, cldnn::tensor(batch(b), feature(f), spatial(x, y)) }),
        reshape("reshape_4_to_6", input_info("input_6d"), cldnn::tensor(batch(b), feature(f_reshape), spatial(x, y, z_reshape, w_reshape))),
        permute("permute", input_info("reshape_4_to_6"), permute_order),
        reshape("reshape_6_to_4", input_info("permute"), cldnn::tensor(batch(b), feature(f), spatial(x, y))),
        reorder("output_4d", input_info("reshape_6_to_4"), { data_types::f32, format::bfyx, cldnn::tensor(batch(b), feature(f), spatial(x, y)) })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output_4d");

    auto output = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < output_ptr.size(); ++i)
    {
        ASSERT_EQ(expected_out[i], output_ptr[i]);
    }
}
TEST(permute_gpu_f32, basic_bfzyx_permute_0_4_1_2_3)
{
    //  Input               : bfzyx:2x2x2x2x3
    //  Permute order       : { 0,4,1,2,3 }

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 2, 3, 2, 2 } });

    set_values(input, {
        1.0f,  2.0f, -15.f, //B0, F0,   // z0 y0 x-3
        3.0f,  4.0f, -15.f,             // z0 y1
        2.0f,  3.0f, -15.f,             // z1 y0
        4.0f,  5.0f, -15.f,             // z1  y1

        5.0f,  6.0f, -15.f, //b0, f1
        7.0f,  8.0f, -15.f,
        6.0f,  7.0f, -15.f,
        8.0f,  9.0f, -15.f,

        0.0f,  0.0f, -15.f, //b1, f0
        0.5f, -0.5f, -15.f,
        1.0f,  1.0f, -15.f,
        1.5f,  0.5f, -15.f,

        1.5f,  5.2f, -15.f, //b1, f1
        12.0f, 8.0f, -15.f,
        2.5f,  6.2f, -15.f,
        13.0f, 9.0f, -15.f,
    });

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 4, 1, 2, 3 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();
    auto l = output->get_layout();
    ASSERT_EQ(l.batch(), 2);
    ASSERT_EQ(l.feature(), 3);
    ASSERT_EQ(l.spatial(0), 2);
    ASSERT_EQ(l.spatial(1), 2);
    ASSERT_EQ(l.spatial(2), 2);

    ASSERT_EQ(output->get_layout().format, cldnn::format::bfzyx);

    float answers[48] = {
        1.0f, 3.0f, 2.0f, 4.0f,
        5.0f, 7.0f, 6.0f, 8.0f,
        2.0f, 4.0f, 3.0f, 5.0f,
        6.0f, 8.0f, 7.0f, 9.0f,
        -15.0f, -15.0f, -15.0f, -15.0f,
        -15.0f, -15.0f, -15.0f, -15.0f,
        0.0f, 0.5f, 1.0f, 1.5f,
        1.5f, 12.0f, 2.5f, 13.0f,
        0.0f, -0.5f, 1.0f, 0.5f,
        5.2f, 8.0f, 6.2f, 9.0f,
        -15.0f, -15.0f, -15.0f, -15.0f,
        -15.0f, -15.0f, -15.0f, -15.0f,
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (int i = 0; i < 48; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }

}

/**
 * Test cases for permute_tile_8x8_4x4 kernel
 *
 * This TCs are enabled only when batch axis move to the last.
 * i.e permute order is 0,2,3,1 or 0,4,1,2,3 or 0,5,1,2,3,4
 */
TEST(permute_gpu_f32_tile_8x8_4x4, normal_bfyx_0_2_3_1) {
    //  Input               : bfyx:2x8x2x8
    //  Permute order       : { 0,3,1,2 }

    constexpr size_t array_size = 256;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 8, 8, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i = 0; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  16.f,  32.f,  48.f,  64.f,  80.f,  96.f, 112.f,   1.f,  17.f,  33.f,  49.f,  65.f,  81.f,  97.f, 113.f,
        2.f,  18.f,  34.f,  50.f,  66.f,  82.f,  98.f, 114.f,   3.f,  19.f,  35.f,  51.f,  67.f,  83.f,  99.f, 115.f,
        4.f,  20.f,  36.f,  52.f,  68.f,  84.f, 100.f, 116.f,   5.f,  21.f,  37.f,  53.f,  69.f,  85.f, 101.f, 117.f,
        6.f,  22.f,  38.f,  54.f,  70.f,  86.f, 102.f, 118.f,   7.f,  23.f,  39.f,  55.f,  71.f,  87.f, 103.f, 119.f,
        8.f,  24.f,  40.f,  56.f,  72.f,  88.f, 104.f, 120.f,   9.f,  25.f,  41.f,  57.f,  73.f,  89.f, 105.f, 121.f,
        10.f,  26.f,  42.f,  58.f,  74.f,  90.f, 106.f, 122.f,  11.f,  27.f,  43.f,  59.f,  75.f,  91.f, 107.f, 123.f,
        12.f,  28.f,  44.f,  60.f,  76.f,  92.f, 108.f, 124.f,  13.f,  29.f,  45.f,  61.f,  77.f,  93.f, 109.f, 125.f,
        14.f,  30.f,  46.f,  62.f,  78.f,  94.f, 110.f, 126.f,  15.f,  31.f,  47.f,  63.f,  79.f,  95.f, 111.f, 127.f,
        128.f, 144.f, 160.f, 176.f, 192.f, 208.f, 224.f, 240.f, 129.f, 145.f, 161.f, 177.f, 193.f, 209.f, 225.f, 241.f,
        130.f, 146.f, 162.f, 178.f, 194.f, 210.f, 226.f, 242.f, 131.f, 147.f, 163.f, 179.f, 195.f, 211.f, 227.f, 243.f,
        132.f, 148.f, 164.f, 180.f, 196.f, 212.f, 228.f, 244.f, 133.f, 149.f, 165.f, 181.f, 197.f, 213.f, 229.f, 245.f,
        134.f, 150.f, 166.f, 182.f, 198.f, 214.f, 230.f, 246.f, 135.f, 151.f, 167.f, 183.f, 199.f, 215.f, 231.f, 247.f,
        136.f, 152.f, 168.f, 184.f, 200.f, 216.f, 232.f, 248.f, 137.f, 153.f, 169.f, 185.f, 201.f, 217.f, 233.f, 249.f,
        138.f, 154.f, 170.f, 186.f, 202.f, 218.f, 234.f, 250.f, 139.f, 155.f, 171.f, 187.f, 203.f, 219.f, 235.f, 251.f,
        140.f, 156.f, 172.f, 188.f, 204.f, 220.f, 236.f, 252.f, 141.f, 157.f, 173.f, 189.f, 205.f, 221.f, 237.f, 253.f,
        142.f, 158.f, 174.f, 190.f, 206.f, 222.f, 238.f, 254.f, 143.f, 159.f, 175.f, 191.f, 207.f, 223.f, 239.f, 255.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, f_remainder_bfyx_0_2_3_1) {
    //  Input               : bfyx:2x5x2x8
    //  Permute order       : { 0,3,1,2 }

    constexpr size_t array_size = 160;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 5, 8, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  16.f,  32.f,  48.f,  64.f,   1.f,  17.f,  33.f,  49.f,  65.f,   2.f,  18.f,  34.f,  50.f,  66.f,   3.f,
        19.f,  35.f,  51.f,  67.f,   4.f,  20.f,  36.f,  52.f,  68.f,   5.f,  21.f,  37.f,  53.f,  69.f,   6.f,  22.f,
        38.f,  54.f,  70.f,   7.f,  23.f,  39.f,  55.f,  71.f,   8.f,  24.f,  40.f,  56.f,  72.f,   9.f,  25.f,  41.f,
        57.f,  73.f,  10.f,  26.f,  42.f,  58.f,  74.f,  11.f,  27.f,  43.f,  59.f,  75.f,  12.f,  28.f,  44.f,  60.f,
        76.f,  13.f,  29.f,  45.f,  61.f,  77.f,  14.f,  30.f,  46.f,  62.f,  78.f,  15.f,  31.f,  47.f,  63.f,  79.f,
        80.f,  96.f, 112.f, 128.f, 144.f,  81.f,  97.f, 113.f, 129.f, 145.f,  82.f,  98.f, 114.f, 130.f, 146.f,  83.f,
        99.f, 115.f, 131.f, 147.f,  84.f, 100.f, 116.f, 132.f, 148.f,  85.f, 101.f, 117.f, 133.f, 149.f,  86.f, 102.f,
        118.f, 134.f, 150.f,  87.f, 103.f, 119.f, 135.f, 151.f,  88.f, 104.f, 120.f, 136.f, 152.f,  89.f, 105.f, 121.f,
        137.f, 153.f,  90.f, 106.f, 122.f, 138.f, 154.f,  91.f, 107.f, 123.f, 139.f, 155.f,  92.f, 108.f, 124.f, 140.f,
        156.f,  93.f, 109.f, 125.f, 141.f, 157.f,  94.f, 110.f, 126.f, 142.f, 158.f,  95.f, 111.f, 127.f, 143.f, 159.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, x_remainder_bfyx_0_2_3_1) {
    //  Input               : bfyx:2x8x2x5
    //  Permute order       : { 0,3,1,2 }

    constexpr size_t array_size = 160;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 8, 5, 2 } });

    set_values(input, {
        0.f,   1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,   9.f,  10.f,  11.f,  12.f,  13.f,  14.f,  15.f,
        16.f,  17.f,  18.f,  19.f,  20.f,  21.f,  22.f,  23.f,  24.f,  25.f,  26.f,  27.f,  28.f,  29.f,  30.f,  31.f,
        32.f,  33.f,  34.f,  35.f,  36.f,  37.f,  38.f,  39.f,  40.f,  41.f,  42.f,  43.f,  44.f,  45.f,  46.f,  47.f,
        48.f,  49.f,  50.f,  51.f,  52.f,  53.f,  54.f,  55.f,  56.f,  57.f,  58.f,  59.f,  60.f,  61.f,  62.f,  63.f,
        64.f,  65.f,  66.f,  67.f,  68.f,  69.f,  70.f,  71.f,  72.f,  73.f,  74.f,  75.f,  76.f,  77.f,  78.f,  79.f,
        80.f,  81.f,  82.f,  83.f,  84.f,  85.f,  86.f,  87.f,  88.f,  89.f,  90.f,  91.f,  92.f,  93.f,  94.f,  95.f,
        96.f,  97.f,  98.f,  99.f, 100.f, 101.f, 102.f, 103.f, 104.f, 105.f, 106.f, 107.f, 108.f, 109.f, 110.f, 111.f,
        112.f, 113.f, 114.f, 115.f, 116.f, 117.f, 118.f, 119.f, 120.f, 121.f, 122.f, 123.f, 124.f, 125.f, 126.f, 127.f,
        128.f, 129.f, 130.f, 131.f, 132.f, 133.f, 134.f, 135.f, 136.f, 137.f, 138.f, 139.f, 140.f, 141.f, 142.f, 143.f,
        144.f, 145.f, 146.f, 147.f, 148.f, 149.f, 150.f, 151.f, 152.f, 153.f, 154.f, 155.f, 156.f, 157.f, 158.f, 159.f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  10.f,  20.f,  30.f,  40.f,  50.f,  60.f,  70.f,   1.f,  11.f,  21.f,  31.f,  41.f,  51.f,  61.f,  71.f,
        2.f,  12.f,  22.f,  32.f,  42.f,  52.f,  62.f,  72.f,   3.f,  13.f,  23.f,  33.f,  43.f,  53.f,  63.f,  73.f,
        4.f,  14.f,  24.f,  34.f,  44.f,  54.f,  64.f,  74.f,   5.f,  15.f,  25.f,  35.f,  45.f,  55.f,  65.f,  75.f,
        6.f,  16.f,  26.f,  36.f,  46.f,  56.f,  66.f,  76.f,   7.f,  17.f,  27.f,  37.f,  47.f,  57.f,  67.f,  77.f,
        8.f,  18.f,  28.f,  38.f,  48.f,  58.f,  68.f,  78.f,   9.f,  19.f,  29.f,  39.f,  49.f,  59.f,  69.f,  79.f,
        80.f,  90.f, 100.f, 110.f, 120.f, 130.f, 140.f, 150.f,  81.f,  91.f, 101.f, 111.f, 121.f, 131.f, 141.f, 151.f,
        82.f,  92.f, 102.f, 112.f, 122.f, 132.f, 142.f, 152.f,  83.f,  93.f, 103.f, 113.f, 123.f, 133.f, 143.f, 153.f,
        84.f,  94.f, 104.f, 114.f, 124.f, 134.f, 144.f, 154.f,  85.f,  95.f, 105.f, 115.f, 125.f, 135.f, 145.f, 155.f,
        86.f,  96.f, 106.f, 116.f, 126.f, 136.f, 146.f, 156.f,  87.f,  97.f, 107.f, 117.f, 127.f, 137.f, 147.f, 157.f,
        88.f,  98.f, 108.f, 118.f, 128.f, 138.f, 148.f, 158.f,  89.f,  99.f, 109.f, 119.f, 129.f, 139.f, 149.f, 159.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, xf_remainder_bfyx_0_2_3_1) {
    //  Input               : bfyx:2x5x2x5
    //  Permute order       : { 0,3,1,2 }

    constexpr size_t array_size = 100;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 5, 5, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  10.f,  20.f,  30.f,  40.f,   1.f,  11.f,  21.f,  31.f,  41.f,
        2.f,  12.f,  22.f,  32.f,  42.f,   3.f,  13.f,  23.f,  33.f,  43.f,
        4.f,  14.f,  24.f,  34.f,  44.f,   5.f,  15.f,  25.f,  35.f,  45.f,
        6.f,  16.f,  26.f,  36.f,  46.f,   7.f,  17.f,  27.f,  37.f,  47.f,
        8.f,  18.f,  28.f,  38.f,  48.f,   9.f,  19.f,  29.f,  39.f,  49.f,
        50.f,  60.f,  70.f,  80.f,  90.f,  51.f,  61.f,  71.f,  81.f,  91.f,
        52.f,  62.f,  72.f,  82.f,  92.f,  53.f,  63.f,  73.f,  83.f,  93.f,
        54.f,  64.f,  74.f,  84.f,  94.f,  55.f,  65.f,  75.f,  85.f,  95.f,
        56.f,  66.f,  76.f,  86.f,  96.f,  57.f,  67.f,  77.f,  87.f,  97.f,
        58.f,  68.f,  78.f,  88.f,  98.f,  59.f,  69.f,  79.f,  89.f,  99.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4_dynamic, normal_bfyx_0_2_3_1) {
    constexpr size_t array_size = 256;

    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::PartialShape::dynamic(4)}, data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{2, 8, 8, 2}, data_types::f32, format::bfyx};

    auto input = engine.allocate_memory(input_layout_static);

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i = 0; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input_layout_dynamic),
        permute("permute", input_info("input"), { 0, 2, 3, 1 }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("permute");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  16.f,  32.f,  48.f,  64.f,  80.f,  96.f, 112.f,   1.f,  17.f,  33.f,  49.f,  65.f,  81.f,  97.f, 113.f,
        2.f,  18.f,  34.f,  50.f,  66.f,  82.f,  98.f, 114.f,   3.f,  19.f,  35.f,  51.f,  67.f,  83.f,  99.f, 115.f,
        4.f,  20.f,  36.f,  52.f,  68.f,  84.f, 100.f, 116.f,   5.f,  21.f,  37.f,  53.f,  69.f,  85.f, 101.f, 117.f,
        6.f,  22.f,  38.f,  54.f,  70.f,  86.f, 102.f, 118.f,   7.f,  23.f,  39.f,  55.f,  71.f,  87.f, 103.f, 119.f,
        8.f,  24.f,  40.f,  56.f,  72.f,  88.f, 104.f, 120.f,   9.f,  25.f,  41.f,  57.f,  73.f,  89.f, 105.f, 121.f,
        10.f,  26.f,  42.f,  58.f,  74.f,  90.f, 106.f, 122.f,  11.f,  27.f,  43.f,  59.f,  75.f,  91.f, 107.f, 123.f,
        12.f,  28.f,  44.f,  60.f,  76.f,  92.f, 108.f, 124.f,  13.f,  29.f,  45.f,  61.f,  77.f,  93.f, 109.f, 125.f,
        14.f,  30.f,  46.f,  62.f,  78.f,  94.f, 110.f, 126.f,  15.f,  31.f,  47.f,  63.f,  79.f,  95.f, 111.f, 127.f,
        128.f, 144.f, 160.f, 176.f, 192.f, 208.f, 224.f, 240.f, 129.f, 145.f, 161.f, 177.f, 193.f, 209.f, 225.f, 241.f,
        130.f, 146.f, 162.f, 178.f, 194.f, 210.f, 226.f, 242.f, 131.f, 147.f, 163.f, 179.f, 195.f, 211.f, 227.f, 243.f,
        132.f, 148.f, 164.f, 180.f, 196.f, 212.f, 228.f, 244.f, 133.f, 149.f, 165.f, 181.f, 197.f, 213.f, 229.f, 245.f,
        134.f, 150.f, 166.f, 182.f, 198.f, 214.f, 230.f, 246.f, 135.f, 151.f, 167.f, 183.f, 199.f, 215.f, 231.f, 247.f,
        136.f, 152.f, 168.f, 184.f, 200.f, 216.f, 232.f, 248.f, 137.f, 153.f, 169.f, 185.f, 201.f, 217.f, 233.f, 249.f,
        138.f, 154.f, 170.f, 186.f, 202.f, 218.f, 234.f, 250.f, 139.f, 155.f, 171.f, 187.f, 203.f, 219.f, 235.f, 251.f,
        140.f, 156.f, 172.f, 188.f, 204.f, 220.f, 236.f, 252.f, 141.f, 157.f, 173.f, 189.f, 205.f, 221.f, 237.f, 253.f,
        142.f, 158.f, 174.f, 190.f, 206.f, 222.f, 238.f, 254.f, 143.f, 159.f, 175.f, 191.f, 207.f, 223.f, 239.f, 255.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4_dynamic, f_remainder_bfyx_0_2_3_1) {
    constexpr size_t array_size = 160;

    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::PartialShape::dynamic(4)}, data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{2, 5, 8, 2}, data_types::f32, format::bfyx};

    auto input = engine.allocate_memory(input_layout_static);

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input_layout_dynamic),
        permute("permute", input_info("input"), { 0, 2, 3, 1 }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("permute");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  16.f,  32.f,  48.f,  64.f,   1.f,  17.f,  33.f,  49.f,  65.f,   2.f,  18.f,  34.f,  50.f,  66.f,   3.f,
        19.f,  35.f,  51.f,  67.f,   4.f,  20.f,  36.f,  52.f,  68.f,   5.f,  21.f,  37.f,  53.f,  69.f,   6.f,  22.f,
        38.f,  54.f,  70.f,   7.f,  23.f,  39.f,  55.f,  71.f,   8.f,  24.f,  40.f,  56.f,  72.f,   9.f,  25.f,  41.f,
        57.f,  73.f,  10.f,  26.f,  42.f,  58.f,  74.f,  11.f,  27.f,  43.f,  59.f,  75.f,  12.f,  28.f,  44.f,  60.f,
        76.f,  13.f,  29.f,  45.f,  61.f,  77.f,  14.f,  30.f,  46.f,  62.f,  78.f,  15.f,  31.f,  47.f,  63.f,  79.f,
        80.f,  96.f, 112.f, 128.f, 144.f,  81.f,  97.f, 113.f, 129.f, 145.f,  82.f,  98.f, 114.f, 130.f, 146.f,  83.f,
        99.f, 115.f, 131.f, 147.f,  84.f, 100.f, 116.f, 132.f, 148.f,  85.f, 101.f, 117.f, 133.f, 149.f,  86.f, 102.f,
        118.f, 134.f, 150.f,  87.f, 103.f, 119.f, 135.f, 151.f,  88.f, 104.f, 120.f, 136.f, 152.f,  89.f, 105.f, 121.f,
        137.f, 153.f,  90.f, 106.f, 122.f, 138.f, 154.f,  91.f, 107.f, 123.f, 139.f, 155.f,  92.f, 108.f, 124.f, 140.f,
        156.f,  93.f, 109.f, 125.f, 141.f, 157.f,  94.f, 110.f, 126.f, 142.f, 158.f,  95.f, 111.f, 127.f, 143.f, 159.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4_dynamic, x_remainder_bfyx_0_2_3_1) {
    constexpr size_t array_size = 160;

    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::PartialShape::dynamic(4)}, data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{2, 8, 5, 2}, data_types::f32, format::bfyx};

    auto input = engine.allocate_memory(input_layout_static);

    set_values(input, {
        0.f,   1.f,   2.f,   3.f,   4.f,   5.f,   6.f,   7.f,   8.f,   9.f,  10.f,  11.f,  12.f,  13.f,  14.f,  15.f,
        16.f,  17.f,  18.f,  19.f,  20.f,  21.f,  22.f,  23.f,  24.f,  25.f,  26.f,  27.f,  28.f,  29.f,  30.f,  31.f,
        32.f,  33.f,  34.f,  35.f,  36.f,  37.f,  38.f,  39.f,  40.f,  41.f,  42.f,  43.f,  44.f,  45.f,  46.f,  47.f,
        48.f,  49.f,  50.f,  51.f,  52.f,  53.f,  54.f,  55.f,  56.f,  57.f,  58.f,  59.f,  60.f,  61.f,  62.f,  63.f,
        64.f,  65.f,  66.f,  67.f,  68.f,  69.f,  70.f,  71.f,  72.f,  73.f,  74.f,  75.f,  76.f,  77.f,  78.f,  79.f,
        80.f,  81.f,  82.f,  83.f,  84.f,  85.f,  86.f,  87.f,  88.f,  89.f,  90.f,  91.f,  92.f,  93.f,  94.f,  95.f,
        96.f,  97.f,  98.f,  99.f, 100.f, 101.f, 102.f, 103.f, 104.f, 105.f, 106.f, 107.f, 108.f, 109.f, 110.f, 111.f,
        112.f, 113.f, 114.f, 115.f, 116.f, 117.f, 118.f, 119.f, 120.f, 121.f, 122.f, 123.f, 124.f, 125.f, 126.f, 127.f,
        128.f, 129.f, 130.f, 131.f, 132.f, 133.f, 134.f, 135.f, 136.f, 137.f, 138.f, 139.f, 140.f, 141.f, 142.f, 143.f,
        144.f, 145.f, 146.f, 147.f, 148.f, 149.f, 150.f, 151.f, 152.f, 153.f, 154.f, 155.f, 156.f, 157.f, 158.f, 159.f
    });

    topology topology(
        input_layout("input", input_layout_dynamic),
        permute("permute", input_info("input"), { 0, 2, 3, 1 }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("permute");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  10.f,  20.f,  30.f,  40.f,  50.f,  60.f,  70.f,   1.f,  11.f,  21.f,  31.f,  41.f,  51.f,  61.f,  71.f,
        2.f,  12.f,  22.f,  32.f,  42.f,  52.f,  62.f,  72.f,   3.f,  13.f,  23.f,  33.f,  43.f,  53.f,  63.f,  73.f,
        4.f,  14.f,  24.f,  34.f,  44.f,  54.f,  64.f,  74.f,   5.f,  15.f,  25.f,  35.f,  45.f,  55.f,  65.f,  75.f,
        6.f,  16.f,  26.f,  36.f,  46.f,  56.f,  66.f,  76.f,   7.f,  17.f,  27.f,  37.f,  47.f,  57.f,  67.f,  77.f,
        8.f,  18.f,  28.f,  38.f,  48.f,  58.f,  68.f,  78.f,   9.f,  19.f,  29.f,  39.f,  49.f,  59.f,  69.f,  79.f,
        80.f,  90.f, 100.f, 110.f, 120.f, 130.f, 140.f, 150.f,  81.f,  91.f, 101.f, 111.f, 121.f, 131.f, 141.f, 151.f,
        82.f,  92.f, 102.f, 112.f, 122.f, 132.f, 142.f, 152.f,  83.f,  93.f, 103.f, 113.f, 123.f, 133.f, 143.f, 153.f,
        84.f,  94.f, 104.f, 114.f, 124.f, 134.f, 144.f, 154.f,  85.f,  95.f, 105.f, 115.f, 125.f, 135.f, 145.f, 155.f,
        86.f,  96.f, 106.f, 116.f, 126.f, 136.f, 146.f, 156.f,  87.f,  97.f, 107.f, 117.f, 127.f, 137.f, 147.f, 157.f,
        88.f,  98.f, 108.f, 118.f, 128.f, 138.f, 148.f, 158.f,  89.f,  99.f, 109.f, 119.f, 129.f, 139.f, 149.f, 159.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4_dynamic, xf_remainder_bfyx_0_2_3_1) {
    constexpr size_t array_size = 100;

    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape{ov::PartialShape::dynamic(4)}, data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{2, 5, 5, 2}, data_types::f32, format::bfyx};

    auto input = engine.allocate_memory(input_layout_static);

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input_layout_dynamic),
        permute("permute", input_info("input"), { 0, 2, 3, 1 }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("permute");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  10.f,  20.f,  30.f,  40.f,   1.f,  11.f,  21.f,  31.f,  41.f,
        2.f,  12.f,  22.f,  32.f,  42.f,   3.f,  13.f,  23.f,  33.f,  43.f,
        4.f,  14.f,  24.f,  34.f,  44.f,   5.f,  15.f,  25.f,  35.f,  45.f,
        6.f,  16.f,  26.f,  36.f,  46.f,   7.f,  17.f,  27.f,  37.f,  47.f,
        8.f,  18.f,  28.f,  38.f,  48.f,   9.f,  19.f,  29.f,  39.f,  49.f,
        50.f,  60.f,  70.f,  80.f,  90.f,  51.f,  61.f,  71.f,  81.f,  91.f,
        52.f,  62.f,  72.f,  82.f,  92.f,  53.f,  63.f,  73.f,  83.f,  93.f,
        54.f,  64.f,  74.f,  84.f,  94.f,  55.f,  65.f,  75.f,  85.f,  95.f,
        56.f,  66.f,  76.f,  86.f,  96.f,  57.f,  67.f,  77.f,  87.f,  97.f,
        58.f,  68.f,  78.f,  88.f,  98.f,  59.f,  69.f,  79.f,  89.f,  99.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}


TEST(permute_gpu_f32_tile_8x8_4x4, normal_bfzyx_0_2_3_4_1) {
    //  Input               : bfzyx:2x8x2x2x8
    //  Permute order       : { 0,4,1,2,3 }

    constexpr size_t array_size = 512;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 8, 8, 2, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 4, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  32.f,  64.f,  96.f, 128.f, 160.f, 192.f, 224.f,   1.f,  33.f,  65.f,  97.f, 129.f, 161.f, 193.f, 225.f,   2.f,  34.f,  66.f,  98.f, 130.f, 162.f, 194.f, 226.f,
        3.f,  35.f,  67.f,  99.f, 131.f, 163.f, 195.f, 227.f,   4.f,  36.f,  68.f, 100.f, 132.f, 164.f, 196.f, 228.f,   5.f,  37.f,  69.f, 101.f, 133.f, 165.f, 197.f, 229.f,
        6.f,  38.f,  70.f, 102.f, 134.f, 166.f, 198.f, 230.f,   7.f,  39.f,  71.f, 103.f, 135.f, 167.f, 199.f, 231.f,   8.f,  40.f,  72.f, 104.f, 136.f, 168.f, 200.f, 232.f,
        9.f,  41.f,  73.f, 105.f, 137.f, 169.f, 201.f, 233.f,  10.f,  42.f,  74.f, 106.f, 138.f, 170.f, 202.f, 234.f,  11.f,  43.f,  75.f, 107.f, 139.f, 171.f, 203.f, 235.f,
        12.f,  44.f,  76.f, 108.f, 140.f, 172.f, 204.f, 236.f,  13.f,  45.f,  77.f, 109.f, 141.f, 173.f, 205.f, 237.f,  14.f,  46.f,  78.f, 110.f, 142.f, 174.f, 206.f, 238.f,
        15.f,  47.f,  79.f, 111.f, 143.f, 175.f, 207.f, 239.f,  16.f,  48.f,  80.f, 112.f, 144.f, 176.f, 208.f, 240.f,  17.f,  49.f,  81.f, 113.f, 145.f, 177.f, 209.f, 241.f,
        18.f,  50.f,  82.f, 114.f, 146.f, 178.f, 210.f, 242.f,  19.f,  51.f,  83.f, 115.f, 147.f, 179.f, 211.f, 243.f,  20.f,  52.f,  84.f, 116.f, 148.f, 180.f, 212.f, 244.f,
        21.f,  53.f,  85.f, 117.f, 149.f, 181.f, 213.f, 245.f,  22.f,  54.f,  86.f, 118.f, 150.f, 182.f, 214.f, 246.f,  23.f,  55.f,  87.f, 119.f, 151.f, 183.f, 215.f, 247.f,
        24.f,  56.f,  88.f, 120.f, 152.f, 184.f, 216.f, 248.f,  25.f,  57.f,  89.f, 121.f, 153.f, 185.f, 217.f, 249.f,  26.f,  58.f,  90.f, 122.f, 154.f, 186.f, 218.f, 250.f,
        27.f,  59.f,  91.f, 123.f, 155.f, 187.f, 219.f, 251.f,  28.f,  60.f,  92.f, 124.f, 156.f, 188.f, 220.f, 252.f,  29.f,  61.f,  93.f, 125.f, 157.f, 189.f, 221.f, 253.f,
        30.f,  62.f,  94.f, 126.f, 158.f, 190.f, 222.f, 254.f,  31.f,  63.f,  95.f, 127.f, 159.f, 191.f, 223.f, 255.f, 256.f, 288.f, 320.f, 352.f, 384.f, 416.f, 448.f, 480.f,
        257.f, 289.f, 321.f, 353.f, 385.f, 417.f, 449.f, 481.f, 258.f, 290.f, 322.f, 354.f, 386.f, 418.f, 450.f, 482.f, 259.f, 291.f, 323.f, 355.f, 387.f, 419.f, 451.f, 483.f,
        260.f, 292.f, 324.f, 356.f, 388.f, 420.f, 452.f, 484.f, 261.f, 293.f, 325.f, 357.f, 389.f, 421.f, 453.f, 485.f, 262.f, 294.f, 326.f, 358.f, 390.f, 422.f, 454.f, 486.f,
        263.f, 295.f, 327.f, 359.f, 391.f, 423.f, 455.f, 487.f, 264.f, 296.f, 328.f, 360.f, 392.f, 424.f, 456.f, 488.f, 265.f, 297.f, 329.f, 361.f, 393.f, 425.f, 457.f, 489.f,
        266.f, 298.f, 330.f, 362.f, 394.f, 426.f, 458.f, 490.f, 267.f, 299.f, 331.f, 363.f, 395.f, 427.f, 459.f, 491.f, 268.f, 300.f, 332.f, 364.f, 396.f, 428.f, 460.f, 492.f,
        269.f, 301.f, 333.f, 365.f, 397.f, 429.f, 461.f, 493.f, 270.f, 302.f, 334.f, 366.f, 398.f, 430.f, 462.f, 494.f, 271.f, 303.f, 335.f, 367.f, 399.f, 431.f, 463.f, 495.f,
        272.f, 304.f, 336.f, 368.f, 400.f, 432.f, 464.f, 496.f, 273.f, 305.f, 337.f, 369.f, 401.f, 433.f, 465.f, 497.f, 274.f, 306.f, 338.f, 370.f, 402.f, 434.f, 466.f, 498.f,
        275.f, 307.f, 339.f, 371.f, 403.f, 435.f, 467.f, 499.f, 276.f, 308.f, 340.f, 372.f, 404.f, 436.f, 468.f, 500.f, 277.f, 309.f, 341.f, 373.f, 405.f, 437.f, 469.f, 501.f,
        278.f, 310.f, 342.f, 374.f, 406.f, 438.f, 470.f, 502.f, 279.f, 311.f, 343.f, 375.f, 407.f, 439.f, 471.f, 503.f, 280.f, 312.f, 344.f, 376.f, 408.f, 440.f, 472.f, 504.f,
        281.f, 313.f, 345.f, 377.f, 409.f, 441.f, 473.f, 505.f, 282.f, 314.f, 346.f, 378.f, 410.f, 442.f, 474.f, 506.f, 283.f, 315.f, 347.f, 379.f, 411.f, 443.f, 475.f, 507.f,
        284.f, 316.f, 348.f, 380.f, 412.f, 444.f, 476.f, 508.f, 285.f, 317.f, 349.f, 381.f, 413.f, 445.f, 477.f, 509.f, 286.f, 318.f, 350.f, 382.f, 414.f, 446.f, 478.f, 510.f,
        287.f, 319.f, 351.f, 383.f, 415.f, 447.f, 479.f, 511.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, f_remainder_bfzyx_0_2_3_4_1) {
    //  Input               : bfzyx:2x5x2x2x8
    //  Permute order       : { 0,4,1,2,3 }

    constexpr size_t array_size = 320;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 5, 8, 2, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 4, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  32.f,  64.f,  96.f, 128.f,   1.f,  33.f,  65.f,  97.f, 129.f,   2.f,  34.f,  66.f,  98.f, 130.f,   3.f,  35.f,  67.f,  99.f, 131.f,   4.f,  36.f,  68.f, 100.f,
        132.f,   5.f,  37.f,  69.f, 101.f, 133.f,   6.f,  38.f,  70.f, 102.f, 134.f,   7.f,  39.f,  71.f, 103.f, 135.f,   8.f,  40.f,  72.f, 104.f, 136.f,   9.f,  41.f,  73.f,
        105.f, 137.f,  10.f,  42.f,  74.f, 106.f, 138.f,  11.f,  43.f,  75.f, 107.f, 139.f,  12.f,  44.f,  76.f, 108.f, 140.f,  13.f,  45.f,  77.f, 109.f, 141.f,  14.f,  46.f,
        78.f, 110.f, 142.f,  15.f,  47.f,  79.f, 111.f, 143.f,  16.f,  48.f,  80.f, 112.f, 144.f,  17.f,  49.f,  81.f, 113.f, 145.f,  18.f,  50.f,  82.f, 114.f, 146.f,  19.f,
        51.f,  83.f, 115.f, 147.f,  20.f,  52.f,  84.f, 116.f, 148.f,  21.f,  53.f,  85.f, 117.f, 149.f,  22.f,  54.f,  86.f, 118.f, 150.f,  23.f,  55.f,  87.f, 119.f, 151.f,
        24.f,  56.f,  88.f, 120.f, 152.f,  25.f,  57.f,  89.f, 121.f, 153.f,  26.f,  58.f,  90.f, 122.f, 154.f,  27.f,  59.f,  91.f, 123.f, 155.f,  28.f,  60.f,  92.f, 124.f,
        156.f,  29.f,  61.f,  93.f, 125.f, 157.f,  30.f,  62.f,  94.f, 126.f, 158.f,  31.f,  63.f,  95.f, 127.f, 159.f, 160.f, 192.f, 224.f, 256.f, 288.f, 161.f, 193.f, 225.f,
        257.f, 289.f, 162.f, 194.f, 226.f, 258.f, 290.f, 163.f, 195.f, 227.f, 259.f, 291.f, 164.f, 196.f, 228.f, 260.f, 292.f, 165.f, 197.f, 229.f, 261.f, 293.f, 166.f, 198.f,
        230.f, 262.f, 294.f, 167.f, 199.f, 231.f, 263.f, 295.f, 168.f, 200.f, 232.f, 264.f, 296.f, 169.f, 201.f, 233.f, 265.f, 297.f, 170.f, 202.f, 234.f, 266.f, 298.f, 171.f,
        203.f, 235.f, 267.f, 299.f, 172.f, 204.f, 236.f, 268.f, 300.f, 173.f, 205.f, 237.f, 269.f, 301.f, 174.f, 206.f, 238.f, 270.f, 302.f, 175.f, 207.f, 239.f, 271.f, 303.f,
        176.f, 208.f, 240.f, 272.f, 304.f, 177.f, 209.f, 241.f, 273.f, 305.f, 178.f, 210.f, 242.f, 274.f, 306.f, 179.f, 211.f, 243.f, 275.f, 307.f, 180.f, 212.f, 244.f, 276.f,
        308.f, 181.f, 213.f, 245.f, 277.f, 309.f, 182.f, 214.f, 246.f, 278.f, 310.f, 183.f, 215.f, 247.f, 279.f, 311.f, 184.f, 216.f, 248.f, 280.f, 312.f, 185.f, 217.f, 249.f,
        281.f, 313.f, 186.f, 218.f, 250.f, 282.f, 314.f, 187.f, 219.f, 251.f, 283.f, 315.f, 188.f, 220.f, 252.f, 284.f, 316.f, 189.f, 221.f, 253.f, 285.f, 317.f, 190.f, 222.f,
        254.f, 286.f, 318.f, 191.f, 223.f, 255.f, 287.f, 319.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, x_remainder_bfzyx_0_2_3_4_1) {
    //  Input               : bfzyx:2x8x2x2x5
    //  Permute order       : { 0,4,1,2,3 }

    constexpr size_t array_size = 320;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 8, 5, 2, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 4, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  20.f,  40.f,  60.f,  80.f, 100.f, 120.f, 140.f,   1.f,  21.f,  41.f,  61.f,  81.f, 101.f, 121.f, 141.f,   2.f,  22.f,  42.f,  62.f,  82.f, 102.f, 122.f, 142.f,
        3.f,  23.f,  43.f,  63.f,  83.f, 103.f, 123.f, 143.f,   4.f,  24.f,  44.f,  64.f,  84.f, 104.f, 124.f, 144.f,   5.f,  25.f,  45.f,  65.f,  85.f, 105.f, 125.f, 145.f,
        6.f,  26.f,  46.f,  66.f,  86.f, 106.f, 126.f, 146.f,   7.f,  27.f,  47.f,  67.f,  87.f, 107.f, 127.f, 147.f,   8.f,  28.f,  48.f,  68.f,  88.f, 108.f, 128.f, 148.f,
        9.f,  29.f,  49.f,  69.f,  89.f, 109.f, 129.f, 149.f,  10.f,  30.f,  50.f,  70.f,  90.f, 110.f, 130.f, 150.f,  11.f,  31.f,  51.f,  71.f,  91.f, 111.f, 131.f, 151.f,
        12.f,  32.f,  52.f,  72.f,  92.f, 112.f, 132.f, 152.f,  13.f,  33.f,  53.f,  73.f,  93.f, 113.f, 133.f, 153.f,  14.f,  34.f,  54.f,  74.f,  94.f, 114.f, 134.f, 154.f,
        15.f,  35.f,  55.f,  75.f,  95.f, 115.f, 135.f, 155.f,  16.f,  36.f,  56.f,  76.f,  96.f, 116.f, 136.f, 156.f,  17.f,  37.f,  57.f,  77.f,  97.f, 117.f, 137.f, 157.f,
        18.f,  38.f,  58.f,  78.f,  98.f, 118.f, 138.f, 158.f,  19.f,  39.f,  59.f,  79.f,  99.f, 119.f, 139.f, 159.f, 160.f, 180.f, 200.f, 220.f, 240.f, 260.f, 280.f, 300.f,
        161.f, 181.f, 201.f, 221.f, 241.f, 261.f, 281.f, 301.f, 162.f, 182.f, 202.f, 222.f, 242.f, 262.f, 282.f, 302.f, 163.f, 183.f, 203.f, 223.f, 243.f, 263.f, 283.f, 303.f,
        164.f, 184.f, 204.f, 224.f, 244.f, 264.f, 284.f, 304.f, 165.f, 185.f, 205.f, 225.f, 245.f, 265.f, 285.f, 305.f, 166.f, 186.f, 206.f, 226.f, 246.f, 266.f, 286.f, 306.f,
        167.f, 187.f, 207.f, 227.f, 247.f, 267.f, 287.f, 307.f, 168.f, 188.f, 208.f, 228.f, 248.f, 268.f, 288.f, 308.f, 169.f, 189.f, 209.f, 229.f, 249.f, 269.f, 289.f, 309.f,
        170.f, 190.f, 210.f, 230.f, 250.f, 270.f, 290.f, 310.f, 171.f, 191.f, 211.f, 231.f, 251.f, 271.f, 291.f, 311.f, 172.f, 192.f, 212.f, 232.f, 252.f, 272.f, 292.f, 312.f,
        173.f, 193.f, 213.f, 233.f, 253.f, 273.f, 293.f, 313.f, 174.f, 194.f, 214.f, 234.f, 254.f, 274.f, 294.f, 314.f, 175.f, 195.f, 215.f, 235.f, 255.f, 275.f, 295.f, 315.f,
        176.f, 196.f, 216.f, 236.f, 256.f, 276.f, 296.f, 316.f, 177.f, 197.f, 217.f, 237.f, 257.f, 277.f, 297.f, 317.f, 178.f, 198.f, 218.f, 238.f, 258.f, 278.f, 298.f, 318.f,
        179.f, 199.f, 219.f, 239.f, 259.f, 279.f, 299.f, 319.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, xf_remainder_bfzyx_0_2_3_4_1) {
    //  Input               : bfzyx:2x5x2x2x5
    //  Permute order       : { 0,4,1,2,3 }

    constexpr size_t array_size = 200;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 2, 5, 5, 2, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 4, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  20.f,  40.f,  60.f,  80.f,   1.f,  21.f,  41.f,  61.f,  81.f,   2.f,  22.f,  42.f,  62.f,  82.f,
        3.f,  23.f,  43.f,  63.f,  83.f,   4.f,  24.f,  44.f,  64.f,  84.f,   5.f,  25.f,  45.f,  65.f,  85.f,
        6.f,  26.f,  46.f,  66.f,  86.f,   7.f,  27.f,  47.f,  67.f,  87.f,   8.f,  28.f,  48.f,  68.f,  88.f,
        9.f,  29.f,  49.f,  69.f,  89.f,  10.f,  30.f,  50.f,  70.f,  90.f,  11.f,  31.f,  51.f,  71.f,  91.f,
        12.f,  32.f,  52.f,  72.f,  92.f,  13.f,  33.f,  53.f,  73.f,  93.f,  14.f,  34.f,  54.f,  74.f,  94.f,
        15.f,  35.f,  55.f,  75.f,  95.f,  16.f,  36.f,  56.f,  76.f,  96.f,  17.f,  37.f,  57.f,  77.f,  97.f,
        18.f,  38.f,  58.f,  78.f,  98.f,  19.f,  39.f,  59.f,  79.f,  99.f, 100.f, 120.f, 140.f, 160.f, 180.f,
        101.f, 121.f, 141.f, 161.f, 181.f, 102.f, 122.f, 142.f, 162.f, 182.f, 103.f, 123.f, 143.f, 163.f, 183.f,
        104.f, 124.f, 144.f, 164.f, 184.f, 105.f, 125.f, 145.f, 165.f, 185.f, 106.f, 126.f, 146.f, 166.f, 186.f,
        107.f, 127.f, 147.f, 167.f, 187.f, 108.f, 128.f, 148.f, 168.f, 188.f, 109.f, 129.f, 149.f, 169.f, 189.f,
        110.f, 130.f, 150.f, 170.f, 190.f, 111.f, 131.f, 151.f, 171.f, 191.f, 112.f, 132.f, 152.f, 172.f, 192.f,
        113.f, 133.f, 153.f, 173.f, 193.f, 114.f, 134.f, 154.f, 174.f, 194.f, 115.f, 135.f, 155.f, 175.f, 195.f,
        116.f, 136.f, 156.f, 176.f, 196.f, 117.f, 137.f, 157.f, 177.f, 197.f, 118.f, 138.f, 158.f, 178.f, 198.f,
        119.f, 139.f, 159.f, 179.f, 199.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, normal_bfwzyx_0_2_3_4_5_1) {
    //  Input               : bfwzyx:2x8x2x2x2x8
    //  Permute order       : { 0,2,3,4,5,1 }

    constexpr size_t array_size = 1024;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx,{ 2, 8, 8, 2, 2, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 4, 5, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  64.f, 128.f, 192.f, 256.f, 320.f, 384.f, 448.f,   1.f,  65.f, 129.f, 193.f, 257.f, 321.f, 385.f, 449.f,   2.f,  66.f, 130.f, 194.f, 258.f, 322.f, 386.f, 450.f,   3.f,  67.f, 131.f, 195.f, 259.f, 323.f, 387.f, 451.f,
        4.f,  68.f, 132.f, 196.f, 260.f, 324.f, 388.f, 452.f,   5.f,  69.f, 133.f, 197.f, 261.f, 325.f, 389.f, 453.f,   6.f,  70.f, 134.f, 198.f, 262.f, 326.f, 390.f, 454.f,   7.f,  71.f, 135.f, 199.f, 263.f, 327.f, 391.f, 455.f,
        8.f,  72.f, 136.f, 200.f, 264.f, 328.f, 392.f, 456.f,   9.f,  73.f, 137.f, 201.f, 265.f, 329.f, 393.f, 457.f,  10.f,  74.f, 138.f, 202.f, 266.f, 330.f, 394.f, 458.f,  11.f,  75.f, 139.f, 203.f, 267.f, 331.f, 395.f, 459.f,
        12.f,  76.f, 140.f, 204.f, 268.f, 332.f, 396.f, 460.f,  13.f,  77.f, 141.f, 205.f, 269.f, 333.f, 397.f, 461.f,  14.f,  78.f, 142.f, 206.f, 270.f, 334.f, 398.f, 462.f,  15.f,  79.f, 143.f, 207.f, 271.f, 335.f, 399.f, 463.f,
        16.f,  80.f, 144.f, 208.f, 272.f, 336.f, 400.f, 464.f,  17.f,  81.f, 145.f, 209.f, 273.f, 337.f, 401.f, 465.f,  18.f,  82.f, 146.f, 210.f, 274.f, 338.f, 402.f, 466.f,  19.f,  83.f, 147.f, 211.f, 275.f, 339.f, 403.f, 467.f,
        20.f,  84.f, 148.f, 212.f, 276.f, 340.f, 404.f, 468.f,  21.f,  85.f, 149.f, 213.f, 277.f, 341.f, 405.f, 469.f,  22.f,  86.f, 150.f, 214.f, 278.f, 342.f, 406.f, 470.f,  23.f,  87.f, 151.f, 215.f, 279.f, 343.f, 407.f, 471.f,
        24.f,  88.f, 152.f, 216.f, 280.f, 344.f, 408.f, 472.f,  25.f,  89.f, 153.f, 217.f, 281.f, 345.f, 409.f, 473.f,  26.f,  90.f, 154.f, 218.f, 282.f, 346.f, 410.f, 474.f,  27.f,  91.f, 155.f, 219.f, 283.f, 347.f, 411.f, 475.f,
        28.f,  92.f, 156.f, 220.f, 284.f, 348.f, 412.f, 476.f,  29.f,  93.f, 157.f, 221.f, 285.f, 349.f, 413.f, 477.f,  30.f,  94.f, 158.f, 222.f, 286.f, 350.f, 414.f, 478.f,  31.f,  95.f, 159.f, 223.f, 287.f, 351.f, 415.f, 479.f,
        32.f,  96.f, 160.f, 224.f, 288.f, 352.f, 416.f, 480.f,  33.f,  97.f, 161.f, 225.f, 289.f, 353.f, 417.f, 481.f,  34.f,  98.f, 162.f, 226.f, 290.f, 354.f, 418.f, 482.f,  35.f,  99.f, 163.f, 227.f, 291.f, 355.f, 419.f, 483.f,
        36.f, 100.f, 164.f, 228.f, 292.f, 356.f, 420.f, 484.f,  37.f, 101.f, 165.f, 229.f, 293.f, 357.f, 421.f, 485.f,  38.f, 102.f, 166.f, 230.f, 294.f, 358.f, 422.f, 486.f,  39.f, 103.f, 167.f, 231.f, 295.f, 359.f, 423.f, 487.f,
        40.f, 104.f, 168.f, 232.f, 296.f, 360.f, 424.f, 488.f,  41.f, 105.f, 169.f, 233.f, 297.f, 361.f, 425.f, 489.f,  42.f, 106.f, 170.f, 234.f, 298.f, 362.f, 426.f, 490.f,  43.f, 107.f, 171.f, 235.f, 299.f, 363.f, 427.f, 491.f,
        44.f, 108.f, 172.f, 236.f, 300.f, 364.f, 428.f, 492.f,  45.f, 109.f, 173.f, 237.f, 301.f, 365.f, 429.f, 493.f,  46.f, 110.f, 174.f, 238.f, 302.f, 366.f, 430.f, 494.f,  47.f, 111.f, 175.f, 239.f, 303.f, 367.f, 431.f, 495.f,
        48.f, 112.f, 176.f, 240.f, 304.f, 368.f, 432.f, 496.f,  49.f, 113.f, 177.f, 241.f, 305.f, 369.f, 433.f, 497.f,  50.f, 114.f, 178.f, 242.f, 306.f, 370.f, 434.f, 498.f,  51.f, 115.f, 179.f, 243.f, 307.f, 371.f, 435.f, 499.f,
        52.f, 116.f, 180.f, 244.f, 308.f, 372.f, 436.f, 500.f,  53.f, 117.f, 181.f, 245.f, 309.f, 373.f, 437.f, 501.f,  54.f, 118.f, 182.f, 246.f, 310.f, 374.f, 438.f, 502.f,  55.f, 119.f, 183.f, 247.f, 311.f, 375.f, 439.f, 503.f,
        56.f, 120.f, 184.f, 248.f, 312.f, 376.f, 440.f, 504.f,  57.f, 121.f, 185.f, 249.f, 313.f, 377.f, 441.f, 505.f,  58.f, 122.f, 186.f, 250.f, 314.f, 378.f, 442.f, 506.f,  59.f, 123.f, 187.f, 251.f, 315.f, 379.f, 443.f, 507.f,
        60.f, 124.f, 188.f, 252.f, 316.f, 380.f, 444.f, 508.f,  61.f, 125.f, 189.f, 253.f, 317.f, 381.f, 445.f, 509.f,  62.f, 126.f, 190.f, 254.f, 318.f, 382.f, 446.f, 510.f,  63.f, 127.f, 191.f, 255.f, 319.f, 383.f, 447.f, 511.f,
        512.f, 576.f, 640.f, 704.f, 768.f, 832.f, 896.f, 960.f, 513.f, 577.f, 641.f, 705.f, 769.f, 833.f, 897.f, 961.f, 514.f, 578.f, 642.f, 706.f, 770.f, 834.f, 898.f, 962.f, 515.f, 579.f, 643.f, 707.f, 771.f, 835.f, 899.f, 963.f,
        516.f, 580.f, 644.f, 708.f, 772.f, 836.f, 900.f, 964.f, 517.f, 581.f, 645.f, 709.f, 773.f, 837.f, 901.f, 965.f, 518.f, 582.f, 646.f, 710.f, 774.f, 838.f, 902.f, 966.f, 519.f, 583.f, 647.f, 711.f, 775.f, 839.f, 903.f, 967.f,
        520.f, 584.f, 648.f, 712.f, 776.f, 840.f, 904.f, 968.f, 521.f, 585.f, 649.f, 713.f, 777.f, 841.f, 905.f, 969.f, 522.f, 586.f, 650.f, 714.f, 778.f, 842.f, 906.f, 970.f, 523.f, 587.f, 651.f, 715.f, 779.f, 843.f, 907.f, 971.f,
        524.f, 588.f, 652.f, 716.f, 780.f, 844.f, 908.f, 972.f, 525.f, 589.f, 653.f, 717.f, 781.f, 845.f, 909.f, 973.f, 526.f, 590.f, 654.f, 718.f, 782.f, 846.f, 910.f, 974.f, 527.f, 591.f, 655.f, 719.f, 783.f, 847.f, 911.f, 975.f,
        528.f, 592.f, 656.f, 720.f, 784.f, 848.f, 912.f, 976.f, 529.f, 593.f, 657.f, 721.f, 785.f, 849.f, 913.f, 977.f, 530.f, 594.f, 658.f, 722.f, 786.f, 850.f, 914.f, 978.f, 531.f, 595.f, 659.f, 723.f, 787.f, 851.f, 915.f, 979.f,
        532.f, 596.f, 660.f, 724.f, 788.f, 852.f, 916.f, 980.f, 533.f, 597.f, 661.f, 725.f, 789.f, 853.f, 917.f, 981.f, 534.f, 598.f, 662.f, 726.f, 790.f, 854.f, 918.f, 982.f, 535.f, 599.f, 663.f, 727.f, 791.f, 855.f, 919.f, 983.f,
        536.f, 600.f, 664.f, 728.f, 792.f, 856.f, 920.f, 984.f, 537.f, 601.f, 665.f, 729.f, 793.f, 857.f, 921.f, 985.f, 538.f, 602.f, 666.f, 730.f, 794.f, 858.f, 922.f, 986.f, 539.f, 603.f, 667.f, 731.f, 795.f, 859.f, 923.f, 987.f,
        540.f, 604.f, 668.f, 732.f, 796.f, 860.f, 924.f, 988.f, 541.f, 605.f, 669.f, 733.f, 797.f, 861.f, 925.f, 989.f, 542.f, 606.f, 670.f, 734.f, 798.f, 862.f, 926.f, 990.f, 543.f, 607.f, 671.f, 735.f, 799.f, 863.f, 927.f, 991.f,
        544.f, 608.f, 672.f, 736.f, 800.f, 864.f, 928.f, 992.f, 545.f, 609.f, 673.f, 737.f, 801.f, 865.f, 929.f, 993.f, 546.f, 610.f, 674.f, 738.f, 802.f, 866.f, 930.f, 994.f, 547.f, 611.f, 675.f, 739.f, 803.f, 867.f, 931.f, 995.f,
        548.f, 612.f, 676.f, 740.f, 804.f, 868.f, 932.f, 996.f, 549.f, 613.f, 677.f, 741.f, 805.f, 869.f, 933.f, 997.f, 550.f, 614.f, 678.f, 742.f, 806.f, 870.f, 934.f, 998.f, 551.f, 615.f, 679.f, 743.f, 807.f, 871.f, 935.f, 999.f,
        552.f, 616.f, 680.f, 744.f, 808.f, 872.f, 936.f, 1000.f, 553.f, 617.f, 681.f, 745.f, 809.f, 873.f, 937.f, 1001.f, 554.f, 618.f, 682.f, 746.f, 810.f, 874.f, 938.f, 1002.f, 555.f, 619.f, 683.f, 747.f, 811.f, 875.f, 939.f, 1003.f,
        556.f, 620.f, 684.f, 748.f, 812.f, 876.f, 940.f, 1004.f, 557.f, 621.f, 685.f, 749.f, 813.f, 877.f, 941.f, 1005.f, 558.f, 622.f, 686.f, 750.f, 814.f, 878.f, 942.f, 1006.f, 559.f, 623.f, 687.f, 751.f, 815.f, 879.f, 943.f, 1007.f,
        560.f, 624.f, 688.f, 752.f, 816.f, 880.f, 944.f, 1008.f, 561.f, 625.f, 689.f, 753.f, 817.f, 881.f, 945.f, 1009.f, 562.f, 626.f, 690.f, 754.f, 818.f, 882.f, 946.f, 1010.f, 563.f, 627.f, 691.f, 755.f, 819.f, 883.f, 947.f, 1011.f,
        564.f, 628.f, 692.f, 756.f, 820.f, 884.f, 948.f, 1012.f, 565.f, 629.f, 693.f, 757.f, 821.f, 885.f, 949.f, 1013.f, 566.f, 630.f, 694.f, 758.f, 822.f, 886.f, 950.f, 1014.f, 567.f, 631.f, 695.f, 759.f, 823.f, 887.f, 951.f, 1015.f,
        568.f, 632.f, 696.f, 760.f, 824.f, 888.f, 952.f, 1016.f, 569.f, 633.f, 697.f, 761.f, 825.f, 889.f, 953.f, 1017.f, 570.f, 634.f, 698.f, 762.f, 826.f, 890.f, 954.f, 1018.f, 571.f, 635.f, 699.f, 763.f, 827.f, 891.f, 955.f, 1019.f,
        572.f, 636.f, 700.f, 764.f, 828.f, 892.f, 956.f, 1020.f, 573.f, 637.f, 701.f, 765.f, 829.f, 893.f, 957.f, 1021.f, 574.f, 638.f, 702.f, 766.f, 830.f, 894.f, 958.f, 1022.f, 575.f, 639.f, 703.f, 767.f, 831.f, 895.f, 959.f, 1023.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, f_remainder_bfwzyx_0_2_3_4_5_1) {
    //  Input               : bfwzyx:2x5x2x2x2x8
    //  Permute order       : { 0,2,3,4,5,1 }

    constexpr size_t array_size = 640;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx,{ 2, 5, 8, 2, 2, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 4, 5, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  64.f, 128.f, 192.f, 256.f,   1.f,  65.f, 129.f, 193.f, 257.f,   2.f,  66.f, 130.f, 194.f, 258.f,   3.f,  67.f, 131.f, 195.f, 259.f,   4.f,  68.f, 132.f, 196.f, 260.f,   5.f,  69.f, 133.f, 197.f, 261.f,   6.f,  70.f,
        134.f, 198.f, 262.f,   7.f,  71.f, 135.f, 199.f, 263.f,   8.f,  72.f, 136.f, 200.f, 264.f,   9.f,  73.f, 137.f, 201.f, 265.f,  10.f,  74.f, 138.f, 202.f, 266.f,  11.f,  75.f, 139.f, 203.f, 267.f,  12.f,  76.f, 140.f, 204.f,
        268.f,  13.f,  77.f, 141.f, 205.f, 269.f,  14.f,  78.f, 142.f, 206.f, 270.f,  15.f,  79.f, 143.f, 207.f, 271.f,  16.f,  80.f, 144.f, 208.f, 272.f,  17.f,  81.f, 145.f, 209.f, 273.f,  18.f,  82.f, 146.f, 210.f, 274.f,  19.f,
        83.f, 147.f, 211.f, 275.f,  20.f,  84.f, 148.f, 212.f, 276.f,  21.f,  85.f, 149.f, 213.f, 277.f,  22.f,  86.f, 150.f, 214.f, 278.f,  23.f,  87.f, 151.f, 215.f, 279.f,  24.f,  88.f, 152.f, 216.f, 280.f,  25.f,  89.f, 153.f,
        217.f, 281.f,  26.f,  90.f, 154.f, 218.f, 282.f,  27.f,  91.f, 155.f, 219.f, 283.f,  28.f,  92.f, 156.f, 220.f, 284.f,  29.f,  93.f, 157.f, 221.f, 285.f,  30.f,  94.f, 158.f, 222.f, 286.f,  31.f,  95.f, 159.f, 223.f, 287.f,
        32.f,  96.f, 160.f, 224.f, 288.f,  33.f,  97.f, 161.f, 225.f, 289.f,  34.f,  98.f, 162.f, 226.f, 290.f,  35.f,  99.f, 163.f, 227.f, 291.f,  36.f, 100.f, 164.f, 228.f, 292.f,  37.f, 101.f, 165.f, 229.f, 293.f,  38.f, 102.f,
        166.f, 230.f, 294.f,  39.f, 103.f, 167.f, 231.f, 295.f,  40.f, 104.f, 168.f, 232.f, 296.f,  41.f, 105.f, 169.f, 233.f, 297.f,  42.f, 106.f, 170.f, 234.f, 298.f,  43.f, 107.f, 171.f, 235.f, 299.f,  44.f, 108.f, 172.f, 236.f,
        300.f,  45.f, 109.f, 173.f, 237.f, 301.f,  46.f, 110.f, 174.f, 238.f, 302.f,  47.f, 111.f, 175.f, 239.f, 303.f,  48.f, 112.f, 176.f, 240.f, 304.f,  49.f, 113.f, 177.f, 241.f, 305.f,  50.f, 114.f, 178.f, 242.f, 306.f,  51.f,
        115.f, 179.f, 243.f, 307.f,  52.f, 116.f, 180.f, 244.f, 308.f,  53.f, 117.f, 181.f, 245.f, 309.f,  54.f, 118.f, 182.f, 246.f, 310.f,  55.f, 119.f, 183.f, 247.f, 311.f,  56.f, 120.f, 184.f, 248.f, 312.f,  57.f, 121.f, 185.f,
        249.f, 313.f,  58.f, 122.f, 186.f, 250.f, 314.f,  59.f, 123.f, 187.f, 251.f, 315.f,  60.f, 124.f, 188.f, 252.f, 316.f,  61.f, 125.f, 189.f, 253.f, 317.f,  62.f, 126.f, 190.f, 254.f, 318.f,  63.f, 127.f, 191.f, 255.f, 319.f,
        320.f, 384.f, 448.f, 512.f, 576.f, 321.f, 385.f, 449.f, 513.f, 577.f, 322.f, 386.f, 450.f, 514.f, 578.f, 323.f, 387.f, 451.f, 515.f, 579.f, 324.f, 388.f, 452.f, 516.f, 580.f, 325.f, 389.f, 453.f, 517.f, 581.f, 326.f, 390.f,
        454.f, 518.f, 582.f, 327.f, 391.f, 455.f, 519.f, 583.f, 328.f, 392.f, 456.f, 520.f, 584.f, 329.f, 393.f, 457.f, 521.f, 585.f, 330.f, 394.f, 458.f, 522.f, 586.f, 331.f, 395.f, 459.f, 523.f, 587.f, 332.f, 396.f, 460.f, 524.f,
        588.f, 333.f, 397.f, 461.f, 525.f, 589.f, 334.f, 398.f, 462.f, 526.f, 590.f, 335.f, 399.f, 463.f, 527.f, 591.f, 336.f, 400.f, 464.f, 528.f, 592.f, 337.f, 401.f, 465.f, 529.f, 593.f, 338.f, 402.f, 466.f, 530.f, 594.f, 339.f,
        403.f, 467.f, 531.f, 595.f, 340.f, 404.f, 468.f, 532.f, 596.f, 341.f, 405.f, 469.f, 533.f, 597.f, 342.f, 406.f, 470.f, 534.f, 598.f, 343.f, 407.f, 471.f, 535.f, 599.f, 344.f, 408.f, 472.f, 536.f, 600.f, 345.f, 409.f, 473.f,
        537.f, 601.f, 346.f, 410.f, 474.f, 538.f, 602.f, 347.f, 411.f, 475.f, 539.f, 603.f, 348.f, 412.f, 476.f, 540.f, 604.f, 349.f, 413.f, 477.f, 541.f, 605.f, 350.f, 414.f, 478.f, 542.f, 606.f, 351.f, 415.f, 479.f, 543.f, 607.f,
        352.f, 416.f, 480.f, 544.f, 608.f, 353.f, 417.f, 481.f, 545.f, 609.f, 354.f, 418.f, 482.f, 546.f, 610.f, 355.f, 419.f, 483.f, 547.f, 611.f, 356.f, 420.f, 484.f, 548.f, 612.f, 357.f, 421.f, 485.f, 549.f, 613.f, 358.f, 422.f,
        486.f, 550.f, 614.f, 359.f, 423.f, 487.f, 551.f, 615.f, 360.f, 424.f, 488.f, 552.f, 616.f, 361.f, 425.f, 489.f, 553.f, 617.f, 362.f, 426.f, 490.f, 554.f, 618.f, 363.f, 427.f, 491.f, 555.f, 619.f, 364.f, 428.f, 492.f, 556.f,
        620.f, 365.f, 429.f, 493.f, 557.f, 621.f, 366.f, 430.f, 494.f, 558.f, 622.f, 367.f, 431.f, 495.f, 559.f, 623.f, 368.f, 432.f, 496.f, 560.f, 624.f, 369.f, 433.f, 497.f, 561.f, 625.f, 370.f, 434.f, 498.f, 562.f, 626.f, 371.f,
        435.f, 499.f, 563.f, 627.f, 372.f, 436.f, 500.f, 564.f, 628.f, 373.f, 437.f, 501.f, 565.f, 629.f, 374.f, 438.f, 502.f, 566.f, 630.f, 375.f, 439.f, 503.f, 567.f, 631.f, 376.f, 440.f, 504.f, 568.f, 632.f, 377.f, 441.f, 505.f,
        569.f, 633.f, 378.f, 442.f, 506.f, 570.f, 634.f, 379.f, 443.f, 507.f, 571.f, 635.f, 380.f, 444.f, 508.f, 572.f, 636.f, 381.f, 445.f, 509.f, 573.f, 637.f, 382.f, 446.f, 510.f, 574.f, 638.f, 383.f, 447.f, 511.f, 575.f, 639.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, x_remainder_bfwzyx_0_2_3_4_5_1) {
    //  Input               : bfwzyx:2x8x2x2x2x5
    //  Permute order       : { 0,2,3,4,5,1 }

    constexpr size_t array_size = 640;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx,{ 2, 8, 5, 2, 2, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 4, 5, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  40.f,  80.f, 120.f, 160.f, 200.f, 240.f, 280.f,   1.f,  41.f,  81.f, 121.f, 161.f, 201.f, 241.f, 281.f,   2.f,  42.f,  82.f, 122.f, 162.f, 202.f, 242.f, 282.f,   3.f,  43.f,  83.f, 123.f, 163.f, 203.f, 243.f, 283.f,
        4.f,  44.f,  84.f, 124.f, 164.f, 204.f, 244.f, 284.f,   5.f,  45.f,  85.f, 125.f, 165.f, 205.f, 245.f, 285.f,   6.f,  46.f,  86.f, 126.f, 166.f, 206.f, 246.f, 286.f,   7.f,  47.f,  87.f, 127.f, 167.f, 207.f, 247.f, 287.f,
        8.f,  48.f,  88.f, 128.f, 168.f, 208.f, 248.f, 288.f,   9.f,  49.f,  89.f, 129.f, 169.f, 209.f, 249.f, 289.f,  10.f,  50.f,  90.f, 130.f, 170.f, 210.f, 250.f, 290.f,  11.f,  51.f,  91.f, 131.f, 171.f, 211.f, 251.f, 291.f,
        12.f,  52.f,  92.f, 132.f, 172.f, 212.f, 252.f, 292.f,  13.f,  53.f,  93.f, 133.f, 173.f, 213.f, 253.f, 293.f,  14.f,  54.f,  94.f, 134.f, 174.f, 214.f, 254.f, 294.f,  15.f,  55.f,  95.f, 135.f, 175.f, 215.f, 255.f, 295.f,
        16.f,  56.f,  96.f, 136.f, 176.f, 216.f, 256.f, 296.f,  17.f,  57.f,  97.f, 137.f, 177.f, 217.f, 257.f, 297.f,  18.f,  58.f,  98.f, 138.f, 178.f, 218.f, 258.f, 298.f,  19.f,  59.f,  99.f, 139.f, 179.f, 219.f, 259.f, 299.f,
        20.f,  60.f, 100.f, 140.f, 180.f, 220.f, 260.f, 300.f,  21.f,  61.f, 101.f, 141.f, 181.f, 221.f, 261.f, 301.f,  22.f,  62.f, 102.f, 142.f, 182.f, 222.f, 262.f, 302.f,  23.f,  63.f, 103.f, 143.f, 183.f, 223.f, 263.f, 303.f,
        24.f,  64.f, 104.f, 144.f, 184.f, 224.f, 264.f, 304.f,  25.f,  65.f, 105.f, 145.f, 185.f, 225.f, 265.f, 305.f,  26.f,  66.f, 106.f, 146.f, 186.f, 226.f, 266.f, 306.f,  27.f,  67.f, 107.f, 147.f, 187.f, 227.f, 267.f, 307.f,
        28.f,  68.f, 108.f, 148.f, 188.f, 228.f, 268.f, 308.f,  29.f,  69.f, 109.f, 149.f, 189.f, 229.f, 269.f, 309.f,  30.f,  70.f, 110.f, 150.f, 190.f, 230.f, 270.f, 310.f,  31.f,  71.f, 111.f, 151.f, 191.f, 231.f, 271.f, 311.f,
        32.f,  72.f, 112.f, 152.f, 192.f, 232.f, 272.f, 312.f,  33.f,  73.f, 113.f, 153.f, 193.f, 233.f, 273.f, 313.f,  34.f,  74.f, 114.f, 154.f, 194.f, 234.f, 274.f, 314.f,  35.f,  75.f, 115.f, 155.f, 195.f, 235.f, 275.f, 315.f,
        36.f,  76.f, 116.f, 156.f, 196.f, 236.f, 276.f, 316.f,  37.f,  77.f, 117.f, 157.f, 197.f, 237.f, 277.f, 317.f,  38.f,  78.f, 118.f, 158.f, 198.f, 238.f, 278.f, 318.f,  39.f,  79.f, 119.f, 159.f, 199.f, 239.f, 279.f, 319.f,
        320.f, 360.f, 400.f, 440.f, 480.f, 520.f, 560.f, 600.f, 321.f, 361.f, 401.f, 441.f, 481.f, 521.f, 561.f, 601.f, 322.f, 362.f, 402.f, 442.f, 482.f, 522.f, 562.f, 602.f, 323.f, 363.f, 403.f, 443.f, 483.f, 523.f, 563.f, 603.f,
        324.f, 364.f, 404.f, 444.f, 484.f, 524.f, 564.f, 604.f, 325.f, 365.f, 405.f, 445.f, 485.f, 525.f, 565.f, 605.f, 326.f, 366.f, 406.f, 446.f, 486.f, 526.f, 566.f, 606.f, 327.f, 367.f, 407.f, 447.f, 487.f, 527.f, 567.f, 607.f,
        328.f, 368.f, 408.f, 448.f, 488.f, 528.f, 568.f, 608.f, 329.f, 369.f, 409.f, 449.f, 489.f, 529.f, 569.f, 609.f, 330.f, 370.f, 410.f, 450.f, 490.f, 530.f, 570.f, 610.f, 331.f, 371.f, 411.f, 451.f, 491.f, 531.f, 571.f, 611.f,
        332.f, 372.f, 412.f, 452.f, 492.f, 532.f, 572.f, 612.f, 333.f, 373.f, 413.f, 453.f, 493.f, 533.f, 573.f, 613.f, 334.f, 374.f, 414.f, 454.f, 494.f, 534.f, 574.f, 614.f, 335.f, 375.f, 415.f, 455.f, 495.f, 535.f, 575.f, 615.f,
        336.f, 376.f, 416.f, 456.f, 496.f, 536.f, 576.f, 616.f, 337.f, 377.f, 417.f, 457.f, 497.f, 537.f, 577.f, 617.f, 338.f, 378.f, 418.f, 458.f, 498.f, 538.f, 578.f, 618.f, 339.f, 379.f, 419.f, 459.f, 499.f, 539.f, 579.f, 619.f,
        340.f, 380.f, 420.f, 460.f, 500.f, 540.f, 580.f, 620.f, 341.f, 381.f, 421.f, 461.f, 501.f, 541.f, 581.f, 621.f, 342.f, 382.f, 422.f, 462.f, 502.f, 542.f, 582.f, 622.f, 343.f, 383.f, 423.f, 463.f, 503.f, 543.f, 583.f, 623.f,
        344.f, 384.f, 424.f, 464.f, 504.f, 544.f, 584.f, 624.f, 345.f, 385.f, 425.f, 465.f, 505.f, 545.f, 585.f, 625.f, 346.f, 386.f, 426.f, 466.f, 506.f, 546.f, 586.f, 626.f, 347.f, 387.f, 427.f, 467.f, 507.f, 547.f, 587.f, 627.f,
        348.f, 388.f, 428.f, 468.f, 508.f, 548.f, 588.f, 628.f, 349.f, 389.f, 429.f, 469.f, 509.f, 549.f, 589.f, 629.f, 350.f, 390.f, 430.f, 470.f, 510.f, 550.f, 590.f, 630.f, 351.f, 391.f, 431.f, 471.f, 511.f, 551.f, 591.f, 631.f,
        352.f, 392.f, 432.f, 472.f, 512.f, 552.f, 592.f, 632.f, 353.f, 393.f, 433.f, 473.f, 513.f, 553.f, 593.f, 633.f, 354.f, 394.f, 434.f, 474.f, 514.f, 554.f, 594.f, 634.f, 355.f, 395.f, 435.f, 475.f, 515.f, 555.f, 595.f, 635.f,
        356.f, 396.f, 436.f, 476.f, 516.f, 556.f, 596.f, 636.f, 357.f, 397.f, 437.f, 477.f, 517.f, 557.f, 597.f, 637.f, 358.f, 398.f, 438.f, 478.f, 518.f, 558.f, 598.f, 638.f, 359.f, 399.f, 439.f, 479.f, 519.f, 559.f, 599.f, 639.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_tile_8x8_4x4, xf_remainder_bfwzyx_0_2_3_4_5_1) {
    //  Input               : bfwzyx:2x5x2x2x2x5
    //  Permute order       : { 0,2,3,4,5,1 }

    constexpr size_t array_size = 400;

    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx,{ 2, 5, 5, 2, 2, 2 } });

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i=0 ; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input->get_layout()),
        permute("permute", input_info("input"), { 0, 2, 3, 4, 5, 1 }));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  40.f,  80.f, 120.f, 160.f,   1.f,  41.f,  81.f, 121.f, 161.f,   2.f,  42.f,  82.f, 122.f, 162.f,   3.f,  43.f,  83.f, 123.f, 163.f,
        4.f,  44.f,  84.f, 124.f, 164.f,   5.f,  45.f,  85.f, 125.f, 165.f,   6.f,  46.f,  86.f, 126.f, 166.f,   7.f,  47.f,  87.f, 127.f, 167.f,
        8.f,  48.f,  88.f, 128.f, 168.f,   9.f,  49.f,  89.f, 129.f, 169.f,  10.f,  50.f,  90.f, 130.f, 170.f,  11.f,  51.f,  91.f, 131.f, 171.f,
        12.f,  52.f,  92.f, 132.f, 172.f,  13.f,  53.f,  93.f, 133.f, 173.f,  14.f,  54.f,  94.f, 134.f, 174.f,  15.f,  55.f,  95.f, 135.f, 175.f,
        16.f,  56.f,  96.f, 136.f, 176.f,  17.f,  57.f,  97.f, 137.f, 177.f,  18.f,  58.f,  98.f, 138.f, 178.f,  19.f,  59.f,  99.f, 139.f, 179.f,
        20.f,  60.f, 100.f, 140.f, 180.f,  21.f,  61.f, 101.f, 141.f, 181.f,  22.f,  62.f, 102.f, 142.f, 182.f,  23.f,  63.f, 103.f, 143.f, 183.f,
        24.f,  64.f, 104.f, 144.f, 184.f,  25.f,  65.f, 105.f, 145.f, 185.f,  26.f,  66.f, 106.f, 146.f, 186.f,  27.f,  67.f, 107.f, 147.f, 187.f,
        28.f,  68.f, 108.f, 148.f, 188.f,  29.f,  69.f, 109.f, 149.f, 189.f,  30.f,  70.f, 110.f, 150.f, 190.f,  31.f,  71.f, 111.f, 151.f, 191.f,
        32.f,  72.f, 112.f, 152.f, 192.f,  33.f,  73.f, 113.f, 153.f, 193.f,  34.f,  74.f, 114.f, 154.f, 194.f,  35.f,  75.f, 115.f, 155.f, 195.f,
        36.f,  76.f, 116.f, 156.f, 196.f,  37.f,  77.f, 117.f, 157.f, 197.f,  38.f,  78.f, 118.f, 158.f, 198.f,  39.f,  79.f, 119.f, 159.f, 199.f,
        200.f, 240.f, 280.f, 320.f, 360.f, 201.f, 241.f, 281.f, 321.f, 361.f, 202.f, 242.f, 282.f, 322.f, 362.f, 203.f, 243.f, 283.f, 323.f, 363.f,
        204.f, 244.f, 284.f, 324.f, 364.f, 205.f, 245.f, 285.f, 325.f, 365.f, 206.f, 246.f, 286.f, 326.f, 366.f, 207.f, 247.f, 287.f, 327.f, 367.f,
        208.f, 248.f, 288.f, 328.f, 368.f, 209.f, 249.f, 289.f, 329.f, 369.f, 210.f, 250.f, 290.f, 330.f, 370.f, 211.f, 251.f, 291.f, 331.f, 371.f,
        212.f, 252.f, 292.f, 332.f, 372.f, 213.f, 253.f, 293.f, 333.f, 373.f, 214.f, 254.f, 294.f, 334.f, 374.f, 215.f, 255.f, 295.f, 335.f, 375.f,
        216.f, 256.f, 296.f, 336.f, 376.f, 217.f, 257.f, 297.f, 337.f, 377.f, 218.f, 258.f, 298.f, 338.f, 378.f, 219.f, 259.f, 299.f, 339.f, 379.f,
        220.f, 260.f, 300.f, 340.f, 380.f, 221.f, 261.f, 301.f, 341.f, 381.f, 222.f, 262.f, 302.f, 342.f, 382.f, 223.f, 263.f, 303.f, 343.f, 383.f,
        224.f, 264.f, 304.f, 344.f, 384.f, 225.f, 265.f, 305.f, 345.f, 385.f, 226.f, 266.f, 306.f, 346.f, 386.f, 227.f, 267.f, 307.f, 347.f, 387.f,
        228.f, 268.f, 308.f, 348.f, 388.f, 229.f, 269.f, 309.f, 349.f, 389.f, 230.f, 270.f, 310.f, 350.f, 390.f, 231.f, 271.f, 311.f, 351.f, 391.f,
        232.f, 272.f, 312.f, 352.f, 392.f, 233.f, 273.f, 313.f, 353.f, 393.f, 234.f, 274.f, 314.f, 354.f, 394.f, 235.f, 275.f, 315.f, 355.f, 395.f,
        236.f, 276.f, 316.f, 356.f, 396.f, 237.f, 277.f, 317.f, 357.f, 397.f, 238.f, 278.f, 318.f, 358.f, 398.f, 239.f, 279.f, 319.f, 359.f, 399.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++)
    {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

struct TiledPermuteParam {
    std::vector<cldnn::tensor::value_type> sizes;
    cldnn::format format_fsv;
};

class TiledPermuteTest : public ::testing::TestWithParam<TiledPermuteParam> {
public:
    cldnn::engine& engine = get_test_engine();
    TiledPermuteTest(): engine(get_test_engine()) { }

    template<typename T>
    void compare_value(T a, T b) const {
        ASSERT_EQ(a, b);
    }

    template<typename T>
    void set_random_values(const cldnn::memory::ptr mem) const {
        tests::set_random_values<T>(mem);
    }

    template<data_types Data_Type>
    void run_test(const std::vector<cldnn::tensor::value_type>& sizes, cldnn::format format_fsv,
                  const std::string & permute_opt = "permute_tile_8x8_4x4_fsv",
                  std::vector<uint16_t> permute_order = {}, bool is_caching_test = false);

    static std::string PrintToStringParamName(const testing::TestParamInfo<TiledPermuteParam>& info) {
        std::ostringstream result;
        result << "InputShape=" << vec2str(info.param.sizes) << "_";
        result << "Format=" << fmt_to_str(info.param.format_fsv);
        return result.str();
    }
};

template<>
void TiledPermuteTest::compare_value(float a, float b) const {
    ASSERT_FLOAT_EQ(a, b);
}

// f16 format
template<>
void TiledPermuteTest::compare_value(ov::float16 a, ov::float16 b) const {
    ASSERT_FLOAT_EQ(static_cast<float>(a), static_cast<float>(b));
}

template<>
void TiledPermuteTest::set_random_values<int8_t>(const cldnn::memory::ptr mem) const {
    // tests::set_random_values<int8_t>() is not supported
    std::mt19937 gen;
    static std::uniform_int_distribution<int32_t> uid(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    cldnn::mem_lock<int8_t> ptr(mem, get_test_stream());
    for (auto it = ptr.begin(); it != ptr.end(); ++it) {
        *it = static_cast<int8_t>(uid(gen));
    }
}

template<data_types Data_Type>
void TiledPermuteTest::run_test(const std::vector<cldnn::tensor::value_type>& sizes, cldnn::format format_fsv,
                                const std::string & permute_opt, std::vector<uint16_t> permute_order, bool is_caching_test)
{
    // convert ov::float16 to ov::float16
    using type_ = typename ov::element_type_traits<Data_Type>::value_type;
    using type = typename std::conditional<std::is_same<type_, ov::float16>::value, ov::float16, type_>::type;

    std::vector<cldnn::tensor::value_type> internal_sizes(sizes);
    std::swap(internal_sizes.at(2), internal_sizes.back());
    cldnn::tensor tensor(internal_sizes);

    cldnn::format format = sizes.size() == 4 ? cldnn::format::bfyx : cldnn::format::bfzyx;

    std::vector<uint16_t> order = {0};
    if (permute_order.empty()) {
        for (uint16_t i = 1; i < (sizes.size() - 1); ++i) {
            order.push_back(i+1);
        }
        order.push_back(1);
    } else {
        std::swap(order, permute_order);
    }

    auto input = engine.allocate_memory({Data_Type, format, tensor});
    set_random_values<type>(input);

    topology topology_ref = topology(
        input_layout("input", input->get_layout()),
        reorder("reorder", input_info("input"), {Data_Type, format_fsv, tensor}),
        permute("output", input_info("reorder"), order )
    );

    // run with permute_ref
    ov::intel_gpu::ExecutionConfig config_ref = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc permute_ref = { format_fsv, "permute_ref" };
    config_ref.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"output", permute_ref} }));

    cldnn::network::ptr network_ref = get_network(engine, topology_ref, config_ref, get_test_stream_ptr(), is_caching_test);
    network_ref->set_input_data("input", input);
    auto outputs_ref = network_ref->execute();
    auto output_ref = outputs_ref.begin()->second.get_memory();
    cldnn::mem_lock<type, mem_lock_type::read> output_ref_ptr(output_ref, get_test_stream());

    // run with optimized kernel, e.g. permute_tile_8x8_4x4_fsv16
    ExecutionConfig config_tile = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc permute_tile_opt = { format_fsv, permute_opt };
    config_tile.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"output", permute_tile_opt} }));

    cldnn::network::ptr network_tile = get_network(engine, topology_ref, config_tile, get_test_stream_ptr(), is_caching_test);
    network_tile->set_input_data("input", input);
    auto outputs_tile = network_tile->execute();
    auto output_tile = outputs_tile.begin()->second.get_memory();
    cldnn::mem_lock<type, mem_lock_type::read> output_tile_ptr(output_tile, get_test_stream());

    // compare results
    const size_t output_size= output_ref->get_layout().get_linear_size();
    for (size_t i = 0; i < output_size; i++)
    {
        compare_value<type>(output_ref_ptr[i], output_tile_ptr[i]);
        if (output_ref_ptr[i] != output_tile_ptr[i]) {
            break;
        }
    }
}

class permute_tile_fsv_4d: public TiledPermuteTest {};

INSTANTIATE_TEST_SUITE_P(, permute_tile_fsv_4d,
    ::testing::ValuesIn(std::vector<TiledPermuteParam> {
        // b_fs_yx_fsv16
        // normal cases
        {{1, 16, 16, 3}, format::b_fs_yx_fsv16},
        // f_not_aligned
        {{1, 16 - 7, 16, 2}, format::b_fs_yx_fsv16},
        {{1, 16 - 15, 16, 2}, format::b_fs_yx_fsv16},
        // y_not_aligned
        {{1, 16, 16 - 1, 2}, format::b_fs_yx_fsv16},
        {{1, 16, 16 - 9, 2}, format::b_fs_yx_fsv16},
        // fy_not_aligned
        {{1, 16 - 15, 16 - 1, 2}, format::b_fs_yx_fsv16},
        {{1, 16 - 1, 16 - 7, 2}, format::b_fs_yx_fsv16},
        {{1, 16 - 7, 16 - 9, 2}, format::b_fs_yx_fsv16},
        {{1, 16 - 9, 16 - 15, 2}, format::b_fs_yx_fsv16},

        // b_fs_yx_fsv32
        // normal cases
        {{1, 32, 32, 3}, format::b_fs_yx_fsv32},
        // f_not_aligned
        {{1, 32 - 7, 32, 2}, format::b_fs_yx_fsv32},
        {{1, 32 - 15, 32, 2}, format::b_fs_yx_fsv32},
        // y_not_aligned
        {{1, 32, 32 - 1, 2}, format::b_fs_yx_fsv32},
        {{1, 32, 32 - 9, 2}, format::b_fs_yx_fsv32},
        // fy_not_aligned
        {{1, 32 - 15, 32 - 1, 2}, format::b_fs_yx_fsv32},
        {{1, 32 - 1, 32 - 7, 2}, format::b_fs_yx_fsv32},
        {{1, 32 - 7, 32 - 9, 2}, format::b_fs_yx_fsv32},
        {{1, 32 - 9, 32 - 15, 2}, format::b_fs_yx_fsv32},

        // b_fs_yx_fsv4
        // normal cases
        {{1, 4, 4, 2}, format::b_fs_yx_fsv4},
        // f_not_aligned
        {{1, 4 - 1, 4, 2}, format::b_fs_yx_fsv4},
        {{1, 4 - 3, 4, 2}, format::b_fs_yx_fsv4},
        // y_not_aligned
        {{1, 4, 4 - 1, 2}, format::b_fs_yx_fsv4},
        {{1, 4, 4 - 3, 2}, format::b_fs_yx_fsv4},
        // fy_not_aligned
        {{1, 4 - 3, 4 - 1, 2}, format::b_fs_yx_fsv4},
        {{1, 4 - 1, 4 - 3, 2}, format::b_fs_yx_fsv4},
    }));

TEST_P(permute_tile_fsv_4d, f16) {
    auto p = GetParam();
    run_test<cldnn::data_types::f16>(p.sizes, p.format_fsv);
}

TEST_P(permute_tile_fsv_4d, f32) {
    auto p = GetParam();
    run_test<cldnn::data_types::f32>(p.sizes, p.format_fsv);
}

TEST_P(permute_tile_fsv_4d, i8) {
    auto p = GetParam();
    run_test<cldnn::data_types::i8>(p.sizes, p.format_fsv);
}

TEST_P(permute_tile_fsv_4d, i32) {
    auto p = GetParam();
    run_test<cldnn::data_types::i32>(p.sizes, p.format_fsv);
}

TEST_P(permute_tile_fsv_4d, i64) {
    auto p = GetParam();
    run_test<cldnn::data_types::i64>(p.sizes, p.format_fsv);
}

class permute_tile_fsv_5d: public TiledPermuteTest {};

INSTANTIATE_TEST_SUITE_P(, permute_tile_fsv_5d,
    ::testing::ValuesIn(std::vector<TiledPermuteParam> {
        // b_fs_zyx_fsv16
        // normal cases
        {{1, 16, 16, 3, 2}, format::b_fs_zyx_fsv16},
        // f_not_aligned
        {{1, 16 - 7, 16, 2, 2}, format::b_fs_zyx_fsv16},
        {{1, 16 - 15, 16, 2, 2}, format::b_fs_zyx_fsv16},
        // z_not_aligned
        {{1, 16, 16 - 1, 2, 2}, format::b_fs_zyx_fsv16},
        {{1, 16, 16 - 9, 2, 2}, format::b_fs_zyx_fsv16},
        // fz_not_aligned
        {{1, 16 - 15, 16 - 1, 2, 2}, format::b_fs_zyx_fsv16},
        {{1, 16 - 1, 16 - 7, 2, 2}, format::b_fs_zyx_fsv16},
        {{1, 16 - 7, 16 - 9, 2, 2}, format::b_fs_zyx_fsv16},
        {{1, 16 - 9, 16 - 15, 2, 2}, format::b_fs_zyx_fsv16},

        // b_fs_zyx_fsv32
        // normal cases
        {{1, 32, 32, 3, 2}, format::b_fs_zyx_fsv32},
        // f_not_aligned
        {{1, 32 - 7, 32, 2, 2}, format::b_fs_zyx_fsv32},
        {{1, 32 - 15, 32, 2, 2}, format::b_fs_zyx_fsv32},
        // z_not_aligned
        {{1, 32, 32 - 1, 2, 2}, format::b_fs_zyx_fsv32},
        {{1, 32, 32 - 9, 2, 2}, format::b_fs_zyx_fsv32},
        // fz_not_aligned
        {{1, 32 - 15, 32 - 1, 2, 2}, format::b_fs_zyx_fsv32},
        {{1, 32 - 1, 32 - 7, 2, 2}, format::b_fs_zyx_fsv32},
        {{1, 32 - 7, 32 - 9, 2, 2}, format::b_fs_zyx_fsv32},
        {{1, 32 - 9, 32 - 15, 2, 2}, format::b_fs_zyx_fsv32},
    }));

TEST_P(permute_tile_fsv_5d, f16) {
    auto p = GetParam();
    run_test<cldnn::data_types::f16>(p.sizes, p.format_fsv);
}

TEST_P(permute_tile_fsv_5d, f32) {
    auto p = GetParam();
    run_test<cldnn::data_types::f32>(p.sizes, p.format_fsv);
}

TEST_P(permute_tile_fsv_5d, i8) {
    auto p = GetParam();
    run_test<cldnn::data_types::i8>(p.sizes, p.format_fsv);
}

TEST_P(permute_tile_fsv_5d, i32) {
    auto p = GetParam();
    run_test<cldnn::data_types::i32>(p.sizes, p.format_fsv);
}

TEST_P(permute_tile_fsv_5d, i64) {
    auto p = GetParam();
    run_test<cldnn::data_types::i64>(p.sizes, p.format_fsv);
}

TEST(permute_gpu_f32_dynamic, bfyx_0_2_3_1) {
    constexpr size_t array_size = 100;

    auto& engine = get_test_engine();

    auto input_layout_dynamic = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{2, 5, 5, 2}, data_types::f32, format::bfyx};

    auto input = engine.allocate_memory(input_layout_static);

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i = 0; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    set_values(input, input_data);

    topology topology(
        input_layout("input", input_layout_dynamic),
        permute("permute", input_info("input"), { 0, 2, 3, 1 }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("permute");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "permute");

    auto output = outputs.begin()->second.get_memory();

    float answers[array_size] = {
        0.f,  10.f,  20.f,  30.f,  40.f,   1.f,  11.f,  21.f,  31.f,  41.f,
        2.f,  12.f,  22.f,  32.f,  42.f,   3.f,  13.f,  23.f,  33.f,  43.f,
        4.f,  14.f,  24.f,  34.f,  44.f,   5.f,  15.f,  25.f,  35.f,  45.f,
        6.f,  16.f,  26.f,  36.f,  46.f,   7.f,  17.f,  27.f,  37.f,  47.f,
        8.f,  18.f,  28.f,  38.f,  48.f,   9.f,  19.f,  29.f,  39.f,  49.f,
        50.f,  60.f,  70.f,  80.f,  90.f,  51.f,  61.f,  71.f,  81.f,  91.f,
        52.f,  62.f,  72.f,  82.f,  92.f,  53.f,  63.f,  73.f,  83.f,  93.f,
        54.f,  64.f,  74.f,  84.f,  94.f,  55.f,  65.f,  75.f,  85.f,  95.f,
        56.f,  66.f,  76.f,  86.f,  96.f,  57.f,  67.f,  77.f,  87.f,  97.f,
        58.f,  68.f,  78.f,  88.f,  98.f,  59.f,  69.f,  79.f,  89.f,  99.f
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++) {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

TEST(permute_gpu_f32_dynamic, fused_op_has_dynamic_shape) {
    auto& engine = get_test_engine();

    auto input1_layout = layout{ov::PartialShape{2, 512, 30}, data_types::f32, format::bfyx};
    auto input2_layout = layout{ov::PartialShape{2, 30, 512}, data_types::f32, format::bfyx};
    auto input3_layout = layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx};

    topology topology(
        input_layout("input1", input1_layout),
        input_layout("input2", input2_layout),
        input_layout("input3", input3_layout),
        permute("permute", input_info("input1"), { 0, 2, 1 }),
        eltwise("add", input_info("permute"), input_info("input2"), eltwise_mode::sum),
        eltwise("multiply", input_info("add"), input_info("input3"), eltwise_mode::prod),
        permute("result", input_info("multiply"), {0, 1, 2})
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto program = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(program, nullptr);
}

TEST(permute_f_y_axes_fallback, b_fs_yx_fsv16) {
    constexpr size_t array_size = 128;

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input_layout_static = layout{ov::PartialShape{1, 8, 16, 1}, data_types::f32, format::bfyx};
    auto input = engine.allocate_memory(input_layout_static);

    std::vector<float> input_data;
    input_data.reserve(array_size);
    for (size_t i = 0; i < array_size; ++i)
        input_data.push_back(static_cast<float>(i));

    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 8, 16, 1, 1 } });

    std::vector<float> weights_data;
    weights_data.reserve(array_size);
    for (size_t i = 0; i < array_size; ++i)
      weights_data.push_back(static_cast<float>(1.0));

    set_values(weights, weights_data);
    set_values(input, input_data);

    auto impl_desc_onednn = ov::intel_gpu::ImplementationDesc{format::b_fs_yx_fsv16, "", impl_types::onednn};
    auto impl_forcing_map = ov::intel_gpu::ImplForcingMap{{"conv", impl_desc_onednn}};

    topology topology;
    topology.add(input_layout("input", input_layout_static));
    topology.add(permute("permute", input_info("input"), { 0, 2, 1, 3 }));
    topology.add(data("weights", weights));
    topology.add(convolution("conv", input_info("permute"), "weights", "", 1, {1,1}, {1,1}, {0,0}, {0,0}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(impl_forcing_map));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "conv");

    auto output = outputs.begin()->second.get_memory();

    float answers[] = {
        120.f, 120.f, 120.f, 120.f, 120.f, 120.f, 120.f, 120.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        376.f, 376.f, 376.f, 376.f, 376.f, 376.f, 376.f, 376.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        632.f, 632.f, 632.f, 632.f, 632.f, 632.f, 632.f, 632.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        888.f, 888.f, 888.f, 888.f, 888.f, 888.f, 888.f, 888.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        1144.f, 1144.f, 1144.f, 1144.f, 1144.f, 1144.f, 1144.f, 1144.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        1400.f, 1400.f, 1400.f, 1400.f, 1400.f, 1400.f, 1400.f, 1400.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        1656.f, 1656.f, 1656.f, 1656.f, 1656.f, 1656.f, 1656.f, 1656.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        1912.f, 1912.f, 1912.f, 1912.f, 1912.f, 1912.f, 1912.f, 1912.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < array_size; i++) {
        ASSERT_FLOAT_EQ(answers[i], output_ptr[i]);
    }
}

class permute_bfzyx_to_bfyxz: public TiledPermuteTest {};

INSTANTIATE_TEST_SUITE_P(, permute_bfzyx_to_bfyxz,
    ::testing::ValuesIn(std::vector<TiledPermuteParam> {
            {{1, 3, 85, 20, 20}, format::bfzyx},
            {{1, 3, 85, 40, 40}, format::bfzyx},
            {{1, 3, 85, 80, 80}, format::bfzyx},
            {{1, 192, 2, 64, 33}, format::bfzyx}
        }));

TEST_P(permute_bfzyx_to_bfyxz, combined) {
    auto p = GetParam();
    run_test<cldnn::data_types::f32>(p.sizes, p.format_fsv, "permute_bfzyx_to_bfyxz", {0, 1, 3, 4, 2});
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(permute_tile_fsv_4d, f16_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::f16>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

TEST_P(permute_tile_fsv_4d, f32_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::f32>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

TEST_P(permute_tile_fsv_4d, i8_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::i8>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

TEST_P(permute_tile_fsv_4d, i32_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::i32>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

TEST_P(permute_tile_fsv_4d, i64_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::i64>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

TEST_P(permute_tile_fsv_5d, f16_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::f16>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

TEST_P(permute_tile_fsv_5d, f32_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::f32>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

TEST_P(permute_tile_fsv_5d, i8_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::i8>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

TEST_P(permute_tile_fsv_5d, i32_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::i32>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

TEST_P(permute_bfzyx_to_bfyxz, combined_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::f32>(p.sizes, p.format_fsv, "permute_bfzyx_to_bfyxz", {0, 1, 3, 4, 2}, true);
}
#endif
TEST_P(permute_tile_fsv_5d, i64_cached) {
    auto p = GetParam();
    run_test<cldnn::data_types::i64>(p.sizes, p.format_fsv, "permute_tile_8x8_4x4_fsv", {}, true);
}

class permute_f_y_axes_tile: public TiledPermuteTest {};

INSTANTIATE_TEST_SUITE_P(smoke_permute_f_y_axes_tile,
                         permute_f_y_axes_tile,
                         ::testing::ValuesIn(std::vector<TiledPermuteParam>{
                             {{1, 4, 8, 1}, format::bfyx},                // permute_f_y_axes
                             {{1, 64, 32, 1}, format::bfyx},              // permute_f_y_axes
                             {{1, 32, 256, 512}, format::b_fs_yx_fsv32},  // THREE_DIM_TRANSPOSE
                             {{1, 32, 256, 512}, format::bfyx},           // PERMUTE_SIMPLE_MEM_COPY
                             {{1, 256, 256, 1}, format::b_fs_yx_fsv32},   // permute_f_y_axes
                             {{1, 32, 16, 4}, format::b_fs_yx_fsv16},     // THREE_DIM_TRANSPOSE
                             //4 batch version
                             {{4, 4, 4, 1}, format::bfyx},                // permute_f_y_axes
                             //32 batch version
                             {{32, 4, 8, 1}, format::bfyx},                // permute_f_y_axes
                             {{32, 64, 32, 1}, format::bfyx},              // permute_f_y_axes
                             {{32, 128, 196, 1}, format::bfyx},              // permute_f_y_axes
                             {{32, 196, 4, 16}, format::bfyx},              // permute_f_y_axes
                             {{32, 196, 4, 32}, format::bfyx},              // permute_f_y_axes
                             {{32, 4, 196, 32}, format::bfyx},              // permute_f_y_axes
                             {{32, 196, 128, 1}, format::bfyx},              // permute_f_y_axes
                             {{32, 196, 8, 16}, format::bfyx},              // permute_f_y_axes
                             {{16, 32, 128, 512}, format::b_fs_yx_fsv32},  // THREE_DIM_TRANSPOSE
                             {{16, 32, 128, 512}, format::bfyx},           // PERMUTE_SIMPLE_MEM_COPY
                             {{32, 256, 256, 1}, format::b_fs_yx_fsv32},   // permute_f_y_axes
                             {{32, 32, 16, 4}, format::b_fs_yx_fsv16},     // THREE_DIM_TRANSPOSE
                             {{32, 16, 16, 16}, format::bfyx}, 
                             {{32, 16, 8, 16}, format::bfyx}, 
                             {{32, 16, 16, 64}, format::bfyx},
                             {{32, 16, 8, 32}, format::bfyx}, 
                             {{32, 8, 16, 32}, format::bfyx},
                             {{32, 196, 8, 64}, format::bfyx},           // permute_f_y_axes
                             {{1, 512, 30, 1}, format::bfyx},            // fix for JTIMES=0
                             {{1, 2, 512, 10}, format::bfyx},            //case trying to set vec size(4) bigger than x divisor(2) in case of f16
                         }),
                         TiledPermuteTest::PrintToStringParamName);

TEST_P(permute_f_y_axes_tile, combined) {
    auto p = GetParam();
    run_test<cldnn::data_types::f32>(p.sizes, p.format_fsv, "permute_f_y_axes", {0, 2, 1, 3});
    run_test<cldnn::data_types::f16>(p.sizes, p.format_fsv, "permute_f_y_axes", {0, 2, 1, 3});
    run_test<cldnn::data_types::u8>(p.sizes, p.format_fsv, "permute_f_y_axes", {0, 2, 1, 3});
    run_test<cldnn::data_types::i8>(p.sizes, p.format_fsv, "permute_f_y_axes", {0, 2, 1, 3});
    run_test<cldnn::data_types::i32>(p.sizes, p.format_fsv, "permute_f_y_axes", {0, 2, 1, 3});
    run_test<cldnn::data_types::i64>(p.sizes, p.format_fsv, "permute_f_y_axes", {0, 2, 1, 3});
}

struct TiledPerformancePermuteTest : TiledPermuteTest
{
    static double get_exectime(const std::map<cldnn::primitive_id, cldnn::network_output>& outputs,
                                const std::string& primitive_id)
    {
        using namespace std::chrono;
        std::shared_ptr<event> e = outputs.at(primitive_id).get_event();
        e->wait(); // should ensure execution completion, if not segfault will occur
        double avg_time = 0.0;
        auto intervals = e->get_profiling_info();
        for (const auto& q : intervals)
        {
            if (q.stage != instrumentation::profiling_stage::executing) {
                continue;
            }
            avg_time = duration_cast<duration<double, microseconds::period>>(q.value->value()).count();
            break;
        }
        return avg_time;
    }

    static void print_all_perf(std::map<primitive_id, network_output> outputs)
    {
        std::cout << "Print last run time" << std::endl;
        using namespace std::chrono;
        for( const auto &n : outputs ) {
            std::shared_ptr<event> e = n.second.get_event();
            auto intervals = e->get_profiling_info();
            double time = 0.0;
            for (const auto& q : intervals)
            {
                if (q.stage == instrumentation::profiling_stage::executing) {
                    continue;
                }
                time = duration_cast<duration<double, microseconds::period>>(q.value->value()).count();
                break;
            }
            std::cout << n.first << ":" << time << std::endl;
        }
        std::cout << std::endl;
    }
    
    template<data_types Data_Type>
    void execute_perf_test(const std::vector<cldnn::tensor::value_type>& sizes, cldnn::format format_fsv,
                            const std::string & kernel_name, std::vector<uint16_t> permute_order)
    {
        auto& engine = get_test_engine();
        // convert half_t to FLOAT16
        using type = typename ov::element_type_traits<Data_Type>::value_type;

        std::vector<cldnn::tensor::value_type> internal_sizes(sizes);
        std::swap(internal_sizes.at(2), internal_sizes.back());
        cldnn::tensor tensor(internal_sizes);

        cldnn::format format = sizes.size() == 4 ? cldnn::format::bfyx : cldnn::format::bfzyx;

        std::vector<uint16_t> order = {0};
        if (permute_order.empty()) {
            for (uint16_t i = 1; i < (sizes.size() - 1); ++i) {
                order.push_back(i+1);
            }
            order.push_back(1);
        } else {
            std::swap(order, permute_order);
        }

        auto input_ref = engine.allocate_memory({Data_Type, format, tensor});
        set_random_values<type>(input_ref);
        topology topology_ref = topology(
            input_layout("input", input_ref->get_layout()),
            reorder("reorder", input_info("input"), {Data_Type, format_fsv, tensor}),
            permute("output", input_info("reorder"), order)
        );
        // run with permute_ref
        ExecutionConfig config_ref(ov::enable_profiling(true));
        ov::intel_gpu::ImplementationDesc permute_ref = {format_fsv, "permute_ref"};
        config_ref.set_property(
            ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"output", permute_ref}}));
        cldnn::network network_ref(engine, topology_ref, config_ref);
        network_ref.set_input_data("input", input_ref);

        // run with optimized kernel, e.g. permute_tile_8x8_4x4_fsv16
        auto input_opt = engine.allocate_memory({Data_Type, format, tensor});
        set_random_values<type>(input_opt);
        topology topology_opt = topology(
            input_layout("input", input_opt->get_layout()),
            reorder("reorder", input_info("input"), {Data_Type, format_fsv, tensor}),
            permute("output", input_info("reorder"), order)
        );
        ExecutionConfig config_tile(ov::enable_profiling(true));
        ov::intel_gpu::ImplementationDesc permute_tile_opt = {format_fsv, kernel_name};
        config_tile.set_property(
            ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"output", permute_tile_opt}}));
        cldnn::network network_tile(engine, topology_opt, config_tile);
        network_tile.set_input_data("input", input_opt);

        // first execution of opt
        std::map<primitive_id, network_output> output_permute_ref;
        std::map<primitive_id, network_output> output_permute_opt;

        for (int i = 0; i < 10; ++i) {
            output_permute_ref = network_ref.execute();
            output_permute_opt = network_tile.execute();
        }

        auto r = 100;
        double exectime_ref = 0.f;
        double exectime_opt = 0.f;
        for (int i = 0; i < r; ++i) {
            output_permute_opt = network_tile.execute();
            auto t_opt = get_exectime(output_permute_opt, "output");
            exectime_opt += t_opt;

            output_permute_ref = network_ref.execute();
            auto t_ref = get_exectime(output_permute_ref, "output");
            exectime_ref += t_ref;
        }
        exectime_ref /= r;
        exectime_opt /= r;
        std::cout << std::endl;
        auto output_layout_ref = network_ref.get_program()->get_node("output").get_output_layout();
        auto output_layout_opt = network_tile.get_program()->get_node("output").get_output_layout();
        std::string frm_str = cldnn::format(format).to_string();
        std::string input_type = ov::element::Type(Data_Type).get_type_name();

        std::cout << "Exectued time " << " " << "permute_ref" << " " << " input(" << tensor.to_string()
                  << ") output(" <<  output_layout_ref.to_string() << ") "
                  << frm_str << " " << input_type << " " << exectime_ref << std::endl;
        std::cout << "Exectued time " << " " << kernel_name << " " << " input(" << tensor.to_string()
                  << ") output(" <<  output_layout_opt.to_string() << ") "
                  << frm_str << " " << input_type << " " << exectime_opt << std::endl;

    }
    
};


// No need to run performance tests on CI
TEST_P(TiledPerformancePermuteTest, DISABLED_f32) {
    auto p = GetParam();
    execute_perf_test<cldnn::data_types::f32>(p.sizes, p.format_fsv, "permute_f_y_axes", {0, 2, 1, 3});
}


INSTANTIATE_TEST_SUITE_P(, TiledPerformancePermuteTest,
    ::testing::ValuesIn(std::vector<TiledPermuteParam> {
        // b_fs_zy_fsv16
        // normal cases
        {{1, 512, 16384, 1}, format::bfyx},
        {{1, 512, 16384, 1}, format::b_fs_yx_fsv16},
        {{1, 256, 128, 256}, format::bfyx},
        {{1, 256, 256, 128}, format::b_fs_yx_fsv16},
    }));
