// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/convolution.hpp"
#include "api/eltwise.hpp"
#include "api/reorder.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/data.hpp>
#include <api/depth_to_space.hpp>

#include <api_extension/fused_conv_eltwise.hpp>

#include <cassert>
#include <cmath>
#include <gmock/gmock.h>
#include <limits>
#include <iostream>

using namespace cldnn;
using namespace tests;
using namespace testing;

TEST(fused_conv_eltwise, yolov5_fused_eltw_pattern_with_ref_b_fs_yx_fsv16_f32)
{
    //Test pattern of Conv_208/WithoutBiases in yolov5s-gpu-rg.xml
#ifdef DUMP_CL_KERNEL_BUILD_LOG
    engine_configuration configuration =
        engine_configuration(
            false,          // profiling
            false,          // decorate_kernel_names
            false,          // dump_custom_program
            "",             // options
            "",             // single_kernel
            true,           // primitives_parallelisation
            "",             // engine_log
            "C:\\Users\\ahnyoung\\sources\\error_dump"             // sources_dumps_dir
            );
    cldnn::engine engine(configuration);
#else
    const auto& engine = get_test_engine();
#endif

    auto input = memory::allocate(engine, { data_types::f32, format::b_fs_yx_fsv16, { 1, 128, 40, 40 } /*memory order*/ }); //memory order
    auto weights = memory::allocate(engine, { data_types::f32, format::os_is_yx_isv16_osv16, { 128, 128, 1, 1 } });
    auto sum_input = memory::allocate(engine, { data_types::f32, format::b_fs_yx_fsv16, { 1, 1, 1, 1 } });

    const int32_t total_size = 128 * 40 * 40;
    std::vector<float> inputVec(total_size);
    for (int i = 0; i < total_size; i++)
    {
        inputVec[i] = float(i+1);
    }

    set_values(input, inputVec);

    topology topology_act(
        input_layout("input", input.get_layout()),
        data("sum_input", sum_input),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise1", "conv", "sum_input", eltwise_mode::sum),
        eltwise("eltwise2", "eltwise1", "conv", eltwise_mode::prod),
        reorder("out_reorder", "eltwise2", format::b_fs_yx_fsv16, data_types::f32));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : fused_conv_eltwise_trial_new_pattern01_b_fs_yx_fsv16_fp32" << std::endl;
    std::cout << "input : f32, b_fs_yx_fsv16, {1, 128, 40, 40}" << std::endl;
    std::cout << "weights : f32, os_is_yx_osv16_isv16 {128, 128, 1, 1}" << std::endl;
    std::cout << "sum_input : f32, b_fs_yx_fsv16 {1, 1, 1, 1}" << std::endl;

    std::cout << "topology topology(" << std::endl;
    std::cout << "    input_layout(\"input\", input.get_layout())," << std::endl;
    std::cout << "    data(\"sum_input\", sum_input)," << std::endl;
    std::cout << "    data(\"weights\", weights)," << std::endl;
    std::cout << "    convolution(\"conv\", \"input\", { \"weights\" })," << std::endl;
    std::cout << "    eltwise(\"eltwise1\", \"conv\", \"sum_input\", eltwise_mode::sum)," << std::endl;
    std::cout << "    eltwise(\"eltwise2\", \"eltwise1\", \"conv\", eltwise_mode::prod)," << std::endl;
    std::cout << "    reorder(\"out_reorder\", \"eltwise\", format::bfyx, data_types::f32));" << std::endl << std::endl;


    build_options opt_act;
#ifdef BUILD_OPTION_GRAPH_COMPILE
    opt_act.set_option(build_option::graph_dumps_dir("/home/yblee/conv_fusing"));
#endif
    opt_act.set_option(build_option::optimize_data(true));
    network network_act(engine, topology_act, opt_act);
    network_act.set_input_data("input", input);

    auto outputs_act = network_act.execute();
    EXPECT_EQ(outputs_act.size(), size_t(1));
    EXPECT_EQ(outputs_act.begin()->first, "out_reorder");

    auto output_act = outputs_act.begin()->second.get_memory();
    auto&& out_layout_act = output_act.get_layout();
    auto out_act_ptr = output_act.pointer<uint8_t>();

    EXPECT_EQ(out_layout_act.format, format::b_fs_yx_fsv16);
    EXPECT_EQ(out_layout_act.size.batch[0], 1);
    EXPECT_EQ(out_layout_act.size.feature[0], 128);
    EXPECT_EQ(out_layout_act.size.spatial[0], 40);
    EXPECT_EQ(out_layout_act.size.spatial[1], 40);

    topology topology_ref(
        input_layout("input", input.get_layout()),
        data("sum_input", sum_input),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise1", "conv", "sum_input", eltwise_mode::sum),
        eltwise("eltwise2", "eltwise1", "conv", eltwise_mode::prod),
        reorder("out_reorder", "eltwise2", format::b_fs_yx_fsv16, data_types::f32));

    build_options opt_ref;
    // opt_ref.set_option(build_option::graph_dumps_dir("/home/yblee/conv_fusing"));
    opt_ref.set_option(build_option::optimize_data(false));
    network network_ref(engine, topology_ref, opt_ref);
    network_ref.set_input_data("input", input);

    auto outputs_ref = network_ref.execute();
    EXPECT_EQ(outputs_ref.size(), size_t(1));
    EXPECT_EQ(outputs_ref.begin()->first, "out_reorder");

    auto output_ref = outputs_ref.begin()->second.get_memory();
    auto&& out_layout_ref = output_ref.get_layout();
    auto out_ref_ptr = output_ref.pointer<uint8_t>();

    EXPECT_EQ(out_layout_ref.format, format::b_fs_yx_fsv16);
    EXPECT_EQ(out_layout_ref.size.batch[0], 1);
    EXPECT_EQ(out_layout_ref.size.feature[0], 128);
    EXPECT_EQ(out_layout_ref.size.spatial[0], 40);
    EXPECT_EQ(out_layout_ref.size.spatial[1], 40);

    for (int i = 0;i < 3 * 256 * 4;i++) {
        EXPECT_EQ(out_act_ptr[i], out_ref_ptr[i]);
    }
}


TEST(fused_conv_eltwise, yolov5_fused_eltw_pattern_b_fs_yx_fsv16_f32)
{
    //Test pattern of Conv_208/WithoutBiases in yolov5s-gpu-rg.xml
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::b_fs_yx_fsv16, { 1, 128, 40, 40 } /*memory order*/ }); //memory order
    auto weights = memory::allocate(engine, { data_types::f32, format::os_is_yx_osv16_isv16, { 128, 128, 1, 1 } });
    auto sum_input = memory::allocate(engine, { data_types::f32, format::b_fs_yx_fsv16, { 1, 1, 1, 1 } });

    const int32_t total_size = 128 * 40 * 40;
    std::vector<float> inputVec(total_size);
    for (int i = 0; i < total_size; i++)
    {
        inputVec[i] = float(i+1);
    }

    set_values(input, inputVec);

    topology topology(
        input_layout("input", input.get_layout()),
        data("sum_input", sum_input),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise1", "conv", "sum_input", eltwise_mode::sum),
        eltwise("eltwise2", "eltwise1", "conv", eltwise_mode::prod),
        reorder("out_reorder", "eltwise2", format::b_fs_yx_fsv16, data_types::f32));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : fused_conv_eltwise_trial_new_pattern01_b_fs_yx_fsv16_fp32" << std::endl;
    std::cout << "input : f32, b_fs_yx_fsv16, {1, 128, 40, 40}" << std::endl;
    std::cout << "weights : f32, os_is_yx_osv16_isv16 {128, 128, 1, 1}" << std::endl;
    std::cout << "sum_input : f32, b_fs_yx_fsv16 {1, 1, 1, 1}" << std::endl;

    std::cout << "topology topology(" << std::endl;
    std::cout << "    input_layout(\"input\", input.get_layout())," << std::endl;
    std::cout << "    data(\"sum_input\", sum_input)," << std::endl;
    std::cout << "    data(\"weights\", weights)," << std::endl;
    std::cout << "    convolution(\"conv\", \"input\", { \"weights\" })," << std::endl;
    std::cout << "    eltwise(\"eltwise1\", \"conv\", \"sum_input\", eltwise_mode::sum)," << std::endl;
    std::cout << "    eltwise(\"eltwise2\", \"eltwise1\", \"conv\", eltwise_mode::prod)," << std::endl;
    std::cout << "    reorder(\"out_reorder\", \"eltwise\", format::bfyx, data_types::f32));" << std::endl << std::endl;

    build_options opt;
    // dump graph
    // opt.set_option(build_option::graph_dumps_dir("/home/yblee/conv_fusing"));
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::b_fs_yx_fsv16);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 128);
    EXPECT_EQ(out_layout.size.spatial[0], 40);
    EXPECT_EQ(out_layout.size.spatial[1], 40);
}

TEST(fused_conv_eltwise, origin_yxfb_f16)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::yxfb, { 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::f16, format::yxfb, { 1, 1, 1, 1 } });

    set_values(input, {
        FLOAT16(1.0f),  FLOAT16(2.0f), FLOAT16(-15.f),  FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(-15.f), FLOAT16(5.0f),  FLOAT16(6.0f), FLOAT16(-15.f), FLOAT16(7.0f),
        FLOAT16(-15.f), FLOAT16(0.0f),  FLOAT16(0.0f), FLOAT16(-15.f), FLOAT16(0.5f), FLOAT16(-0.5f), FLOAT16(-15.f), FLOAT16(8.0f),  FLOAT16(1.5f),  FLOAT16(5.2f)
    });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out_reorder", "eltwise", format::yxfb, data_types::f16));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : fused_conv_eltwise_origin_yxfb_f16" << std::endl;
    std::cout << "input : f16, yxfb, {4,5,1,1}" << std::endl;
    std::cout << "weights : f16, yxfb {1,1,1,1}" << std::endl;

    std::cout << "topology topology(\n\tinput_layout(\"input\", input.get_layout()),\n";
    std::cout << "\tdata(\"weights\", weights),\n";
    std::cout << "\tconvolution(\"conv\", \"input\", { \"weights\" }),\n";
    std::cout << "\teltwise(\"eltwise\", \"input\", \"conv\", eltwise_mode::sum),\n";
    std::cout << "\treorder(\"out_reorder\", \"eltwise\", format::yxfb, data_types::f16));" << std::endl << std::endl;

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::yxfb);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 5);
    EXPECT_EQ(out_layout.size.spatial[1], 4);
}

TEST(fused_conv_eltwise, origin_f32_b_fs_zyx_fsv16_os_is_zyx_osv16_isv16)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::b_fs_zyx_fsv16, { 1, 128, 40, 40, 1 } /*memory order*/ }); //memory order
    auto weights = memory::allocate(engine, { data_types::f32, format::os_is_zyx_isv16_osv16, { 128, 128, 1, 1, 1 } });

    const int32_t total_size = 128 * 40 * 40;
    std::vector<float> inputVec(total_size);
    for (int i = 0; i < total_size; i++)
    {
        inputVec[i] = float(i+1);
    }
    set_values(input, inputVec);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out_reorder", "eltwise", format::b_fs_zyx_fsv16, data_types::f32));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : fused_conv_eltwise_origin_f32_b_fs_zyx_fsv16_os_is_zyx_osv16_isv16" << std::endl;
    std::cout << "input : f32, b_fs_zyx_fsv16, { 1, 128, 1, 40, 40 }" << std::endl;
    std::cout << "weights : f32, os_is_zyx_isv16_osv16, { 128, 128, 1, 1, 1 }" << std::endl;

    std::cout << "topology topology(\n\tinput_layout(\"input\", input.get_layout()),\n";
    std::cout << "\tdata(\"weights\", weights),\n";
    std::cout << "\tconvolution(\"conv\", \"input\", { \"weights\" }),\n";
    std::cout << "\teltwise(\"eltwise\", \"input\", \"conv\", eltwise_mode::sum),\n";
    std::cout << "\treorder(\"out_reorder\", \"eltwise\", format::b_fs_zyx_fsv16, data_types::f32));" << std::endl << std::endl;

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::b_fs_zyx_fsv16);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 128);
    EXPECT_EQ(out_layout.size.spatial[0], 40);
    EXPECT_EQ(out_layout.size.spatial[1], 40);
    EXPECT_EQ(out_layout.size.spatial[2], 1);
}

TEST(fused_conv_eltwise, origin_f16_b_fs_zyx_fsv16_os_is_zyx_osv16_isv16)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::b_fs_zyx_fsv16, { 1, 128, 40, 40, 1 } /*memory order*/ }); //memory order
    auto weights = memory::allocate(engine, { data_types::f16, format::os_is_zyx_isv16_osv16, { 128, 128, 1, 1, 1 } });

    const int32_t total_size = 128 * 40 * 40;
    std::vector<FLOAT16> inputVec(total_size);
    for (int i = 0; i < total_size; i++)
    {
        inputVec[i] = FLOAT16(i+1);
    }
    set_values(input, inputVec);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out_reorder", "eltwise", format::b_fs_zyx_fsv16, data_types::f16));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : fused_conv_eltwise_origin_f16_b_fs_zyx_fsv16_os_is_zyx_osv16_isv16" << std::endl;
    std::cout << "input : f16, b_fs_zyx_fsv16, { 1, 128, 1, 40, 40 }" << std::endl;
    std::cout << "weights : f16, os_is_zyx_isv16_osv16, { 128, 128, 1, 1, 1 }" << std::endl;

    std::cout << "topology topology(\n\tinput_layout(\"input\", input.get_layout()),\n";
    std::cout << "\tdata(\"weights\", weights),\n";
    std::cout << "\tconvolution(\"conv\", \"input\", { \"weights\" }),\n";
    std::cout << "\teltwise(\"eltwise\", \"input\", \"conv\", eltwise_mode::sum),\n";
    std::cout << "\treorder(\"out_reorder\", \"eltwise\", format::b_fs_zyx_fsv16, data_types::f16));" << std::endl << std::endl;

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::b_fs_zyx_fsv16);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 128);
    EXPECT_EQ(out_layout.size.spatial[0], 40);
    EXPECT_EQ(out_layout.size.spatial[1], 40);
    EXPECT_EQ(out_layout.size.spatial[2], 1);
}

TEST(fused_conv_eltwise, origin_f16_b_fs_yx_fsv16_os_is_yx_osv16_isv16)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::b_fs_yx_fsv16, { 1, 128, 40, 40 } /*memory order*/ }); //memory order
    auto weights = memory::allocate(engine, { data_types::f16, format::os_is_yx_osv16_isv16, { 128, 128, 1, 1 } });

    const int32_t total_size = 128 * 40 * 40;
    std::vector<FLOAT16> inputVec(total_size);
    for (int i = 0; i < total_size; i++)
    {
        inputVec[i] = FLOAT16(i+1);
    }
    set_values(input, inputVec);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out_reorder", "eltwise", format::b_fs_yx_fsv16, data_types::f16));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : origin_f16_b_fs_yx_fsv16_os_is_yx_osv16_isv16" << std::endl;
    std::cout << "input : f16, b_fs_yx_fsv16, { 1, 128, 40, 40 }" << std::endl;
    std::cout << "weights : f16, os_is_yx_osv16_isv16, { 128, 128, 1, 1 }" << std::endl;

    std::cout << "topology topology(\n\tinput_layout(\"input\", input.get_layout()),\n";
    std::cout << "\tdata(\"weights\", weights),\n";
    std::cout << "\tconvolution(\"conv\", \"input\", { \"weights\" }),\n";
    std::cout << "\teltwise(\"eltwise\", \"input\", \"conv\", eltwise_mode::sum),\n";
    std::cout << "\treorder(\"out_reorder\", \"eltwise\", format::b_fs_yx_fsv16, data_types::f16));" << std::endl << std::endl;

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::b_fs_yx_fsv16);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 128);
    EXPECT_EQ(out_layout.size.spatial[0], 40);
    EXPECT_EQ(out_layout.size.spatial[1], 40);
}

TEST(fused_conv_eltwise, origin_f32_b_fs_yx_fsv16_os_is_yx_osv16_isv16)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::b_fs_yx_fsv16, { 1, 128, 40, 40 } /*memory order*/ }); //memory order
    auto weights = memory::allocate(engine, { data_types::f32, format::os_is_yx_osv16_isv16, { 128, 128, 1, 1 } });

    const int32_t total_size = 128 * 40 * 40;
    std::vector<float> inputVec(total_size);
    for (int i = 0; i < total_size; i++)
    {
        inputVec[i] = float(i+1);
    }
    set_values(input, inputVec);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out_reorder", "eltwise", format::b_fs_yx_fsv16, data_types::f32));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : origin_f32_b_fs_yx_fsv16_os_is_yx_osv16_isv16" << std::endl;
    std::cout << "input : f32, b_fs_yx_fsv16, { 1, 128, 40, 40 }" << std::endl;
    std::cout << "weights : f32, os_is_yx_osv16_isv16, { 128, 128, 1, 1 }" << std::endl;

    std::cout << "topology topology(\n\tinput_layout(\"input\", input.get_layout()),\n";
    std::cout << "\tdata(\"weights\", weights),\n";
    std::cout << "\tconvolution(\"conv\", \"input\", { \"weights\" }),\n";
    std::cout << "\teltwise(\"eltwise\", \"input\", \"conv\", eltwise_mode::sum),\n";
    std::cout << "\treorder(\"out_reorder\", \"eltwise\", format::b_fs_yx_fsv16, data_types::f32));" << std::endl << std::endl;

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::b_fs_yx_fsv16);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 128);
    EXPECT_EQ(out_layout.size.spatial[0], 40);
    EXPECT_EQ(out_layout.size.spatial[1], 40);
}

TEST(fused_conv_eltwise, origin_bfyx_u8)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::bfyx, { 1, 1, 4, 5 } });
    auto weights = memory::allocate(engine, { data_types::u8, format::bfyx, { 1, 1, 1, 1 } });

    std::vector<int8_t> inputVec = {
        1,  2, 15, 3,  4, 15, 5,  6, 15, 7,
        15, 0, 0,  15, 1, 1,  15, 8, 2,  5
    };
    set_values(input, inputVec);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out_reorder", "eltwise", format::bfyx, data_types::u8));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : fused_conv_eltwise_origin_bfyx_u8" << std::endl;
    std::cout << "input : u8, bfyx, {1,1,5,4}" << std::endl;
    std::cout << "weights : u8, bfyx {1,1,1,1}" << std::endl;

    std::cout << "topology topology(\n\tinput_layout(\"input\", input.get_layout()),\n";
    std::cout << "\tdata(\"weights\", weights),\n";
    std::cout << "\tconvolution(\"conv\", \"input\", { \"weights\" }),\n";
    std::cout << "\teltwise(\"eltwise\", \"input\", \"conv\", eltwise_mode::sum),\n";
    std::cout << "\treorder(\"out_reorder\", \"eltwise\", format::bfyx, data_types::u8));" << std::endl << std::endl;

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

TEST(fused_conv_eltwise, origin_bfyx_i8)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::i8, format::bfyx, { 1, 1, 4, 5 } });
    auto weights = memory::allocate(engine, { data_types::i8, format::bfyx, { 1, 1, 1, 1 } });

    std::vector<int8_t> inputVec = {
        1,  2, 15, 3,  4, 15, 5,  6, 15, 7,
        15, 0, 0,  15, 1, 1,  15, 8, 2,  5
    };
    set_values(input, inputVec);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out_reorder", "eltwise", format::bfyx, data_types::i8));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : fused_conv_eltwise_origin_bfyx_i8" << std::endl;
    std::cout << "input : i8, bfyx, {1,1,5,4}" << std::endl;
    std::cout << "weights : i8, bfyx {1,1,1,1}" << std::endl;

    std::cout << "topology topology(\n\tinput_layout(\"input\", input.get_layout()),\n";
    std::cout << "\tdata(\"weights\", weights),\n";
    std::cout << "\tconvolution(\"conv\", \"input\", { \"weights\" }),\n";
    std::cout << "\teltwise(\"eltwise\", \"input\", \"conv\", eltwise_mode::sum),\n";
    std::cout << "\treorder(\"out_reorder\", \"eltwise\", format::bfyx, data_types::i8));" << std::endl << std::endl;

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

TEST(fused_conv_eltwise, origin_bfyx_f16)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 1, 4, 5 } });
    auto weights = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        FLOAT16(1.0f),  FLOAT16(2.0f), FLOAT16(-15.f),  FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(-15.f), FLOAT16(5.0f),  FLOAT16(6.0f), FLOAT16(-15.f), FLOAT16(7.0f),
        FLOAT16(-15.f), FLOAT16(0.0f),  FLOAT16(0.0f), FLOAT16(-15.f), FLOAT16(0.5f), FLOAT16(-0.5f), FLOAT16(-15.f), FLOAT16(8.0f),  FLOAT16(1.5f),  FLOAT16(5.2f)
    });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out_reorder", "eltwise", format::bfyx, data_types::f16));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : fused_conv_eltwise_origin_bfyx_f16" << std::endl;
    std::cout << "input : f16, bfyx, {1,1,5,4}" << std::endl;
    std::cout << "weights : f16, bfyx {1,1,1,1}" << std::endl;

    std::cout << "topology topology(\n\tinput_layout(\"input\", input.get_layout()),\n";
    std::cout << "\tdata(\"weights\", weights),\n";
    std::cout << "\tconvolution(\"conv\", \"input\", { \"weights\" }),\n";
    std::cout << "\teltwise(\"eltwise\", \"input\", \"conv\", eltwise_mode::sum),\n";
    std::cout << "\treorder(\"out_reorder\", \"eltwise\", format::bfyx, data_types::f16));" << std::endl << std::endl;

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

TEST(fused_conv_eltwise, origin_bfyx_f32)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out_reorder", "eltwise", format::bfyx, data_types::f32));

    std::cout << "*************************************************************" << std::endl;
    std::cout << "Test : fused_conv_eltwise_origin_bfyx_f32" << std::endl;
    std::cout << "input : f32, bfyx, {1,1,5,4}" << std::endl;
    std::cout << "weights : f32, bfyx {1,1,1,1}" << std::endl;

    std::cout << "topology topology(\n\tinput_layout(\"input\", input.get_layout()),\n";
    std::cout << "\tdata(\"weights\", weights),\n";
    std::cout << "\tconvolution(\"conv\", \"input\", { \"weights\" }),\n";
    std::cout << "\teltwise(\"eltwise\", \"input\", \"conv\", eltwise_mode::sum),\n";
    std::cout << "\treorder(\"out_reorder\", \"eltwise\", format::bfyx, data_types::f32));" << std::endl << std::endl;

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out_reorder");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

TEST(fused_conv_eltwise, basic_0)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out", "eltwise", format::bfyx, data_types::f32));

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

TEST(fused_conv_eltwise, basic_image2d)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 4, 128, 2 } });
    auto input2 = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 3, 256, 4 } });
    auto weights = memory::allocate(engine, { data_types::f16, format::bfyx, { 12, 4, 1, 1 } });

    auto input_data1 = generate_random_4d<FLOAT16>(1, 4, 2, 128, -1, 1);
    auto input_data1_bfyx = flatten_4d(format::bfyx, input_data1);
    set_values(input, input_data1_bfyx);

    auto input_data2 = generate_random_4d<FLOAT16>(1, 3, 4, 256, -1, 1);
    auto input_data2_bfyx = flatten_4d(format::bfyx, input_data2);
    set_values(input2, input_data2_bfyx);

    auto weights_data= generate_random_4d<FLOAT16>(12, 4, 1, 1, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    set_values(weights, weights_data_bfyx);

    topology topology_act(
        input_layout("input", input.get_layout()),
        input_layout("input2", input2.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        depth_to_space("depth_to_space", "conv", 2, depth_to_space_mode::blocks_first),
        eltwise("eltwise", "input2", "depth_to_space", eltwise_mode::sum)
    );

    build_options opt_act;
    opt_act.set_option(build_option::optimize_data(true));
    network network_act(engine, topology_act, opt_act);
    network_act.set_input_data("input", input);
    network_act.set_input_data("input2", input2);

    auto outputs_act = network_act.execute();
    EXPECT_EQ(outputs_act.size(), size_t(1));
    EXPECT_EQ(outputs_act.begin()->first, "eltwise");

    auto output_act = outputs_act.begin()->second.get_memory();
    auto out_act_ptr = output_act.pointer<uint8_t>();

    topology topology_ref(
        input_layout("input", input.get_layout()),
        input_layout("input2", input2.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        depth_to_space("depth_to_space", "conv", 2, depth_to_space_mode::blocks_first),
        eltwise("eltwise", "input2", "depth_to_space", eltwise_mode::sum),
        reorder("out", "eltwise", format::image_2d_rgba, data_types::u8));

    build_options opt_ref;
    opt_ref.set_option(build_option::optimize_data(false));
    network network_ref(engine, topology_ref, opt_ref);
    network_ref.set_input_data("input", input);
    network_ref.set_input_data("input2", input2);

    auto outputs_ref = network_ref.execute();
    EXPECT_EQ(outputs_ref.size(), size_t(1));
    EXPECT_EQ(outputs_ref.begin()->first, "out");

    auto output_ref = outputs_ref.begin()->second.get_memory();
    auto out_ref_ptr = output_ref.pointer<uint8_t>();

    for (int i = 0;i < 3 * 256 * 4;i++) {
        EXPECT_EQ(out_act_ptr[i], out_ref_ptr[i]);
    }
}

TEST(fused_conv_eltwise, dont_fuse_if_conv_elt_are_outputs)
{
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 5 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
        });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("out", "input", "conv", eltwise_mode::sum));

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output.get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

template<typename InputTy,
         typename OutputTy>
class FusedConvTest : public testing::Test
{
protected:
    static constexpr bool is_pure_float = std::is_same<InputTy, float>::value;
    using OutputPreActivationTy = typename std::conditional<is_pure_float, float, int32_t>::type;
    using WeightsTy = typename std::conditional<is_pure_float, float, int8_t>::type;
    using BiasesTy = typename std::conditional<is_pure_float, float, int32_t>::type;

    topology the_topology;

    std::vector<InputTy> input_values;
    std::vector<WeightsTy> weights_values;
    std::vector<BiasesTy> biases_values;

    // Eltw part.
    std::vector<InputTy> non_conv_input_values;
    std::vector<OutputPreActivationTy> output_pre_relu;

    void add_feature(std::vector<InputTy> input,
                     std::vector<WeightsTy> weights,
                     BiasesTy bias,
                     std::vector<InputTy> non_conv_input,
                     std::vector<OutputPreActivationTy> output)
    {
        assert(non_conv_input.size() == output.size());
        input_values.insert(input_values.end(), input.begin(), input.end());
        weights_values.insert(
            weights_values.end(), weights.begin(), weights.end());
        biases_values.push_back(bias);
        non_conv_input_values.insert(non_conv_input_values.end(),
                                     non_conv_input.begin(),
                                     non_conv_input.end());
        output_pre_relu.insert(
            output_pre_relu.end(), output.begin(), output.end());
    }

    void do_test(const fused_conv_eltwise& fused_prim)
    {
        const auto& engine = get_test_engine();

        int n_features = static_cast<int>(biases_values.size());

        auto input_shape = tensor(1, n_features, 4, 1);
        auto weights_shape = tensor(n_features, n_features, 3, 1);
        auto biases_shape = tensor(1, n_features, 1, 1);
        auto sum_input_shape = tensor(1, n_features, 2, 1);

        auto input = memory::allocate(
            engine,
            {type_to_data_type<InputTy>::value, format::bfyx, input_shape});
        auto weights = memory::allocate(
            engine,
            {type_to_data_type<WeightsTy>::value, format::bfyx, weights_shape});

        auto biases = memory::allocate(
            engine,
            {type_to_data_type<BiasesTy>::value, format::bfyx, biases_shape});
        auto sum_input = memory::allocate(
            engine,
            {type_to_data_type<InputTy>::value, format::bfyx, sum_input_shape});

        set_values(input, input_values);
        std::vector<WeightsTy> post_processed_weights_values(n_features
                                                             * n_features * 3);
        for (int output_feature = 0; output_feature < n_features; ++output_feature)
            for (int input_feature = 0; input_feature < n_features;
                 ++input_feature)
                for (int x = 0; x < 3; ++x)
                {
                    int idx =
                        output_feature * n_features * 3 + input_feature * 3 + x;
                    if (input_feature == output_feature)
                        post_processed_weights_values[idx] =
                            weights_values[input_feature * 3 + x];
                    else
                        post_processed_weights_values[idx] = 0;
                }
        set_values(weights, post_processed_weights_values);
        set_values(biases, biases_values);
        set_values(sum_input, non_conv_input_values);

        the_topology.add(input_layout("input", input.get_layout()));
        the_topology.add(data("weights", weights));
        the_topology.add(data("biases", biases));
        the_topology.add(data("sum_input", sum_input));
        the_topology.add(fused_prim);

        build_options opts;
        opts.set_option(build_option::optimize_data(false));

        network network(engine, the_topology, opts);
        network.set_input_data("input", input);

        auto outputs = network.execute();

        auto output_memory = outputs.at("fused_conv").get_memory();
        auto output_layout = output_memory.get_layout();
        auto output_ptr = output_memory.pointer<OutputTy>();
        int y_size = output_layout.size.spatial[1];
        int x_size = output_layout.size.spatial[0];
        int f_size = output_layout.size.feature[0];
        int b_size = output_layout.size.batch[0];
        EXPECT_EQ(output_layout.format, format::bfyx);
        EXPECT_EQ(y_size, 1);
        EXPECT_EQ(x_size, 2);
        EXPECT_EQ(f_size, n_features);
        EXPECT_EQ(b_size, 1);

        for (int f = 0; f < f_size; f++)
            for (int x = 0; x < x_size; ++x)
            {
                // printf("f: %d, x: %d\n", f, x);
                OutputPreActivationTy expected =
                    pre_relu_to_output(output_pre_relu[f * x_size + x]);
                auto actual = static_cast<OutputPreActivationTy>(
                    output_ptr[f * x_size + x]);
                expect_eq(expected, actual);
            }
    }

private:
    template<typename T = OutputPreActivationTy>
    static typename std::enable_if<std::is_floating_point<T>::value>::type
    expect_eq(const OutputPreActivationTy& lhs, const OutputPreActivationTy& rhs)
    {
        EXPECT_NEAR(lhs, rhs, 0.001f);
    }

    template<typename T = OutputPreActivationTy>
    static typename std::enable_if<std::is_integral<T>::value>::type
    expect_eq(const OutputPreActivationTy& lhs, const OutputPreActivationTy& rhs)
    {
        EXPECT_EQ(lhs, rhs);
    }

    template <typename T>
    static T pre_relu_to_output(T pre_relu) {
      // No std::clamp before C++17 :(
      return std::min(
          static_cast<T>(std::numeric_limits<OutputTy>::max()),
          std::max(static_cast<T>(std::numeric_limits<OutputTy>::lowest()),
                   std::max(static_cast<T>(0), pre_relu)));
    }
};

class FusedConvTest_all_float : public FusedConvTest<float, float>
{};

TEST_F(FusedConvTest_all_float, DISABLED_basic) {
    add_feature({125.0f, 125.0f, 0.0f, 1.0f}, // input
                {2.0f, 0.0f, 1.0f},           // weights
                1.0f,                         // bias
                {-10.0f, -10.0f},             // non_conv_input
                {241.0f, 242.0f});            // output_pre_relu

    add_feature({125.0f, 125.0f, 0.0f, 1.0f}, // input
                {2.0f, 0.0f, 1.0f},           // weights
                0.0f,                         // bias
                {-10.0f, -11.0f},             // non_conv_input
                {480.0f, 480.0f});            // output_pre_relu

    do_test(fused_conv_eltwise("fused_conv",
                               "input",
                               "sum_input",
                               eltwise_mode::sum,
                               {"weights"},
                               {"biases"},
                               {{1, 1, 1, 1}}, // eltw_stride
                               {1, 1, 1, 1},   // stride
                               {0, 0, 0, 0},   // input_offset
                               {1, 1, 1, 1},   // dilation
                               false,          // conv_with_activation
                               0.0f,           // con_activation_slp
                               true,           // eltw_activation
                               0.0f));         // eltw_activation_slp
}
