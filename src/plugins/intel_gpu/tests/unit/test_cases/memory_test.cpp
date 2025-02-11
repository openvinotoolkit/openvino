// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/pooling.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "intel_gpu/primitives/fully_connected.hpp"
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/grid_sample.hpp>
#include <fully_connected_inst.h>

using namespace cldnn;
using namespace ::tests;

#if 0
TEST(memory_tests, DISABLED_execution_loop)
{
    engine eng;

    memory in = memory::allocate(eng, layout{ data_types::f32, format::bfyx, { 1, 1, 1000, 1000 } });

    topology tpl{
        input_layout("in", in->get_layout()),
        activation("out", "in", activation_func::linear)
    };

    network net(eng, tpl);

    while (true)
    {
        net.set_input_data("in", in);
        net.execute();
    }
}

TEST(memory_tests, DISABLED_network_creation_loop)
{
    engine eng;

    memory in = memory::allocate(eng, layout{ data_types::f32, format::bfyx, { 1, 1, 1000, 1000 } });

    topology tpl{
        input_layout("in", in->get_layout()),
        activation("out", "in", activation_func::linear)
    };

    while (true)
    {
        network net(eng, tpl);
    }
}
#endif
namespace {
class memory_pool: public ::testing::Test {
public:
    void test_basic_non_padded_relu_pipe(bool is_caching_test) {
        // We need a new engine here to get correct get_max_used_device_memory() result
        // If we reuse common engine, then max memory value will be taken from some previously executed tests
        // as it's tracked within engine instance
        auto engine = create_test_engine();
        auto batch_num = 1;
        auto feature_num = 4;
        auto x_size = 1;
        auto y_size = 1;

        auto input = engine->allocate_memory({ data_types::f32, format::bfyx, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(activation("relu", input_info("input"), activation_func::relu));
        topology.add(activation("relu1", input_info("relu"), activation_func::relu));
        topology.add(activation("relu2", input_info("relu1"), activation_func::relu));
        topology.add(activation("relu3", input_info("relu2"), activation_func::relu));
        topology.add(activation("relu4", input_info("relu3"), activation_func::relu));
        topology.add(activation("relu5", input_info("relu4"), activation_func::relu));

        std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
        set_values(input, input_vec);
        ExecutionConfig config = get_test_default_config(*engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(*engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);
        auto outputs = network->execute();

        ASSERT_EQ(engine->get_max_used_device_memory(), (uint64_t)64);
    }

    void test_basic_non_padded_relu_and_pooling_pipe(bool is_caching_test) {
        // We need a new engine here to get correct get_max_used_device_memory() result
        // If we reuse common engine, then max memory value will be taken from some previously executed tests
        // as it's tracked within engine instance
        auto engine = create_test_engine();
        auto batch_num = 1;
        auto feature_num = 4;
        auto x_size = 4;
        auto y_size = 4;

        auto input = engine->allocate_memory({ data_types::f32, format::bfyx, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(activation("relu", input_info("input"), activation_func::relu));
        topology.add(activation("relu1", input_info("relu"), activation_func::relu));
        topology.add(pooling("pool1", input_info("relu1"), pooling_mode::max, { 3, 3 }, { 2, 2 }));
        topology.add(activation("relu2", input_info("pool1"), activation_func::relu));
        topology.add(activation("relu3", input_info("relu2"), activation_func::relu));
        topology.add(activation("relu4", input_info("relu3"), activation_func::relu));
        topology.add(activation("relu5", input_info("relu4"), activation_func::relu));

        ExecutionConfig config = get_test_default_config(*engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(*engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);
        auto outputs = network->execute();

        ASSERT_EQ(engine->get_max_used_device_memory(), (uint64_t)896);
    }

    void test_multi_outputs_network(bool is_caching_test) {
        //            -- relu -- relu1 -- relu4
        //     input<
        //            -- relu2 --  relu3 -- relu5--relu6--relu7
        // neither of relu5, relu6 nor relu7 can share resource with relu4.

        auto engine = create_test_engine();
        auto batch_num = 1;
        auto feature_num = 4;
        auto x_size = 4;
        auto y_size = 4;

        auto input = engine->allocate_memory({ data_types::f32, format::bfyx, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(activation("relu", input_info("input"), activation_func::relu));
        topology.add(activation("relu1", input_info("relu"), activation_func::relu));
        topology.add(activation("relu2", input_info("input"), activation_func::relu));
        topology.add(activation("relu3", input_info("relu2"), activation_func::relu));
        topology.add(activation("relu4", input_info("relu1"), activation_func::relu));
        topology.add(activation("relu5", input_info("relu3"), activation_func::relu));
        topology.add(activation("relu6", input_info("relu5"), activation_func::relu));
        topology.add(activation("relu7", input_info("relu6"), activation_func::relu));

        ExecutionConfig config = get_test_default_config(*engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(*engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);
        auto outputs = network->execute();

        ASSERT_EQ(engine->get_max_used_device_memory(), (uint64_t) 1536);
    }

    void test_oooq(bool is_caching_test) {
        /*          -- relu1 - concat1- relu4 --
            input<  -- relu2 /                   >-- concat2 -- relu6
                    -- relu3 --  relu5 ---------
        neither of relu5, relu6 nor relu7 can share resource with relu4. */

        // We need a new engine here to get correct get_max_used_device_memory() result
        // If we reuse common engine, then max memory value will be taken from some previously executed tests
        // as it's tracked within engine instance
        auto engine = create_test_engine();
        auto batch_num = 1;
        auto feature_num = 4;
        auto x_size = 4;
        auto y_size = 4;

        auto input = engine->allocate_memory({ data_types::f32, format::bfyx, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(activation("relu1", input_info("input"), activation_func::relu));
        topology.add(activation("relu2", input_info("input"), activation_func::relu));
        topology.add(activation("relu3", input_info("input"), activation_func::relu));
        topology.add(concatenation("concat1", { input_info("relu1"), input_info("relu2") }, 1));
        topology.add(activation("relu4", input_info("concat1"), activation_func::relu));
        topology.add(activation("relu5", input_info("relu3"), activation_func::relu));
        topology.add(concatenation("concat2", { input_info("relu4"), input_info("relu5") }, 1));
        topology.add(activation("relu6", input_info("concat2"), activation_func::relu));

        ExecutionConfig config = get_test_default_config(*engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(*engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);
        auto outputs = network->execute();

        ASSERT_EQ(engine->get_max_used_device_memory(), (uint64_t) 2560);
    }

    void test_shared_mem_pool_same_topology_twice() {
        /*                -- relu1 - concat1- relu4 --
        input<  -- relu2 |                             >-- concat2 -- relu6
                        -- relu3 --  relu5 ---------
        neither of relu5, relu6 nor relu7 can share resource with relu4. */

        // We need a new engine here to get correct get_max_used_device_memory() result
        // If we reuse common engine, then max memory value will be taken from some previously executed tests
        // as it's tracked within engine instance
        auto engine = create_test_engine();
        auto batch_num = 1;
        auto feature_num = 4;
        auto inp_x_size = 4;
        auto inp_y_size = 4;

        auto input = engine->allocate_memory({ data_types::f32, format::bfyx, { tensor(spatial(inp_x_size, inp_y_size), feature(feature_num), batch(batch_num)) } });

        set_values(input,
        {   1.0f, 2.5f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 6.1f, 4.7f, 1.0f, 1.0f, 8.2f, 1.0f, 2.0f, 1.0f,
            5.0f, 2.0f, 2.0f, 3.0f, 5.0f, 2.0f, 2.0f, 3.0f, 1.1f, 2.4f, 1.0f, 1.0f, 4.0f, 6.0f, 3.0f, 3.6f,
            4.0f, 6.0f, 3.0f, 3.0f, 1.0f, 1.0f, 1.5f, 1.0f, 4.0f, 6.5f, 3.0f, 3.0f, 4.0f, 6.0f, 1.8f, 3.5f,
            3.0f, 5.0f, 1.0f, 1.0f, 1.3f, 1.0f, 0.4f, 1.3f, 4.0f, 7.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.9f, 4.0f
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(activation("relu1", input_info("input"), activation_func::relu));
        topology.add(activation("relu2", input_info("input"), activation_func::sqrt));
        topology.add(activation("relu3", input_info("input"), activation_func::square));
        topology.add(concatenation("concat1", { input_info("relu1"), input_info("relu2") }, 1));
        topology.add(activation("relu4", input_info("concat1"), activation_func::relu));
        topology.add(activation("relu5", input_info("relu3"), activation_func::relu));
        topology.add(concatenation("concat2", { input_info("relu4"), input_info("relu5") }, 1));
        topology.add(activation("relu6", input_info("concat2"), activation_func::linear, { 1.0f, 0.5f }));

        ExecutionConfig config = get_test_default_config(*engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network network_first(*engine, topology, config);
        network_first.set_input_data("input", input);
        auto outputs = network_first.execute();

        auto output_memory_first = outputs.at("relu6").get_memory();
        auto output_layout_first = output_memory_first->get_layout();
        cldnn::mem_lock<float> output_ptr_first(output_memory_first, get_test_stream());

        ASSERT_EQ(engine->get_max_used_device_memory(), (uint64_t) 2560);

        network network_second(*engine, topology, config);
        network_second.set_input_data("input", input);
        auto outputs_second = network_second.execute();

        auto output_memory_second = outputs_second.at("relu6").get_memory();
        auto output_layout_second = output_memory_second->get_layout();
        cldnn::mem_lock<float> output_ptr_second(output_memory_second, get_test_stream());

        ASSERT_EQ(engine->get_max_used_device_memory(), (uint64_t) 3328);

        ASSERT_EQ(output_layout_first, output_layout_second);

        int y_size = output_layout_first.spatial(1);
        int x_size = output_layout_first.spatial(0);
        int f_size = output_layout_first.feature();
        int b_size = output_layout_first.batch();
        int f_offset = y_size*x_size;
        int b_offset = f_size * f_offset;
        for (int b = 0; b < b_size; ++b)
        {
            for (int f = 0; f < f_size; ++f)
            {
                for (int y = 0; y < y_size; ++y)
                {
                    for (int x = 0; x < x_size; ++x)
                    {
                        int idx = b * b_offset + f * f_offset + y * x_size + x;
                        ASSERT_EQ(output_ptr_first[idx], output_ptr_second[idx]);
                    }
                }
            }
        }
    }

    void test_shared_mem_pool_same_topology_twice_weights() {
        // We need a new engine here to get correct get_max_used_device_memory() result
        // If we reuse common engine, then max memory value will be taken from some previously executed tests
        // as it's tracked within engine instance
        auto engine = create_test_engine();
        auto batch_num = 1;
        auto feature_num = 3;
        auto inp_x_size = 4;
        auto inp_y_size = 4;

        auto input= engine->allocate_memory({ data_types::f32, format::bfyx, { tensor(spatial(inp_x_size, inp_y_size), feature(feature_num), batch(batch_num)) } });
        auto weights = engine->allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 2 } });

        std::vector<float> dummy_input_data_1 = {
        /*f0 xy*/ 0.8f, 0.65f, 0.1f, 1.0f, 1.0f, 0.5f, 0.11f, 0.33f, 0.66f, 0.11f, 0.22f, 0.33f, 0.99f, 0.8f, 0.7f, 0.5f,
        /*f1 xy*/ 0.48f, 0.05f, 0.35f, 1.0f, 1.0f, 0.51f, 0.51f, 0.13f, 0.86f, 0.10f, 0.29f, 0.53f, 0.99f, 0.4f, 0.3f, 0.1f,
        /*f2 xy*/ 0.98f, 0.35f, 0.3f, 0.01f, 0.9f, 0.55f, 0.15f, 0.39f, 0.36f, 0.01f, 0.32f, 0.4f, 0.3f, 0.2f, 0.1f, 0.5f,
        };

        set_values(input, dummy_input_data_1);
        set_values(weights, { 0.10f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f });

        topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            convolution("conv", input_info("input"), "weights", "", 1, {1, 2}, {1, 1}, {0, 0}, {0, 0}, false),
            softmax("softmax", input_info("conv")));

        ExecutionConfig config = get_test_default_config(*engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network network_first(*engine, topology, config);
        network_first.set_input_data("input", input);
        auto outputs = network_first.execute();
        uint64_t cl_mem_result = 824;
        uint64_t usm_result = 1208; // USM have a higher peak, since transfering memory to device adds temporay memory bytes allocated. Old memory is deallocated quickly, but max peak is higher.
        auto is_correct = engine->get_max_used_device_memory() == cl_mem_result
            || engine->get_max_used_device_memory() == usm_result;
        ASSERT_TRUE(is_correct) << "Memory max peak is not correct";

        auto output_memory_first = outputs.at("softmax").get_memory();
        auto output_layout_first = output_memory_first->get_layout();
        cldnn::mem_lock<float> output_ptr_first(output_memory_first, get_test_stream());

        network network_second(*engine, topology, config);
        network_second.set_input_data("input", input);
        auto outputs_second = network_second.execute();

        auto output_memory_second = outputs_second.at("softmax").get_memory();
        auto output_layout_second = output_memory_second->get_layout();
        cldnn::mem_lock<float> output_ptr_second(output_memory_second, get_test_stream());

        cl_mem_result = 1224;
        usm_result = 1992; // USM have a higher peak, since transfering memory to device adds temporay memory bytes allocated. Old memory is deallocated quickly, but max peak is higher.
        is_correct = engine->get_max_used_device_memory() == cl_mem_result
            || engine->get_max_used_device_memory() == usm_result;
        ASSERT_TRUE(is_correct) << "Memory max peak is not correct";
        ASSERT_EQ(output_layout_first, output_layout_second);

        int y_size = output_layout_first.spatial(1);
        int x_size = output_layout_first.spatial(0);
        int f_size = output_layout_first.feature();
        int b_size = output_layout_first.batch();
        int f_offset = y_size * x_size;
        int b_offset = f_size * f_offset;
        for (int b = 0; b < b_size; ++b)
        {
            for (int f = 0; f < f_size; ++f)
            {
                for (int y = 0; y < y_size; ++y)
                {
                    for (int x = 0; x < x_size; ++x)
                    {
                        int idx = b * b_offset + f * f_offset + y * x_size + x;
                        ASSERT_EQ(output_ptr_first[idx], output_ptr_second[idx]);
                    }
                }
            }
        }
    }

    void test_shared_mem_pool_diff_batches(bool is_caching_test) {
        tests::random_generator rg(GET_SUITE_NAME);
        // We need a new engine here to get correct get_max_used_device_memory() result
        // If we reuse common engine, then max memory value will be taken from some previously executed tests
        // as it's tracked within engine instance
        auto engine = create_test_engine();
        auto batch_8 = 8;
        auto batch_1 = 1;
        auto feature_num = 3;
        auto inp_x_size = 4;
        auto inp_y_size = 4;
        auto dt = data_types::f32;
        auto fmt = format::bfyx;
        layout lay_batch_1 = { dt, fmt, { tensor(spatial(inp_x_size, inp_y_size), feature(feature_num), batch(batch_1)) }};
        layout lay_batch_8 = { dt, fmt, { tensor(spatial(inp_x_size, inp_y_size), feature(feature_num), batch(batch_8)) }};
        auto input_1 = engine->allocate_memory(lay_batch_1);
        auto input_8 = engine->allocate_memory(lay_batch_8);
        auto weights = engine->allocate_memory({ dt, fmt, { 1, 3, 3, 2 } });

        std::vector<float> dummy_input_data_1 = rg.generate_random_1d<float>(batch_1 * feature_num * inp_x_size * inp_y_size, 0, 1);
        std::vector<float> dummy_input_data_8 = rg.generate_random_1d<float>(batch_8 * feature_num * inp_x_size * inp_y_size, 0, 1);

        set_values(input_1, dummy_input_data_1);
        set_values(input_8, dummy_input_data_8);
        set_values(weights, { 0.10f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f,
                            0.10f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f,
                            0.10f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f });

        topology topo(
            input_layout("input", input_8->get_layout()),
            data("weights", weights),
            convolution("conv", input_info("input"), { "weights" }, {}, 1, { 2, 1 }, {1, 1}, {0, 0}, {0, 0}, false),
            softmax("softmax", input_info("conv")));

        ExecutionConfig config = get_test_default_config(*engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network_first = get_network(*engine, topo, config, get_test_stream_ptr(), is_caching_test);
        network_first->set_input_data("input", input_8);
        auto outputs = network_first->execute();

        auto dev_info = engine->get_device_info();
        ASSERT_EQ(engine->get_max_used_device_memory(), (uint64_t)4744);

        topo.change_input_layout("input", input_1->get_layout());//change input layout to batch=1

        network::ptr network_second = get_network(*engine, topo, config, get_test_stream_ptr(), is_caching_test);
        network_second->set_input_data("input", input_1);
        auto outputs_second = network_second->execute();
        ASSERT_EQ(engine->get_max_used_device_memory(), (uint64_t)5912);
    }

    void test_shared_dep_two_output(bool is_caching_test) {
        // We need a new engine here to get correct get_max_used_device_memory() result
        // If we reuse common engine, then max memory value will be taken from some previously executed tests
        // as it's tracked within engine instance
        auto engine = create_test_engine();

        auto input_1 = engine->allocate_memory({ {1, 1, 4, 4}, data_types::f32, format::bfyx });
        set_random_values<float>(input_1);

        //build and execute network
        topology topo;
        topo.add(cldnn::data("constant_0_0", input_1));
        topo.add(cldnn::concatenation("result_1_0", { input_info("constant_0_0") }, 0));
        topo.add(cldnn::concatenation("result_2_0", { input_info("constant_0_0") }, 0));

        ExecutionConfig config = get_test_default_config(*engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(*engine, topo, config, get_test_stream_ptr(), is_caching_test);
        auto outputs = network->execute();
        ASSERT_EQ(engine->get_max_used_device_memory(), (uint64_t)192);
    }

    void test_non_opt_intermidate_opt_after(bool is_caching_test) {
        auto& engine = get_test_engine();
        auto input_layout1 = layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1, 1, 2, 2 });
        auto input_layout2 = layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1, 1, 2, 2 });

        auto input_memory1 = engine.allocate_memory(input_layout1);
        auto input_memory2 = engine.allocate_memory(input_layout2);
        auto scale_memory = engine.allocate_memory(layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1, 1, 1, 1 }));
        auto data_memory = cldnn::data("scale_mem", scale_memory);

        set_values(input_memory1, { 1.0f, 2.0f, 3.0f, 4.0f });
        set_values(input_memory2, { 5.0f, 6.0f, 7.0f, 8.0f });
        set_values(scale_memory, { 1.0f });

        auto reshape_tensor = cldnn::tensor(8, 1, 1, 1);
        auto input = cldnn::input_layout("input1", input_layout1);
        auto input2 = cldnn::input_layout("input2", input_layout2);
        auto concat = cldnn::concatenation("concat", { input_info("input1"), input_info("input2") }, 0);
        auto reshape = cldnn::reshape("reshape", input_info("concat"), reshape_tensor);
        auto crop1 = cldnn::crop("crop1", input_info("reshape"), { 1, 1, 1, 1 }, { 0, 0, 0, 0 });
        auto crop2 = cldnn::crop("crop2", input_info("reshape"), { 1, 1, 1, 1 }, { 1, 0, 0, 0 });
        auto eltwise1 = cldnn::eltwise("elt1", { input_info("crop1"), input_info("scale_mem") }, eltwise_mode::prod);
        auto eltwise2 = cldnn::eltwise("elt2", { input_info("crop2"), input_info("scale_mem") }, eltwise_mode::prod);

        auto topology = cldnn::topology(
            input, input2,
            concat,
            reshape,
            crop1, crop2,
            eltwise1, eltwise2,
            data_memory
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(false));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input1", input_memory1);
        network->set_input_data("input2", input_memory2);
        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), static_cast<size_t>(2));

        auto out1 = outputs.at("elt1");
        auto out2 = outputs.at("elt2");

        cldnn::mem_lock<float> out1_ptr(out1.get_memory(), get_test_stream());
        cldnn::mem_lock<float> out2_ptr(out2.get_memory(), get_test_stream());
        ASSERT_EQ(out1_ptr[0], 1.0f);
        ASSERT_EQ(out2_ptr[0], 2.0f);
    }

    void test_add_mem_dep(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input_layout1 = layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1, 2, 2, 2 });

        auto input_memory1 = engine.allocate_memory(input_layout1);
        auto scale_memory = engine.allocate_memory(layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1, 1, 1, 1 }));
        auto data_memory = cldnn::data("scale_mem", scale_memory);

        set_values(input_memory1, { 1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f});
        set_values(scale_memory, { 1.0f });

        auto input = cldnn::input_layout("input1", input_layout1);
        auto actv1 = cldnn::activation("input_activ1", input_info("input1"), activation_func::abs);
        auto actv2 = cldnn::activation("input_activ2", input_info("input1"), activation_func::abs);
        auto crop1 = cldnn::crop("crop1", input_info("input_activ1"), { 1, 1, 2, 2 }, { 0, 0, 0, 0 });
        auto crop2 = cldnn::crop("crop2", input_info("input_activ2"), { 1, 1, 2, 2 }, { 0, 1, 0, 0 });
        auto eltwise1 = cldnn::eltwise("elt1", { input_info("crop1"), input_info("scale_mem") }, eltwise_mode::prod);
        auto eltwise2 = cldnn::eltwise("elt2", { input_info("crop2"), input_info("scale_mem") }, eltwise_mode::prod);
        auto actv3 = cldnn::activation("out3", input_info("elt1"), activation_func::abs);
        auto actv4 = cldnn::activation("out4", input_info("elt2"), activation_func::abs);

        auto topology = cldnn::topology(
            input,
            crop1, crop2,
            actv1, actv2,
            eltwise1, eltwise2,
            data_memory,
            actv3, actv4
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input1", input_memory1);
        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), static_cast<size_t>(2));

        auto out1 = outputs.at("out3");
        auto out2 = outputs.at("out4");

        cldnn::mem_lock<float> out1_ptr(out1.get_memory(), get_test_stream());
        cldnn::mem_lock<float> out2_ptr(out2.get_memory(), get_test_stream());
        ASSERT_EQ(out1_ptr[0], 1.0f);
        ASSERT_EQ(out1_ptr[1], 2.0f);
        ASSERT_EQ(out1_ptr[2], 3.0f);
        ASSERT_EQ(out1_ptr[3], 4.0f);

        ASSERT_EQ(out2_ptr[0], 5.0f);
        ASSERT_EQ(out2_ptr[1], 6.0f);
        ASSERT_EQ(out2_ptr[2], 7.0f);
        ASSERT_EQ(out2_ptr[3], 8.0f);
    }

    void test_dynamic_mem_reuse() {
        auto& engine = get_test_engine();

        const int32_t input_f = 3, weight_b = 4;

        auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(), input_f }, data_types::f32,format::bfyx };
        auto input_actual_layout1 = layout{ ov::PartialShape{ 2, input_f }, data_types::f32,format::bfyx};
        auto input_actual_layout2 = layout{ ov::PartialShape{ 1, input_f }, data_types::f32,format::bfyx};
        auto input_data1 = engine.allocate_memory(input_actual_layout1);
        auto input_data2 = engine.allocate_memory(input_actual_layout2);
        auto fc_weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx});

        set_values(input_data1, { -0.5f, 2.0f, 0.5f, 
                                   0.5f, -2.0f, -0.5f});
        set_values(input_data2, { 0.5f, -2.0f, -0.5f });
        set_values(fc_weights_data, { 1.5f, 1.0f, 0.5f,
                                -1.0f, 0.0f, 0.5f,
                                0.5f, -0.5f, -2.0f,
                                -0.5f, 1.0f, 1.5f });

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            activation("relu1", input_info("input"), activation_func::relu),
            eltwise("elt1", { input_info("input"), input_info("relu1") }, eltwise_mode::prod),
            activation("relu2", input_info("elt1"), activation_func::sqrt),
            eltwise("elt2", { input_info("elt1"), input_info("relu2") }, eltwise_mode::prod),
            data("fc_weights", fc_weights_data),
            fully_connected("fc", input_info("elt2"), "fc_weights")
        };

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network(engine, topology, config);

        {
            network.set_input_data("input", input_data1);

            auto outputs = network.execute();

            ASSERT_EQ(std::static_pointer_cast<fully_connected_inst>(network.get_primitive("relu1"))->output_memory_ptr()->buffer_ptr(), 
                      std::static_pointer_cast<fully_connected_inst>(network.get_primitive("relu2"))->output_memory_ptr()->buffer_ptr());
        }

        {
            network.set_input_data("input", input_data2);

            auto outputs = network.execute();

            ASSERT_EQ(std::static_pointer_cast<fully_connected_inst>(network.get_primitive("relu1"))->output_memory_ptr()->buffer_ptr(), 
                      std::static_pointer_cast<fully_connected_inst>(network.get_primitive("relu2"))->output_memory_ptr()->buffer_ptr());
        }
    }

    void test_dynamic_mem_reuse_for_null_sel_impl() {
        auto& engine = get_test_engine();

        const int32_t input_f = 3;

        auto input_static_layout = layout{ ov::PartialShape{ 2, input_f, 1, 2 }, data_types::f32, format::bfyx };
        auto input_dyn_layout = layout{ ov::PartialShape{ 2, input_f, ov::Dimension(), 2 }, data_types::f32, format::bfyx };
        auto input_data1 = engine.allocate_memory(input_static_layout);
        auto fc_weights_data = engine.allocate_memory({ ov::PartialShape{ 2, input_f, 1, 1 }, data_types::f32, format::bfyx});

        set_values(input_data1, { 0.5f, -2.0f, -0.5f, -0.5f, 2.0f, 0.5f,
                                -0.5f, 2.0f, 0.5f, -0.5f, 2.0f, 0.5f });
        set_values(fc_weights_data, { 1.5f, 1.0f, 0.5f,
                                -1.0f, 0.0f, 0.5f});

        GridSampleOp::Attributes attributes(false, GridSampleOp::InterpolationMode::NEAREST, GridSampleOp::PaddingMode::ZEROS);

        cldnn::topology topology{
            input_layout("input", input_static_layout),
            input_layout("input_dyn", input_dyn_layout),
            activation("relu1", input_info("input"), activation_func::relu),
            eltwise("elt1", { input_info("input"), input_info("relu1") }, eltwise_mode::prod),
            activation("relu2", input_info("elt1"), activation_func::sqrt),
            // The 'grid_sample' layer should be dynamic, not 'shape agnostic kernel' and user of 'elt1'. This is a key condition of this test.
            grid_sample("grid_sample", { input_info("elt1"), input_info("input_dyn") }, attributes),
            data("fc_weights", fc_weights_data),
            fully_connected("fc", input_info("grid_sample"), "fc_weights")
        };

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network(engine, topology, config);

        {
            network.set_input_data("input", input_data1);
            network.set_input_data("input_dyn", input_data1);

            auto outputs = network.execute();

            // Should be false for memory-reuse
            ASSERT_NE(std::static_pointer_cast<fully_connected_inst>(network.get_primitive("relu1"))->output_memory_ptr()->buffer_ptr(),
                      std::static_pointer_cast<fully_connected_inst>(network.get_primitive("relu2"))->output_memory_ptr()->buffer_ptr());
        }
    }
};

TEST_F(memory_pool, basic_non_padded_relu_pipe) {
    this->test_basic_non_padded_relu_pipe(false);
}

TEST_F(memory_pool, basic_non_padded_relu_and_pooling_pipe) {
    this->test_basic_non_padded_relu_and_pooling_pipe(false);
}

TEST_F(memory_pool, multi_outputs_network) {
    this->test_multi_outputs_network(false);
}

TEST_F(memory_pool, oooq) {
    this->test_oooq(false);
}

TEST_F(memory_pool, DISABLED_shared_mem_pool_same_topology_twice) {
    this->test_shared_mem_pool_same_topology_twice();
}

TEST_F(memory_pool, DISABLED_shared_mem_pool_same_topology_twice_weights) {
    this->test_shared_mem_pool_same_topology_twice_weights();
}

TEST_F(memory_pool, shared_mem_pool_diff_batches) {
    this->test_shared_mem_pool_diff_batches(false);
}

TEST_F(memory_pool, shared_dep_two_output) {
    this->test_shared_dep_two_output(false);
}

TEST_F(memory_pool, non_opt_intermidate_opt_after) {
    this->test_non_opt_intermidate_opt_after(false);
}

TEST_F(memory_pool, add_mem_dep_test) {
    this->test_add_mem_dep(false);
}

TEST_F(memory_pool, dynamic_mem_reuse) {
    this->test_dynamic_mem_reuse();
}

TEST_F(memory_pool, dynamic_mem_reuse_for_null_sel_impl) {
    this->test_dynamic_mem_reuse_for_null_sel_impl();
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_F(memory_pool, basic_non_padded_relu_pipe_cached) {
    this->test_basic_non_padded_relu_pipe(true);
}

TEST_F(memory_pool, basic_non_padded_relu_and_pooling_pipe_cached) {
    this->test_basic_non_padded_relu_and_pooling_pipe(true);
}

TEST_F(memory_pool, multi_outputs_network_cached) {
    this->test_multi_outputs_network(true);
}

TEST_F(memory_pool, oooq_cached) {
    this->test_oooq(true);
}

TEST_F(memory_pool, shared_mem_pool_diff_batches_cached) {
    this->test_shared_mem_pool_diff_batches(true);
}

TEST_F(memory_pool, shared_dep_two_output_cached) {
    this->test_shared_dep_two_output(true);
}

TEST_F(memory_pool, non_opt_intermidate_opt_after_cached) {
    this->test_non_opt_intermidate_opt_after(true);
}
#endif

TEST_F(memory_pool, add_mem_dep_test_cached) {
    this->test_add_mem_dep(true);
}
}
