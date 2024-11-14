// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>

using namespace cldnn;
using namespace ::tests;

class gpu_streams: public ::testing::Test {
public:
    void test_can_create_networks_for_stream(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
        set_values(input,
                { 1.0f, -2.0f, -3.0f, 4.0f, 5.0f,
                    2.0f, 2.0f, 3.0f, 4.0f, -6.0f,
                    3.0f, -3.0f, 3.0f, 5.0f, 1.0f,
                    1.0f, 1.0f, 1.0f, -1.0f, 1.0f });
        VF<float> output_vec = {
                1.0f, -1.0f, -1.5f, 4.0f, 5.0f,
                2.0f, 2.0f, 3.0f, 4.0f, -3.0f,
                3.0f, -1.5f, 3.0f, 5.0f, 1.0f,
                1.0f, 1.0f, 1.0f, -0.5f, 1.0f };

        topology topology(
                input_layout("input", input->get_layout()),
                activation("relu", input_info("input"), activation_func::relu_negative_slope, activation_additional_params{ 0.5f, 0.f }));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);
        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "relu");

        auto output_memory = outputs.at("relu").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();
        ASSERT_EQ(output_layout.format, format::yxfb);
        ASSERT_EQ(y_size, 4);
        ASSERT_EQ(x_size, 5);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);

        for (size_t i = 0; i < output_vec.size(); ++i) {
            ASSERT_FLOAT_EQ(output_vec[i], output_ptr[i]);
        }
    }

    void test_check_networks_can_use_the_same_weights(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto weights = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 3, 2 } });

        VVF<float> output_vec = {
                { 20.0f, 27.0f, 38.0f },
                { 17.0f, 19.0f, 19.0f } };

        layout input0_layout(data_types::f32, format::yxfb, { 1, 1, 5, 4 });

        topology topology(
                input_layout("input", input0_layout),
                data("weights", weights),
                convolution("conv", input_info("input"), "weights", "", 1, {2, 1}, {1, 1}, {0, 0}, {0, 0}, false));

        set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });

        cldnn::network::ptr network0;
        cldnn::network::ptr network1;
        if (is_caching_test) {
            membuf mem_buf;
            {
                std::ostream out_mem(&mem_buf);
                BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);
                ob.set_stream(get_test_stream_ptr().get());
                program::build_program(engine, topology, get_test_default_config(engine))->save(ob);
            }
            {
                std::istream in_mem(&mem_buf);
                BinaryInputBuffer ib = BinaryInputBuffer(in_mem, engine);
                auto imported_prog = std::make_shared<cldnn::program>(engine, get_test_default_config(engine));
                imported_prog->load(ib);
                network0 = std::make_shared<cldnn::network>(imported_prog, 0);
                network1 = std::make_shared<cldnn::network>(imported_prog, 1);
            }
        } else {
            auto prog = program::build_program(engine, topology, get_test_default_config(engine));
            network0 = std::make_shared<cldnn::network>(prog, 0);
            network1 = std::make_shared<cldnn::network>(prog, 1);
        }


        auto input0 = engine.allocate_memory(input0_layout);
        auto input1 = engine.allocate_memory(input0_layout);
        set_values(input0, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
        set_values(input1, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });

        network0->set_input_data("input", input0);
        network1->set_input_data("input", input1);

        auto outputs0 = network0->execute();
        auto outputs1 = network1->execute();
        ASSERT_EQ(outputs0.size(), size_t(1));
        ASSERT_EQ(outputs1.size(), size_t(1));
        ASSERT_EQ(outputs0.begin()->first, "conv");
        ASSERT_EQ(outputs1.begin()->first, "conv");

        auto output_memory0 = outputs0.at("conv").get_memory();
        auto output_memory1 = outputs1.at("conv").get_memory();
        auto output_layout = output_memory0->get_layout();
        cldnn::mem_lock<float> output_ptr0(output_memory0, get_test_stream());
        cldnn::mem_lock<float> output_ptr1(output_memory1, get_test_stream());

        auto wmem0 = network0->get_output_memory("weights");
        auto wmem1 = network1->get_output_memory("weights");

        if (!is_caching_test) {
            ASSERT_EQ(wmem0, wmem1);
        }

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();
        ASSERT_EQ(output_layout.format, format::yxfb);
        ASSERT_EQ(y_size, 2);
        ASSERT_EQ(x_size, 3);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_EQ(output_vec[y][x], output_ptr0[y * x_size + x]);
                ASSERT_EQ(output_vec[y][x], output_ptr1[y * x_size + x]);
            }
        }
    }

    void test_check_networks_use_unique_mutable_data_per_stream(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto weights = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 3, 2 } });

        VVF<float> output_vec = {
                { 20.0f, 27.0f, 38.0f },
                { 17.0f, 19.0f, 19.0f } };

        layout input0_layout(data_types::f32, format::bfyx, { 1, 1, 5, 4 });

        topology topology(
                input_layout("input", input0_layout),
                mutable_data("weights", weights),
                convolution("conv", input_info("input"), "weights", "", 1, {2, 1}, {1, 1}, {0, 0}, {0, 0}, false));

        set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });

        cldnn::network::ptr network0;
        cldnn::network::ptr network1;
        if (is_caching_test) {
            membuf mem_buf;
            {
                std::ostream out_mem(&mem_buf);
                BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);
                ob.set_stream(get_test_stream_ptr().get());
                program::build_program(engine, topology, get_test_default_config(engine))->save(ob);
            }
            {
                std::istream in_mem(&mem_buf);
                BinaryInputBuffer ib = BinaryInputBuffer(in_mem, engine);
                auto imported_prog = std::make_shared<cldnn::program>(engine, get_test_default_config(engine));
                imported_prog->load(ib);
                network0 = std::make_shared<cldnn::network>(imported_prog, 0);
                network1 = std::make_shared<cldnn::network>(imported_prog, 1);
            }
        } else {
            auto prog = program::build_program(engine, topology, get_test_default_config(engine));
            network0 = std::make_shared<cldnn::network>(prog, 0);
            network1 = std::make_shared<cldnn::network>(prog, 1);
        }


        auto input0 = engine.allocate_memory(input0_layout);
        auto input1 = engine.allocate_memory(input0_layout);
        set_values(input0, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
        set_values(input1, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });

        network0->set_input_data("input", input0);
        network1->set_input_data("input", input1);

        auto outputs0 = network0->execute();
        auto outputs1 = network1->execute();
        ASSERT_EQ(outputs0.size(), size_t(1));
        ASSERT_EQ(outputs1.size(), size_t(1));
        ASSERT_EQ(outputs0.begin()->first, "conv");
        ASSERT_EQ(outputs1.begin()->first, "conv");

        auto output_memory0 = outputs0.at("conv").get_memory();
        auto output_memory1 = outputs1.at("conv").get_memory();
        auto output_layout = output_memory0->get_layout();
        cldnn::mem_lock<float> output_ptr0(output_memory0, get_test_stream());
        cldnn::mem_lock<float> output_ptr1(output_memory1, get_test_stream());

        auto wmem0 = network0->get_output_memory("weights");
        auto wmem1 = network1->get_output_memory("weights");

        // check that each stream has unique weights data
        ASSERT_NE(wmem0, wmem1);

        // check that initial memory is reused by the primary stream
        if (!is_caching_test) {
            ASSERT_EQ(wmem0, weights);
        }

        int y_size = output_layout.spatial(1);
        int x_size = output_layout.spatial(0);
        int f_size = output_layout.feature();
        int b_size = output_layout.batch();
        ASSERT_EQ(output_layout.format, format::bfyx);
        ASSERT_EQ(y_size, 2);
        ASSERT_EQ(x_size, 3);
        ASSERT_EQ(f_size, 1);
        ASSERT_EQ(b_size, 1);
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                ASSERT_EQ(output_vec[y][x], output_ptr0[y * x_size + x]);
                ASSERT_EQ(output_vec[y][x], output_ptr1[y * x_size + x]);
            }
        }
    }
};

TEST_F(gpu_streams, can_create_networks_for_stream) {
    this->test_can_create_networks_for_stream(false);
}

TEST_F(gpu_streams, check_networks_can_use_the_same_weights) {
    this->test_check_networks_can_use_the_same_weights(false);
}

TEST_F(gpu_streams, check_networks_use_unique_mutable_data_per_stream) {
    this->test_check_networks_use_unique_mutable_data_per_stream(false);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_F(gpu_streams, can_create_networks_for_stream_cached) {
    this->test_can_create_networks_for_stream(true);
}

TEST_F(gpu_streams, check_networks_can_use_the_same_weights_cached) {
    this->test_check_networks_can_use_the_same_weights(true);
}
#endif
TEST_F(gpu_streams, check_networks_use_unique_mutable_data_per_stream_cached) {
    this->test_check_networks_use_unique_mutable_data_per_stream(true);
}
