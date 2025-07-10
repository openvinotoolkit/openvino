// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/strided_slice.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "strided_slice_inst.h"

using namespace cldnn;
using namespace ::tests;

class strided_slice_gpu: public ::testing::Test {
public:
    void test_2x2x2x2_full_legacy_activation(bool is_caching_test) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): 1x1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2, 2 }, data_types::f32, format::bfyx });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                -0.2f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.8f
        });

        set_values<int64_t>(begin, {0, 0, 0, 0});
        set_values<int64_t>(end, {2, 2, 2, 2});
        set_values<int64_t>(strides, {1, 1, 1, 1});

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {2, 2, 2, 2}));
        topology.add(activation("out", input_info("strided_slice"), activation_func::clamp, {0.f, 15.0f}));
        topology.add(reorder("out_reorder", input_info("out"), format::bfyx, data_types::f32));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "out_reorder");

        auto output = outputs.at("out_reorder").get_memory();

        std::vector<float> answers = {
                0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_full(bool is_caching_test) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): 1x1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2, 2 }, data_types::f32, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        std::vector<int64_t> begin_data = { 0, 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {2, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2x2_full(bool is_caching_test) {
        // Input (BFZYX): 2x2x2x2x2
        // Begin (BFZYX): 0x0x0x0x0
        // End (BFZYX): 2x2x2x2x2
        // Stride (BFZYX): 1x1x1x1x1
        // Output (BFZYX): 2x2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2, 2, 2 }, data_types::f32, format::bfzyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f,
                23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
        });
        std::vector<int64_t> begin_data = { 0, 0, 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 2, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {2, 2, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f,
            23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }


    void test_2x2x2x2x2x2_full(bool is_caching_test) {
        // Input (BFWZYX): 2x2x2x2x2x2
        // Begin (BFWZYX): 0x0x0x0x0x0
        // End (BFWZYX): 2x2x2x2x2x2
        // Stride (BFWZYX): 1x1x1x1x1x1
        // Output (BFWZYX): 2x2x2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2, 2, 2, 2 }, data_types::f32, format::bfwzyx });

        set_values(input, {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
            24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
            32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
            48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
            56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f,
        });
        std::vector<int64_t> begin_data = { 0, 0, 0, 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 2, 2, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {2, 2, 2, 2, 2, 2}));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfwzyx, "", impl_types::ocl}} }));

        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
            16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
            24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
            32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
            40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
            48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
            56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f,
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_full_pad(bool is_caching_test) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): 1x1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2, 2 }, data_types::f32, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        std::vector<int64_t> begin_data = { 0, 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 1 };

        padding in_pad({0, 0, 1, 1}, {0, 0, 1, 1});
        auto padded_layout = input->get_layout().with_padding(in_pad);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("input_reorder", input_info("input"), padded_layout));
        topology.add(strided_slice("strided_slice", input_info("input_reorder"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {2, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_ignore(bool is_caching_test) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 1x1x1x1
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): 1x1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        std::vector<int64_t> begin_data = { 1, 1, 1, 1 };
        std::vector<int64_t> end_data = { 2, 2, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}, {2, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_single(bool is_caching_test, impl_types impl_type = impl_types::any) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 1x1x1x1
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): 1x1x1x1
        // Output (BFYX): 1x1x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        std::vector<int64_t> begin_data = { 1, 1, 1, 1 };
        std::vector<int64_t> end_data = { 2, 2, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {1, 1, 1, 1}));

        auto config = get_test_default_config(engine);
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = { 15.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x4x3_stride(bool is_caching_test) {
        // Input (BFYX): 2x2x4x3
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 2x2x4x3
        // Stride (BFYX): 1x1x2x1
        // Output (BFYX): 2x2x2x3

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 3, 4 } });

        set_values(input, {
                0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f,
                18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f,
                27.f, 28.f, 29.f, 30.f, 31.f, 32.f, 33.f, 34.f, 35.f,
                36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f,
                45.f, 46.f, 47.f
        });
        std::vector<int64_t> begin_data = { 0, 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 4, 3 };
        std::vector<int64_t> strides_data = { 1, 1, 2, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}, {2, 2, 2, 3}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.f, 1.f, 2.f, 6.f, 7.f, 8.f, 12.f, 13.f, 14.f, 18.f, 19.f, 20.f,
                24.f, 25.f, 26.f, 30.f, 31.f, 32.f, 36.f, 37.f, 38.f, 42.f, 43.f, 44.f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x4x4_part_stride(bool is_caching_test) {
        // Input (BFYX): 2x2x4x4
        // Begin (BFYX): 1x0x0x1
        // End (BFYX): 2x2x4x4
        // Stride (BFYX): 1x1x1x2
        // Output (BFYX): 1x2x4x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 4, 4 } });

        set_values(input, {
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
                60.0f, 61.0f, 62.0f, 63.0f
        });
        std::vector<int64_t> begin_data = { 1, 0, 0, 1 };
        std::vector<int64_t> end_data = { 2, 2, 4, 4 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 2 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {0, 1, 1, 0}, {}, {}, {}, {}, {1, 2, 4, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                33.0f, 35.0f,
                37.0f, 39.0f,
                41.0f, 43.0f,
                45.0f, 47.0f,

                49.0f, 51.0f,
                53.0f, 55.0f,
                57.0f, 59.0f,
                61.0f, 63.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x4x1_new_axis_mask(bool is_caching_test) {
        // Input (BFYX): 2x2x4x1
        // New_axis_mask: 1
        // Output (BFYX): 1x2x2x4

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 4 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        std::vector<int64_t> begin_data = { 1, 0, 1, 0 };
        std::vector<int64_t> end_data = { 2, 2, 4, 4 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 2 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, { 1 }, {}, {}, {2, 2, 4, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x1x1_new_axis_mask_2(bool is_caching_test) {
        // Input (BFYX): 2x2x1x1
        // New_axis_mask: 101
        // Output (BFYX): 1x2x1x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f
        });
        std::vector<int64_t> begin_data = { 1, 0, 1, 0 };
        std::vector<int64_t> end_data = { 2, 2, 4, 4 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 2 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, { 1, 0, 1 }, {}, {}, {2, 2, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x1x1(bool is_caching_test, impl_types impl_type = impl_types::any, bool disable_usm = false) {
        // Input (BFYX): 2x2x1x1
        // Output (BFYX): 2x2x1x1

        auto engine = create_test_engine(engine_types::ocl, runtime_types::ocl, !disable_usm);
        auto input = engine->allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f
        });
        std::vector<int64_t> begin_data = { 0, 0 };
        std::vector<int64_t> end_data = { 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {1, 0}, {}, {}, {}, {}, {2, 2, 1, 1}));

        auto config = get_test_default_config(*engine);
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        cldnn::network::ptr network = get_network(*engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x1x1(bool is_caching_test) {
        // Input (BFZYX): 2x2x2x1x1
        // Output (BFZYX): 1x2x2x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { 2, 2, 1, 1, 2 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        });
        std::vector<int64_t> begin_data = { 0, 0, 0 };
        std::vector<int64_t> end_data = { 1, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {1, 2, 2, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x1x1_2(bool is_caching_test, impl_types impl_type = impl_types::any) {
        // Input (BFZYX): 2x2x2x1x1
        // Output (BFZYX): 2x1x1x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { 2, 2, 1, 1, 2 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        });
        std::vector<int64_t> begin_data = { 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 2, 2 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {2, 1, 1, 1}));

        auto config = get_test_default_config(engine);
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 4.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2x1x1(bool is_caching_test) {
        // Input (BFWZYX): 2x2x2x2x1x1
        // Output (BFWZYX): 1x2x2x2x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx, { 2, 2, 1, 1, 2, 2 }});

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        });
        std::vector<int64_t> begin_data = { 0, 0, 0, 0, 0, 0 };
        std::vector<int64_t> end_data = { 1, 2, 2, 2, 1, 1 };
        std::vector<int64_t> strides_data = { 1, 1, 1, 1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {1, 2, 2, 2, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2x1x1_2(bool is_caching_test, impl_types impl_type = impl_types::any) {
        // Input (BFWZYX): 2x2x2x2x1x1
        // Output (BFWZYX): 2x1x1x1x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfwzyx, { 2, 2, 1, 1, 2, 2 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        });
        std::vector<int64_t> begin_data = { 0, 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 2, 2, 2 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {2, 1, 1, 1, 1}));

        auto config = get_test_default_config(engine);
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfwzyx, "", impl_types::cpu}} }));

        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 8.0f,
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_full_negative_stride(bool is_caching_test) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): -1x-1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2, 2 }, data_types::f32, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        std::vector<int64_t> begin_data = { 0, 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 2, 2 };
        std::vector<int64_t> strides_data = { -1, -1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {2, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                12.f, 13.f, 14.f, 15.f, 8.f, 9.f, 10.f, 11.f, 4.f, 5.f, 6.f, 7.f, 0.f, 1.f, 2.f, 3.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_full_negative_stride_pad(bool is_caching_test) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): -1x-1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2, 2 }, data_types::f32, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        std::vector<int64_t> begin_data = { 0, 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 2, 2 };
        std::vector<int64_t> strides_data = { -1, -1, 1, 1 };

        padding in_pad({0, 0, 1, 1}, {0, 0, 1, 1});
        auto padded_layout = input->get_layout().with_padding(in_pad);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("input_reorder", input_info("input"), padded_layout));
        topology.add(strided_slice("strided_slice", input_info("input_reorder"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {2, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                12.f, 13.f, 14.f, 15.f, 8.f, 9.f, 10.f, 11.f, 4.f, 5.f, 6.f, 7.f, 0.f, 1.f, 2.f, 3.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x1x1_2_negative_all(bool is_caching_test) {
        // Input (BFZYX): 2x2x2x1x1
        // Output (BFZYX): 2x1x1x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { 2, 2, 1, 1, 2 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        });
        std::vector<int64_t> begin_data = { 0, 0, 0 };
        std::vector<int64_t> end_data = { 2, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 2, 2 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {2, 1, 1, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 4.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x1x1_2_negative_all_dynamic(impl_types impl_type = impl_types::any) {
        // Input (BFZYX): 2x2x2x1x1
        // Output (BFZYX): 2x1x1x1x1

        auto& engine = get_test_engine();
        auto input_lay = layout{ ov::PartialShape::dynamic(3), data_types::f32, format::bfyx };
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2 }, data_types::f32, format::bfyx });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 3 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 3 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 3 }, data_types::i64, format::bfyx });

        set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
        set_values<int64_t>(begin, {0, 0, 0});
        set_values<int64_t>(end, {2, 2, 2});
        set_values<int64_t>(strides, {1, 2, 2});

        topology topology;
        topology.add(input_layout("input", input_lay));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {}));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        network network(engine, topology, config);

        network.set_input_data("input", input);

        auto inst = network.get_primitive("strided_slice");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network.execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 4.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2_all_dynamic_bcast(impl_types impl_type = impl_types::any) {
        auto& engine = get_test_engine();
        auto input_lay = layout{ ov::PartialShape::dynamic(3), data_types::f32, format::bfyx };
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2 }, data_types::f32, format::bfyx });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 1 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 1 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 1 }, data_types::i64, format::bfyx });

        set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
        set_values<int64_t>(begin, {1});
        set_values<int64_t>(end, {2});
        set_values<int64_t>(strides, {1});

        topology topology;
        topology.add(input_layout("input", input_lay));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {}));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        network network(engine, topology, config);

        network.set_input_data("input", input);

        auto inst = network.get_primitive("strided_slice");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network.execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        ov::PartialShape expected_shape{1, 2, 2};

        ASSERT_EQ(output->get_layout().get_partial_shape(), expected_shape);

        std::vector<float> answers = {
                4.0f, 5.0f, 6.0f, 7.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)         {
            ASSERT_EQ(answers[i], output_ptr[i]) << " i = " << i;
        }
    }

    void test_2x2x2x1x1_2_negative_all_dynamic_begin(impl_types impl_type = impl_types::any) {
        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ ov::PartialShape{ 2, 2, 2 }, data_types::f32, format::bfyx });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 3 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 3 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 3 }, data_types::i64, format::bfyx });

        set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
        set_values<int64_t>(begin, {0, 0, 0});
        set_values<int64_t>(end, {2, 2, 2});
        set_values<int64_t>(strides, {1, 2, 2});

        topology topology;
        topology.add(data("input", input));
        topology.add(input_layout("input2", begin->get_layout()));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {}));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        network network(engine, topology, config);

        network.set_input_data("input2", begin);

        auto inst = network.get_primitive("strided_slice");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network.execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 4.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_3d_all_dynamic_with_new_axis() {
        auto& engine = get_test_engine();
        auto l_input = layout{ ov::PartialShape::dynamic(3), data_types::f32, format::bfyx };
        auto l_begin = layout{ ov::PartialShape{ 3 }, data_types::i64, format::bfyx };
        auto l_end = layout{ ov::PartialShape{ 3 }, data_types::i64, format::bfyx };

        auto stride = engine.allocate_memory({ ov::PartialShape{ 3 }, data_types::i64, format::bfyx });
        set_values<int64_t>(stride, {1, 1, 1});

        topology topology(
            input_layout("input", l_input),
            input_layout("begin", l_begin),
            input_layout("end", l_end),
            data("stride", stride),
            strided_slice("strided_slice",
                input_info("input"),
                input_info("begin"),
                input_info("end"),
                input_info("stride"),
                {}, {}, {1, 0, 0}, {}, {}, {})
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto program = program::build_program(engine, topology, config, false, false, false);
        ASSERT_NE(program, nullptr);

        auto out_size_act = program->get_outputs()[0]->get_output_layouts(0)[0].get_partial_shape().size();
        auto out_size_exp = size_t(4);
        ASSERT_EQ(out_size_act, out_size_exp);
    }

    void test_3d_all_dynamic_with_shrink_axis() {
        auto& engine = get_test_engine();

        auto l_input = layout{ ov::PartialShape::dynamic(3), data_types::f32, format::bfyx };
        auto l_begin = layout{ ov::PartialShape{ 3 }, data_types::i64, format::bfyx };
        auto l_end = layout{ ov::PartialShape{ 3 }, data_types::i64, format::bfyx };

        auto stride = engine.allocate_memory({ ov::PartialShape{ 3 }, data_types::i64, format::bfyx });
        set_values<int64_t>(stride, {1, 1, 1});

        topology topology(
            input_layout("input", l_input),
            input_layout("begin", l_begin),
            input_layout("end", l_end),
            data("stride", stride),
            strided_slice("strided_slice",
                input_info("input"),
                input_info("begin"),
                input_info("end"),
                input_info("stride"),
                {}, {}, {}, {1, 0, 0}, {}, {})
        );

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto program = program::build_program(engine, topology, config, false, false, false);
        ASSERT_NE(program, nullptr);

        auto out_size_act = program->get_outputs()[0]->get_output_layouts(0)[0].get_partial_shape().size();
        auto out_size_exp = size_t(2);
        ASSERT_EQ(out_size_act, out_size_exp);
    }
};

class strided_slice_gpu_constants: public ::testing::Test {
public:
    void test_2x2x2x2_full(bool is_caching_test, impl_types impl_type = impl_types::any) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): 1x1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        set_values<int64_t>(begin, {
                0, 0, 0, 0
        });
        set_values<int64_t>(end, {
                2, 2, 2, 2
        });
        set_values<int64_t>(strides, {
                1, 1, 1, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {2, 2, 2, 2}));

        auto config = get_test_default_config(engine);
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_ignore(bool is_caching_test) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 1x1x1x1
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): 1x1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        set_values<int64_t>(begin, {
                1, 1, 1, 1
        });
        set_values<int64_t>(end, {
                2, 2, 2, 2
        });
        set_values<int64_t>(strides, {
                1, 1, 1, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}, {2, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_single(bool is_caching_test, impl_types impl_type = impl_types::any) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 1x1x1x1
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): 1x1x1x1
        // Output (BFYX): 1x1x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
                });
        set_values<int64_t>(begin, {
                1, 1, 1, 1
                });
        set_values<int64_t>(end, {
                2, 2, 2, 2
                });
        set_values<int64_t>(strides, {
                1, 1, 1, 1
                });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {1, 1, 1, 1}));

        auto config = get_test_default_config(engine);
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = { 15.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x4x3_stride(bool is_caching_test, impl_types impl_type = impl_types::any) {
        // Input (BFYX): 2x2x4x3
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 2x2x4x3
        // Stride (BFYX): 1x1x2x1
        // Output (BFYX): 2x2x2x3

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 3, 4 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f,
                18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f,
                27.f, 28.f, 29.f, 30.f, 31.f, 32.f, 33.f, 34.f, 35.f,
                36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f,
                45.f, 46.f, 47.f
        });
        set_values<int64_t>(begin, {
                0, 0, 0, 0
        });
        set_values<int64_t>(end, {
                2, 2, 4, 3
        });
        set_values<int64_t>(strides, {
                1, 1, 2, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}, {2, 2, 2, 3}));

        auto config = get_test_default_config(engine);
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.f, 1.f, 2.f, 6.f, 7.f, 8.f, 12.f, 13.f, 14.f, 18.f, 19.f, 20.f,
                24.f, 25.f, 26.f, 30.f, 31.f, 32.f, 36.f, 37.f, 38.f, 42.f, 43.f, 44.f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x4x4_part_stride(bool is_caching_test) {
        // Input (BFYX): 2x2x4x4
        // Begin (BFYX): 1x0x0x1
        // End (BFYX): 2x2x4x4
        // Stride (BFYX): 1x1x1x2
        // Output (BFYX): 1x2x4x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 4, 4 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
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
                60.0f, 61.0f, 62.0f, 63.0f
        });
        set_values<int64_t>(begin, {
                1, 0, 0, 1
        });
        set_values<int64_t>(end, {
                2, 2, 4, 4
        });
        set_values<int64_t>(strides, {
                1, 1, 1, 2
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(input_layout("input2", begin->get_layout()));
        topology.add(input_layout("input3", end->get_layout()));
        topology.add(input_layout("input4", strides->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {0, 1, 1, 0}, {}, {}, {}, {}, {1, 2, 4, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);
        network->set_input_data("input2", begin);
        network->set_input_data("input3", end);
        network->set_input_data("input4", strides);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                33.0f, 35.0f,
                37.0f, 39.0f,
                41.0f, 43.0f,
                45.0f, 47.0f,

                49.0f, 51.0f,
                53.0f, 55.0f,
                57.0f, 59.0f,
                61.0f, 63.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x4x1_new_axis_mask(bool is_caching_test, impl_types impl_type = impl_types::any) {
        // Input (BFYX): 2x2x4x1
        // New_axis_mask: 1
        // Output (BFYX): 1x2x2x4

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 4 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        set_values<int64_t>(begin, {
                1, 0, 1, 0
        });
        set_values<int64_t>(end, {
                2, 2, 4, 4
        });
        set_values<int64_t>(strides, {
                1, 1, 1, 2
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, { 1 }, {}, {}, {2, 2, 4, 1}));

        auto config = get_test_default_config(engine);
        if (impl_type != impl_types::any)
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"strided_slice", {format::bfyx, "", impl_types::cpu}} }));

        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x1x1_new_axis_mask_2(bool is_caching_test) {
        // Input (BFYX): 2x2x1x1
        // New_axis_mask: 101
        // Output (BFYX): 1x2x1x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f
        });
        set_values<int64_t>(begin, {
                1, 0, 1, 0
        });
        set_values<int64_t>(end, {
                2, 2, 4, 4
        });
        set_values<int64_t>(strides, {
                1, 1, 1, 2
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, { 1, 0, 1 }, {}, {}, {2, 2, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x1x1(bool is_caching_test) {
        // Input (BFYX): 2x2x1x1
        // Output (BFYX): 2x2x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
        auto begin = engine.allocate_memory({ data_types::i64, format::bfyx, { 2, 1, 1, 1 } });
        auto end = engine.allocate_memory({ data_types::i64, format::bfyx, { 2, 1, 1, 1 } });
        auto strides = engine.allocate_memory({ data_types::i64, format::bfyx, { 2, 1, 1, 1 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f
        });

        set_values<int64_t>(begin, {
                0, 0
        });
        set_values<int64_t>(end, {
                2, 2
        });
        set_values<int64_t>(strides, {
                1, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {1, 0}, {}, {}, {}, {}, {2, 2, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x1x1(bool is_caching_test) {
        // Input (BFZYX): 2x2x2x1x1
        // Output (BFZYX): 1x2x2x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { 2, 2, 1, 1, 2 } });
        auto begin = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });
        auto end = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });
        auto strides = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        });
        set_values<int64_t>(begin, {
                0, 0, 0
        });
        set_values<int64_t>(end, {
                1, 2, 2
        });
        set_values<int64_t>(strides, {
                1, 1, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {1, 2, 2, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x1x1_2(bool is_caching_test) {
        // Input (BFZYX): 2x2x2x1x1
        // Output (BFZYX): 2x1x1x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { 2, 2, 1, 1, 2 } });
        auto begin = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });
        auto end = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });
        auto strides = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        });
        set_values<int64_t>(begin, {
                0, 0, 0
        });
        set_values<int64_t>(end, {
                2, 2, 2
        });
        set_values<int64_t>(strides, {
                1, 2, 2
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {2, 1, 1, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 4.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_full_negative_stride(bool is_caching_test) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): -1x-1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        set_values<int64_t>(begin, {
                0, 0, 0, 0
        });
        set_values<int64_t>(end, {
                2, 2, 2, 2
        });
        set_values<int64_t>(strides, {
                -1, -1, 1, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {2, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                12.f, 13.f, 14.f, 15.f, 8.f, 9.f, 10.f, 11.f, 4.f, 5.f, 6.f, 7.f, 0.f, 1.f, 2.f, 3.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_full_negative_stride_f_axis(bool is_caching_test) {
        // Input (BFYX): 1x2x2x2
        // Begin (BFYX): 0x0x0x0
        // End (BFYX): 1x2x2x2
        // Stride (BFYX): 1x-1x1x1
        // Output (BFYX): 1x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f, 7.0f
        });
        std::vector<int64_t> begin = {
                0, 0, 0, 0
        };
        std::vector<int64_t> end = {
                1, 2, 2, 2
        };
        std::vector<int64_t> strides = {
                1, -1, 1, 1
        };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin, end, strides, {}, {}, {}, {}, {}, {1, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                4.f, 5.f, 6.f, 7.f, 0.f, 1.f, 2.f, 3.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_full_negative_stride_negative_begin_f_axis(bool is_caching_test) {
        // Input (BFYX): 1x2x2x2
        // Begin (BFYX): 0x-1
        // End (BFYX): 100x-100
        // Stride (BFYX): 1x-1
        // Output (BFYX): 1x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f, 7.0f
        });
        std::vector<int64_t> begin = {
                0, -1
        };
        std::vector<int64_t> end= {
                100, -100
        };
        std::vector<int64_t> strides= {
                1, -1
        };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin, end, strides, {}, {}, {}, {}, {}, {1, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                4.f, 5.f, 6.f, 7.f, 0.f, 1.f, 2.f, 3.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_full_negative_stride_f_axis_clamp(bool is_caching_test) {
        // Input (BFYX): 1x2x2x2
        // Begin (BFYX): 0x100
        // End (BFYX): 100x-100
        // Stride (BFYX): 1x-1
        // Output (BFYX): 1x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f, 7.0f
        });
        std::vector<int64_t> begin = {
                0, 100
        };
        std::vector<int64_t> end= {
                100, -100
        };
        std::vector<int64_t> strides= {
                1, -1
        };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin, end, strides, {}, {}, {}, {}, {}, {1, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                4.f, 5.f, 6.f, 7.f, 0.f, 1.f, 2.f, 3.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x1x1_2_negative_all(bool is_caching_test) {
        // Input (BFZYX): 2x2x2x1x1
        // Output (BFZYX): 2x1x1x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { 2, 2, 1, 1, 2 } });
        auto begin = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });
        auto end = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });
        auto strides = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        });
        set_values<int64_t>(begin, {
                0, 0, 0
        });
        set_values<int64_t>(end, {
                2, 2, 2
        });
        set_values<int64_t>(strides, {
                1, 2, 2
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {2, 1, 1, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 4.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_1x3x1x3_negative(bool is_caching_test) {
        // Input (BFZYX):  1x3x1x3
        // Output (BFZYX): 1x1x1x3
        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 1, 3} });
        auto begin = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });
        auto end = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1 } });
        auto strides = engine.allocate_memory({ data_types::i64, format::bfyx, { 3, 1, 1, 1} });

        set_values(input, {
                0.0f, 1.0f, 2.0f,
                3.0f, 4.0f, 5.0f,
                6.0f, 7.0f, 8.0f,
        });

        set_values<int64_t>(begin, {
                0, -1, 0
        });
        set_values<int64_t>(end, {
                1, -2, 3
        });
        set_values<int64_t>(strides, {
                1, -1, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {},  {1, 1, 1, 3}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                6.0f, 7.0f, 8.0f,
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_1x1x1x3_negative(bool is_caching_test) {
        // Input (BFZYX):  1x1x1x3
        // Output (BFZYX): 1x1x1x3
        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 3, 1} });
        auto begin = engine.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
        auto end = engine.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1 } });
        auto strides = engine.allocate_memory({ data_types::i64, format::bfyx, { 4, 1, 1, 1} });

        set_values(input, {
                6.0f, 7.0f, 8.0f,
        });

        set_values<int64_t>(begin, {
                0, -1, 0, 0
        });
        set_values<int64_t>(end, {
                1, -2, 1, 3
        });
        set_values<int64_t>(strides, {
                1, -1, 1, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {},  {1, 1, 1, 3}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                6.0f, 7.0f, 8.0f,
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2_negative_begin_end_positive_stride(bool is_caching_test) {
        // Input (BFYX): 2x2x2x2
        // Begin (BFYX): -1x-1x-1x-1
        // End (BFYX): 2x2x2x2
        // Stride (BFYX): -1x-1x1x1
        // Output (BFYX): 2x2x2x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                8.0f, 9.0f, 10.0f, 11.0f,
                12.0f, 13.0f, 14.0f, 15.0f
        });
        set_values<int64_t>(begin, {
                -1, -1, 0, 0
        });
        set_values<int64_t>(end, {
                -3, -3, 2, 2
        });
        set_values<int64_t>(strides, {
                1, 1, 1, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {2, 2, 2, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                12.f, 13.f, 14.f, 15.f, 8.f, 9.f, 10.f, 11.f, 4.f, 5.f, 6.f, 7.f, 0.f, 1.f, 2.f, 3.f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_1x1x1x10_pos_begin_end_neg_stride2(bool is_caching_test) {
        // Input (BFYX): 1x1x1x10
        // Begin (BFYX): 0x0x0x3
        // End (BFYX): 1x1x1x6
        // Stride (BFYX): 1x1x1x-2
        // Output (BFYX): 1x1x1x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 10, 1 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f
        });
        set_values<int64_t>(begin, {
                0, 0, 0, 3
        });
        set_values<int64_t>(end, {
                1, 1, 1, 6
        });
        set_values<int64_t>(strides, {
                1, 1, 1, -2
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {1, 1, 1, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = { 5.0f, 3.0f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_1x1x1x10_neg_begin_end_neg_stride2(bool is_caching_test) {
        // Input (BFYX): 1x1x1x10
        // Begin (BFYX): 0x0x0x-5
        // End (BFYX): 1x1x1x-8
        // Stride (BFYX): 1x1x1x-2
        // Output (BFYX): 1x1x1x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 10, 1 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f
        });
        set_values<int64_t>(begin, {
                0, 0, 0, -5
        });
        set_values<int64_t>(end, {
                1, 1, 1, -8
        });
        set_values<int64_t>(strides, {
                1, 1, 1, -2
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, {}, {}, {}, {1, 1, 1, 2}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = { 5.0f, 3.0f };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), answers.size());
        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }
};

class strided_slice_gpu_four_inputs: public ::testing::Test {
public:
    void test_2x2x4x1_new_axis_mask(bool is_caching_test) {
        // Input (BFYX): 2x2x4x1
        // New_axis_mask: 1
        // Output (BFYX): 1x2x2x4

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 4 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        });
        set_values<int64_t>(begin, {
                1, 0, 1, 0
        });
        set_values<int64_t>(end, {
                2, 2, 4, 4
        });
        set_values<int64_t>(strides, {
                1, 1, 1, 2
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(input_layout("input2", begin->get_layout()));
        topology.add(input_layout("input3", end->get_layout()));
        topology.add(input_layout("input4", strides->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, { 1 }, {}, {}, {2, 2, 4, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);
        network->set_input_data("input2", begin);
        network->set_input_data("input3", end);
        network->set_input_data("input4", strides);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x1x1_new_axis_mask_2(bool is_caching_test) {
        // Input (BFYX): 2x2x1x1
        // New_axis_mask: 101
        // Output (BFYX): 1x2x1x2

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 4 }, data_types::i64, format::bfyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f
        });
        set_values<int64_t>(begin, {
                1, 0, 1, 0
        });
        set_values<int64_t>(end, {
                2, 2, 4, 4
        });
        set_values<int64_t>(strides, {
                1, 1, 1, 2
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(input_layout("input2", begin->get_layout()));
        topology.add(input_layout("input3", end->get_layout()));
        topology.add(input_layout("input4", strides->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {}, {}, { 1, 0, 1 }, {}, {}, {2, 2, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);
        network->set_input_data("input2", begin);
        network->set_input_data("input3", end);
        network->set_input_data("input4", strides);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }
};

class strided_slice_gpu_i8: public ::testing::Test {
public:
    void test_2x2x2x1x1(bool is_caching_test) {
        // Input (BFZYX): 2x2x2x1x1
        // Output (BFZYX): 1x2x2x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::i8, format::bfzyx, { 2, 2, 1, 1, 2 } });

        set_values<int8_t>(input, {
                0, 1, 2, 3, 4, 5, 6, 7
        });
        std::vector<int64_t> begin_data = { 0, 0, 0 };
        std::vector<int64_t> end_data = { 1, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {1, 2, 2, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<int8_t> answers = {
                0, 1, 2, 3
        };

        cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i) {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }

    void test_2x2x2x2x1x1(bool is_caching_test) {
        // Input (BFWZYX): 2x2x2x2x1x1
        // Output (BFWZYX): 1x2x2x2x1x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::i8, format::bfwzyx, { 2, 2, 1, 1, 2, 2 } });

        set_values<int8_t>(input, {
                0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15,
        });
        std::vector<int64_t> begin_data = { 0, 0, 0 };
        std::vector<int64_t> end_data = { 1, 2, 2 };
        std::vector<int64_t> strides_data = { 1, 1, 1 };

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, {1, 2, 2, 2, 1, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<int8_t> answers = {
                0, 1, 2, 3, 4, 5, 6, 7,
        };

        cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i) {
            ASSERT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }
};

class strided_slice_gpu_f32_i32: public ::testing::Test {
public:
    void test_1x1x1x8x1_new_axis_5d(bool is_caching_test) {
        // Input (BFYX): 1x8x1x1
        // Output (BFZYX): 1x1x1x8x1

        auto& engine = get_test_engine();
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 8, 1, 1 } });
        auto begin = engine.allocate_memory({ ov::PartialShape{ 5 }, data_types::i32, format::bfzyx });
        auto end = engine.allocate_memory({ ov::PartialShape{ 5 }, data_types::i32, format::bfzyx });
        auto strides = engine.allocate_memory({ ov::PartialShape{ 5 }, data_types::i32, format::bfzyx });

        set_values(input, {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        });
        set_values(begin, {
                0, 0, 0, 0, 0
        });
        set_values(end, {
                0, 0, 0, 0, 0
        });
        set_values(strides, {
                1, 1, 1, 1, 1
        });

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(data("input2", begin));
        topology.add(data("input3", end));
        topology.add(data("input4", strides));
        topology.add(strided_slice("strided_slice", input_info("input"), input_info("input2"), input_info("input3"), input_info("input4"), {1, 0, 0, 1, 0}, {1, 0, 0, 1, 0}, {0, 1, 1, 0, 1}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {1, 1, 1, 8, 1}));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "strided_slice");

        auto output = outputs.at("strided_slice").get_memory();

        std::vector<float> answers = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
        };

        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < answers.size(); ++i)
        {
            EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
        }
    }
};

TEST_F(strided_slice_gpu, test_2x2x2x2_full) {
    this->test_2x2x2x2_full(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full) {
    this->test_2x2x2x2_full(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x2x2_full) {
    this->test_2x2x2x2x2_full(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x2x2x2_full) {
    this->test_2x2x2x2x2x2_full(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_full_pad) {
    this->test_2x2x2x2_full_pad(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_ignore) {
    this->test_2x2x2x2_ignore(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_ignore) {
    this->test_2x2x2x2_ignore(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_single) {
    this->test_2x2x2x2_single(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_single) {
    this->test_2x2x2x2_single(false);
}

TEST_F(strided_slice_gpu, test_2x2x4x3_stride) {
    this->test_2x2x4x3_stride(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x4x3_stride) {
    this->test_2x2x4x3_stride(false);
}

TEST_F(strided_slice_gpu, test_2x2x4x4_part_stride) {
    this->test_2x2x4x4_part_stride(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x4x4_part_stride) {
    this->test_2x2x4x4_part_stride(false);
}

TEST_F(strided_slice_gpu, test_2x2x4x1_new_axis_mask) {
    this->test_2x2x4x1_new_axis_mask(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x4x1_new_axis_mask) {
    this->test_2x2x4x1_new_axis_mask(false);
}

TEST_F(strided_slice_gpu_four_inputs, test_2x2x4x1_new_axis_mask) {
    this->test_2x2x4x1_new_axis_mask(false);
}

TEST_F(strided_slice_gpu, test_2x2x1x1_new_axis_mask_2) {
    this->test_2x2x1x1_new_axis_mask_2(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x1x1_new_axis_mask_2) {
    this->test_2x2x1x1_new_axis_mask_2(false);
}

TEST_F(strided_slice_gpu_four_inputs, test_2x2x1x1_new_axis_mask_2) {
    this->test_2x2x1x1_new_axis_mask_2(false);
}

TEST_F(strided_slice_gpu, test_2x2x1x1) {
    this->test_2x2x1x1(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x1x1) {
    this->test_2x2x1x1(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x1x1) {
    this->test_2x2x2x1x1(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x1x1) {
    this->test_2x2x2x1x1(false);
}

TEST_F(strided_slice_gpu_i8, test_2x2x2x1x1) {
    this->test_2x2x2x1x1(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x1x1_2) {
    this->test_2x2x2x1x1_2(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x1x1_2) {
    this->test_2x2x2x1x1_2(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x2x1x1) {
    this->test_2x2x2x2x1x1(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x2x1x1_2) {
    this->test_2x2x2x2x1x1_2(false);
}

TEST_F(strided_slice_gpu_i8, test_2x2x2x2x1x1) {
    this->test_2x2x2x2x1x1(false);
}

TEST_F(strided_slice_gpu_f32_i32, test_1x1x1x8x1_new_axis_5d) {
    this->test_1x1x1x8x1_new_axis_5d(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_full_negative_stride) {
    this->test_2x2x2x2_full_negative_stride(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full_negative_stride) {
    this->test_2x2x2x2_full_negative_stride(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_full_negative_stride_pad) {
    this->test_2x2x2x2_full_negative_stride_pad(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full_negative_stride_f_axis) {
    this->test_2x2x2x2_full_negative_stride_f_axis(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full_negative_stride_negative_begin_f_axis) {
    this->test_2x2x2x2_full_negative_stride_negative_begin_f_axis(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full_negative_stride_f_axis_clamp) {
    this->test_2x2x2x2_full_negative_stride_f_axis_clamp(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x1x1_2_negative_all) {
    this->test_2x2x2x1x1_2_negative_all(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x1x1_2_negative_all) {
    this->test_2x2x2x1x1_2_negative_all(false);
}

TEST_F(strided_slice_gpu, test_2x2x2x1x1_2_negative_all_dynamic) {
    this->test_2x2x2x1x1_2_negative_all_dynamic();
}

TEST_F(strided_slice_gpu, test_2x2x2x1x1_2_negative_all_dynamic_begin) {
    this->test_2x2x2x1x1_2_negative_all_dynamic_begin();
}

TEST_F(strided_slice_gpu, test_2x2x2_all_dynamic_bcast) {
    this->test_2x2x2_all_dynamic_bcast();
}

class strided_slice_cpu_impl : public strided_slice_gpu {};
TEST_F(strided_slice_cpu_impl, test_2x2x2x1x1_2_negative_all_dynamic) {
    this->test_2x2x2x1x1_2_negative_all_dynamic(impl_types::cpu);
}

TEST_F(strided_slice_cpu_impl, test_2x2x2x1x1_2_negative_all_dynamic_begin) {
    this->test_2x2x2x1x1_2_negative_all_dynamic_begin(impl_types::cpu);
}

TEST_F(strided_slice_cpu_impl, test_2x2x2_all_dynamic_bcast) {
    this->test_2x2x2_all_dynamic_bcast(impl_types::cpu);
}

TEST_F(strided_slice_cpu_impl, test_2x2x1x1) {
    this->test_2x2x1x1(false, impl_types::cpu);
}

TEST_F(strided_slice_cpu_impl, test_2x2x1x1_disable_usm) {
    this->test_2x2x1x1(false, impl_types::cpu, true);
}

TEST_F(strided_slice_cpu_impl, test_2x2x2x1x1_2) {
    this->test_2x2x2x1x1_2(false, impl_types::cpu);
}

TEST_F(strided_slice_cpu_impl, test_2x2x2x2_single) {
    this->test_2x2x2x2_single(false, impl_types::cpu);
}

class strided_slice_cpu_impl_constants : public strided_slice_gpu_constants {};
TEST_F(strided_slice_cpu_impl_constants, test_2x2x2x2_full) {
    this->test_2x2x2x2_full(false, impl_types::cpu);
}

TEST_F(strided_slice_cpu_impl_constants, test_2x2x2x2_single) {
    this->test_2x2x2x2_single(false, impl_types::cpu);
}

TEST_F(strided_slice_cpu_impl_constants, test_2x2x4x3_stride) {
    this->test_2x2x4x3_stride(false, impl_types::cpu);
}

TEST_F(strided_slice_cpu_impl_constants, DISABLED_test_2x2x4x1_new_axis_mask) { // Issue 129991
    this->test_2x2x4x1_new_axis_mask(false, impl_types::cpu);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_F(strided_slice_gpu, test_2x2x2x2_full_cached) {
    this->test_2x2x2x2_full(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full_cached) {
    this->test_2x2x2x2_full(true);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_full_pad_cached) {
    this->test_2x2x2x2_full_pad(true);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_ignore_cached) {
    this->test_2x2x2x2_ignore(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_ignore_cached) {
    this->test_2x2x2x2_ignore(true);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_single_cached) {
    this->test_2x2x2x2_single(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_single_cached) {
    this->test_2x2x2x2_single(true);
}

TEST_F(strided_slice_gpu, test_2x2x4x3_stride_cached) {
    this->test_2x2x4x3_stride(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x4x3_stride_cached) {
    this->test_2x2x4x3_stride(true);
}

TEST_F(strided_slice_gpu, test_2x2x4x4_part_stride_cached) {
    this->test_2x2x4x4_part_stride(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x4x4_part_stride_cached) {
    this->test_2x2x4x4_part_stride(true);
}

TEST_F(strided_slice_gpu, test_2x2x4x1_new_axis_mask_cached) {
    this->test_2x2x4x1_new_axis_mask(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x4x1_new_axis_mask_cached) {
    this->test_2x2x4x1_new_axis_mask(true);
}

TEST_F(strided_slice_gpu_four_inputs, test_2x2x4x1_new_axis_mask_cached) {
    this->test_2x2x4x1_new_axis_mask(true);
}

TEST_F(strided_slice_gpu, test_2x2x1x1_new_axis_mask_2_cached) {
    this->test_2x2x1x1_new_axis_mask_2(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x1x1_new_axis_mask_2_cached) {
    this->test_2x2x1x1_new_axis_mask_2(true);
}

TEST_F(strided_slice_gpu_four_inputs, test_2x2x1x1_new_axis_mask_2_cached) {
    this->test_2x2x1x1_new_axis_mask_2(true);
}

TEST_F(strided_slice_gpu, test_2x2x1x1_cached) {
    this->test_2x2x1x1(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x1x1_cached) {
    this->test_2x2x1x1(true);
}

TEST_F(strided_slice_gpu, test_2x2x2x1x1_cached) {
    this->test_2x2x2x1x1(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x1x1_cached) {
    this->test_2x2x2x1x1(true);
}

TEST_F(strided_slice_gpu_i8, test_2x2x2x1x1_cached) {
    this->test_2x2x2x1x1(true);
}

TEST_F(strided_slice_gpu, test_2x2x2x1x1_2_cached) {
    this->test_2x2x2x1x1_2(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x1x1_2_cached) {
    this->test_2x2x2x1x1_2(true);
}

TEST_F(strided_slice_gpu_f32_i32, test_1x1x1x8x1_new_axis_5d_cached) {
    this->test_1x1x1x8x1_new_axis_5d(true);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_full_negative_stride_cached) {
    this->test_2x2x2x2_full_negative_stride(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full_negative_stride_cached) {
    this->test_2x2x2x2_full_negative_stride(true);
}

TEST_F(strided_slice_gpu, test_2x2x2x2_full_negative_stride_pad_cached) {
    this->test_2x2x2x2_full_negative_stride_pad(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full_negative_stride_f_axis_cached) {
    this->test_2x2x2x2_full_negative_stride_f_axis(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full_negative_stride_negative_begin_f_axis_cached) {
    this->test_2x2x2x2_full_negative_stride_negative_begin_f_axis(true);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_full_negative_stride_f_axis_clamp_cached) {
    this->test_2x2x2x2_full_negative_stride_f_axis_clamp(true);
}

TEST_F(strided_slice_gpu, test_2x2x2x1x1_2_negative_all_cached) {
    this->test_2x2x2x1x1_2_negative_all(true);
}
#endif
TEST_F(strided_slice_gpu_constants, test_2x2x2x1x1_2_negative_all_cached) {
    this->test_2x2x2x1x1_2_negative_all(true);
}

// test_2x2x2x2_full_activation
TEST_F(strided_slice_gpu, test_2x2x2x2_full_legacy_activation) {
    this->test_2x2x2x2_full_legacy_activation(true);
}


TEST_F(strided_slice_gpu, test_3d_all_dynamic_with_new_axis) {
    this->test_3d_all_dynamic_with_new_axis();
}

TEST_F(strided_slice_gpu, test_3d_all_dynamic_with_shrink_axis) {
    this->test_3d_all_dynamic_with_shrink_axis();
}

TEST_F(strided_slice_gpu_constants, test_1x3x1x3_negative) {
    this->test_1x3x1x3_negative(false);
}

TEST_F(strided_slice_gpu_constants, test_1x1x1x3_negative) {
    this->test_1x1x1x3_negative(false);
}

TEST_F(strided_slice_gpu_constants, test_2x2x2x2_negative_begin_end_positive_stride) {
    this->test_2x2x2x2_negative_begin_end_positive_stride(false);
}

TEST_F(strided_slice_gpu_constants, test_1x1x1x10_pos_begin_end_neg_stride2) {
    this->test_1x1x1x10_pos_begin_end_neg_stride2(false);
}

TEST_F(strided_slice_gpu_constants, test_1x1x1x10_neg_begin_end_neg_stride2) {
    this->test_1x1x1x10_neg_begin_end_neg_stride2(false);
}
