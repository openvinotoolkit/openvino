// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include "intel_gpu/primitives/concatenation.hpp"

using namespace cldnn;
using namespace ::tests;

class spatial_concatenate_f32_gpu: public ::testing::Test {
public:
    void test_test01(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

        set_values(input1, {
            1.0f, 2.0f,
            3.0f, 4.0f
        });

        set_values(input2, {
            5.0f, 6.0f,
            7.0f, 8.0f
        });

        const auto expected_output = std::vector<float>{
            1.0f, 2.0f, 5.0f, 6.0f,
            3.0f, 4.0f, 7.0f, 8.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        tpl.add(concatenation("conc", { input_info("in1"), input_info("in2") }, 3));

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1));
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0) + input2->get_layout().spatial(0));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_test02(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });

        set_values(input1, {
            1.0f, 2.0f,
            3.0f, 4.0f
        });

        set_values(input2, {
            5.0f, 6.0f,
            7.0f, 8.0f
        });

        const auto expected_output = std::vector<float>{
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        tpl.add(concatenation("conc", { input_info("in1"), input_info("in2") }, 2));

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0));
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1) + input2->get_layout().spatial(1));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_test03(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });

        set_values(input1, {
            1.0f, 2.0f,
            3.0f, 4.0f
        });

        set_values(input2, {
            5.0f, 6.0f,
            7.0f, 8.0f
        });

        const auto expected_output = std::vector<float>{
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 2.0f, 0.0f,
            0.0f, 3.0f, 4.0f, 0.0f,
            0.0f, 5.0f, 6.0f, 0.0f,
            0.0f, 7.0f, 8.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        auto concat = concatenation("conc", { input_info("in1"), input_info("in2") }, 2);
        concat.output_paddings = { padding({ 0, 0, 1, 1 }, 0.0f) };
        tpl.add(concat);

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0));
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1) + input2->get_layout().spatial(1));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_test04(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfyx,{ 1, 1, 2, 2 }, padding({ 0, 0, 0, 0 }, { 0, 0, 0, 1 }) });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfyx,{ 1, 1, 2, 2 }, padding({ 0, 0, 1, 0 }, 0.0f) });

        set_values(input1, {
            1.0f, 2.0f, 0.0f,
            3.0f, 4.0f, 0.0f
        });

        set_values(input2, {
            0.0f, 0.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,
            0.0f, 0.0f
        });

        const auto expected_output = std::vector<float>{
            0.0f, 0.0f, 1.0f, 2.0f, 5.0f, 6.0f,
            0.0f, 0.0f, 3.0f, 4.0f, 7.0f, 8.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        auto concat = concatenation("conc", { input_info("in1"), input_info("in2") }, 3);
        concat.output_paddings = {padding({ 0, 0, 0, 2 }, { 0, 0, 0, 0 }) };
        tpl.add(concat);

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1));
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0) + input2->get_layout().spatial(0));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_inputs_3(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
        memory::ptr input3 = engine.allocate_memory(layout{ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

        set_values(input1, {
            1.0f, 2.0f,
            3.0f, 4.0f
            });

        set_values(input2, {
            5.0f, 6.0f,
            7.0f, 8.0f
            });

        set_values(input3, {
            9.0f, 10.0f,
            11.0f, 12.0f
            });

        const auto expected_output = std::vector<float>{
            1.0f, 2.0f, 5.0f, 6.0f, 9.0f, 10.0f,
            3.0f, 4.0f, 7.0f, 8.0f, 11.0f, 12.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        tpl.add(input_layout("in3", input3->get_layout()));
        tpl.add(concatenation("conc", { input_info("in1"), input_info("in2"), input_info("in3") }, 3));

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);
        net->set_input_data("in3", input3);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1));
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0) + input2->get_layout().spatial(0) + input3->get_layout().spatial(0));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_inputs_3_uneven_axis_b(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfyx, { 3, 1, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
        memory::ptr input3 = engine.allocate_memory(layout{ data_types::f32, format::bfyx, { 2, 1, 2, 2 } });

        set_values(input1, {
            1.0f, 2.0f,
            3.0f, 4.0f,

            5.0f, 6.0f,
            7.0f, 8.0f,

            9.0f, 10.0f,
            11.0f, 12.0f
            });

        set_values(input2, {
            13.0f, 14.0f,
            15.0f, 16.0f
            });

        set_values(input3, {
            17.0f, 18.0f,
            19.0f, 20.0f,

            21.0f, 22.0f,
            23.0f, 24.0f
            });

        const auto expected_output = std::vector<float>{
            // input1
            1.0f, 2.0f,
            3.0f, 4.0f,

            5.0f, 6.0f,
            7.0f, 8.0f,

            9.0f, 10.0f,
            11.0f, 12.0f,

            // input2
            13.0f, 14.0f,
            15.0f, 16.0f,

            // input3
            17.0f, 18.0f,
            19.0f, 20.0f,

            21.0f, 22.0f,
            23.0f, 24.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        tpl.add(input_layout("in3", input3->get_layout()));
        tpl.add(concatenation("conc", { input_info("in1"), input_info("in2"), input_info("in3") }, 0));

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);
        net->set_input_data("in3", input3);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch() + input2->get_layout().batch() + input3->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1));
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_inputs3d_axis_x(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });

        set_values(input1, {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f
            });

        set_values(input2, {
            9.0f, 10.0f,
            11.0f, 12.0f,
            13.0f, 14.0f,
            15.0f, 16.0f
            });

        const auto expected_output = std::vector<float>{
            1.0f, 2.0f, 9.0f, 10.0f,
            3.0f, 4.0f, 11.0f, 12.0f,
            5.0f, 6.0f, 13.0f, 14.0f,
            7.0f, 8.0f, 15.0f, 16.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        tpl.add(concatenation("conc", { input_info("in1"), input_info("in2") }, 4));

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0) + input2->get_layout().spatial(0));
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1));
        ASSERT_EQ(output_layout.spatial(2), input1->get_layout().spatial(2));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_inputs3d_axis_y(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });

        set_values(input1, {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f
            });

        set_values(input2, {
            9.0f, 10.0f,
            11.0f, 12.0f,
            13.0f, 14.0f,
            15.0f, 16.0f
            });

        const auto expected_output = std::vector<float>{
            1.0f, 2.0f,
            3.0f, 4.0f,
            9.0f, 10.0f,
            11.0f, 12.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,
            13.0f, 14.0f,
            15.0f, 16.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        tpl.add(concatenation("conc", { input_info("in1"), input_info("in2") }, 3));

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0));
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1) + input2->get_layout().spatial(1));
        ASSERT_EQ(output_layout.spatial(2), input1->get_layout().spatial(2));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_inputs3d_axis_z(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });

        set_values(input1, {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f
            });

        set_values(input2, {
            9.0f, 10.0f,
            11.0f, 12.0f,
            13.0f, 14.0f,
            15.0f, 16.0f
            });

        const auto expected_output = std::vector<float>{
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f,
            13.0f, 14.0f,
            15.0f, 16.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        tpl.add(concatenation("conc", { input_info("in1"), input_info("in2") }, 2));

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0));
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1));
        ASSERT_EQ(output_layout.spatial(2), input1->get_layout().spatial(2) + input2->get_layout().spatial(2));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_inputs3d_axis_b(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 2, 1, 2, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });

        set_values(input1, {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,

            9.0f, 10.0f,
            11.0f, 12.0f,
            13.0f, 14.0f,
            15.0f, 16.0f
            });

        set_values(input2, {
            17.0f, 18.0f,
            19.0f, 20.0f,
            21.0f, 22.0f,
            23.0f, 24.0f
            });

        const auto expected_output = std::vector<float>{
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,

            9.0f, 10.0f,
            11.0f, 12.0f,
            13.0f, 14.0f,
            15.0f, 16.0f,

            17.0f, 18.0f,
            19.0f, 20.0f,
            21.0f, 22.0f,
            23.0f, 24.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        tpl.add(concatenation("conc", { input_info("in1"), input_info("in2") }, 0));

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch() + input2->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0));
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1));
        ASSERT_EQ(output_layout.spatial(2), input1->get_layout().spatial(2));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }

    void test_inputs3d_3_uneven_axis_b(bool is_caching_test) {
        auto& engine = get_test_engine();

        memory::ptr input1 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 3, 1, 2, 2, 2 } });
        memory::ptr input2 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });
        memory::ptr input3 = engine.allocate_memory(layout{ data_types::f32, format::bfzyx, { 2, 1, 2, 2, 2 } });

        set_values(input1, {
            //b0
            //z0
            //x0  //x1
            1.0f, 2.0f,//y0
            3.0f, 4.0f,//y1
            //z1
            5.0f, 6.0f,//y0
            7.0f, 8.0f,//y1

            //b1
            //z0
            9.0f, 10.0f,
            11.0f, 12.0f,
            //z1
            13.0f, 14.0f,
            15.0f, 16.0f,

            //b2
            //z0
            17.0f, 18.0f,
            19.0f, 20.0f,
            //z1
            12.0f, 22.0f,
            23.0f, 24.0f
            });

        set_values(input2, {
            //b0
            //z0
            //x0  //x1
            25.0f, 26.0f,//y0
            27.0f, 28.0f,//y1
            //z1
            29.0f, 30.0f,//y0
            31.0f, 32.0f//y1
            });

        set_values(input3, {
            //b0
            //z0
            //x0  //x1
            33.0f, 34.0f,//y0
            35.0f, 36.0f,//y1
            //z1
            37.0f, 38.0f,//y0
            39.0f, 40.0f,//y1

            //b1
            //z0
            41.0f, 42.0f,
            43.0f, 44.0f,
            //z1
            45.0f, 46.0f,
            47.0f, 48.0f
            });

        const auto expected_output = std::vector<float>{
            //input1
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f,
            13.0f, 14.0f,
            15.0f, 16.0f,
            17.0f, 18.0f,
            19.0f, 20.0f,
            12.0f, 22.0f,
            23.0f, 24.0f,

            //input2
            25.0f, 26.0f,
            27.0f, 28.0f,
            29.0f, 30.0f,
            31.0f, 32.0f,

            //input3
            33.0f, 34.0f,
            35.0f, 36.0f,
            37.0f, 38.0f,
            39.0f, 40.0f,
            41.0f, 42.0f,
            43.0f, 44.0f,
            45.0f, 46.0f,
            47.0f, 48.0f
        };

        topology tpl;
        tpl.add(input_layout("in1", input1->get_layout()));
        tpl.add(input_layout("in2", input2->get_layout()));
        tpl.add(input_layout("in3", input3->get_layout()));
        tpl.add(concatenation("conc", { input_info("in1"), input_info("in2"), input_info("in3") }, 0));

        cldnn::network::ptr net = get_network(engine, tpl, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        net->set_input_data("in1", input1);
        net->set_input_data("in2", input2);
        net->set_input_data("in3", input3);

        auto outputs = net->execute();
        ASSERT_TRUE(outputs.size() == 1 && outputs.count("conc") == 1);

        auto output_mem = outputs.at("conc").get_memory();
        auto output_layout = output_mem->get_layout();

        ASSERT_EQ(output_layout.batch(), input1->get_layout().batch() + input2->get_layout().batch() + input3->get_layout().batch());
        ASSERT_EQ(output_layout.feature(), input1->get_layout().feature());
        ASSERT_EQ(output_layout.spatial(0), input1->get_layout().spatial(0));
        ASSERT_EQ(output_layout.spatial(1), input1->get_layout().spatial(1));
        ASSERT_EQ(output_layout.spatial(2), input1->get_layout().spatial(2));

        ASSERT_EQ(output_mem->get_layout().get_linear_size(), expected_output.size());
        {
            cldnn::mem_lock<const float> out_ptr(output_mem, get_test_stream());

            size_t idx = 0;
            for (auto const& value : out_ptr)
            {
                ASSERT_FLOAT_EQ(value, expected_output[idx++]);
            }
        }
    }
};

TEST_F(spatial_concatenate_f32_gpu, test01) {
    this->test_test01(false);
}

TEST_F(spatial_concatenate_f32_gpu, test02) {
    this->test_test02(false);
}

TEST_F(spatial_concatenate_f32_gpu, test03) {
    this->test_test03(false);
}

TEST_F(spatial_concatenate_f32_gpu, test04) {
    this->test_test04(false);
}

TEST_F(spatial_concatenate_f32_gpu, inputs_3) {
    this->test_inputs_3(false);
}

TEST_F(spatial_concatenate_f32_gpu, inputs_3_uneven_axis_b) {
    this->test_inputs_3_uneven_axis_b(false);
}

TEST_F(spatial_concatenate_f32_gpu, inputs3d_axis_x) {
    this->test_inputs3d_axis_x(false);
}

TEST_F(spatial_concatenate_f32_gpu, inputs3d_axis_y) {
    this->test_inputs3d_axis_y(false);
}

TEST_F(spatial_concatenate_f32_gpu, inputs3d_axis_z) {
    this->test_inputs3d_axis_z(false);
}

TEST_F(spatial_concatenate_f32_gpu, inputs3d_axis_b) {
    this->test_inputs3d_axis_b(false);
}

TEST_F(spatial_concatenate_f32_gpu, inputs3d_3_uneven_axis_b) {
    this->test_inputs3d_3_uneven_axis_b(false);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_F(spatial_concatenate_f32_gpu, test01_cached) {
    this->test_test01(true);
}

TEST_F(spatial_concatenate_f32_gpu, test02_cached) {
    this->test_test02(true);
}

TEST_F(spatial_concatenate_f32_gpu, test03_cached) {
    this->test_test03(true);
}

TEST_F(spatial_concatenate_f32_gpu, test04_cached) {
    this->test_test04(true);
}

TEST_F(spatial_concatenate_f32_gpu, inputs_3_cached) {
    this->test_inputs_3(true);
}

TEST_F(spatial_concatenate_f32_gpu, inputs_3_uneven_axis_b_cached) {
    this->test_inputs_3_uneven_axis_b(true);
}

TEST_F(spatial_concatenate_f32_gpu, inputs3d_axis_x_cached) {
    this->test_inputs3d_axis_x(true);
}

TEST_F(spatial_concatenate_f32_gpu, inputs3d_axis_y_cached) {
    this->test_inputs3d_axis_y(true);
}

TEST_F(spatial_concatenate_f32_gpu, inputs3d_axis_z_cached) {
    this->test_inputs3d_axis_z(true);
}

TEST_F(spatial_concatenate_f32_gpu, inputs3d_axis_b_cached) {
    this->test_inputs3d_axis_b(true);
}
#endif
TEST_F(spatial_concatenate_f32_gpu, inputs3d_3_uneven_axis_b_cached) {
    this->test_inputs3d_3_uneven_axis_b(true);
}
