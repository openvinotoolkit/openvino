// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_update.hpp>
#include "scatter_update_inst.h"

#include <cstddef>

using namespace cldnn;
using namespace ::tests;


const auto plain_2d_format = format::bfyx;
const std::vector<format::type> formats2D{
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32
};

const auto plain_3d_format = format::bfzyx;
const std::vector<format::type> formats3D{
        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::bs_fs_zyx_bsv16_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16
};


template <typename T>
void test_d2411_axisB(bool is_caching_test) {
    //  Dictionary : 2x4x1x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x4x1x1
    //  Axis : 0
    //  Output : 2x4x1x1
    //  Input values in fp16

    //  Indexes:
    //  1.f, 0.f
    //
    //  Updates:
    //  1.f, 7.f, 2.f, 9.f,
    //  3.f, 6.f, 5.f, 4.f
    //
    //  Dictionary:
    //  0.f, 0.f, 0.f, 0.f,
    //  0.f, 0.f, 0.f, 0.f
    //
    //  Output:
    //  3.f, 6.f, 5.f, 4.f,
    //  1.f, 7.f, 2.f, 9.f

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {

        auto input1 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{2, 4, 1, 1}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 1, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{2, 4, 1, 1}}); // Updates
        auto axis = 0;

        set_values(input1, {
                T(0.0f), T(0.0f), T(0.0f), T(0.0f),
                T(0.0f), T(0.0f), T(0.0f), T(0.0f)
        });

        set_values(input2, {
                1.f, 0.f
        });

        set_values(input3, {
                T(1.0f), T(7.0f), T(2.0f), T(9.0f),
                T(3.0f), T(6.0f), T(5.0f), T(4.0f)
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f16));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::f16));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f16));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("InputDictionary", input1);
        network->set_input_data("InputText", input2);
        network->set_input_data("InputUpdates", input3);

        auto outputs = network->execute();


        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                3.f, 6.f, 5.f, 4.f,
                1.f, 7.f, 2.f, 9.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]))
                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_fp16, d2411_axisB) {
    test_d2411_axisB<ov::float16>(false);
}

TEST(scatter_update_gpu_fp32, d8111_axisB) {
    //  Dictionary : 8x1x1x1
    //  Indexes : 4x1x1x1
    //  Updates : 4x1x1x1
    //  Axis : 0
    //  Output : 8x1x1x1
    //  Input values in fp32

    //  Indexes:
    //  4.f, 3.f, 1.f, 7.f
    //
    //  Updates:
    //  9.f, 10.f, 11.f, 12.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f
    //
    //  Output:
    //  1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f


    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{8, 1, 1, 1}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{4, 1, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{4, 1, 1, 1}}); // Updates
        auto axis = 0;

        set_values(input1, {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
        });

        set_values(input2, {
                4.f, 3.f, 1.f, 7.f
        });

        set_values(input3, {
                9.0f, 10.0f, 11.0f, 12.0f
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f32));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::f32));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f32));

        network network(engine, topology, get_test_default_config(engine));


        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i])
                                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_fp16, d4311_axisB) {
    //  Dictionary : 4x3x1x1
    //  Indexes : 2x2x1x1
    //  Updates : 2x2x3x1
    //  Axis : 0
    //  Output : 4x3x1x1
    //  Input values in fp16

    //  Indexes:
    //  3.f, 1.f,
    //  2.f, 0.f
    //
    //  Updates:
    //  7.f, 7.f, 7.f,
    //  8.f, 8.f, 8.f,
    //
    //  6.f, 6.f, 6.f,
    //  9.f, 10.f, 11.f
    //
    //  Dictionary:
    //  1.f, 1.f, 1.f,
    //  2.f, 2.f, 2.f,
    //  0.f, 0.f, 0.f,
    //  3.f, 3.f, 3.f
    //
    //  Output:
    //  9.f, 10.f, 11.f,
    //  8.f, 8.f, 8.f,
    //  6.f, 6.f, 6.f,
    //  7.f, 7.f, 7.f

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{4, 3, 1, 1}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 2, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{2, 2, 1, 3}}); // Updates
        auto axis = 0;

        set_values(input1, {
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(2.0f), ov::float16(2.0f), ov::float16(2.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(3.0f), ov::float16(3.0f), ov::float16(3.0f)
        });

        set_values(input2, {
                3.f, 1.f,
                2.f, 0.f
        });

        set_values(input3, {
                ov::float16(7.0f), ov::float16(7.0f), ov::float16(7.0f),
                ov::float16(8.0f), ov::float16(8.0f), ov::float16(8.0f),

                ov::float16(6.0f), ov::float16(6.0f), ov::float16(6.0f),
                ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f)
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f16));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::f16));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f16));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                9.f, 10.f, 11.f,
                8.f, 8.f, 8.f,
                6.f, 6.f, 6.f,
                7.f, 7.f, 7.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]))
                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_fp16, d2521_axisF) {
    //  Dictionary : 2x5x2x1
    //  Indexes : 2x2x1x1
    //  Updates : 2x2x2x2
    //  Axis : 1
    //  Output : 2x5x2x1
    //  Input values in fp16

    //  Indexes:
    //  0.f, 2.f,
    //  4.f, 1.f
    //
    //  Updates:
    //  21.f, 31.f,
    //  41.f, 51.f,
    //
    //  61.f, 71.f,
    //  81.f, 91.f,
    //
    //  101.f, 111.f,
    //  121.f, 131.f,
    //
    //  141.f, 151.f,
    //  161.f, 171.f
    //
    //  Dictionary:
    //  0.f, 1.f,
    //  2.f, 3.f,
    //  4.f, 5.f,
    //  6.f, 7.f,
    //  8.f, 9.f,
    //
    //  10.f, 11.f,
    //  12.f, 13.f,
    //  14.f, 15.f,
    //  16.f, 17.f,
    //  18.f, 19.f
    //
    //  Output:
    //  21.f, 31.f,
    //  81.f, 91.f,
    //  41.f, 51.f,
    //  6.f, 7.f,
    //  61.f, 71.f,
    //
    //  101.f, 111.f,
    //  161.f, 171.f,
    //  121.f, 131.f,
    //  16.f, 17.f,
    //  141.f, 151.f

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{2, 5, 1, 2}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 2, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{2, 2, 2, 2}}); // Updates
        auto axis = 1;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(1.0f),
                ov::float16(2.0f), ov::float16(3.0f),
                ov::float16(4.0f), ov::float16(5.0f),
                ov::float16(6.0f), ov::float16(7.0f),
                ov::float16(8.0f), ov::float16(9.0f),

                ov::float16(10.0f), ov::float16(11.0f),
                ov::float16(12.0f), ov::float16(13.0f),
                ov::float16(14.0f), ov::float16(15.0f),
                ov::float16(16.0f), ov::float16(17.0f),
                ov::float16(18.0f), ov::float16(19.0f)
        });

        set_values(input2, {
                0.f, 2.f,
                4.f, 1.f
        });

        set_values(input3, {
                ov::float16(21.0f), ov::float16(31.0f),
                ov::float16(41.0f), ov::float16(51.0f),
                ov::float16(61.0f), ov::float16(71.0f),
                ov::float16(81.0f), ov::float16(91.0f),

                ov::float16(101.0f), ov::float16(111.0f),
                ov::float16(121.0f), ov::float16(131.0f),
                ov::float16(141.0f), ov::float16(151.0f),
                ov::float16(161.0f), ov::float16(171.0f)
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f16));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::f16));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f16));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                21.f, 31.f,
                81.f, 91.f,
                41.f, 51.f,
                6.f, 7.f,
                61.f, 71.f,

                101.f, 111.f,
                161.f, 171.f,
                121.f, 131.f,
                16.f, 17.f,
                141.f, 151.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]))
                                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_fp16, d2241_axisY) {
    //  Dictionary : 2x2x4x1
    //  Indexes : 2x2x1x1
    //  Updates : 2x2x2x2
    //  Axis : 2
    //  Output : 2x2x4x1
    //  Input values in fp16

    //  Indexes:
    //  0.f, 2.f,
    //  3.f, 1.f
    //
    //  Updates:
    //  0.f, 20.f,
    //  30.f, 40.f,
    //
    //  50.f, 60.f,
    //  70.f, 80.f,
    //
    //  90.f, 100.f,
    //  110.f, 120.f,
    //
    //  130.f, 140.f,
    //  150.f, 160.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f,
    //  5.f, 6.f, 7.f, 8.f,
    //  11.f, 10.f, 11.f, 12.f,
    //  13.f, 14.f, 15.f, 16.f
    //
    //  Output:
    //  0.f, 40.f, 20.f, 30.f,
    //  50.f, 80.f, 60.f, 70.f,
    //  90.f, 120.f, 100.f, 110.f,
    //  130.f, 160.f, 140.f, 150.f

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{2, 2, 1, 4}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 2, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{2, 2, 2, 2}}); // Updates
        auto axis = 2;

        set_values(input1, {
                ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
                ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
                ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),
                ov::float16(13.0f), ov::float16(14.0f), ov::float16(15.0f), ov::float16(16.0f)
        });

        set_values(input2, {
                0.f, 2.f,
                3.f, 1.f
        });

        set_values(input3, {
                ov::float16(0.0f), ov::float16(20.0f),
                ov::float16(30.0f), ov::float16(40.0f),
                ov::float16(50.0f), ov::float16(60.0f),
                ov::float16(70.0f), ov::float16(80.0f),

                ov::float16(90.0f), ov::float16(100.0f),
                ov::float16(110.0f), ov::float16(120.0f),
                ov::float16(130.0f), ov::float16(140.0f),
                ov::float16(150.0f), ov::float16(160.0f)
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f16));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::f16));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f16));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.f, 40.f, 20.f, 30.f,
                50.f, 80.f, 60.f, 70.f,
                90.f, 120.f, 100.f, 110.f,
                130.f, 160.f, 140.f, 150.f
        };


        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]))
                                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_fp16, d8x2x20x1_axisB) {
    //  Dictionary : 8x2x20x1
    //  Indexes : 2x3x1x1
    //  Updates : 2x3x2x20
    //  Axis : 0
    //  Output : 8x2x20x1
    //  Input values in fp16

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{8, 2, 1, 20}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 3, 1, 1}});  // Indexes
        auto input3 = engine.allocate_memory({data_types::f16, plain_2d_format, tensor{2, 3, 20, 2}}); // Updates
        auto axis = 0;

        set_values(input1, {
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),

                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),

                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),

                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),

                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),

                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),

                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),

                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f), ov::float16(0.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
                ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f)
        });

        set_values(input2, {
                3.f, 1.f, 6.f,
                2.f, 7.f, 4.f
        });

        set_values(input3, {
                ov::float16(0), ov::float16(1), ov::float16(2), ov::float16(3), ov::float16(4), ov::float16(5), ov::float16(6), ov::float16(7),
                ov::float16(8), ov::float16(9), ov::float16(10), ov::float16(11), ov::float16(12), ov::float16(13), ov::float16(14), ov::float16(15),
                ov::float16(16), ov::float16(17), ov::float16(18), ov::float16(19),
                ov::float16(20), ov::float16(21), ov::float16(22), ov::float16(23), ov::float16(24), ov::float16(25), ov::float16(26), ov::float16(27),
                ov::float16(28), ov::float16(29), ov::float16(30), ov::float16(31), ov::float16(32), ov::float16(33), ov::float16(34), ov::float16(35),
                ov::float16(36), ov::float16(37), ov::float16(38), ov::float16(39),

                ov::float16(40), ov::float16(41), ov::float16(42), ov::float16(43), ov::float16(44), ov::float16(45), ov::float16(46), ov::float16(47),
                ov::float16(48), ov::float16(49), ov::float16(50), ov::float16(51), ov::float16(52), ov::float16(53), ov::float16(54), ov::float16(55),
                ov::float16(56), ov::float16(57), ov::float16(58), ov::float16(59),
                ov::float16(60), ov::float16(61), ov::float16(62), ov::float16(63), ov::float16(64), ov::float16(65), ov::float16(66), ov::float16(67),
                ov::float16(68), ov::float16(69), ov::float16(70), ov::float16(71), ov::float16(72), ov::float16(73), ov::float16(74), ov::float16(75),
                ov::float16(76), ov::float16(77), ov::float16(78), ov::float16(79),

                ov::float16(80), ov::float16(81), ov::float16(82), ov::float16(83), ov::float16(84), ov::float16(85), ov::float16(86), ov::float16(87),
                ov::float16(88), ov::float16(89), ov::float16(90), ov::float16(91), ov::float16(92), ov::float16(93), ov::float16(94), ov::float16(95),
                ov::float16(96), ov::float16(97), ov::float16(98), ov::float16(99),
                ov::float16(100), ov::float16(101), ov::float16(102), ov::float16(103), ov::float16(104), ov::float16(105), ov::float16(106),
                ov::float16(107), ov::float16(108), ov::float16(109), ov::float16(110), ov::float16(111), ov::float16(112), ov::float16(113),
                ov::float16(114), ov::float16(115), ov::float16(116), ov::float16(117), ov::float16(118), ov::float16(119),

                ov::float16(120), ov::float16(121), ov::float16(122), ov::float16(123), ov::float16(124), ov::float16(125), ov::float16(126),
                ov::float16(127), ov::float16(128), ov::float16(129), ov::float16(130), ov::float16(131), ov::float16(132), ov::float16(133),
                ov::float16(134), ov::float16(135), ov::float16(136), ov::float16(137), ov::float16(138), ov::float16(139),
                ov::float16(140), ov::float16(141), ov::float16(142), ov::float16(143), ov::float16(144), ov::float16(145), ov::float16(146),
                ov::float16(147), ov::float16(148), ov::float16(149), ov::float16(150), ov::float16(151), ov::float16(152), ov::float16(153),
                ov::float16(154), ov::float16(155), ov::float16(156), ov::float16(157), ov::float16(158), ov::float16(159),

                ov::float16(160), ov::float16(161), ov::float16(162), ov::float16(163), ov::float16(164), ov::float16(165), ov::float16(166),
                ov::float16(167), ov::float16(168), ov::float16(169), ov::float16(170), ov::float16(171), ov::float16(172), ov::float16(173),
                ov::float16(174), ov::float16(175), ov::float16(176), ov::float16(177), ov::float16(178), ov::float16(179),
                ov::float16(180), ov::float16(181), ov::float16(182), ov::float16(183), ov::float16(184), ov::float16(185), ov::float16(186),
                ov::float16(187), ov::float16(188), ov::float16(189), ov::float16(190), ov::float16(191), ov::float16(192), ov::float16(193),
                ov::float16(194), ov::float16(195), ov::float16(196), ov::float16(197), ov::float16(198), ov::float16(199),

                ov::float16(200), ov::float16(201), ov::float16(202), ov::float16(203), ov::float16(204), ov::float16(205), ov::float16(206),
                ov::float16(207), ov::float16(208), ov::float16(209), ov::float16(210), ov::float16(211), ov::float16(212), ov::float16(213),
                ov::float16(214), ov::float16(215), ov::float16(216), ov::float16(217), ov::float16(218), ov::float16(219),
                ov::float16(220), ov::float16(221), ov::float16(222), ov::float16(223), ov::float16(224), ov::float16(225), ov::float16(226),
                ov::float16(227), ov::float16(228), ov::float16(229), ov::float16(230), ov::float16(231), ov::float16(232), ov::float16(233),
                ov::float16(234), ov::float16(235), ov::float16(236), ov::float16(237), ov::float16(238), ov::float16(239)
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f16));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::f16));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f16));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,

                40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f,
                57.f, 58.f, 59.f,
                60.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 67.f, 68.f, 69.f, 70.f, 71.f, 72.f, 73.f, 74.f, 75.f, 76.f,
                77.f, 78.f, 79.f,

                120.f, 121.f, 122.f, 123.f, 124.f, 125.f, 126.f, 127.f, 128.f, 129.f, 130.f, 131.f, 132.f, 133.f, 134.f,
                135.f, 136.f, 137.f, 138.f, 139.f,
                140.f, 141.f, 142.f, 143.f, 144.f, 145.f, 146.f, 147.f, 148.f, 149.f, 150.f, 151.f, 152.f, 153.f, 154.f,
                155.f, 156.f, 157.f, 158.f, 159.f,

                0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f,
                19.f,
                20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f,
                37.f, 38.f, 39.f,

                200.f, 201.f, 202.f, 203.f, 204.f, 205.f, 206.f, 207.f, 208.f, 209.f, 210.f, 211.f, 212.f, 213.f, 214.f,
                215.f, 216.f, 217.f, 218.f, 219.f,
                220.f, 221.f, 222.f, 223.f, 224.f, 225.f, 226.f, 227.f, 228.f, 229.f, 230.f, 231.f, 232.f, 233.f, 234.f,
                235.f, 236.f, 237.f, 238.f, 239.f,

                0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,

                80.f, 81.f, 82.f, 83.f, 84.f, 85.f, 86.f, 87.f, 88.f, 89.f, 90.f, 91.f, 92.f, 93.f, 94.f, 95.f, 96.f,
                97.f, 98.f, 99.f,
                100.f, 101.f, 102.f, 103.f, 104.f, 105.f, 106.f, 107.f, 108.f, 109.f, 110.f, 111.f, 112.f, 113.f, 114.f,
                115.f, 116.f, 117.f, 118.f, 119.f,

                160.f, 161.f, 162.f, 163.f, 164.f, 165.f, 166.f, 167.f, 168.f, 169.f, 170.f, 171.f, 172.f, 173.f, 174.f,
                175.f, 176.f, 177.f, 178.f, 179.f,
                180.f, 181.f, 182.f, 183.f, 184.f, 185.f, 186.f, 187.f, 188.f, 189.f, 190.f, 191.f, 192.f, 193.f, 194.f,
                195.f, 196.f, 197.f, 198.f, 199.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]))
                                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_fp32, d2214_axisX) {
    //  Dictionary : 2x2x1x4
    //  Indexes : 3x1x1x1
    //  Updates : 2x2x1x3
    //  Axis : 3
    //  Output : 2x2x1x4
    //  Input values in fp32

    //  Indexes:
    //  2.f, 0.f, 3.f
    //
    //  Updates:
    //  20.f, 30.f, 40.f,
    //  50.f, 60.f, 70.f,
    //
    //  80.f, 90.f, 100.f,
    //  110.f, 120.f, 130.f
    //
    //  Dictionary:
    //  0.f, 1.f, 2.f, 3.f,
    //  4.f, 5.f, 6.f, 7.f,
    //
    //  8.f, 9.f, 10.f, 11.f,
    //  12.f, 13.f, 14.f, 15.f
    //
    //  Output:
    //  30.f, 1.f, 20.f, 40.f,
    //  60.f, 5.f, 50.f, 70.f,
    //
    //  90.f, 9.f, 80.f, 100.f,
    //  120.f, 13.f, 110.f, 130.f

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 2, 4, 1}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{3, 1, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 2, 3, 1}}); // Updates
        auto axis = 3;

        set_values(input1, {
                0.f, 1.f, 2.f, 3.f,
                4.f, 5.f, 6.f, 7.f,
                8.f, 9.f, 10.f, 11.f,
                12.f, 13.f, 14.f, 15.f
        });

        set_values(input2, {
                2.f, 0.f, 3.f
        });

        set_values(input3, {
                20.f, 30.f, 40.f,
                50.f, 60.f, 70.f,
                80.f, 90.f, 100.f,
                110.f, 120.f, 130.f
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f32));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::f32));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f32));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                30.f, 1.f, 20.f, 40.f,
                60.f, 5.f, 50.f, 70.f,
                90.f, 9.f, 80.f, 100.f,
                120.f, 13.f, 110.f, 130.f
        };


        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i])
                                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_int32, d6211_axisB) {
    //  Dictionary : 6x2x1x1
    //  Indexes : 1x2x2x1
    //  Updates : 1x2x2x2
    //  Axis : 0
    //  Output : 6x2x1x1
    //  Input values in int32

    //  Indexes:
    //  3,   1,
    //  5,   2
    //
    //  Updates:
    //  20,  30,
    //  40,  50
    //
    //  60,  70,
    //  80,  90
    //
    //  Dictionary:
    //  1,   2,
    //  3,   4,
    //  5,   6,
    //  7,   8,
    //  9,   10,
    //  11,  12
    //
    //  Output:
    //   1,  2,
    //  40,  50,
    //  80,  90,
    //  20,  30,
    //   9,  10,
    //  60,  70

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::i32, plain_2d_format, tensor{6, 2, 1, 1}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::i32, plain_2d_format, tensor{1, 2, 1, 2}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::i32, plain_2d_format, tensor{1, 2, 2, 2}}); // Updates
        auto axis = 0;

        set_values(input1, {
                1, 2,
                3, 4,
                5, 6,
                7, 8,
                9, 10,
                11, 12
        });

        set_values(input2, {
                3, 1,
                5, 2
        });

        set_values(input3, {
                20, 30,
                40, 50,
                60, 70,
                80, 90
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::i32));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::i32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::i32));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::i32));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<int> output_ptr(output, get_test_stream());

        std::vector<int> expected_results = {
                1, 2,
                40, 50,
                80, 90,
                20, 30,
                9, 10,
                60, 70
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i])
                                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_int32, d3151_axisY) {
    //  Dictionary : 3x1x5x1
    //  Indexes : 2x2x1x1
    //  Updates : 3x1x2x2
    //  Axis : 2
    //  Output : 3x1x5x1
    //  Input values in int32

    //  Indexes:
    //  3,   2,
    //  0,   4
    //
    //  Updates:
    //  200,  20,
    //  30,  40
    //
    //  50,  60,
    //  70,  80
    //
    //  90,  100,
    //  110,  120
    //
    //  Dictionary:
    //  1,  2,  3,  4,  5,
    //  6,  7,  8,  9,  10,
    //  11, 12, 13, 14, 15
    //
    //  Output:
    //   30,  1,  20, 200, 40,
    //   70,  6,  60,  50, 80,
    //   110, 11, 100, 90, 120

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::i32, plain_2d_format, tensor{3, 1, 1, 5}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::i32, plain_2d_format, tensor{2, 2, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::i32, plain_2d_format, tensor{3, 1, 2, 2}}); // Updates
        auto axis = 2;

        set_values(input1, {
                0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
                10, 11, 12, 13, 14
        });

        set_values(input2, {
                3, 2,
                0, 4
        });

        set_values(input3, {
                200, 20,
                30, 40,
                50, 60,
                70, 80,
                90, 100,
                110, 120
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::i32));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::i32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::i32));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::i32));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<int> output_ptr(output, get_test_stream());

        std::vector<int> expected_results = {
                30, 1, 20, 200, 40,
                70, 6, 60, 50, 80,
                110, 11, 100, 90, 120
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i])
                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_fp32, d24111_axisF_bfzyx) {
    //  Dictionary : 2x4x1x1
    //  Indexes : 1x1x1x2
    //  Updates : 2x1x1x1x2
    //  Axis : 1
    //  Output : 2x4x1x1x1
    //  Input values in fp32

    //  Indexes:
    //  2.f, 0.f
    //
    //  Updates:
    //  1.f, 2.f,
    //  3.f, 4.f
    //
    //  Dictionary:
    //  0.f, 0.f, 0.f, 0.f,
    //  0.f, 0.f, 0.f, 0.f
    //
    //  Output:
    //  2.f, 0.f, 1.f, 0.f,
    //  4.f, 0.f, 3.f, 0.f

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        for (const auto target_format_3d: formats3D) {
            auto input1 = engine.allocate_memory(
                    {data_types::f32, plain_2d_format, tensor{2, 4, 1, 1}});      // Dictionary
            auto input2 = engine.allocate_memory(
                    {data_types::f32, plain_2d_format, tensor{1, 1, 2, 1}});      // Indexes
            auto input3 = engine.allocate_memory({data_types::f32, plain_3d_format, tensor{2, 1, 1, 2, 1}});  // Updates
            auto axis = 1;

            set_values(input1, {
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f
            });

            set_values(input2, {
                    2.f, 0.f
            });

            set_values(input3, {
                    1.0f, 2.0f,
                    3.0f, 4.0f
            });

            topology topology;
            topology.add(input_layout("InputDictionary", input1->get_layout()));
            topology.add(input_layout("InputText", input2->get_layout()));
            topology.add(input_layout("InputUpdates", input3->get_layout()));
            topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f32));
            topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
            topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format_3d, data_types::f32));
            topology.add(
                    scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
            );
            topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f32));

            network network(engine, topology, get_test_default_config(engine));

            network.set_input_data("InputDictionary", input1);
            network.set_input_data("InputText", input2);
            network.set_input_data("InputUpdates", input3);

            auto outputs = network.execute();

            auto output = outputs.at("out").get_memory();
            cldnn::mem_lock<float> output_ptr(output, get_test_stream());

            std::vector<float> expected_results = {
                    2.f, 0.f, 1.f, 0.f,
                    4.f, 0.f, 3.f, 0.f
            };

            for (size_t i = 0; i < expected_results.size(); ++i) {
                ASSERT_EQ(expected_results[i], output_ptr[i])
                                    << "i=" << i
                                    << ", target_format_2d=" << target_format
                                    << ", target_format_3d=" << target_format_3d;
            }
        }
    }
}

TEST(scatter_update_gpu_int32, d121251_bfwzyx_axisB) {
    //  Dictionary : 1x2x1x2x5x1
    //  Indexes : 1x2x2x1
    //  Updates : 1x2x1x2x2x2
    //  Axis : 4
    //  Output : 1x2x1x2x5x1
    //  Input values in int32

    //  Indexes:
    //  2,   1,
    //  0,   4
    //
    //  Updates:
    //  20,  30,
    //  40,  50
    //
    //  60,  70,
    //  80,  90,
    //
    //  100,  110,
    //  120,  130,
    //
    //  140,  150,
    //  160,  170
    //
    //  Dictionary:
    //  0, 1, 2, 3, 4,
    //  5, 6, 7, 8, 9,
    //  10, 11, 12, 13, 14,
    //  15, 16, 17, 18, 19
    //
    //  Output:
    //  40,  30,   20,  3, 50,
    //  80,  70,   60,  8, 90,
    //  120, 110, 100, 13, 130,
    //  160, 150, 140, 18, 170

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory(
                {data_types::i32, format::bfwzyx, tensor{batch(1), feature(2), spatial(1, 5, 2, 1)}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::i32, plain_2d_format,
                                              tensor{2, 2, 1, 1}});                                       // Indexes
        auto input3 = engine.allocate_memory(
                {data_types::i32, format::bfwzyx, tensor{batch(1), feature(2), spatial(2, 2, 2, 1)}}); // Updates
        auto axis = 4;

        set_values(input1, {
                0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
                10, 11, 12, 13, 14,
                15, 16, 17, 18, 19
        });

        set_values(input2, {
                2, 1,
                0, 4
        });

        set_values(input3, {
                20, 30,
                40, 50,
                60, 70,
                80, 90,
                100, 110,
                120, 130,
                140, 150,
                160, 170
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::i32));
        topology.add(
                scatter_update("scatter_update", input_info("InputDictionary"), input_info("TextReordered"), input_info("InputUpdates"), axis)
        );

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("scatter_update").get_memory();
        cldnn::mem_lock<int> output_ptr(output, get_test_stream());

        std::vector<int> expected_results = {
                40, 30, 20, 3, 50,
                80, 70, 60, 8, 90,
                120, 110, 100, 13, 130,
                160, 150, 140, 18, 170
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i])
                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_fp32, d21511_bfzyx_axisX) {
    //  Dictionary : 2x1x5x1x1
    //  Indexes : 2x1x2x1
    //  Updates : 2x1x2x1x2
    //  Axis : 2
    //  Output : 2x1x5x1x1
    //  Input values in fp32

    //  Indexes:
    //  3.f, 4.f
    //  0.f, 1.f
    //
    //  Updates:
    //  10.f, 20.f,
    //  30.f, 40.f,
    //  50.f, 60.f,
    //  70.f, 80.f
    //
    //  Dictionary:
    //  0.f, 1.f, 2.f, 3.f, 4.f
    //  5.f, 6.f, 7.f, 8.f, 9.f
    //
    //  Output:
    //  30.f, 40.f, 2.f, 10.f, 20.f,
    //  70.f, 80.f, 7.f, 50.f, 60.f
    //

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        for (const auto target_format_3d: formats3D) {
            auto input1 = engine.allocate_memory(
                    {data_types::f32, plain_3d_format, tensor{2, 1, 1, 1, 5}}); // Dictionary
            auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 2, 1, 1}});     // Indices
            auto input3 = engine.allocate_memory({data_types::f32, plain_3d_format, tensor{2, 1, 1, 2, 2}}); // Updates
            auto axis = 2;

            set_values(input1, {
                    0.f, 1.f, 2.f, 3.f, 4.f,
                    5.f, 6.f, 7.f, 8.f, 9.f
            });

            set_values(input2, {
                    3.f, 4.f,
                    0.f, 1.f
            });

            set_values(input3, {
                    10.f, 20.f,
                    30.f, 40.f,
                    50.f, 60.f,
                    70.f, 80.f
            });

            topology topology;
            topology.add(input_layout("InputDictionary", input1->get_layout()));
            topology.add(input_layout("InputText", input2->get_layout()));
            topology.add(input_layout("InputUpdates", input3->get_layout()));
            topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format_3d, data_types::f32));
            topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
            topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format_3d, data_types::f32));
            topology.add(
                    scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
            );
            topology.add(reorder("out", input_info("scatter_update"), plain_3d_format, data_types::f32));

            network network(engine, topology, get_test_default_config(engine));


            network.set_input_data("InputDictionary", input1);
            network.set_input_data("InputText", input2);
            network.set_input_data("InputUpdates", input3);

            auto outputs = network.execute();

            auto output = outputs.at("out").get_memory();
            cldnn::mem_lock<float> output_ptr(output, get_test_stream());

            std::vector<float> expected_results = {
                    30.f, 40.f, 2.f, 10.f, 20.f,
                    70.f, 80.f, 7.f, 50.f, 60.f
            };

            for (size_t i = 0; i < expected_results.size(); ++i) {
                ASSERT_EQ(expected_results[i], output_ptr[i])
                                    << "i=" << i
                                    << ", target_format_2d=" << target_format
                                    << ", target_format_3d=" << target_format_3d;
            }
        }
    }
}

TEST(scatter_update_gpu_fp32, d1252_axisY_bfwzyx) {
    //  Dictionary : 1x2x5x2
    //  Indexes : 2x1x2x1
    //  Updates : 1x2x2x1x2x2
    //  Axis : 2
    //  Output : 1x2x5x2
    //  Input values in fp32

    //  Indexes:
    //  2.f, 0.f,
    //  3.f, 4.f
    //
    //  Updates:
    //  20.f, 30.f,
    //  40.f, 50.f
    //
    //  60.f, 70.f,
    //  80.f, 90.f
    //
    //  100.f, 110.f,
    //  120.f, 130.f
    //
    //  140.f, 150.f,
    //  160.f, 170.f
    //
    //  Dictionary:
    //  0.f, 1.f,     2.f, 3.f,     4.f, 5.f,     6.f, 7.f,     8.f, 9.f,
    //  10.f, 11.f,   12.f, 13.f,   14.f, 15.f,   16.f, 17.f,   18.f, 19.f
    //
    //  Output:
    //  40.f, 50.f,     2.f, 3.f,     20.f, 30.f,     60.f, 70.f,     80.f, 90.f,
    //  120.f, 130.f,   12.f, 13.f,   100.f, 110.f,   140.f, 150.f,   160.f, 170.f

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{1, 2, 2,
                                                                                       5}});                                         // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format,
                                              tensor{2, 1, 1, 2}});                                         // Indices
        auto input3 = engine.allocate_memory(
                {data_types::f32, format::bfwzyx, tensor{batch(1), feature(2), spatial(2, 2, 1, 2)}});  // Updates
        auto axis = 2;

        set_values(input1, {
                0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
                10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f
        });

        set_values(input2, {
                2.f, 0.f,
                3.f, 4.f
        });

        set_values(input3, {
                20.f, 30.f,
                40.f, 50.f,

                60.f, 70.f,
                80.f, 90.f,

                100.f, 110.f,
                120.f, 130.f,

                140.f, 150.f,
                160.f, 170.f
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f32));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("InputUpdates"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f32));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                40.f, 50.f, 2.f, 3.f, 20.f, 30.f, 60.f, 70.f, 80.f, 90.f,
                120.f, 130.f, 12.f, 13.f, 100.f, 110.f, 140.f, 150.f, 160.f, 170.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i])
                                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_int32, d2115_axisX_bfwzyx) {
    //  Dictionary : 2x1x1x5
    //  Indexes : 2x2x1x1
    //  Updates : 2x1x1x2x2x1
    //  Axis : 3
    //  Output : 2x1x1x5
    //  Input values in int32

    //  Indexes:
    //  2,   1,
    //  4,   3
    //
    //  Updates:
    //  20,  30,
    //  40,  50
    //
    //  60,  70,
    //  80,  90
    //
    //  Dictionary:
    //  0, 1, 2, 3, 4,
    //  5, 6, 7, 8, 9
    //
    //  Output:
    //  0,  30,   20,  50, 40,
    //  5,  70,   60,  90, 80

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::i32, plain_2d_format,
                                              tensor{2, 1, 5, 1}});                                        // Dictionary
        auto input2 = engine.allocate_memory({data_types::i32, plain_2d_format,
                                              tensor{2, 2, 1, 1}});                                       // Indexes
        auto input3 = engine.allocate_memory(
                {data_types::i32, format::bfwzyx, tensor{batch(2), feature(1), spatial(1, 2, 2, 1)}}); // Updates
        auto axis = 3;

        set_values(input1, {
                0, 1, 2, 3, 4,
                5, 6, 7, 8, 9
        });

        set_values(input2, {
                2, 1,
                4, 3
        });

        set_values(input3, {
                20, 30,
                40, 50,
                60, 70,
                80, 90
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));

        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::i32));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::i32));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("InputUpdates"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::i32));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<int> output_ptr(output, get_test_stream());

        std::vector<int> expected_results = {
                0, 30, 20, 50, 40,
                5, 70, 60, 90, 80
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i])
                                << "i=" << i << ", target_format=" << target_format;
        }
    }
}

template <typename T>
void test_d21214_bfzyx_axisX_bfwzyx(bool is_caching_test) {
    //  Dictionary : 2x1x2x1x4
    //  Indexes : 1x3x1x1
    //  Updates : 2x1x2x1x1x3
    //  Axis : 4
    //  Output : 2x1x2x1x4
    //  Input values in fp16

    //  Indexes:
    //  3.f, 2.f, 1.f
    //
    //  Updates:
    //  20.f, 30.f, 40.f,
    //  50.f, 60.f, 70.f,
    //  80.f, 90.f, 100.f,
    //  110.f, 120.f, 130.f
    //
    //  Dictionary:
    //  0.f, 1.f, 2.f, 3.f,
    //  4.f, 5.f, 6.f, 7.f,
    //  8.f, 9.f, 10.f, 11.f,
    //  12.f, 13.f, 14.f, 15.f
    //
    //  Output:
    //  0.f, 40.f, 30.f, 20.f,
    //  4.f, 70.f, 60.f, 50.f,
    //  8.f, 100.f, 90.f, 80.f,
    //  12.f, 130.f, 120.f, 110.f

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        for (const auto target_format_3d: formats3D) {
            auto input1 = engine.allocate_memory({data_types::f16, plain_3d_format, tensor{2, 1, 4, 1,
                                                                                           2}});                                    // Dictionary
            auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{1, 3, 1,
                                                                                           1}});                                        // Indexes
            auto input3 = engine.allocate_memory(
                    {data_types::f16, format::bfwzyx, tensor{batch(2), feature(1), spatial(3, 1, 1, 2)}}); // Updates
            auto axis = -1;

            set_values(input1, {
                    T(0.0f), T(1.0f), T(2.0f), T(3.0f),
                    T(4.0f), T(5.0f), T(6.0f), T(7.0f),
                    T(8.0f), T(9.0f), T(10.0f), T(11.0f),
                    T(12.0f), T(13.0f), T(14.0f), T(15.0f)
            });

            set_values(input2, {
                    3.f, 2.f, 1.f
            });

            set_values(input3, {
                    T(20.0f), T(30.0f), T(40.0f),
                    T(50.0f), T(60.0f), T(70.0f),
                    T(80.0f), T(90.0f), T(100.0f),
                    T(110.0f), T(120.0f), T(130.0f)
            });

            topology topology;
            topology.add(input_layout("InputDictionary", input1->get_layout()));
            topology.add(input_layout("InputText", input2->get_layout()));
            topology.add(input_layout("InputUpdates", input3->get_layout()));
            topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format_3d, data_types::f16));
            topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
            topology.add(
                    scatter_update("scatter_update", input_info("InputDictionary"), input_info("InputText"), input_info("InputUpdates"), axis)
            );
            topology.add(reorder("out", input_info("scatter_update"), plain_3d_format, data_types::f16));

            cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

            network->set_input_data("InputDictionary", input1);
            network->set_input_data("InputText", input2);
            network->set_input_data("InputUpdates", input3);

            auto outputs = network->execute();

            auto output = outputs.at("out").get_memory();
            cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

            std::vector<float> expected_results = {
                    0.f, 40.f, 30.f, 20.f,
                    4.f, 70.f, 60.f, 50.f,
                    8.f, 100.f, 90.f, 80.f,
                    12.f, 130.f, 120.f, 110.f
            };

            for (size_t i = 0; i < expected_results.size(); ++i) {
                ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]))
                                    << "i=" << i
                                    << ", target_format_2d=" << target_format
                                    << ", target_format_3d=" << target_format_3d;
            }
        }
    }
}

TEST(scatter_update_gpu_fp16, d21214_bfzyx_axisX_bfwzyx) {
    test_d21214_bfzyx_axisX_bfwzyx<ov::float16>(false);
}

TEST(scatter_update_gpu_fp32, dynamic) {
    //  Dictionary : 1x2x5x2
    //  Indexes : 2x1x2x1
    //  Updates : 1x2x2x1x2x2
    //  Axis : 2
    //  Output : 1x2x5x2
    //  Input values in fp32

    auto& engine = get_test_engine();

    auto input1_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto input2_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto input3_layout = layout{ ov::PartialShape::dynamic(6), data_types::f32, format::bfyx };

    auto input1 = engine.allocate_memory({{1, 2, 5, 2},       data_types::f32, format::bfyx});   // Dictionary
    auto input2 = engine.allocate_memory({{2, 1, 2, 1},       data_types::f32, format::bfyx});   // Indices
    auto input3 = engine.allocate_memory({{1, 2, 2, 1, 2, 2}, data_types::f32, format::bfwzyx}); // Updates
    auto axis = 2;

    set_values(input1, {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f
    });

    set_values(input2, {
        2.f, 0.f,
        3.f, 4.f
    });

    set_values(input3, {
        20.f, 30.f,
        40.f, 50.f,
        60.f, 70.f,
        80.f, 90.f,
        100.f, 110.f,
        120.f, 130.f,
        140.f, 150.f,
        160.f, 170.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1_layout));
    topology.add(input_layout("InputText", input2_layout));
    topology.add(input_layout("InputUpdates", input3_layout));

    topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), format::bfyx, data_types::f32));
    topology.add(reorder("TextReordered", input_info("InputText"), format::bfyx, data_types::f32));
    topology.add(scatter_update("scatter_update",
                                input_info("DictionaryReordered"),
                                input_info("TextReordered"),
                                input_info("InputUpdates"),
                                axis)
    );
    topology.add(reorder("out", input_info("scatter_update"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto inst = network.get_primitive("scatter_update");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    auto output = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        40.f, 50.f, 2.f, 3.f, 20.f, 30.f, 60.f, 70.f, 80.f, 90.f,
        120.f, 130.f, 12.f, 13.f, 100.f, 110.f, 140.f, 150.f, 160.f, 170.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_fp32, mixed_input_with_dynamic_static) {
    //  Dictionary : 1x2x5x2
    //  Indexes : 2x1x2x1
    //  Updates : 1x2x2x1x2x2
    //  Axis : 2
    //  Output : 1x2x5x2
    //  Input values in fp32

    auto& engine = get_test_engine();

    auto input1_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto input2_layout = layout{ ov::PartialShape{2, 1, 2, 1}, data_types::f32, format::bfyx };
    auto input3_layout = layout{ ov::PartialShape::dynamic(6), data_types::f32, format::bfyx };

    auto input1 = engine.allocate_memory({{1, 2, 5, 2},       data_types::f32, format::bfyx});   // Dictionary
    auto input2 = engine.allocate_memory({{2, 1, 2, 1},       data_types::f32, format::bfyx});   // Indices
    auto input3 = engine.allocate_memory({{1, 2, 2, 1, 2, 2}, data_types::f32, format::bfwzyx}); // Updates
    auto axis = 2;

    set_values(input1, {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f
    });

    set_values(input2, {
        2.f, 0.f,
        3.f, 4.f
    });

    set_values(input3, {
        20.f, 30.f,
        40.f, 50.f,
        60.f, 70.f,
        80.f, 90.f,
        100.f, 110.f,
        120.f, 130.f,
        140.f, 150.f,
        160.f, 170.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1_layout));
    topology.add(input_layout("InputText", input2_layout));
    topology.add(input_layout("InputUpdates", input3_layout));

    topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), format::bfyx, data_types::f32));
    topology.add(reorder("TextReordered", input_info("InputText"), format::bfyx, data_types::f32));
    topology.add(scatter_update("scatter_update",
                                input_info("DictionaryReordered"),
                                input_info("TextReordered"),
                                input_info("InputUpdates"),
                                axis)
    );
    topology.add(reorder("out", input_info("scatter_update"), format::bfyx, data_types::f32));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto inst = network.get_primitive("scatter_update");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    auto output = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        40.f, 50.f, 2.f, 3.f, 20.f, 30.f, 60.f, 70.f, 80.f, 90.f,
        120.f, 130.f, 12.f, 13.f, 100.f, 110.f, 140.f, 150.f, 160.f, 170.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_cpu_impl_fp32, dynamic) {
    //  Dictionary : 1x2x5x2
    //  Indexes : 2x1x2x1
    //  Updates : 1x2x2x1x2x2
    //  Axis : 2
    //  Output : 1x2x5x2
    //  Input values in fp32

    auto& engine = get_test_engine();

    auto input1_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto input2_layout = layout{ ov::PartialShape::dynamic(4), data_types::i32, format::bfyx };
    auto input3_layout = layout{ ov::PartialShape::dynamic(6), data_types::f32, format::bfyx };

    auto input1 = engine.allocate_memory({{1, 2, 5, 2},       data_types::f32, format::bfyx});   // Dictionary
    auto input2 = engine.allocate_memory({{2, 2},             data_types::i32, format::bfyx});   // Indices
    auto input3 = engine.allocate_memory({{1, 2, 2, 2, 2},    data_types::f32, format::bfzyx}); // Updates
    auto axis = 2;

    set_values(input1, {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f
    });

    set_values(input2, {
        2, 0,
        3, 4
    });

    set_values(input3, {
        20.f, 30.f,
        40.f, 50.f,
        60.f, 70.f,
        80.f, 90.f,
        100.f, 110.f,
        120.f, 130.f,
        140.f, 150.f,
        160.f, 170.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1_layout));
    topology.add(input_layout("InputText", input2_layout));
    topology.add(input_layout("InputUpdates", input3_layout));

    topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), format::bfyx, data_types::f32));
    topology.add(reorder("TextReordered", input_info("InputText"), format::bfyx, data_types::i32));
    topology.add(scatter_update("scatter_update",
                                input_info("DictionaryReordered"),
                                input_info("TextReordered"),
                                input_info("InputUpdates"),
                                axis)
    );
    topology.add(reorder("out", input_info("scatter_update"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"scatter_update", {format::bfyx, "", impl_types::cpu}} }));
    network network(engine, topology, config);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto inst = network.get_primitive("scatter_update");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    auto output = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        40.f, 50.f, 2.f, 3.f, 20.f, 30.f, 60.f, 70.f, 80.f, 90.f,
        120.f, 130.f, 12.f, 13.f, 100.f, 110.f, 140.f, 150.f, 160.f, 170.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(scatter_update_gpu_fp16, d21214_bfzyx_axisX_bfwzyx_cached) {
    test_d21214_bfzyx_axisX_bfwzyx<ov::float16>(true);
}
#endif
TEST(scatter_update_gpu_fp16, d2411_axisB_cached) {
    test_d2411_axisB<ov::float16>(true);
}

TEST(scatter_update_gpu_fp32, output_padding) {
    //  Dictionary : 2x2x1x4
    //  Indexes : 3x1x1x1
    //  Updates : 2x2x1x3
    //  Axis : 3
    //  Output : 2x2x1x4
    //  Input values in fp32

    //  Indexes:
    //  2.f, 0.f, 3.f
    //
    //  Updates:
    //  20.f, 30.f, 40.f,
    //  50.f, 60.f, 70.f,
    //
    //  80.f, 90.f, 100.f,
    //  110.f, 120.f, 130.f
    //
    //  Dictionary:
    //  0.f, 1.f, 2.f, 3.f,
    //  4.f, 5.f, 6.f, 7.f,
    //
    //  8.f, 9.f, 10.f, 11.f,
    //  12.f, 13.f, 14.f, 15.f
    //
    //  Output:
    //  30.f, 1.f, 20.f, 40.f,
    //  60.f, 5.f, 50.f, 70.f,
    //
    //  90.f, 9.f, 80.f, 100.f,
    //  120.f, 13.f, 110.f, 130.f

    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 2, 4, 1}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{3, 1, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{2, 2, 3, 1}}); // Updates
        auto axis = 3;

        set_values(input1, {
                0.f, 1.f, 2.f, 3.f,
                4.f, 5.f, 6.f, 7.f,
                8.f, 9.f, 10.f, 11.f,
                12.f, 13.f, 14.f, 15.f
        });

        set_values(input2, {
                2.f, 0.f, 3.f
        });

        set_values(input3, {
                20.f, 30.f, 40.f,
                50.f, 60.f, 70.f,
                80.f, 90.f, 100.f,
                110.f, 120.f, 130.f
        });

        padding output_padding = padding({1,1}, {1,1});

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f32));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::f32));
        auto scatter_upd = scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis);
        scatter_upd.output_paddings = { output_padding };
        topology.add(scatter_upd);
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f32));

        network network(engine, topology, get_test_default_config(engine));

        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                30.f,   1.f,    20.f,   40.f,
                60.f,   5.f,    50.f,   70.f,
                90.f,   9.f,    80.f,   100.f,
                120.f,  13.f,   110.f,  130.f
        };


        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i])
                            << "i=" << i << ", target_format=" << target_format;
        }
    }
}

TEST(scatter_update_gpu_fp32, d8111_axisB_first_iteration_kernel_check) {
    //  Dictionary : 8x1x1x1
    //  Indexes : 4x1x1x1
    //  Updates : 4x1x1x1
    //  Axis : 0
    //  Output : 8x1x1x1
    //  Input values in fp32

    //  Indexes:
    //  4.f, 3.f, 1.f, 7.f
    //
    //  Updates:
    //  9.f, 10.f, 11.f, 12.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f
    //
    //  Output:
    //  1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f


    auto& engine = get_test_engine();

    for(const auto target_format : formats2D) {
        auto input1 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{8, 1, 1, 1}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{1, 1, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{1, 1, 1, 1}}); // Updates
        auto axis = 0;

        set_values(input1, {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
        });

        set_values(input2, {
                4.f
        });

        set_values(input3, {
                9.0f
        });

        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("DictionaryReordered", input_info("InputDictionary"), target_format, data_types::f32));
        topology.add(reorder("TextReordered", input_info("InputText"), target_format, data_types::f32));
        topology.add(reorder("UpdatesReordered", input_info("InputUpdates"), target_format, data_types::f32));
        topology.add(
                scatter_update("scatter_update", input_info("DictionaryReordered"), input_info("TextReordered"), input_info("UpdatesReordered"), axis)
        );
        topology.add(reorder("out", input_info("scatter_update"), plain_2d_format, data_types::f32));

        network network(engine, topology, get_test_default_config(engine));


        network.set_input_data("InputDictionary", input1);
        network.set_input_data("InputText", input2);
        network.set_input_data("InputUpdates", input3);

        // allocate new output memory
        layout out_l = network.get_output_memory("out")->get_layout();
        //auto output_mem = engine.allocate_memory({data_types::f32, plain_2d_format, tensor{8, 1, 1, 1}});
        auto output_mem = engine.allocate_memory(out_l);
        set_values(output_mem, {
                -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f
        });

        network.set_output_memory("out", output_mem);
        auto outputs = network.execute();

        auto output = outputs.at("out").get_memory();
        ASSERT_TRUE(engine.is_the_same_buffer(*output_mem, *output));
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 6.0f, 7.0f, 8.0f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            ASSERT_EQ(expected_results[i], output_ptr[i])
                                << "i=" << i << ", target_format=" << target_format;
        }
    }
}
