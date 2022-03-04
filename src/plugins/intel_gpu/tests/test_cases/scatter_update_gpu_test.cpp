// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_update.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;


TEST(scatter_update_gpu_fp16, d2411_axisB) {
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

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 4, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 4, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f)
    });

    set_values(input2, {
        1.f, 0.f
    });

    set_values(input3, {
        FLOAT16(1.0f), FLOAT16(7.0f), FLOAT16(2.0f), FLOAT16(9.0f),
        FLOAT16(3.0f), FLOAT16(6.0f), FLOAT16(5.0f), FLOAT16(4.0f)
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);


    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        3.f, 6.f, 5.f, 4.f,
        1.f, 7.f, 2.f, 9.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
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

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 8, 1, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 4, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, { 4, 1, 1, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

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
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);


    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 11.f, 3.f, 10.f, 9.f, 6.f, 7.f, 12.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
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

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 3, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 3 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(2.0f), FLOAT16(2.0f),
        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(3.0f), FLOAT16(3.0f), FLOAT16(3.0f)
    });

    set_values(input2, {
        3.f, 1.f,
        2.f, 0.f
    });

    set_values(input3, {
        FLOAT16(7.0f), FLOAT16(7.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(8.0f), FLOAT16(8.0f),

        FLOAT16(6.0f), FLOAT16(6.0f), FLOAT16(6.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f)
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        9.f, 10.f, 11.f,
        8.f, 8.f, 8.f,
        6.f, 6.f, 6.f,
        7.f, 7.f, 7.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
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

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 5, 1, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 2, 2 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_f;

    set_values(input1, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f),
        FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f),

        FLOAT16(10.0f), FLOAT16(11.0f),
        FLOAT16(12.0f), FLOAT16(13.0f),
        FLOAT16(14.0f), FLOAT16(15.0f),
        FLOAT16(16.0f), FLOAT16(17.0f),
        FLOAT16(18.0f), FLOAT16(19.0f)
    });

    set_values(input2, {
        0.f, 2.f,
        4.f, 1.f
    });

    set_values(input3, {
        FLOAT16(21.0f), FLOAT16(31.0f),
        FLOAT16(41.0f), FLOAT16(51.0f),
        FLOAT16(61.0f), FLOAT16(71.0f),
        FLOAT16(81.0f), FLOAT16(91.0f),

        FLOAT16(101.0f), FLOAT16(111.0f),
        FLOAT16(121.0f), FLOAT16(131.0f),
        FLOAT16(141.0f), FLOAT16(151.0f),
        FLOAT16(161.0f), FLOAT16(171.0f)
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
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
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
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

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 4 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 2, 2 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_y;

    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
        FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f), FLOAT16(16.0f)
    });

    set_values(input2, {
        0.f, 2.f,
        3.f, 1.f
    });

    set_values(input3, {
        FLOAT16(0.0f), FLOAT16(20.0f),
        FLOAT16(30.0f), FLOAT16(40.0f),
        FLOAT16(50.0f), FLOAT16(60.0f),
        FLOAT16(70.0f), FLOAT16(80.0f),

        FLOAT16(90.0f), FLOAT16(100.0f),
        FLOAT16(110.0f), FLOAT16(120.0f),
        FLOAT16(130.0f), FLOAT16(140.0f),
        FLOAT16(150.0f), FLOAT16(160.0f)
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 40.f, 20.f, 30.f,
        50.f, 80.f, 60.f, 70.f,
        90.f, 120.f, 100.f, 110.f,
        130.f, 160.f, 140.f, 150.f
    };


    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
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

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 8, 2, 1, 20 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 3, 1, 1 } });  // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 20, 2 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

    set_values(input1, {
        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),

        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),

        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),

        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),

        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),

        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),

        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),

        FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f),FLOAT16(1.0f), FLOAT16(1.0f)
    });

    set_values(input2, {
        3.f, 1.f, 6.f,
        2.f, 7.f, 4.f
    });

    set_values(input3, {
        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(3), FLOAT16(4), FLOAT16(5), FLOAT16(6), FLOAT16(7), FLOAT16(8), FLOAT16(9), FLOAT16(10), FLOAT16(11), FLOAT16(12), FLOAT16(13), FLOAT16(14), FLOAT16(15), FLOAT16(16), FLOAT16(17), FLOAT16(18), FLOAT16(19),
        FLOAT16(20), FLOAT16(21), FLOAT16(22), FLOAT16(23), FLOAT16(24), FLOAT16(25), FLOAT16(26), FLOAT16(27), FLOAT16(28), FLOAT16(29), FLOAT16(30), FLOAT16(31), FLOAT16(32), FLOAT16(33), FLOAT16(34), FLOAT16(35), FLOAT16(36), FLOAT16(37), FLOAT16(38), FLOAT16(39),

        FLOAT16(40), FLOAT16(41), FLOAT16(42), FLOAT16(43), FLOAT16(44), FLOAT16(45), FLOAT16(46), FLOAT16(47), FLOAT16(48), FLOAT16(49), FLOAT16(50), FLOAT16(51), FLOAT16(52), FLOAT16(53), FLOAT16(54), FLOAT16(55), FLOAT16(56), FLOAT16(57), FLOAT16(58), FLOAT16(59),
        FLOAT16(60), FLOAT16(61), FLOAT16(62), FLOAT16(63), FLOAT16(64), FLOAT16(65), FLOAT16(66), FLOAT16(67), FLOAT16(68), FLOAT16(69), FLOAT16(70), FLOAT16(71), FLOAT16(72), FLOAT16(73), FLOAT16(74), FLOAT16(75), FLOAT16(76), FLOAT16(77), FLOAT16(78), FLOAT16(79),

        FLOAT16(80), FLOAT16(81), FLOAT16(82), FLOAT16(83), FLOAT16(84), FLOAT16(85), FLOAT16(86), FLOAT16(87), FLOAT16(88), FLOAT16(89), FLOAT16(90), FLOAT16(91), FLOAT16(92), FLOAT16(93), FLOAT16(94), FLOAT16(95), FLOAT16(96), FLOAT16(97), FLOAT16(98), FLOAT16(99),
        FLOAT16(100), FLOAT16(101), FLOAT16(102), FLOAT16(103), FLOAT16(104), FLOAT16(105), FLOAT16(106), FLOAT16(107), FLOAT16(108), FLOAT16(109), FLOAT16(110), FLOAT16(111), FLOAT16(112), FLOAT16(113), FLOAT16(114), FLOAT16(115), FLOAT16(116), FLOAT16(117), FLOAT16(118), FLOAT16(119),

        FLOAT16(120), FLOAT16(121), FLOAT16(122), FLOAT16(123), FLOAT16(124), FLOAT16(125), FLOAT16(126), FLOAT16(127), FLOAT16(128), FLOAT16(129), FLOAT16(130), FLOAT16(131), FLOAT16(132), FLOAT16(133), FLOAT16(134), FLOAT16(135), FLOAT16(136), FLOAT16(137), FLOAT16(138), FLOAT16(139),
        FLOAT16(140), FLOAT16(141), FLOAT16(142), FLOAT16(143), FLOAT16(144), FLOAT16(145), FLOAT16(146), FLOAT16(147), FLOAT16(148), FLOAT16(149), FLOAT16(150), FLOAT16(151), FLOAT16(152), FLOAT16(153), FLOAT16(154), FLOAT16(155), FLOAT16(156), FLOAT16(157), FLOAT16(158), FLOAT16(159),

        FLOAT16(160), FLOAT16(161), FLOAT16(162), FLOAT16(163), FLOAT16(164), FLOAT16(165), FLOAT16(166), FLOAT16(167), FLOAT16(168), FLOAT16(169), FLOAT16(170), FLOAT16(171), FLOAT16(172), FLOAT16(173), FLOAT16(174), FLOAT16(175), FLOAT16(176), FLOAT16(177), FLOAT16(178), FLOAT16(179),
        FLOAT16(180), FLOAT16(181), FLOAT16(182), FLOAT16(183), FLOAT16(184), FLOAT16(185), FLOAT16(186), FLOAT16(187), FLOAT16(188), FLOAT16(189), FLOAT16(190), FLOAT16(191), FLOAT16(192), FLOAT16(193), FLOAT16(194), FLOAT16(195), FLOAT16(196), FLOAT16(197), FLOAT16(198), FLOAT16(199),

        FLOAT16(200), FLOAT16(201), FLOAT16(202), FLOAT16(203), FLOAT16(204), FLOAT16(205), FLOAT16(206), FLOAT16(207), FLOAT16(208), FLOAT16(209), FLOAT16(210), FLOAT16(211), FLOAT16(212), FLOAT16(213), FLOAT16(214), FLOAT16(215), FLOAT16(216), FLOAT16(217), FLOAT16(218), FLOAT16(219),
        FLOAT16(220), FLOAT16(221), FLOAT16(222), FLOAT16(223), FLOAT16(224), FLOAT16(225), FLOAT16(226), FLOAT16(227), FLOAT16(228), FLOAT16(229), FLOAT16(230), FLOAT16(231), FLOAT16(232), FLOAT16(233), FLOAT16(234), FLOAT16(235), FLOAT16(236), FLOAT16(237), FLOAT16(238), FLOAT16(239)
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f , 1.f, 1.f, 1.f, 1.f, 1.f,

        40.f, 41.f,  42.f,  43.f,  44.f,  45.f,  46.f,  47.f,  48.f,  49.f,  50.f,  51.f,  52.f,  53.f,  54.f,  55.f,  56.f, 57.f,  58.f,  59.f,
        60.f,  61.f,  62.f,  63.f,  64.f,  65.f,  66.f, 67.f,  68.f,  69.f, 70.f,  71.f,  72.f,  73.f,  74.f,  75.f,  76.f,  77.f,  78.f,  79.f,

        120.f, 121.f, 122.f, 123.f, 124.f, 125.f, 126.f, 127.f, 128.f, 129.f,130.f, 131.f, 132.f, 133.f, 134.f, 135.f, 136.f, 137.f, 138.f, 139.f,
        140.f, 141.f, 142.f, 143.f, 144.f, 145.f, 146.f, 147.f, 148.f, 149.f,150.f, 151.f, 152.f, 153.f, 154.f, 155.f, 156.f, 157.f, 158.f, 159.f,

        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f, 32.f, 33.f, 34.f , 35.f, 36.f, 37.f, 38.f, 39.f,

        200.f, 201.f, 202.f, 203.f, 204.f, 205.f, 206.f, 207.f, 208.f, 209.f,210.f, 211.f, 212.f, 213.f, 214.f, 215.f, 216.f, 217.f, 218.f, 219.f,
        220.f, 221.f, 222.f, 223.f, 224.f, 225.f, 226.f,227.f, 228.f, 229.f, 230.f, 231.f, 232.f, 233.f, 234.f, 235.f, 236.f, 237.f, 238.f, 239.f,

        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f , 1.f, 1.f, 1.f, 1.f, 1.f,

        80.f,  81.f,  82.f,  83.f,  84.f,  85.f,  86.f, 87.f,  88.f,  89.f, 90.f,  91.f,  92.f,  93.f,  94.f,  95.f,  96.f,  97.f,  98.f,  99.f,
        100.f, 101.f, 102.f, 103.f, 104.f, 105.f, 106.f, 107.f, 108.f, 109.f,110.f, 111.f, 112.f, 113.f, 114.f, 115.f, 116.f, 117.f, 118.f, 119.f,

        160.f, 161.f, 162.f, 163.f, 164.f, 165.f, 166.f, 167.f, 168.f, 169.f, 170.f, 171.f, 172.f, 173.f, 174.f, 175.f, 176.f, 177.f, 178.f, 179.f,
        180.f, 181.f, 182.f, 183.f, 184.f, 185.f, 186.f, 187.f, 188.f, 189.f, 190.f, 191.f, 192.f, 193.f, 194.f, 195.f, 196.f, 197.f, 198.f, 199.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
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

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 4, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 3, 1 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_x;

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
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        30.f, 1.f, 20.f, 40.f,
        60.f, 5.f, 50.f, 70.f,
        90.f, 9.f, 80.f, 100.f,
        120.f, 13.f, 110.f, 130.f
    };


    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
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

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, { 6, 2, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 2, 1, 2 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 2, 2, 2 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_b;

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
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
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
        EXPECT_EQ(expected_results[i], output_ptr[i]);
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

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 1, 1, 5 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 1, 2, 2 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_y;

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
        200,  20,
        30,   40,
        50,   60,
        70,   80,
        90,  100,
        110,  120
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
        30, 1, 20, 200, 40,
        70, 6, 60, 50, 80,
        110, 11, 100, 90, 120
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
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

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 4, 1, 1 } });      // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 1 } });      // Indexes
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfzyx, { 2, 1, 1, 2, 1 } });  // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_f;

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
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        2.f, 0.f, 1.f, 0.f,
        4.f, 0.f, 3.f, 0.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
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

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfwzyx, tensor{ batch(1), feature(2), spatial(1, 5, 2, 1) }}); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, { 2, 2, 1, 1 } });                                       // Indexes
    auto input3 = engine.allocate_memory({ data_types::i32, format::bfwzyx, tensor{ batch(1), feature(2), spatial(2, 2, 2, 1) }}); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_y;

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
        100,  110,
        120,  130,
        140,  150,
        160,  170
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
        40,  30,   20,  3, 50,
        80,  70,   60,  8, 90,
        120, 110, 100, 13, 130,
        160, 150, 140, 18, 170
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
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

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfzyx, { 2, 1, 1, 1, 5 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 1, 1 } });     // Indices
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfzyx, { 2, 1, 1, 2, 2 } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_z;

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
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);


    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        30.f, 40.f, 2.f, 10.f, 20.f,
        70.f, 80.f, 7.f, 50.f, 60.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
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

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 5 } });                                         // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 1, 2 } });                                         // Indices
    auto input3 = engine.allocate_memory({ data_types::f32, format::bfwzyx, tensor{ batch(1), feature(2), spatial(2, 2, 1, 2) } });  // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_y;

    set_values(input1, {
        0.f, 1.f,     2.f, 3.f,     4.f, 5.f,     6.f, 7.f,     8.f, 9.f,
        10.f, 11.f,   12.f, 13.f,   14.f, 15.f,   16.f, 17.f,   18.f, 19.f
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
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        40.f, 50.f,     2.f, 3.f,     20.f, 30.f,     60.f, 70.f,     80.f, 90.f,
        120.f, 130.f,   12.f, 13.f,   100.f, 110.f,   140.f, 150.f,   160.f, 170.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
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

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, { 2, 1, 5, 1 }});                                        // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, { 2, 2, 1, 1 } });                                       // Indexes
    auto input3 = engine.allocate_memory({ data_types::i32, format::bfwzyx, tensor{ batch(2), feature(1), spatial(1, 2, 2, 1) }}); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_x;

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
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
        0,  30,  20,  50, 40,
        5,  70,  60,  90, 80
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_update_gpu_fp16, d21214_bfzyx_axisX_bfwzyx) {
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

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 1, 4, 1, 2 } });                                    // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 3, 1, 1 } });                                        // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, tensor{ batch(2), feature(1), spatial(3, 1, 1, 2) } }); // Updates
    auto axis = cldnn::scatter_update::scatter_update_axis::along_x;

    set_values(input1, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f),
        FLOAT16(8.0f), FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f),
        FLOAT16(12.0f), FLOAT16(13.0f), FLOAT16(14.0f), FLOAT16(15.0f)
    });

    set_values(input2, {
        3.f, 2.f, 1.f
    });

    set_values(input3, {
        FLOAT16(20.0f), FLOAT16(30.0f), FLOAT16(40.0f),
        FLOAT16(50.0f), FLOAT16(60.0f), FLOAT16(70.0f),
        FLOAT16(80.0f), FLOAT16(90.0f), FLOAT16(100.0f),
        FLOAT16(110.0f), FLOAT16(120.0f), FLOAT16(130.0f)
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_update("scatter_update", "InputDictionary", "InputText", "InputUpdates", axis)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 40.f, 30.f, 20.f,
        4.f, 70.f, 60.f, 50.f,
        8.f, 100.f, 90.f, 80.f,
        12.f, 130.f, 120.f, 110.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}
