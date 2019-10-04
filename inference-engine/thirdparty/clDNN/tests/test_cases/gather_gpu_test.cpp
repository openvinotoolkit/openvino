// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/input_layout.hpp>
#include <api/memory.hpp>
#include <api/gather.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>

#include <cstddef>
#include <tests/test_utils/test_utils.h>

using namespace cldnn;
using namespace ::tests;

TEST(gather_gpu_fp16, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 1.f, 0.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f),
        FLOAT16(3.0f), FLOAT16(4.0f)
    });

    set_values(input2, {
        0.f, 1.f,
        1.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, tensor(1, 4, 1, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
        FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

        FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
        FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
        0.f, 1.f,
        2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_y;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
        FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

        FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
        FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;

    set_values(input1, {
            FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
            FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

            FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
            FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
            0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp32, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 1.f, 0.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    set_values(input2, {
        0.f, 1.f,
        1.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, tensor(1, 4, 1, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_b;

    set_values(input1, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,

        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 3 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_y;

    set_values(input1, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,

        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f

    engine engine;

    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 3, 1, 2 } }); // Dictionary
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto axis = cldnn::gather::gather_axis::along_f;

    set_values(input1, {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,

            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f
    });

    set_values(input2, {
            0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1.get_layout()));
    topology.add(input_layout("InputText", input2.get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, tensor(2, 2, 2, 2))
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}
