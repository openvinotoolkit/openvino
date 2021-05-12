// Copyright (c) 2021 Intel Corporation
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
#include <api/gather_nd.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>

#include <cstddef>
#include <tests/test_utils/test_utils.h>

using namespace cldnn;
using namespace ::tests;

inline void DoTest(const engine& engine,
    const cldnn::memory& input0,
    const cldnn::memory& input1,
    const std::vector<float>& expected_results,
    const int indices_rank,
    const int batch_dims) {
    topology topology;
    topology.add(input_layout("InputData", input0.get_layout()));
    topology.add(input_layout("InputIndices", input1.get_layout()));
    topology.add(
        gather_nd("gather_nd", "InputData", "InputIndices", indices_rank, batch_dims)
    );

    network network(engine, topology);

    network.set_input_data("InputData", input0);
    network.set_input_data("InputIndices", input1);
    auto outputs = network.execute();
    auto output = outputs.at("gather_nd").get_memory();
    auto output_ptr = output.pointer<uint16_t>();

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_nd_gpu_fp16, d23322_i231312_ir6_batch2) {
    const auto& engine = get_test_engine();

    const int indices_rank = 6;
    const int batch_dims = 2;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 2, 3, 2, 2, 3 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfwzyx, { 2, 3, 2, 1, 3, 1 } }); // indices
    // expected output dim: {6,1,3,1,2}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),  FLOAT16(13), FLOAT16(14),    FLOAT16(15), FLOAT16(16),  FLOAT16(11), FLOAT16(12),    FLOAT16(13), FLOAT16(14),  FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),  FLOAT16(23), FLOAT16(24),    FLOAT16(25), FLOAT16(26),  FLOAT16(21), FLOAT16(22),    FLOAT16(23), FLOAT16(24),  FLOAT16(25), FLOAT16(26),
        FLOAT16(31), FLOAT16(32),  FLOAT16(33), FLOAT16(34),    FLOAT16(35), FLOAT16(36),  FLOAT16(31), FLOAT16(32),    FLOAT16(33), FLOAT16(34),  FLOAT16(35), FLOAT16(36),

        FLOAT16(11), FLOAT16(12),  FLOAT16(13), FLOAT16(14),    FLOAT16(15), FLOAT16(16),  FLOAT16(11), FLOAT16(12),    FLOAT16(13), FLOAT16(14),  FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),  FLOAT16(23), FLOAT16(24),    FLOAT16(25), FLOAT16(26),  FLOAT16(21), FLOAT16(22),    FLOAT16(23), FLOAT16(24),  FLOAT16(25), FLOAT16(26),
        FLOAT16(31), FLOAT16(32),  FLOAT16(33), FLOAT16(34),    FLOAT16(35), FLOAT16(36),  FLOAT16(31), FLOAT16(32),    FLOAT16(33), FLOAT16(34),  FLOAT16(35), FLOAT16(36),
        });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(0),    FLOAT16(2), FLOAT16(0),    FLOAT16(2), FLOAT16(0),
        FLOAT16(0), FLOAT16(1),    FLOAT16(0), FLOAT16(1),    FLOAT16(0), FLOAT16(1),

        FLOAT16(2), FLOAT16(0),    FLOAT16(1), FLOAT16(0),    FLOAT16(1), FLOAT16(0),
        FLOAT16(1), FLOAT16(1),    FLOAT16(2), FLOAT16(1),    FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(1), FLOAT16(0),    FLOAT16(2), FLOAT16(0),
        });

    std::vector<float> expected_results = {
        FLOAT16(15), FLOAT16(16),   FLOAT16(11), FLOAT16(12),   FLOAT16(11), FLOAT16(12),
        FLOAT16(25), FLOAT16(26),   FLOAT16(23), FLOAT16(24),   FLOAT16(23), FLOAT16(24),
        FLOAT16(33), FLOAT16(34),   FLOAT16(33), FLOAT16(34),   FLOAT16(33), FLOAT16(34),

        FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),   FLOAT16(25), FLOAT16(26),   FLOAT16(25), FLOAT16(26),
        FLOAT16(31), FLOAT16(32),   FLOAT16(35), FLOAT16(36),   FLOAT16(33), FLOAT16(34),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d231322_i231321_ir6_batch5) {
    const auto& engine = get_test_engine();

    const int indices_rank = 6;
    const int batch_dims = 5;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfwzyx, { 2, 3, 2, 2, 3, 1 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfwzyx, { 2, 3, 1, 2, 3, 1 } }); // indices
    // expected output dim: {36}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(21), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(31), FLOAT16(30),

        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(29), FLOAT16(30),
        });

    set_values(input1, {
        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),
        });

    std::vector<float> expected_results = {
        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(31),

        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(29),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d23322_i23321_ir5_batch4) {
    const auto& engine = get_test_engine();

    const int indices_rank = 5;
    const int batch_dims = 4;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 2, 3, 2, 2, 3 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 2, 3, 1, 2, 3 } }); // indices
    // expected output dim: {36}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(21), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(31), FLOAT16(30),

        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(29), FLOAT16(30),
        });

    set_values(input1, {
        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),
        });

    std::vector<float> expected_results = {
        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(31),

        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(29),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d23223_i2321_ir4_batch3) {
    const auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 3;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 2, 3, 3, 2, 2 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 3, 1, 2 } }); // indices
    // expected output dim: {2*3*2,3}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12), FLOAT16(13),  FLOAT16(14), FLOAT16(15), FLOAT16(16),  FLOAT16(17), FLOAT16(18),FLOAT16(15),  FLOAT16(16), FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22), FLOAT16(23),  FLOAT16(24), FLOAT16(25), FLOAT16(26),  FLOAT16(27), FLOAT16(28),FLOAT16(25),  FLOAT16(26), FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30), FLOAT16(31),  FLOAT16(32), FLOAT16(33), FLOAT16(34),  FLOAT16(35), FLOAT16(36),FLOAT16(33),  FLOAT16(34), FLOAT16(35), FLOAT16(36),

        FLOAT16(11), FLOAT16(12), FLOAT16(13),  FLOAT16(14), FLOAT16(15), FLOAT16(16),  FLOAT16(17), FLOAT16(18),FLOAT16(15),  FLOAT16(16), FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22), FLOAT16(23),  FLOAT16(24), FLOAT16(25), FLOAT16(26),  FLOAT16(27), FLOAT16(28),FLOAT16(25),  FLOAT16(26), FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30), FLOAT16(31),  FLOAT16(32), FLOAT16(33), FLOAT16(34),  FLOAT16(35), FLOAT16(36),FLOAT16(33),  FLOAT16(34), FLOAT16(35), FLOAT16(36),
        });

    set_values(input1, {
        FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(0),
        FLOAT16(1), FLOAT16(1),

        FLOAT16(0), FLOAT16(0),
        FLOAT16(0), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),
        });

    std::vector<float> expected_results = {
        FLOAT16(14), FLOAT16(15), FLOAT16(16),  FLOAT16(16), FLOAT16(17), FLOAT16(18),
        FLOAT16(24), FLOAT16(25), FLOAT16(26),  FLOAT16(27), FLOAT16(28), FLOAT16(25),
        FLOAT16(32), FLOAT16(33), FLOAT16(34),  FLOAT16(34), FLOAT16(35), FLOAT16(36),

        FLOAT16(11), FLOAT16(12), FLOAT16(13),  FLOAT16(17), FLOAT16(18), FLOAT16(15),
        FLOAT16(21), FLOAT16(22), FLOAT16(23),  FLOAT16(26), FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30), FLOAT16(31),  FLOAT16(35), FLOAT16(36), FLOAT16(33),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d2342_i2312_ir4_batch2) {
    const auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 2;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 3, 2, 4 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 3, 2, 1 } }); // indices
    // expected output dim: {6,1}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30),   FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),

        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30),   FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),
    });

    set_values(input1, {
        FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),
        FLOAT16(2), FLOAT16(1),

        FLOAT16(0), FLOAT16(0),
        FLOAT16(2), FLOAT16(1),
        FLOAT16(2), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(14),
        FLOAT16(21),
        FLOAT16(34),

        FLOAT16(11),
        FLOAT16(26),
        FLOAT16(33),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d234_i2311_ir4_batch2) {
    const auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 2;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 3, 1, 4 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 3, 1, 1 } }); // indices
    // expected output dim: {6,1,1}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2), FLOAT16(3), FLOAT16(4),
        FLOAT16(5), FLOAT16(6), FLOAT16(7), FLOAT16(8),
        FLOAT16(9), FLOAT16(10), FLOAT16(11), FLOAT16(12),

        FLOAT16(13), FLOAT16(14), FLOAT16(15), FLOAT16(16),
        FLOAT16(17), FLOAT16(18), FLOAT16(19), FLOAT16(20),
        FLOAT16(21), FLOAT16(22), FLOAT16(23), FLOAT16(24),

        });

    set_values(input1, {
        FLOAT16(1),
        FLOAT16(0),
        FLOAT16(2),

        FLOAT16(0),
        FLOAT16(2),
        FLOAT16(2),
        });

    std::vector<float> expected_results = {
        FLOAT16(2),
        FLOAT16(5),
        FLOAT16(11),

        FLOAT16(13),
        FLOAT16(19),
        FLOAT16(23),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d234_i21_ir2_batch1) {
    const auto& engine = get_test_engine();

    const int indices_rank = 2;
    const int batch_dims = 1;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 3, 1, 4 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    // expected output dim: {2,4}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2), FLOAT16(3), FLOAT16(4),
        FLOAT16(5), FLOAT16(6), FLOAT16(7), FLOAT16(8),
        FLOAT16(9), FLOAT16(10), FLOAT16(11), FLOAT16(12),

        FLOAT16(13), FLOAT16(14), FLOAT16(15), FLOAT16(16),
        FLOAT16(17), FLOAT16(18), FLOAT16(19), FLOAT16(20),
        FLOAT16(21), FLOAT16(22), FLOAT16(23), FLOAT16(24),

    });

    set_values(input1, {
        FLOAT16(1),
        FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(5), FLOAT16(6), FLOAT16(7), FLOAT16(8),
        FLOAT16(13), FLOAT16(14), FLOAT16(15), FLOAT16(16),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d22_i21_ir2_batch1) {
    const auto& engine = get_test_engine();

    const int indices_rank = 2;
    const int batch_dims = 1;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    // expected output dim: 2

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(3), FLOAT16(4),
    });

    set_values(input1, {
        FLOAT16(1),
        FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(2),
        FLOAT16(3),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d3223_i321113_ir6_batch0) {
    const auto& engine = get_test_engine();

    const int indices_rank = 6;
    const int batch_dims = 0;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 3, 2 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfwzyx, { 3, 2, 3, 1, 1, 1 } }); // indices
    // expected output dim: 321113

    set_values(input0, {
        FLOAT16(11), FLOAT16(12), FLOAT16(13),   FLOAT16(14), FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22), FLOAT16(23),   FLOAT16(24), FLOAT16(25), FLOAT16(26),

        FLOAT16(31), FLOAT16(32), FLOAT16(33),   FLOAT16(34), FLOAT16(35), FLOAT16(36),
        FLOAT16(41), FLOAT16(42), FLOAT16(43),   FLOAT16(44), FLOAT16(45), FLOAT16(46),

        FLOAT16(51), FLOAT16(52), FLOAT16(53),   FLOAT16(54), FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62), FLOAT16(63),   FLOAT16(64), FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(0), FLOAT16(0),

        FLOAT16(0), FLOAT16(1), FLOAT16(0),
        FLOAT16(2), FLOAT16(0), FLOAT16(1),

        FLOAT16(1), FLOAT16(1), FLOAT16(0),
        FLOAT16(0), FLOAT16(0), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(64), FLOAT16(65), FLOAT16(66),
        FLOAT16(31), FLOAT16(32), FLOAT16(33),

        FLOAT16(21), FLOAT16(22), FLOAT16(23),
        FLOAT16(54), FLOAT16(55), FLOAT16(56),

        FLOAT16(41), FLOAT16(42), FLOAT16(43),
        FLOAT16(11), FLOAT16(12), FLOAT16(13),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d3221_i32312_ir3_batch0) {
    const auto& engine = get_test_engine();

    const int indices_rank = 3;
    const int batch_dims = 0;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 3, 2, 2, 1, 3 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 2 } }); // indices
    // expected output dim: 32312

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),     FLOAT16(13), FLOAT16(14),     FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),     FLOAT16(23), FLOAT16(24),     FLOAT16(25), FLOAT16(26),

        FLOAT16(31), FLOAT16(32),     FLOAT16(33), FLOAT16(34),     FLOAT16(35), FLOAT16(36),
        FLOAT16(41), FLOAT16(42),     FLOAT16(43), FLOAT16(44),     FLOAT16(45), FLOAT16(46),

        FLOAT16(51), FLOAT16(52),     FLOAT16(53), FLOAT16(54),     FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62),     FLOAT16(63), FLOAT16(64),     FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1),
        FLOAT16(1), FLOAT16(0),

        FLOAT16(0), FLOAT16(1),
        FLOAT16(2), FLOAT16(0),

        FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(61), FLOAT16(62),     FLOAT16(63), FLOAT16(64),     FLOAT16(65), FLOAT16(66),
        FLOAT16(31), FLOAT16(32),     FLOAT16(33), FLOAT16(34),     FLOAT16(35), FLOAT16(36),

        FLOAT16(21), FLOAT16(22),     FLOAT16(23), FLOAT16(24),     FLOAT16(25), FLOAT16(26),
        FLOAT16(51), FLOAT16(52),     FLOAT16(53), FLOAT16(54),     FLOAT16(55), FLOAT16(56),

        FLOAT16(41), FLOAT16(42),     FLOAT16(43), FLOAT16(44),     FLOAT16(45), FLOAT16(46),
        FLOAT16(11), FLOAT16(12),     FLOAT16(13), FLOAT16(14),     FLOAT16(15), FLOAT16(16),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d3231_i32312_ir3_batch0) {
    const auto& engine = get_test_engine();

    const int indices_rank = 3;
    const int batch_dims = 0;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 3, 2, 2, 1, 3 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 3 } }); // indices
    // expected output dim: {3,2,1,2}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),     FLOAT16(13), FLOAT16(14),     FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),     FLOAT16(23), FLOAT16(24),     FLOAT16(25), FLOAT16(26),

        FLOAT16(31), FLOAT16(32),     FLOAT16(33), FLOAT16(34),     FLOAT16(35), FLOAT16(36),
        FLOAT16(41), FLOAT16(42),     FLOAT16(43), FLOAT16(44),     FLOAT16(45), FLOAT16(46),

        FLOAT16(51), FLOAT16(52),     FLOAT16(53), FLOAT16(54),     FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62),     FLOAT16(63), FLOAT16(64),     FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(0), FLOAT16(2),

        FLOAT16(0), FLOAT16(1), FLOAT16(0),
        FLOAT16(2), FLOAT16(0), FLOAT16(1),

        FLOAT16(1), FLOAT16(1), FLOAT16(2),
        FLOAT16(0), FLOAT16(0), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(63), FLOAT16(64),
        FLOAT16(35), FLOAT16(36),

        FLOAT16(21), FLOAT16(22),
        FLOAT16(53), FLOAT16(54),

        FLOAT16(45), FLOAT16(46),
        FLOAT16(11), FLOAT16(12),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d3112_i3221_ir4_batch0) {
    const auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 0;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 1, 2, 1 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 2 } }); // indices
    // expected output dim: {3,2,2,1,1,2}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(7), FLOAT16(8),
        FLOAT16(13), FLOAT16(14),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(1),

        FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(1),

        FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(1),
    });

    std::vector<float> expected_results = {
        FLOAT16(13), FLOAT16(14),       FLOAT16(7), FLOAT16(8),
        FLOAT16(1), FLOAT16(2),         FLOAT16(7), FLOAT16(8),

        FLOAT16(13), FLOAT16(14),       FLOAT16(7), FLOAT16(8),
        FLOAT16(1), FLOAT16(2),         FLOAT16(7), FLOAT16(8),

        FLOAT16(13), FLOAT16(14),       FLOAT16(7), FLOAT16(8),
        FLOAT16(1), FLOAT16(2),         FLOAT16(7), FLOAT16(8),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d311211_i322111_ir4_batch0) {
    const auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 0;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfwzyx, { 3, 1, 1, 1, 2, 1 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfwzyx, { 3, 2, 1, 1, 1, 2 } }); // indices
    // expected output dim: {3,2,2,1,1,2,1,1}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(7), FLOAT16(8),
        FLOAT16(13), FLOAT16(14),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(1),

        FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(1),

        FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(1),
    });

    std::vector<float> expected_results = {
        FLOAT16(13), FLOAT16(14),       FLOAT16(7), FLOAT16(8),
        FLOAT16(1), FLOAT16(2),         FLOAT16(7), FLOAT16(8),

        FLOAT16(13), FLOAT16(14),       FLOAT16(7), FLOAT16(8),
        FLOAT16(1), FLOAT16(2),         FLOAT16(7), FLOAT16(8),

        FLOAT16(13), FLOAT16(14),       FLOAT16(7), FLOAT16(8),
        FLOAT16(1), FLOAT16(2),         FLOAT16(7), FLOAT16(8),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d3332_i3223_ir4_batch0) {
    const auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 0;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 3, 3, 2 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 3, 2 } }); // indices

    set_values(input0, {
        FLOAT16(1), FLOAT16(2), FLOAT16(3),     FLOAT16(4), FLOAT16(5), FLOAT16(6),
        FLOAT16(7), FLOAT16(8), FLOAT16(9),     FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(13), FLOAT16(14), FLOAT16(15),  FLOAT16(16), FLOAT16(17), FLOAT16(18),

        FLOAT16(19), FLOAT16(20), FLOAT16(21),     FLOAT16(22), FLOAT16(23), FLOAT16(24),
        FLOAT16(25), FLOAT16(26), FLOAT16(27),     FLOAT16(28), FLOAT16(29), FLOAT16(30),
        FLOAT16(31), FLOAT16(32), FLOAT16(33),     FLOAT16(34), FLOAT16(35), FLOAT16(36),

        FLOAT16(41), FLOAT16(42), FLOAT16(43),     FLOAT16(44), FLOAT16(45), FLOAT16(46),
        FLOAT16(51), FLOAT16(52), FLOAT16(53),     FLOAT16(54), FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62), FLOAT16(63),     FLOAT16(64), FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(0), FLOAT16(0),        FLOAT16(2), FLOAT16(2), FLOAT16(0),
        FLOAT16(1), FLOAT16(0), FLOAT16(0),        FLOAT16(1), FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(0), FLOAT16(1),        FLOAT16(1), FLOAT16(1), FLOAT16(1),
        FLOAT16(2), FLOAT16(0), FLOAT16(0),        FLOAT16(2), FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(1), FLOAT16(1),        FLOAT16(0), FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(2), FLOAT16(1),        FLOAT16(0), FLOAT16(2), FLOAT16(1),
    });

    std::vector<float> expected_results = {
        FLOAT16(41), FLOAT16(42), FLOAT16(43),      FLOAT16(61), FLOAT16(62), FLOAT16(63),
        FLOAT16(19), FLOAT16(20), FLOAT16(21),      FLOAT16(25), FLOAT16(26), FLOAT16(27),

        FLOAT16(22), FLOAT16(23), FLOAT16(24),      FLOAT16(28), FLOAT16(29), FLOAT16(30),
        FLOAT16(41), FLOAT16(42), FLOAT16(43),      FLOAT16(51), FLOAT16(52), FLOAT16(53),

        FLOAT16(28), FLOAT16(29), FLOAT16(30),      FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(34), FLOAT16(35), FLOAT16(36),      FLOAT16(16), FLOAT16(17), FLOAT16(18),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d3323_i322_ir3_batch0) {
    const auto& engine = get_test_engine();

    const int indices_rank = 3;
    const int batch_dims = 0;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 3, 3, 2 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 2 } }); // indices

    set_values(input0, {
        FLOAT16(1), FLOAT16(2), FLOAT16(3),     FLOAT16(4), FLOAT16(5), FLOAT16(6),
        FLOAT16(7), FLOAT16(8), FLOAT16(9),     FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(13), FLOAT16(14), FLOAT16(15),  FLOAT16(16), FLOAT16(17), FLOAT16(18),

        FLOAT16(19), FLOAT16(20), FLOAT16(21),     FLOAT16(22), FLOAT16(23), FLOAT16(24),
        FLOAT16(25), FLOAT16(26), FLOAT16(27),     FLOAT16(28), FLOAT16(29), FLOAT16(30),
        FLOAT16(31), FLOAT16(32), FLOAT16(33),     FLOAT16(34), FLOAT16(35), FLOAT16(36),

        FLOAT16(41), FLOAT16(42), FLOAT16(43),     FLOAT16(44), FLOAT16(45), FLOAT16(46),
        FLOAT16(51), FLOAT16(52), FLOAT16(53),     FLOAT16(54), FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62), FLOAT16(63),     FLOAT16(64), FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(0),
        FLOAT16(2), FLOAT16(1),

        FLOAT16(1), FLOAT16(2),
        FLOAT16(1), FLOAT16(0),

        FLOAT16(0), FLOAT16(1),
        FLOAT16(0), FLOAT16(2),
    });

    std::vector<float> expected_results = {
        FLOAT16(41), FLOAT16(42), FLOAT16(43),     FLOAT16(44), FLOAT16(45), FLOAT16(46),
        FLOAT16(51), FLOAT16(52), FLOAT16(53),     FLOAT16(54), FLOAT16(55), FLOAT16(56),

        FLOAT16(31), FLOAT16(32), FLOAT16(33),     FLOAT16(34), FLOAT16(35), FLOAT16(36),
        FLOAT16(19), FLOAT16(20), FLOAT16(21),     FLOAT16(22), FLOAT16(23), FLOAT16(24),

        FLOAT16(7), FLOAT16(8), FLOAT16(9),        FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(13), FLOAT16(14), FLOAT16(15),     FLOAT16(16), FLOAT16(17), FLOAT16(18),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d22_i21_ir2_batch0) {
    const auto& engine = get_test_engine();

    const int indices_rank = 2;
    const int batch_dims = 0;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(3), FLOAT16(4)
    });

    set_values(input1, {
        FLOAT16(1), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(3), FLOAT16(4),
        FLOAT16(1), FLOAT16(2),
    };

    DoTest(engine, input0, input1, expected_results, indices_rank, batch_dims);
}

TEST(gather_nd_gpu_fp16, d22_i32_ir2_batch0) {
    const auto& engine = get_test_engine();

    const int indices_rank = 2;
    const int batch_dims = 0;
    auto input0 = memory::allocate(engine, { data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // data
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // indices

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(3), FLOAT16(4)
    });

    set_values(input1, {
        FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),
        FLOAT16(1), FLOAT16(1),
    });

    std::vector<float> expected_results = {
        FLOAT16(1),
        FLOAT16(3),
        FLOAT16(4),
    };

    DoTest(engine,input0, input1, expected_results, indices_rank, batch_dims);
}

