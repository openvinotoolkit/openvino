// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "ngraph/runtime/reference/scatter_nd_update.hpp"
#include "shape.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_update.hpp>
#include <intel_gpu/primitives/scatter_nd_update.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cstddef>
#include <cstring>
#include <numeric>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;


struct scatter_nd_update_basic_test_params
{
    data_types input_type;
    data_types indices_type;
    data_types updates_type;
    format input_format;
    format indices_format;
    format updates_format;
    format input_result_format;
    format indices_result_format;
    format updates_result_format;
    tensor input_size;
    tensor indices_size;
    tensor updates_size;
    int indices_rank;
};

struct scatter_nd_update_random_test : testing::TestWithParam<scatter_nd_update_basic_test_params>
{
    template<typename T>
    std::vector<T> generate_random_indices(size_t a, int min, int max, int k = 1) 
    {
        srand(time(0));
        std::vector<T> vec(a);
        for(int i = 0; i < a; ++i) {
            int val = rand() % (max + 1);
            if (val >= min && std::find(vec.begin(), vec.end(), static_cast<T>(val)) == vec.end())
                vec[i] = val;
        }
        return vec;
    }

    template<typename T, typename T_size>
    void execute_fp16(const scatter_nd_update_basic_test_params& params)
    {
        // create input, indices, updates using params
        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ params.input_type, params.input_format, params.input_size });
        auto input2 = engine.allocate_memory({ params.indices_type, params.indices_format, params.indices_size });
        auto input3 = engine.allocate_memory({ params.updates_type, params.updates_format, params.updates_size });

        std::vector<int> input_vec(params.input_size.sizes().size());
        for(int i = 0; i < params.input_size.sizes().size(); ++i)
            input_vec[i] = (int)params.input_size.sizes()[i];
        std::vector<int> updates_vec(params.updates_size.sizes().size());
        for(int i = 0; i < params.updates_size.sizes().size(); ++i)
            updates_vec[i] = (int)params.updates_size.sizes()[i];
        std::vector<int> indices_vec(params.indices_rank, -1);
        for(int i = 0; i < params.indices_rank; ++i)
            indices_vec[i] = (int)params.indices_size.sizes()[i];

        auto input_data = generate_random_1d<T>(params.input_size.count(), -127, 127);
        auto indices_data = generate_random_indices<T>(params.indices_size.count(), 0, 23);
        auto updates_data = generate_random_1d<T>(params.updates_size.count(), -127, 127);

        std::vector<float> input_data_fp16(params.input_size.count(), -1);
        for(int i = 0; i < params.input_size.count(); ++i)
            input_data_fp16[i] = (float)input_data[i];
        std::vector<float> indices_data_fp16(params.indices_size.count(), -1);
        for(int i = 0; i < params.indices_size.count(); ++i)
            indices_data_fp16[i] = (float)indices_data[i];
        std::vector<float> updates_data_fp16(params.updates_size.count(), -1);
        for(int i = 0; i < params.updates_size.count(); ++i)
            updates_data_fp16[i] = (float)updates_data[i];

        set_values(input1, input_data);
        set_values(input2, indices_data);
        set_values(input3, updates_data);

        // execute scatter_nd_update
        topology topology(
            input_layout("InputData", input1->get_layout()),
            input_layout("InputIndices", input2->get_layout()),
            input_layout("InputUpdates", input3->get_layout()),
            reorder("reorder1", "InputData", params.input_result_format, params.input_type),
            reorder("reorder2", "InputIndices", params.indices_result_format, params.indices_type),
            reorder("reorder3", "InputUpdates", params.updates_result_format, params.updates_type),
            scatter_nd_update("scatter_nd_update", "reorder1", "reorder2", "reorder3", params.indices_rank),
            reorder("out", "scatter_nd_update", params.input_format, params.input_type)
        );

        network network(engine, topology);

        network.set_input_data("InputData", input1);
        network.set_input_data("InputIndices", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();
        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<T_size> outputs_ptr(output, get_test_stream());

        auto outputs_ref = std::vector<float>(params.input_size.count());
        ngraph::runtime::reference::scatterNdUpdate<float, float>(input_data_fp16.data(), 
                                                                  indices_data_fp16.data(), 
                                                                  updates_data_fp16.data(), 
                                                                  outputs_ref.data(), 
                                                                  ov::Shape(input_vec.begin(), input_vec.end()), 
                                                                  ov::Shape(indices_vec.begin(), indices_vec.end()), 
                                                                  ov::Shape(updates_vec.begin(), updates_vec.end()));

        for (size_t i = 0; i < outputs_ref.size(); ++i) {
            EXPECT_EQ(outputs_ref[i], float16_to_float32(outputs_ptr[i]));
        }
    }

    template<typename T>
    void execute(const scatter_nd_update_basic_test_params& params)
    {
        // create input, indices, updates using params
        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ params.input_type, params.input_format, params.input_size });
        auto input2 = engine.allocate_memory({ params.indices_type, params.indices_format, params.indices_size });
        auto input3 = engine.allocate_memory({ params.updates_type, params.updates_format, params.updates_size });

        std::vector<int> input_vec(params.input_size.sizes().size());
        for(int i = 0; i < params.input_size.sizes().size(); ++i)
            input_vec[i] = (int)params.input_size.sizes()[i];
        std::vector<int> updates_vec(params.updates_size.sizes().size());
        for(int i = 0; i < params.updates_size.sizes().size(); ++i)
            updates_vec[i] = (int)params.updates_size.sizes()[i];
        std::vector<int> indices_vec(params.indices_rank, -1);
        for(int i = 0; i < params.indices_rank; ++i)
            indices_vec[i] = (int)params.indices_size.sizes()[i];

        auto input_data = generate_random_1d<T>(params.input_size.count(), -127, 127);
        auto indices_data = generate_random_indices<T>(params.indices_size.count(), 0, 23);
        auto updates_data = generate_random_1d<T>(params.updates_size.count(), -127, 127);

        set_values(input1, input_data);
        set_values(input2, indices_data);
        set_values(input3, updates_data);

        // execute scatter_nd_update
        topology topology(
            input_layout("InputData", input1->get_layout()),
            input_layout("InputIndices", input2->get_layout()),
            input_layout("InputUpdates", input3->get_layout()),
            reorder("reorder1", "InputData", params.input_result_format, params.input_type),
            reorder("reorder2", "InputIndices", params.indices_result_format, params.indices_type),
            reorder("reorder3", "InputUpdates", params.updates_result_format, params.updates_type),
            scatter_nd_update("scatter_nd_update", "reorder1", "reorder2", "reorder3", params.indices_rank),
            reorder("out", "scatter_nd_update", params.input_format, params.input_type)
        );

        network network(engine, topology);

        network.set_input_data("InputData", input1);
        network.set_input_data("InputIndices", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();
        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<T> outputs_ptr(output, get_test_stream());

        auto outputs_ref = std::vector<T>(params.input_size.count());
        ngraph::runtime::reference::scatterNdUpdate<T, T>(input_data.data(), 
                                                          indices_data.data(), 
                                                          updates_data.data(), 
                                                          outputs_ref.data(), 
                                                          ov::Shape(input_vec.begin(), input_vec.end()), 
                                                          ov::Shape(indices_vec.begin(), indices_vec.end()), 
                                                          ov::Shape(updates_vec.begin(), updates_vec.end()));

        for (size_t i = 0; i < outputs_ref.size(); ++i) {
            EXPECT_EQ(outputs_ref[i], outputs_ptr[i]);
        }
    }
};

TEST_P(scatter_nd_update_random_test, random)
{
    auto param = GetParam();
    if(param.input_type == data_types::u8)
        this->execute<uint8_t>(param);
    else if (param.input_type == data_types::i8)
        this->execute<int8_t>(param);
    else if (param.input_type == data_types::i32)
        this->execute<int32_t>(param);
    else if (param.input_type == data_types::i64)
        this->execute<int64_t>(param);
    else if (param.input_type == data_types::f16)
        this->execute_fp16<FLOAT16, uint16_t>(param);
    else if (param.input_type == data_types::f32)
        this->execute<float>(param);
    else
        std::cout << "unidentified data type" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_bsv32_fsv16_fp32, 
                         scatter_nd_update_random_test,
                         testing::ValuesIn( 
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32, 
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16,
                               {48, 24, 3, 3}, {3, 2, 1, 1}, {3, 3, 1, 3},
                               2 } 
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fsv16_fp32_yx, 
                         scatter_nd_update_random_test,
                         testing::ValuesIn( 
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32, 
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16,
                               {48, 24, 3, 3}, {3, 2, 1, 1}, {3, 3, 1, 3},
                               2 } 
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fsv16_fp32_zyx, 
                         scatter_nd_update_random_test,
                         testing::ValuesIn( 
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32, 
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16, format::b_fs_zyx_fsv16,
                               {48, 24, 3, 3, 10}, {5, 2, 1, 1}, {5, 10, 1, 3, 3},
                               2 } 
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_bsv32_fsv16_fp16, 
                         scatter_nd_update_random_test,
                         testing::ValuesIn( 
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16, 
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16,
                               {48, 24, 3, 3}, {3, 2, 1, 1}, {3, 3, 1, 3},
                               2 } 
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fsv16_fp16, 
                         scatter_nd_update_random_test,
                         testing::ValuesIn( 
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16, 
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16,
                               {48, 24, 3, 3}, {3, 2, 1, 1}, {3, 3, 1, 3},
                               2 } 
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_bsv32_fsv16_i8, 
                         scatter_nd_update_random_test,
                         testing::ValuesIn( 
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8, 
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16,
                               {41, 23, 3, 3}, {3, 2, 1, 1}, {3, 3, 1, 3},
                               2 } 
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_bsv32_fsv32_i8, 
                         scatter_nd_update_random_test,
                         testing::ValuesIn( 
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8, 
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32,
                               {41, 23, 3, 3}, {3, 2, 1, 1}, {3, 3, 1, 3},
                               2 } 
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fsv32_i8, 
                         scatter_nd_update_random_test,
                         testing::ValuesIn( 
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8, 
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32,
                               {41, 23, 3, 3}, {3, 2, 1, 1}, {3, 3, 1, 3},
                               2 } 
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fsv16_i8, 
                         scatter_nd_update_random_test,
                         testing::ValuesIn( 
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8, 
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16,
                               {41, 23, 3, 3}, {3, 2, 1, 1}, {3, 3, 1, 3},
                               2 } 
                         }));
                         

TEST(scatter_nd_update_gpu_fp16_test15, data5_indice3_update5) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 2, 4, 3 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx,  { 1, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 2, 4, 3, 2 } }); // updates

    set_values(input1, {
        // 0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        // 1
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
    });

    set_values(input2, {
        FLOAT16(1.0f),
        FLOAT16(0.0f),
    });

    set_values(input3, {
        // 0
        FLOAT16(91.0f), FLOAT16(2.0f),    FLOAT16(83.0f), FLOAT16(4.0f),      FLOAT16(71.0f), FLOAT16(2.0f),   FLOAT16(63.0f), FLOAT16(4.0f),
        FLOAT16(95.0f), FLOAT16(6.0f),    FLOAT16(87.0f), FLOAT16(8.0f),      FLOAT16(75.0f), FLOAT16(6.0f),   FLOAT16(67.0f), FLOAT16(8.0f),
        FLOAT16(99.0f), FLOAT16(10.0f),   FLOAT16(811.0f), FLOAT16(12.0f),    FLOAT16(79.0f), FLOAT16(10.0f),  FLOAT16(611.0f), FLOAT16(12.0f),

        FLOAT16(91.0f), FLOAT16(2.0f),    FLOAT16(83.0f), FLOAT16(4.0f),      FLOAT16(71.0f), FLOAT16(2.0f),   FLOAT16(63.0f), FLOAT16(4.0f),
        FLOAT16(95.0f), FLOAT16(6.0f),    FLOAT16(87.0f), FLOAT16(8.0f),      FLOAT16(75.0f), FLOAT16(6.0f),   FLOAT16(67.0f), FLOAT16(8.0f),
        FLOAT16(99.0f), FLOAT16(10.0f),   FLOAT16(811.0f), FLOAT16(12.0f),    FLOAT16(79.0f), FLOAT16(10.0f),  FLOAT16(611.0f), FLOAT16(12.0f),
        // 1
        FLOAT16(91.0f), FLOAT16(2.0f),    FLOAT16(83.0f), FLOAT16(4.0f),      FLOAT16(71.0f), FLOAT16(2.0f),   FLOAT16(63.0f), FLOAT16(4.0f),
        FLOAT16(95.0f), FLOAT16(6.0f),    FLOAT16(87.0f), FLOAT16(8.0f),      FLOAT16(75.0f), FLOAT16(6.0f),   FLOAT16(67.0f), FLOAT16(8.0f),
        FLOAT16(99.0f), FLOAT16(10.0f),   FLOAT16(811.0f), FLOAT16(12.0f),    FLOAT16(79.0f), FLOAT16(10.0f),  FLOAT16(611.0f), FLOAT16(12.0f),

        FLOAT16(91.0f), FLOAT16(2.0f),    FLOAT16(83.0f), FLOAT16(4.0f),      FLOAT16(71.0f), FLOAT16(2.0f),   FLOAT16(63.0f), FLOAT16(4.0f),
        FLOAT16(95.0f), FLOAT16(6.0f),    FLOAT16(87.0f), FLOAT16(8.0f),      FLOAT16(75.0f), FLOAT16(6.0f),   FLOAT16(67.0f), FLOAT16(8.0f),
        FLOAT16(99.0f), FLOAT16(10.0f),   FLOAT16(811.0f), FLOAT16(12.0f),    FLOAT16(79.0f), FLOAT16(10.0f),  FLOAT16(611.0f), FLOAT16(12.0f),
    });

    std::vector<float> expected_results = {
        // 0
        FLOAT16(91.0f), FLOAT16(2.0f),    FLOAT16(83.0f), FLOAT16(4.0f),      FLOAT16(71.0f), FLOAT16(2.0f),   FLOAT16(63.0f), FLOAT16(4.0f),
        FLOAT16(95.0f), FLOAT16(6.0f),    FLOAT16(87.0f), FLOAT16(8.0f),      FLOAT16(75.0f), FLOAT16(6.0f),   FLOAT16(67.0f), FLOAT16(8.0f),
        FLOAT16(99.0f), FLOAT16(10.0f),   FLOAT16(811.0f), FLOAT16(12.0f),    FLOAT16(79.0f), FLOAT16(10.0f),  FLOAT16(611.0f), FLOAT16(12.0f),

        FLOAT16(91.0f), FLOAT16(2.0f),    FLOAT16(83.0f), FLOAT16(4.0f),      FLOAT16(71.0f), FLOAT16(2.0f),   FLOAT16(63.0f), FLOAT16(4.0f),
        FLOAT16(95.0f), FLOAT16(6.0f),    FLOAT16(87.0f), FLOAT16(8.0f),      FLOAT16(75.0f), FLOAT16(6.0f),   FLOAT16(67.0f), FLOAT16(8.0f),
        FLOAT16(99.0f), FLOAT16(10.0f),   FLOAT16(811.0f), FLOAT16(12.0f),    FLOAT16(79.0f), FLOAT16(10.0f),  FLOAT16(611.0f), FLOAT16(12.0f),
        // 1
        FLOAT16(91.0f), FLOAT16(2.0f),    FLOAT16(83.0f), FLOAT16(4.0f),      FLOAT16(71.0f), FLOAT16(2.0f),   FLOAT16(63.0f), FLOAT16(4.0f),
        FLOAT16(95.0f), FLOAT16(6.0f),    FLOAT16(87.0f), FLOAT16(8.0f),      FLOAT16(75.0f), FLOAT16(6.0f),   FLOAT16(67.0f), FLOAT16(8.0f),
        FLOAT16(99.0f), FLOAT16(10.0f),   FLOAT16(811.0f), FLOAT16(12.0f),    FLOAT16(79.0f), FLOAT16(10.0f),  FLOAT16(611.0f), FLOAT16(12.0f),

        FLOAT16(91.0f), FLOAT16(2.0f),    FLOAT16(83.0f), FLOAT16(4.0f),      FLOAT16(71.0f), FLOAT16(2.0f),   FLOAT16(63.0f), FLOAT16(4.0f),
        FLOAT16(95.0f), FLOAT16(6.0f),    FLOAT16(87.0f), FLOAT16(8.0f),      FLOAT16(75.0f), FLOAT16(6.0f),   FLOAT16(67.0f), FLOAT16(8.0f),
        FLOAT16(99.0f), FLOAT16(10.0f),   FLOAT16(811.0f), FLOAT16(12.0f),    FLOAT16(79.0f), FLOAT16(10.0f),  FLOAT16(611.0f), FLOAT16(12.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 3)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test14, data5_indice2_update3) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 2, 4, 3 } }); // data 2x2x3x4x2 (bfzyx)
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx,  { 3, 3, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 4, 1, 1, 2 } }); // updates

    set_values(input1, {
        // 0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        // 1
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(0.0f),
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(1.0f),
        });

    set_values(input3, {
        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f), FLOAT16(55.0f), FLOAT16(56.0f), FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(61.0f), FLOAT16(62.0f), FLOAT16(63.0f), FLOAT16(64.0f), FLOAT16(65.0f), FLOAT16(66.0f), FLOAT16(67.0f), FLOAT16(68.0f),
        FLOAT16(71.0f), FLOAT16(72.0f), FLOAT16(73.0f), FLOAT16(74.0f), FLOAT16(75.0f), FLOAT16(76.0f), FLOAT16(77.0f), FLOAT16(78.0f),
        });

    std::vector<float> expected_results = {
        // 0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(71.0f), FLOAT16(72.0f), FLOAT16(73.0f), FLOAT16(74.0f),     FLOAT16(75.0f), FLOAT16(76.0f), FLOAT16(77.0f), FLOAT16(78.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        // 1
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(61.0f), FLOAT16(62.0f), FLOAT16(63.0f), FLOAT16(64.0f),     FLOAT16(65.0f), FLOAT16(66.0f), FLOAT16(67.0f), FLOAT16(68.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f),     FLOAT16(55.0f), FLOAT16(56.0f), FLOAT16(57.0f), FLOAT16(58.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test13, data4_indice2_update2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 4, 2 } }); // data 2x3x2x4 (bfyx)
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 1 } }); // updates

    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f), FLOAT16(4.0f),       FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f), FLOAT16(8.0f),       FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),     FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f), FLOAT16(4.0f),       FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f), FLOAT16(8.0f),       FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),     FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(1.0f),
        FLOAT16(0.0f), FLOAT16(2.0f), FLOAT16(1.0f),
        });

    set_values(input3, {
        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f),
        FLOAT16(61.0f), FLOAT16(62.0f), FLOAT16(63.0f), FLOAT16(64.0f),
        FLOAT16(71.0f), FLOAT16(72.0f), FLOAT16(73.0f), FLOAT16(74.0f),
        });

    std::vector<float> expected_results = {
        FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f), FLOAT16(4.0f),       FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f), FLOAT16(8.0f),       FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),     FLOAT16(71.0f), FLOAT16(72.0f), FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f), FLOAT16(4.0f),       FLOAT16(1.0f), FLOAT16(2.0f),  FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f),    FLOAT16(5.0f), FLOAT16(6.0f),  FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),     FLOAT16(61.0f), FLOAT16(62.0f), FLOAT16(63.0f), FLOAT16(64.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test12, data3_indice3_update1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 1, 4 } }); // data 3x3x4 (bfy)
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 3, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } }); // updates

    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(2.0f), FLOAT16(0.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(0.0f),
        });

    set_values(input3, {
        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f),
        });

    std::vector<float> expected_results = {
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(54.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(53.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(52.0f),

        FLOAT16(51.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test11, data6_indice1_update6) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 2, 3, 4, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 2, 3, 4, 2 } }); // updates

    set_values(input1, {
        // 0, 0, 0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        // 0, 0, 1
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        // 0, 1, 0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),


        // 1, 0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        // 1, 1
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(1.0f),
        });

    set_values(input3, {
        // 0
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(50.0f), FLOAT16(51.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        FLOAT16(150.0f), FLOAT16(151.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        });

    std::vector<float> expected_results = {
        // 0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        // 1
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(50.0f), FLOAT16(51.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        FLOAT16(150.0f), FLOAT16(151.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test10, data5_indice1_update5) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 3, 4, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 3, 4, 2 } }); // updates

    set_values(input1, {
        // 0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        // 1
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(1.0f), FLOAT16(0.0f),
        });

    set_values(input3, {
        // 0
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(50.0f), FLOAT16(51.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        // 1
        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        FLOAT16(150.0f), FLOAT16(151.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        });

    std::vector<float> expected_results = {
        // 0
        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        FLOAT16(150.0f), FLOAT16(151.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        // 1
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(50.0f), FLOAT16(51.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test9, data4_indice1_update4) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 4, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 4, 2 } }); // updates

    set_values(input1, {
        // 0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        // 1
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        // 2
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(2.0f), FLOAT16(0.0f),
        });

    set_values(input3, {
        // 0
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        // 1
        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        });

    std::vector<float> expected_results = {
        // 0
        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),
        // 1
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        // 2
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test8, data6_indice2_update5) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 2, 4, 3, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx,   { 2, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 1, 2, 4, 3 } }); // updates

    set_values(input1, {
        //0,0
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        //0,1
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(0.0f), FLOAT16(0.0f)
        });

    set_values(input3, {
        // 0
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),


        // 1
        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),
        });

    std::vector<float> expected_results = {
        // 0,0
        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        // 0,1
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test7, data5_indice2_update4) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 2, 3, 4, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx,  { 2, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx,  { 2, 2, 1, 3, 4 } }); // updates


    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(0.0f), FLOAT16(0.0f)
        });

    set_values(input3, {
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),
        });

    std::vector<float> expected_results = {
        FLOAT16(151.0f), FLOAT16(152.0f),    FLOAT16(153.0f), FLOAT16(154.0f),      FLOAT16(155.0f), FLOAT16(156.0f),   FLOAT16(157.0f), FLOAT16(158.0f),
        FLOAT16(159.0f), FLOAT16(160.0f),    FLOAT16(161.0f), FLOAT16(162.0f),      FLOAT16(163.0f), FLOAT16(164.0f),   FLOAT16(165.0f), FLOAT16(166.0f),
        FLOAT16(167.0f), FLOAT16(168.0f),    FLOAT16(169.0f), FLOAT16(170.0f),      FLOAT16(171.0f), FLOAT16(172.0f),   FLOAT16(173.0f), FLOAT16(174.0f),

        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16_test6, data4_indice2_update3) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 2, 4 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 2 } }); // updates


    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),   FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),   FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),  FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(1.0f), FLOAT16(1.0f),
        FLOAT16(1.0f), FLOAT16(0.0f),
        FLOAT16(0.0f), FLOAT16(2.0f)
        });

    set_values(input3, {
        FLOAT16(51.0f), FLOAT16(52.0f),    FLOAT16(53.0f), FLOAT16(54.0f),      FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(59.0f), FLOAT16(60.0f),    FLOAT16(61.0f), FLOAT16(62.0f),      FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),    FLOAT16(69.0f), FLOAT16(70.0f),      FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),
        });

    std::vector<float> expected_results = {
        FLOAT16(1.0f), FLOAT16(2.0f),    FLOAT16(3.0f), FLOAT16(4.0f),      FLOAT16(1.0f), FLOAT16(2.0f),     FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f),    FLOAT16(7.0f), FLOAT16(8.0f),      FLOAT16(5.0f), FLOAT16(6.0f),     FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(67.0f), FLOAT16(68.0f),  FLOAT16(69.0f), FLOAT16(70.0f),    FLOAT16(71.0f), FLOAT16(72.0f),   FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(59.0f), FLOAT16(60.0f),  FLOAT16(61.0f), FLOAT16(62.0f),    FLOAT16(63.0f), FLOAT16(64.0f),   FLOAT16(65.0f), FLOAT16(66.0f),
        FLOAT16(51.0f), FLOAT16(52.0f),  FLOAT16(53.0f), FLOAT16(54.0f),    FLOAT16(55.0f), FLOAT16(56.0f),   FLOAT16(57.0f), FLOAT16(58.0f),
        FLOAT16(9.0f), FLOAT16(10.0f),   FLOAT16(11.0f), FLOAT16(12.0f),    FLOAT16(9.0f), FLOAT16(10.0f),    FLOAT16(11.0f), FLOAT16(12.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test5, data3_indice2_update2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 1, 4 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 1 } }); // updates


    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(1.0f), FLOAT16(1.0f),
        FLOAT16(1.0f), FLOAT16(0.0f),
        FLOAT16(0.0f), FLOAT16(2.0f)
        });

    set_values(input3, {
        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f),
        FLOAT16(61.0f), FLOAT16(62.0f), FLOAT16(63.0f), FLOAT16(64.0f),
        FLOAT16(71.0f), FLOAT16(72.0f), FLOAT16(73.0f), FLOAT16(74.0f),
        });

    std::vector<float> expected_results = {
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(71.0f), FLOAT16(72.0f), FLOAT16(73.0f), FLOAT16(74.0f),

        FLOAT16(61.0f), FLOAT16(62.0f), FLOAT16(63.0f), FLOAT16(64.0f),
        FLOAT16(51.0f), FLOAT16(52.0f), FLOAT16(53.0f), FLOAT16(54.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test4, data2_indice2_update1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 1 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 1, 1, 1 } }); // updates


    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
        FLOAT16(2.0f), FLOAT16(1.0f),
        FLOAT16(0.0f), FLOAT16(3.0f),
        FLOAT16(0.0f), FLOAT16(2.0f)
        });

    set_values(input3, {
        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f)
        });

    std::vector<float> expected_results = {
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(23.0f), FLOAT16(22.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(21.0f), FLOAT16(11.0f), FLOAT16(12.0f),
        };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test3, data3_indice1_update3) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 4, 1 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 4, 1 } }); // updates


    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),
        });

    set_values(input2, {
            FLOAT16(2.0f), FLOAT16(0.0f)
        });

    set_values(input3, {
        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
        FLOAT16(25.0f), FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f),
        FLOAT16(29.0f), FLOAT16(30.0f), FLOAT16(31.0f), FLOAT16(32.0f),

        FLOAT16(41.0f), FLOAT16(42.0f), FLOAT16(43.0f), FLOAT16(44.0f),
        FLOAT16(45.0f), FLOAT16(46.0f), FLOAT16(47.0f), FLOAT16(48.0f),
        FLOAT16(49.0f), FLOAT16(50.0f), FLOAT16(51.0f), FLOAT16(52.0f),
        });

    std::vector<float> expected_results = {
        FLOAT16(41.0f), FLOAT16(42.0f), FLOAT16(43.0f), FLOAT16(44.0f),
        FLOAT16(45.0f), FLOAT16(46.0f), FLOAT16(47.0f), FLOAT16(48.0f),
        FLOAT16(49.0f), FLOAT16(50.0f), FLOAT16(51.0f), FLOAT16(52.0f),

        FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(3.0f), FLOAT16(4.0f),
        FLOAT16(5.0f), FLOAT16(6.0f), FLOAT16(7.0f), FLOAT16(8.0f),
        FLOAT16(9.0f), FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f),

        FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f), FLOAT16(24.0f),
        FLOAT16(25.0f), FLOAT16(26.0f), FLOAT16(27.0f), FLOAT16(28.0f),
        FLOAT16(29.0f), FLOAT16(30.0f), FLOAT16(31.0f), FLOAT16(32.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16_test2, data2_indice1_update2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 1 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 4, 1, 1 } }); // updates


    set_values(input1, {
        FLOAT16(13.0f), FLOAT16(12.0f), FLOAT16(11.0f), FLOAT16(10.0f),
        FLOAT16(9.0f), FLOAT16(8.0f), FLOAT16(7.0f), FLOAT16(6.0f),
        FLOAT16(5.0f), FLOAT16(4.0f), FLOAT16(3.0f), FLOAT16(2.0f)
        });

    set_values(input2, {
            FLOAT16(2.0f), FLOAT16(0.0f)
        });

    set_values(input3, {
            FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f),
            FLOAT16(24.0f), FLOAT16(25.0f), FLOAT16(26.0f), FLOAT16(27.0f)
        });

    std::vector<float> expected_results = {
        FLOAT16(24.0f), FLOAT16(25.0f), FLOAT16(26.0f), FLOAT16(27.0f),
        FLOAT16(9.0f), FLOAT16(8.0f), FLOAT16(7.0f), FLOAT16(6.0f),
        FLOAT16(20.0f), FLOAT16(21.0f), FLOAT16(22.0f), FLOAT16(23.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test1, data1_indice1_update1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 8, 1, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } }); // Updates


    set_values(input1, {
        FLOAT16(9.0f), FLOAT16(8.0f), FLOAT16(7.0f), FLOAT16(6.0f), FLOAT16(5.0f), FLOAT16(4.0f), FLOAT16(3.0f), FLOAT16(2.0f)
    });

    set_values(input2, {
        FLOAT16(2.0f), FLOAT16(4.0f), FLOAT16(5.0f), FLOAT16(7.0f)
    });

    set_values(input3, {
        FLOAT16(10.0f), FLOAT16(11.0f), FLOAT16(12.0f), FLOAT16(13.0f)
    });

    std::vector<float> expected_results = {
        9.f, 8.f, 10.f, 6.f, 11.f, 12.f, 3.f, 13.f
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}



TEST(scatter_nd_update_gpu_fp16, d6661_i2311) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x3x1x1
    //  Updates : 2x1x1x1
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 6, 6, 1, 6 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // Updates

    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f), FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f), FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),
        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f), FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f), FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),
        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f), FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),
        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f), FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        FLOAT16(136.f), FLOAT16(137.f), FLOAT16(138.f), FLOAT16(139.f), FLOAT16(140.f), FLOAT16(141.f),
        FLOAT16(142.f), FLOAT16(143.f), FLOAT16(144.f), FLOAT16(145.f), FLOAT16(146.f), FLOAT16(147.f),
        FLOAT16(148.f), FLOAT16(149.f), FLOAT16(150.f), FLOAT16(151.f), FLOAT16(152.f), FLOAT16(153.f),
        FLOAT16(154.f), FLOAT16(155.f), FLOAT16(156.f), FLOAT16(157.f), FLOAT16(158.f), FLOAT16(159.f),
        FLOAT16(160.f), FLOAT16(161.f), FLOAT16(162.f), FLOAT16(163.f), FLOAT16(164.f), FLOAT16(165.f),
        FLOAT16(166.f), FLOAT16(167.f), FLOAT16(168.f), FLOAT16(169.f), FLOAT16(170.f), FLOAT16(171.f),

        FLOAT16(172.f), FLOAT16(173.f), FLOAT16(174.f), FLOAT16(175.f), FLOAT16(176.f), FLOAT16(177.f),
        FLOAT16(178.f), FLOAT16(179.f), FLOAT16(180.f), FLOAT16(181.f), FLOAT16(182.f), FLOAT16(183.f),
        FLOAT16(184.f), FLOAT16(185.f), FLOAT16(186.f), FLOAT16(187.f), FLOAT16(188.f), FLOAT16(189.f),
        FLOAT16(190.f), FLOAT16(191.f), FLOAT16(192.f), FLOAT16(193.f), FLOAT16(194.f), FLOAT16(195.f),
        FLOAT16(196.f), FLOAT16(197.f), FLOAT16(198.f), FLOAT16(199.f), FLOAT16(200.f), FLOAT16(201.f),
        FLOAT16(202.f), FLOAT16(203.f), FLOAT16(204.f), FLOAT16(205.f), FLOAT16(206.f), FLOAT16(207.f),

        FLOAT16(208.f), FLOAT16(209.f), FLOAT16(210.f), FLOAT16(211.f), FLOAT16(212.f), FLOAT16(213.f),
        FLOAT16(214.f), FLOAT16(215.f), FLOAT16(216.f), FLOAT16(217.f), FLOAT16(218.f), FLOAT16(219.f),
        FLOAT16(220.f), FLOAT16(221.f), FLOAT16(222.f), FLOAT16(223.f), FLOAT16(224.f), FLOAT16(225.f),
        FLOAT16(226.f), FLOAT16(227.f), FLOAT16(228.f), FLOAT16(229.f), FLOAT16(230.f), FLOAT16(231.f),
        FLOAT16(232.f), FLOAT16(233.f), FLOAT16(234.f), FLOAT16(235.f), FLOAT16(236.f), FLOAT16(237.f),
        FLOAT16(238.f), FLOAT16(239.f), FLOAT16(240.f), FLOAT16(241.f), FLOAT16(242.f), FLOAT16(243.f),

        FLOAT16(244.f), FLOAT16(245.f), FLOAT16(246.f), FLOAT16(247.f), FLOAT16(248.f), FLOAT16(249.f),
        FLOAT16(250.f), FLOAT16(251.f), FLOAT16(252.f), FLOAT16(253.f), FLOAT16(254.f), FLOAT16(255.f),
        FLOAT16(256.f), FLOAT16(257.f), FLOAT16(258.f), FLOAT16(259.f), FLOAT16(260.f), FLOAT16(261.f),
        FLOAT16(262.f), FLOAT16(263.f), FLOAT16(264.f), FLOAT16(265.f), FLOAT16(266.f), FLOAT16(267.f),
        FLOAT16(268.f), FLOAT16(269.f), FLOAT16(270.f), FLOAT16(271.f), FLOAT16(272.f), FLOAT16(273.f),
        FLOAT16(274.f), FLOAT16(275.f), FLOAT16(276.f), FLOAT16(277.f), FLOAT16(278.f), FLOAT16(279.f),

        FLOAT16(280.f), FLOAT16(281.f), FLOAT16(282.f), FLOAT16(283.f), FLOAT16(284.f), FLOAT16(285.f),
        FLOAT16(286.f), FLOAT16(287.f), FLOAT16(288.f), FLOAT16(289.f), FLOAT16(290.f), FLOAT16(291.f),
        FLOAT16(292.f), FLOAT16(293.f), FLOAT16(294.f), FLOAT16(295.f), FLOAT16(296.f), FLOAT16(297.f),
        FLOAT16(298.f), FLOAT16(299.f), FLOAT16(300.f), FLOAT16(301.f), FLOAT16(302.f), FLOAT16(303.f),
        FLOAT16(304.f), FLOAT16(305.f), FLOAT16(306.f), FLOAT16(307.f), FLOAT16(308.f), FLOAT16(309.f),
        FLOAT16(310.f), FLOAT16(311.f), FLOAT16(312.f), FLOAT16(313.f), FLOAT16(314.f), FLOAT16(315.f),
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f),
        FLOAT16(3.0f), FLOAT16(4.0f), FLOAT16(5.0f)
        });

    set_values(input3, {
        FLOAT16(999.0f), FLOAT16(888.0f)
        });


    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f, 102.f, 103.f, 104.f, 105.f,
        106.f, 107.f, 999.f, 109.f, 110.f, 111.f,
        112.f, 113.f, 114.f, 115.f, 116.f, 117.f,
        118.f, 119.f, 120.f, 121.f, 122.f, 123.f,
        124.f, 125.f, 126.f, 127.f, 128.f, 129.f,
        130.f, 131.f, 132.f, 133.f, 134.f, 135.f,

        136.f, 137.f, 138.f, 139.f, 140.f, 141.f,
        142.f, 143.f, 144.f, 145.f, 146.f, 147.f,
        148.f, 149.f, 150.f, 151.f, 152.f, 153.f,
        154.f, 155.f, 156.f, 157.f, 158.f, 159.f,
        160.f, 161.f, 162.f, 163.f, 164.f, 165.f,
        166.f, 167.f, 168.f, 169.f, 170.f, 171.f,

        172.f, 173.f, 174.f, 175.f, 176.f, 177.f,
        178.f, 179.f, 180.f, 181.f, 182.f, 183.f,
        184.f, 185.f, 186.f, 187.f, 188.f, 189.f,
        190.f, 191.f, 192.f, 193.f, 194.f, 195.f,
        196.f, 197.f, 198.f, 199.f, 200.f, 201.f,
        202.f, 203.f, 204.f, 205.f, 206.f, 207.f,

        208.f, 209.f, 210.f, 211.f, 212.f, 213.f,
        214.f, 215.f, 216.f, 217.f, 218.f, 219.f,
        220.f, 221.f, 222.f, 223.f, 224.f, 225.f,
        226.f, 227.f, 228.f, 229.f, 230.f, 231.f,
        232.f, 233.f, 234.f, 235.f, 236.f, 888.f,
        238.f, 239.f, 240.f, 241.f, 242.f, 243.f,

        244.f, 245.f, 246.f, 247.f, 248.f, 249.f,
        250.f, 251.f, 252.f, 253.f, 254.f, 255.f,
        256.f, 257.f, 258.f, 259.f, 260.f, 261.f,
        262.f, 263.f, 264.f, 265.f, 266.f, 267.f,
        268.f, 269.f, 270.f, 271.f, 272.f, 273.f,
        274.f, 275.f, 276.f, 277.f, 278.f, 279.f,

        280.f, 281.f, 282.f, 283.f, 284.f, 285.f,
        286.f, 287.f, 288.f, 289.f, 290.f, 291.f,
        292.f, 293.f, 294.f, 295.f, 296.f, 297.f,
        298.f, 299.f, 300.f, 301.f, 302.f, 303.f,
        304.f, 305.f, 306.f, 307.f, 308.f, 309.f,
        310.f, 311.f, 312.f, 313.f, 314.f, 315.f,
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16, d6661_i2211) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x2x1x1
    //  Updates : 2x6x1x1
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 6, 6, 1, 6 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 6, 1, 1 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f), FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f), FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),
        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f), FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f), FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),
        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f), FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),
        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f), FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        FLOAT16(136.f), FLOAT16(137.f), FLOAT16(138.f), FLOAT16(139.f), FLOAT16(140.f), FLOAT16(141.f),
        FLOAT16(142.f), FLOAT16(143.f), FLOAT16(144.f), FLOAT16(145.f), FLOAT16(146.f), FLOAT16(147.f),
        FLOAT16(148.f), FLOAT16(149.f), FLOAT16(150.f), FLOAT16(151.f), FLOAT16(152.f), FLOAT16(153.f),
        FLOAT16(154.f), FLOAT16(155.f), FLOAT16(156.f), FLOAT16(157.f), FLOAT16(158.f), FLOAT16(159.f),
        FLOAT16(160.f), FLOAT16(161.f), FLOAT16(162.f), FLOAT16(163.f), FLOAT16(164.f), FLOAT16(165.f),
        FLOAT16(166.f), FLOAT16(167.f), FLOAT16(168.f), FLOAT16(169.f), FLOAT16(170.f), FLOAT16(171.f),

        FLOAT16(172.f), FLOAT16(173.f), FLOAT16(174.f), FLOAT16(175.f), FLOAT16(176.f), FLOAT16(177.f),
        FLOAT16(178.f), FLOAT16(179.f), FLOAT16(180.f), FLOAT16(181.f), FLOAT16(182.f), FLOAT16(183.f),
        FLOAT16(184.f), FLOAT16(185.f), FLOAT16(186.f), FLOAT16(187.f), FLOAT16(188.f), FLOAT16(189.f),
        FLOAT16(190.f), FLOAT16(191.f), FLOAT16(192.f), FLOAT16(193.f), FLOAT16(194.f), FLOAT16(195.f),
        FLOAT16(196.f), FLOAT16(197.f), FLOAT16(198.f), FLOAT16(199.f), FLOAT16(200.f), FLOAT16(201.f),
        FLOAT16(202.f), FLOAT16(203.f), FLOAT16(204.f), FLOAT16(205.f), FLOAT16(206.f), FLOAT16(207.f),

        FLOAT16(208.f), FLOAT16(209.f), FLOAT16(210.f), FLOAT16(211.f), FLOAT16(212.f), FLOAT16(213.f),
        FLOAT16(214.f), FLOAT16(215.f), FLOAT16(216.f), FLOAT16(217.f), FLOAT16(218.f), FLOAT16(219.f),
        FLOAT16(220.f), FLOAT16(221.f), FLOAT16(222.f), FLOAT16(223.f), FLOAT16(224.f), FLOAT16(225.f),
        FLOAT16(226.f), FLOAT16(227.f), FLOAT16(228.f), FLOAT16(229.f), FLOAT16(230.f), FLOAT16(231.f),
        FLOAT16(232.f), FLOAT16(233.f), FLOAT16(234.f), FLOAT16(235.f), FLOAT16(236.f), FLOAT16(237.f),
        FLOAT16(238.f), FLOAT16(239.f), FLOAT16(240.f), FLOAT16(241.f), FLOAT16(242.f), FLOAT16(243.f),

        FLOAT16(244.f), FLOAT16(245.f), FLOAT16(246.f), FLOAT16(247.f), FLOAT16(248.f), FLOAT16(249.f),
        FLOAT16(250.f), FLOAT16(251.f), FLOAT16(252.f), FLOAT16(253.f), FLOAT16(254.f), FLOAT16(255.f),
        FLOAT16(256.f), FLOAT16(257.f), FLOAT16(258.f), FLOAT16(259.f), FLOAT16(260.f), FLOAT16(261.f),
        FLOAT16(262.f), FLOAT16(263.f), FLOAT16(264.f), FLOAT16(265.f), FLOAT16(266.f), FLOAT16(267.f),
        FLOAT16(268.f), FLOAT16(269.f), FLOAT16(270.f), FLOAT16(271.f), FLOAT16(272.f), FLOAT16(273.f),
        FLOAT16(274.f), FLOAT16(275.f), FLOAT16(276.f), FLOAT16(277.f), FLOAT16(278.f), FLOAT16(279.f),

        FLOAT16(280.f), FLOAT16(281.f), FLOAT16(282.f), FLOAT16(283.f), FLOAT16(284.f), FLOAT16(285.f),
        FLOAT16(286.f), FLOAT16(287.f), FLOAT16(288.f), FLOAT16(289.f), FLOAT16(290.f), FLOAT16(291.f),
        FLOAT16(292.f), FLOAT16(293.f), FLOAT16(294.f), FLOAT16(295.f), FLOAT16(296.f), FLOAT16(297.f),
        FLOAT16(298.f), FLOAT16(299.f), FLOAT16(300.f), FLOAT16(301.f), FLOAT16(302.f), FLOAT16(303.f),
        FLOAT16(304.f), FLOAT16(305.f), FLOAT16(306.f), FLOAT16(307.f), FLOAT16(308.f), FLOAT16(309.f),
        FLOAT16(310.f), FLOAT16(311.f), FLOAT16(312.f), FLOAT16(313.f), FLOAT16(314.f), FLOAT16(315.f),
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(3.0f), FLOAT16(4.0f),
        });

    set_values(input3, {
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f, 102.f, 103.f, 104.f, 105.f,
        999.f, 999.f, 999.f, 999.f, 999.f, 999.f,
        112.f, 113.f, 114.f, 115.f, 116.f, 117.f,
        118.f, 119.f, 120.f, 121.f, 122.f, 123.f,
        124.f, 125.f, 126.f, 127.f, 128.f, 129.f,
        130.f, 131.f, 132.f, 133.f, 134.f, 135.f,

        136.f, 137.f, 138.f, 139.f, 140.f, 141.f,
        142.f, 143.f, 144.f, 145.f, 146.f, 147.f,
        148.f, 149.f, 150.f, 151.f, 152.f, 153.f,
        154.f, 155.f, 156.f, 157.f, 158.f, 159.f,
        160.f, 161.f, 162.f, 163.f, 164.f, 165.f,
        166.f, 167.f, 168.f, 169.f, 170.f, 171.f,

        172.f, 173.f, 174.f, 175.f, 176.f, 177.f,
        178.f, 179.f, 180.f, 181.f, 182.f, 183.f,
        184.f, 185.f, 186.f, 187.f, 188.f, 189.f,
        190.f, 191.f, 192.f, 193.f, 194.f, 195.f,
        196.f, 197.f, 198.f, 199.f, 200.f, 201.f,
        202.f, 203.f, 204.f, 205.f, 206.f, 207.f,

        208.f, 209.f, 210.f, 211.f, 212.f, 213.f,
        214.f, 215.f, 216.f, 217.f, 218.f, 219.f,
        220.f, 221.f, 222.f, 223.f, 224.f, 225.f,
        226.f, 227.f, 228.f, 229.f, 230.f, 231.f,
        888.f, 888.f, 888.f, 888.f, 888.f, 888.f,
        238.f, 239.f, 240.f, 241.f, 242.f, 243.f,

        244.f, 245.f, 246.f, 247.f, 248.f, 249.f,
        250.f, 251.f, 252.f, 253.f, 254.f, 255.f,
        256.f, 257.f, 258.f, 259.f, 260.f, 261.f,
        262.f, 263.f, 264.f, 265.f, 266.f, 267.f,
        268.f, 269.f, 270.f, 271.f, 272.f, 273.f,
        274.f, 275.f, 276.f, 277.f, 278.f, 279.f,

        280.f, 281.f, 282.f, 283.f, 284.f, 285.f,
        286.f, 287.f, 288.f, 289.f, 290.f, 291.f,
        292.f, 293.f, 294.f, 295.f, 296.f, 297.f,
        298.f, 299.f, 300.f, 301.f, 302.f, 303.f,
        304.f, 305.f, 306.f, 307.f, 308.f, 309.f,
        310.f, 311.f, 312.f, 313.f, 314.f, 315.f,
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16, d6661_i2111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 6, 6, 1, 6 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 6, 1, 6 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f), FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f), FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),
        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f), FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f), FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),
        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f), FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),
        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f), FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        FLOAT16(136.f), FLOAT16(137.f), FLOAT16(138.f), FLOAT16(139.f), FLOAT16(140.f), FLOAT16(141.f),
        FLOAT16(142.f), FLOAT16(143.f), FLOAT16(144.f), FLOAT16(145.f), FLOAT16(146.f), FLOAT16(147.f),
        FLOAT16(148.f), FLOAT16(149.f), FLOAT16(150.f), FLOAT16(151.f), FLOAT16(152.f), FLOAT16(153.f),
        FLOAT16(154.f), FLOAT16(155.f), FLOAT16(156.f), FLOAT16(157.f), FLOAT16(158.f), FLOAT16(159.f),
        FLOAT16(160.f), FLOAT16(161.f), FLOAT16(162.f), FLOAT16(163.f), FLOAT16(164.f), FLOAT16(165.f),
        FLOAT16(166.f), FLOAT16(167.f), FLOAT16(168.f), FLOAT16(169.f), FLOAT16(170.f), FLOAT16(171.f),

        FLOAT16(172.f), FLOAT16(173.f), FLOAT16(174.f), FLOAT16(175.f), FLOAT16(176.f), FLOAT16(177.f),
        FLOAT16(178.f), FLOAT16(179.f), FLOAT16(180.f), FLOAT16(181.f), FLOAT16(182.f), FLOAT16(183.f),
        FLOAT16(184.f), FLOAT16(185.f), FLOAT16(186.f), FLOAT16(187.f), FLOAT16(188.f), FLOAT16(189.f),
        FLOAT16(190.f), FLOAT16(191.f), FLOAT16(192.f), FLOAT16(193.f), FLOAT16(194.f), FLOAT16(195.f),
        FLOAT16(196.f), FLOAT16(197.f), FLOAT16(198.f), FLOAT16(199.f), FLOAT16(200.f), FLOAT16(201.f),
        FLOAT16(202.f), FLOAT16(203.f), FLOAT16(204.f), FLOAT16(205.f), FLOAT16(206.f), FLOAT16(207.f),

        FLOAT16(208.f), FLOAT16(209.f), FLOAT16(210.f), FLOAT16(211.f), FLOAT16(212.f), FLOAT16(213.f),
        FLOAT16(214.f), FLOAT16(215.f), FLOAT16(216.f), FLOAT16(217.f), FLOAT16(218.f), FLOAT16(219.f),
        FLOAT16(220.f), FLOAT16(221.f), FLOAT16(222.f), FLOAT16(223.f), FLOAT16(224.f), FLOAT16(225.f),
        FLOAT16(226.f), FLOAT16(227.f), FLOAT16(228.f), FLOAT16(229.f), FLOAT16(230.f), FLOAT16(231.f),
        FLOAT16(232.f), FLOAT16(233.f), FLOAT16(234.f), FLOAT16(235.f), FLOAT16(236.f), FLOAT16(237.f),
        FLOAT16(238.f), FLOAT16(239.f), FLOAT16(240.f), FLOAT16(241.f), FLOAT16(242.f), FLOAT16(243.f),

        FLOAT16(244.f), FLOAT16(245.f), FLOAT16(246.f), FLOAT16(247.f), FLOAT16(248.f), FLOAT16(249.f),
        FLOAT16(250.f), FLOAT16(251.f), FLOAT16(252.f), FLOAT16(253.f), FLOAT16(254.f), FLOAT16(255.f),
        FLOAT16(256.f), FLOAT16(257.f), FLOAT16(258.f), FLOAT16(259.f), FLOAT16(260.f), FLOAT16(261.f),
        FLOAT16(262.f), FLOAT16(263.f), FLOAT16(264.f), FLOAT16(265.f), FLOAT16(266.f), FLOAT16(267.f),
        FLOAT16(268.f), FLOAT16(269.f), FLOAT16(270.f), FLOAT16(271.f), FLOAT16(272.f), FLOAT16(273.f),
        FLOAT16(274.f), FLOAT16(275.f), FLOAT16(276.f), FLOAT16(277.f), FLOAT16(278.f), FLOAT16(279.f),

        FLOAT16(280.f), FLOAT16(281.f), FLOAT16(282.f), FLOAT16(283.f), FLOAT16(284.f), FLOAT16(285.f),
        FLOAT16(286.f), FLOAT16(287.f), FLOAT16(288.f), FLOAT16(289.f), FLOAT16(290.f), FLOAT16(291.f),
        FLOAT16(292.f), FLOAT16(293.f), FLOAT16(294.f), FLOAT16(295.f), FLOAT16(296.f), FLOAT16(297.f),
        FLOAT16(298.f), FLOAT16(299.f), FLOAT16(300.f), FLOAT16(301.f), FLOAT16(302.f), FLOAT16(303.f),
        FLOAT16(304.f), FLOAT16(305.f), FLOAT16(306.f), FLOAT16(307.f), FLOAT16(308.f), FLOAT16(309.f),
        FLOAT16(310.f), FLOAT16(311.f), FLOAT16(312.f), FLOAT16(313.f), FLOAT16(314.f), FLOAT16(315.f),
        });

    set_values(input2, {
        FLOAT16(0.0f),
        FLOAT16(3.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(777.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(777.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(777.0f), FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(777.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(777.0f),

        FLOAT16(666.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(666.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(666.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(666.0f), FLOAT16(888.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(666.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(666.0f),
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        777.f, 999.f, 999.f, 999.f, 999.f, 999.f,
        999.f, 777.f, 999.f, 999.f, 999.f, 999.f,
        999.f, 999.f, 777.f, 999.f, 999.f, 999.f,
        999.f, 999.f, 999.f, 777.f, 999.f, 999.f,
        999.f, 999.f, 999.f, 999.f, 777.f, 999.f,
        999.f, 999.f, 999.f, 999.f, 999.f, 777.f,

        136.f, 137.f, 138.f, 139.f, 140.f, 141.f,
        142.f, 143.f, 144.f, 145.f, 146.f, 147.f,
        148.f, 149.f, 150.f, 151.f, 152.f, 153.f,
        154.f, 155.f, 156.f, 157.f, 158.f, 159.f,
        160.f, 161.f, 162.f, 163.f, 164.f, 165.f,
        166.f, 167.f, 168.f, 169.f, 170.f, 171.f,

        172.f, 173.f, 174.f, 175.f, 176.f, 177.f,
        178.f, 179.f, 180.f, 181.f, 182.f, 183.f,
        184.f, 185.f, 186.f, 187.f, 188.f, 189.f,
        190.f, 191.f, 192.f, 193.f, 194.f, 195.f,
        196.f, 197.f, 198.f, 199.f, 200.f, 201.f,
        202.f, 203.f, 204.f, 205.f, 206.f, 207.f,

        666.f, 888.f, 888.f, 888.f, 888.f, 888.f,
        888.f, 666.f, 888.f, 888.f, 888.f, 888.f,
        888.f, 888.f, 666.f, 888.f, 888.f, 888.f,
        888.f, 888.f, 888.f, 666.f, 888.f, 888.f,
        888.f, 888.f, 888.f, 888.f, 666.f, 888.f,
        888.f, 888.f, 888.f, 888.f, 888.f, 666.f,

        244.f, 245.f, 246.f, 247.f, 248.f, 249.f,
        250.f, 251.f, 252.f, 253.f, 254.f, 255.f,
        256.f, 257.f, 258.f, 259.f, 260.f, 261.f,
        262.f, 263.f, 264.f, 265.f, 266.f, 267.f,
        268.f, 269.f, 270.f, 271.f, 272.f, 273.f,
        274.f, 275.f, 276.f, 277.f, 278.f, 279.f,

        280.f, 281.f, 282.f, 283.f, 284.f, 285.f,
        286.f, 287.f, 288.f, 289.f, 290.f, 291.f,
        292.f, 293.f, 294.f, 295.f, 296.f, 297.f,
        298.f, 299.f, 300.f, 301.f, 302.f, 303.f,
        304.f, 305.f, 306.f, 307.f, 308.f, 309.f,
        310.f, 311.f, 312.f, 313.f, 314.f, 315.f,
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d3232_i2411) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 2, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 4, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),
        FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f),
        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),
        FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f),
        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),
        FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f),
        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f)
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(1.0f), FLOAT16(2.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f,
        102.f, 103.f,
        104.f, 105.f,

        106.f, 107.f,
        108.f, 109.f,
        110.f, 777.f,

        112.f, 113.f,
        114.f, 115.f,
        116.f, 117.f,

        118.f, 119.f,
        120.f, 121.f,
        122.f, 123.f,

        124.f, 125.f,
        126.f, 127.f,
        128.f, 129.f,

        130.f, 131.f,
        132.f, 133.f,
        134.f, 999.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d3232_i2311) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 2, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),
        FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f),
        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),
        FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f),
        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),
        FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f),
        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f)
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(2.0f),
        FLOAT16(2.0f), FLOAT16(1.0f), FLOAT16(2.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f,
        102.f, 103.f,
        104.f, 105.f,

        106.f, 107.f,
        108.f, 109.f,
        777.f, 777.f,

        112.f, 113.f,
        114.f, 115.f,
        116.f, 117.f,

        118.f, 119.f,
        120.f, 121.f,
        122.f, 123.f,

        124.f, 125.f,
        126.f, 127.f,
        128.f, 129.f,

        130.f, 131.f,
        132.f, 133.f,
        999.f, 999.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d3232_i2211) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 2, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 1, 2 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),
        FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f),
        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),
        FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f),
        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),
        FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f),
        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f)
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f,
        102.f, 103.f,
        104.f, 105.f,

        777.f, 777.f,
        777.f, 777.f,
        777.f, 777.f,

        112.f, 113.f,
        114.f, 115.f,
        116.f, 117.f,

        118.f, 119.f,
        120.f, 121.f,
        122.f, 123.f,

        124.f, 125.f,
        126.f, 127.f,
        128.f, 129.f,

        999.f, 999.f,
        999.f, 999.f,
        999.f, 999.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d3232_i2111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 2, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 2, 3 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),
        FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f),
        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),
        FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f),
        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),
        FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f),
        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f)
        });

    set_values(input2, {
        FLOAT16(0.0f),
        FLOAT16(2.0f)
        });

    set_values(input3, {
        FLOAT16(666.0f), FLOAT16(666.0f),
        FLOAT16(666.0f), FLOAT16(666.0f),
        FLOAT16(666.0f), FLOAT16(666.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(888.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(888.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        666.f, 666.f,
        666.f, 666.f,
        666.f, 666.f,

        777.f, 777.f,
        777.f, 777.f,
        777.f, 777.f,

        112.f, 113.f,
        114.f, 115.f,
        116.f, 117.f,

        118.f, 119.f,
        120.f, 121.f,
        122.f, 123.f,

        888.f, 888.f,
        888.f, 888.f,
        888.f, 888.f,

        999.f, 999.f,
        999.f, 999.f,
        999.f, 999.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16, d32323_i25111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 2, 3, 2, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 5, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 1, 1, 1, 1 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 2
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 3
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f)
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(2.0f),
        FLOAT16(2.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        124.f, 125.f, 126.f,
        127.f, 128.f, 777.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f,

        // 2
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        124.f, 125.f, 126.f,
        127.f, 128.f, 129.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f,

        // 3
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        124.f, 125.f, 126.f,
        127.f, 999.f, 129.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d32323_i24111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 2, 3, 2, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 4, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 3, 1, 1, 1 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 2
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 3
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f)
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        124.f, 125.f, 126.f,
        777.f, 777.f, 777.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f,

        // 2
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        124.f, 125.f, 126.f,
        127.f, 128.f, 129.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f,

        // 3
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        124.f, 125.f, 126.f,
        999.f, 999.f, 999.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d32323_i23111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 2, 3, 2, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 3, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 1, 1, 3 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 2
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 3
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f)
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(1.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        777.f, 777.f, 777.f,
        777.f, 777.f, 777.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f,

        // 2
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        124.f, 125.f, 126.f,
        127.f, 128.f, 129.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f,

        // 3
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        999.f, 999.f, 999.f,
        999.f, 999.f, 999.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d32323_i22111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 2, 3, 2, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 3, 1, 3, 2 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 2
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 3
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f)
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(2.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(555.0f), FLOAT16(555.0f), FLOAT16(555.0f),
        FLOAT16(555.0f), FLOAT16(555.0f), FLOAT16(555.0f),

        FLOAT16(666.0f), FLOAT16(666.0f), FLOAT16(666.0f),
        FLOAT16(666.0f), FLOAT16(666.0f), FLOAT16(666.0f),

        FLOAT16(444.0f), FLOAT16(444.0f), FLOAT16(444.0f),
        FLOAT16(444.0f), FLOAT16(444.0f), FLOAT16(444.0f),

        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f),

        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        555.f, 555.f, 555.f,
        555.f, 555.f, 555.f,

        666.f, 666.f, 666.f,
        666.f, 666.f, 666.f,

        444.f, 444.f, 444.f,
        444.f, 444.f, 444.f,

        // 2
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        124.f, 125.f, 126.f,
        127.f, 128.f, 129.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f,

        // 3
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        777.f, 777.f, 777.f,
        777.f, 777.f, 777.f,

        888.f, 888.f, 888.f,
        888.f, 888.f, 888.f,

        999.f, 999.f, 999.f,
        999.f, 999.f, 999.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d32323_i21111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 2, 3, 2, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 1, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 3, 2, 3 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 2
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f),

        // 3
        FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f),
        FLOAT16(103.f), FLOAT16(104.f), FLOAT16(105.f),

        FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f),
        FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f), FLOAT16(114.f),
        FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f),

        FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),
        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f),
        FLOAT16(127.f), FLOAT16(128.f), FLOAT16(129.f),

        FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f),
        FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f)
        });

    set_values(input2, {
        FLOAT16(0.0f),
        FLOAT16(2.0f)
        });

    set_values(input3, {
        FLOAT16(555.0f), FLOAT16(555.0f), FLOAT16(555.0f),
        FLOAT16(555.0f), FLOAT16(555.0f), FLOAT16(555.0f),

        FLOAT16(666.0f), FLOAT16(666.0f), FLOAT16(666.0f),
        FLOAT16(666.0f), FLOAT16(666.0f), FLOAT16(666.0f),

        FLOAT16(444.0f), FLOAT16(444.0f), FLOAT16(444.0f),
        FLOAT16(444.0f), FLOAT16(444.0f), FLOAT16(444.0f),

        FLOAT16(555.0f), FLOAT16(555.0f), FLOAT16(555.0f),
        FLOAT16(555.0f), FLOAT16(555.0f), FLOAT16(555.0f),

        FLOAT16(666.0f), FLOAT16(666.0f), FLOAT16(666.0f),
        FLOAT16(666.0f), FLOAT16(666.0f), FLOAT16(666.0f),

        FLOAT16(444.0f), FLOAT16(444.0f), FLOAT16(444.0f),
        FLOAT16(444.0f), FLOAT16(444.0f), FLOAT16(444.0f),

        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f),

        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f),
        FLOAT16(888.0f), FLOAT16(888.0f), FLOAT16(888.0f),

        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        555.f, 555.f, 555.f,
        555.f, 555.f, 555.f,

        666.f, 666.f, 666.f,
        666.f, 666.f, 666.f,

        444.f, 444.f, 444.f,
        444.f, 444.f, 444.f,

        555.f, 555.f, 555.f,
        555.f, 555.f, 555.f,

        666.f, 666.f, 666.f,
        666.f, 666.f, 666.f,

        444.f, 444.f, 444.f,
        444.f, 444.f, 444.f,

        // 2
        100.f, 101.f, 102.f,
        103.f, 104.f, 105.f,

        106.f, 107.f, 108.f,
        109.f, 110.f, 111.f,

        112.f, 113.f, 114.f,
        115.f, 116.f, 117.f,

        118.f, 119.f, 120.f,
        121.f, 122.f, 123.f,

        124.f, 125.f, 126.f,
        127.f, 128.f, 129.f,

        130.f, 131.f, 132.f,
        133.f, 134.f, 135.f,

        // 3
        777.f, 777.f, 777.f,
        777.f, 777.f, 777.f,

        888.f, 888.f, 888.f,
        888.f, 888.f, 888.f,

        999.f, 999.f, 999.f,
        999.f, 999.f, 999.f,

        777.f, 777.f, 777.f,
        777.f, 777.f, 777.f,

        888.f, 888.f, 888.f,
        888.f, 888.f, 888.f,

        999.f, 999.f, 999.f,
        999.f, 999.f, 999.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d222222_i261111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    // memory order is bfxyzw
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 2, 2, 2, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 6, 1, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 1, 1, 1, 1, 1 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),//1

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),//2

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),//3

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),

        FLOAT16(128.f), FLOAT16(129.f),
        FLOAT16(130.f), FLOAT16(131.f),//4

        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f),

        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),//5

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),//6

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),//7

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),//8
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(0.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(0.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f,
        102.f, 103.f,

        104.f, 105.f,
        106.f, 107.f,//1

        108.f, 109.f,
        110.f, 111.f,

        112.f, 113.f,
        114.f, 115.f,//2

        116.f, 117.f,
        118.f, 119.f,

        120.f, 121.f,
        122.f, 123.f,//3

        124.f, 125.f,
        126.f, 127.f,

        128.f, 129.f,
        777.f, 131.f,//4

        132.f, 133.f,
        134.f, 135.f,

        100.f, 101.f,
        102.f, 103.f,//5

        104.f, 105.f,
        106.f, 107.f,

        108.f, 109.f,
        110.f, 111.f,//6

        112.f, 113.f,
        114.f, 115.f,

        116.f, 117.f,
        118.f, 119.f,//7

        120.f, 121.f,
        122.f, 123.f,

        124.f, 125.f,
        999.f, 127.f,//8
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d222222_i251111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    // memory order is bfxyzw
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 2, 2, 2, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 5, 1, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 1, 1, 1, 1 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),//1

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),//2

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),//3

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),

        FLOAT16(128.f), FLOAT16(129.f),
        FLOAT16(130.f), FLOAT16(131.f),//4

        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f),

        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),//5

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),//6

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),//7

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),//8
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f,
        102.f, 103.f,

        104.f, 105.f,
        106.f, 107.f,//1

        108.f, 109.f,
        110.f, 111.f,

        112.f, 113.f,
        114.f, 115.f,//2

        116.f, 117.f,
        118.f, 119.f,

        120.f, 121.f,
        122.f, 123.f,//3

        124.f, 125.f,
        126.f, 127.f,

        128.f, 129.f,
        777.f, 777.f,//4

        132.f, 133.f,
        134.f, 135.f,

        100.f, 101.f,
        102.f, 103.f,//5

        104.f, 105.f,
        106.f, 107.f,

        108.f, 109.f,
        110.f, 111.f,//6

        112.f, 113.f,
        114.f, 115.f,

        116.f, 117.f,
        118.f, 119.f,//7

        120.f, 121.f,
        122.f, 123.f,

        124.f, 125.f,
        999.f, 999.f,//8
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d222222_i241111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    // memory order is bfxyzw
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 2, 2, 2, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 4, 1, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 1, 1, 1, 2 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),//1

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),//2

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),//3

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),

        FLOAT16(128.f), FLOAT16(129.f),
        FLOAT16(130.f), FLOAT16(131.f),//4

        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f),

        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),//5

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),//6

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),//7

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),//8
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f,
        102.f, 103.f,

        104.f, 105.f,
        106.f, 107.f,//1

        108.f, 109.f,
        110.f, 111.f,

        112.f, 113.f,
        114.f, 115.f,//2

        116.f, 117.f,
        118.f, 119.f,

        120.f, 121.f,
        122.f, 123.f,//3

        124.f, 125.f,
        126.f, 127.f,

        777.f, 777.f,
        777.f, 777.f,//4

        132.f, 133.f,
        134.f, 135.f,

        100.f, 101.f,
        102.f, 103.f,//5

        104.f, 105.f,
        106.f, 107.f,

        108.f, 109.f,
        110.f, 111.f,//6

        112.f, 113.f,
        114.f, 115.f,

        116.f, 117.f,
        118.f, 119.f,//7

        120.f, 121.f,
        122.f, 123.f,

        999.f, 999.f,
        999.f, 999.f,//8
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}



TEST(scatter_nd_update_gpu_fp16, d222222_i231111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    // memory order is bfxyzw
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 2, 2, 2, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 3, 1, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 1, 1, 2, 2 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),//1

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),//2

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),//3

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),

        FLOAT16(128.f), FLOAT16(129.f),
        FLOAT16(130.f), FLOAT16(131.f),//4

        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f),

        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),//5

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),//6

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),//7

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),//8
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f), FLOAT16(1.0f),
        FLOAT16(1.0f), FLOAT16(1.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f,
        102.f, 103.f,

        104.f, 105.f,
        106.f, 107.f,//1

        108.f, 109.f,
        110.f, 111.f,

        112.f, 113.f,
        114.f, 115.f,//2

        116.f, 117.f,
        118.f, 119.f,

        120.f, 121.f,
        122.f, 123.f,//3

        777.f, 777.f,
        777.f, 777.f,

        777.f, 777.f,
        777.f, 777.f,//4

        132.f, 133.f,
        134.f, 135.f,

        100.f, 101.f,
        102.f, 103.f,//5

        104.f, 105.f,
        106.f, 107.f,

        108.f, 109.f,
        110.f, 111.f,//6

        112.f, 113.f,
        114.f, 115.f,

        116.f, 117.f,
        118.f, 119.f,//7

        999.f, 999.f,
        999.f, 999.f,

        999.f, 999.f,
        999.f, 999.f,//8
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16, d222222_i221111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    // memory order is bfxyzw
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 2, 2, 2, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 1, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 1, 2, 2, 2 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),//1

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),//2

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),//3

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),

        FLOAT16(128.f), FLOAT16(129.f),
        FLOAT16(130.f), FLOAT16(131.f),//4

        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f),

        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),//5

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),//6

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),//7

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),//8
        });

    set_values(input2, {
        FLOAT16(0.0f), FLOAT16(1.0f),
        FLOAT16(1.0f), FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        100.f, 101.f,
        102.f, 103.f,

        104.f, 105.f,
        106.f, 107.f,//1

        108.f, 109.f,
        110.f, 111.f,

        112.f, 113.f,
        114.f, 115.f,//2

        777.f, 777.f,
        777.f, 777.f,

        777.f, 777.f,
        777.f, 777.f,//3

        777.f, 777.f,
        777.f, 777.f,

        777.f, 777.f,
        777.f, 777.f,//4

        132.f, 133.f,
        134.f, 135.f,

        100.f, 101.f,
        102.f, 103.f,//5

        104.f, 105.f,
        106.f, 107.f,

        108.f, 109.f,
        110.f, 111.f,//6

        999.f, 999.f,
        999.f, 999.f,

        999.f, 999.f,
        999.f, 999.f,//7

        999.f, 999.f,
        999.f, 999.f,

        999.f, 999.f,
        999.f, 999.f,//8
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16, d222222_i211111) {
    //  Dictionary : 6x6x6x1
    //  Indexes : 2x1x1x1
    //  Updates : 2x6x1x6
    //  Output : 6x6x6x1
    //  Input values in fp16
    //

    auto& engine = get_test_engine();

    // memory order is bfxyzw
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 2, 2, 2, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 1, 1, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 2, 2, 2, 2 } }); // Updates


    set_values(input1, {
        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),//1

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),//2

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),//3

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),

        FLOAT16(128.f), FLOAT16(129.f),
        FLOAT16(130.f), FLOAT16(131.f),//4

        FLOAT16(132.f), FLOAT16(133.f),
        FLOAT16(134.f), FLOAT16(135.f),

        FLOAT16(100.f), FLOAT16(101.f),
        FLOAT16(102.f), FLOAT16(103.f),//5

        FLOAT16(104.f), FLOAT16(105.f),
        FLOAT16(106.f), FLOAT16(107.f),

        FLOAT16(108.f), FLOAT16(109.f),
        FLOAT16(110.f), FLOAT16(111.f),//6

        FLOAT16(112.f), FLOAT16(113.f),
        FLOAT16(114.f), FLOAT16(115.f),

        FLOAT16(116.f), FLOAT16(117.f),
        FLOAT16(118.f), FLOAT16(119.f),//7

        FLOAT16(120.f), FLOAT16(121.f),
        FLOAT16(122.f), FLOAT16(123.f),

        FLOAT16(124.f), FLOAT16(125.f),
        FLOAT16(126.f), FLOAT16(127.f),//8
        });

    set_values(input2, {
        FLOAT16(0.0f),
        FLOAT16(1.0f)
        });

    set_values(input3, {
        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(777.0f), FLOAT16(777.0f),
        FLOAT16(777.0f), FLOAT16(777.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f),

        FLOAT16(999.0f), FLOAT16(999.0f),
        FLOAT16(999.0f), FLOAT16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", "InputData", "InputIndices", "InputUpdates", 2)
    );

    network network(engine, topology);


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();


    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        777.f, 777.f,
        777.f, 777.f,

        777.f, 777.f,
        777.f, 777.f,//1

        777.f, 777.f,
        777.f, 777.f,

        777.f, 777.f,
        777.f, 777.f,//2

        777.f, 777.f,
        777.f, 777.f,

        777.f, 777.f,
        777.f, 777.f,//3

        777.f, 777.f,
        777.f, 777.f,

        777.f, 777.f,
        777.f, 777.f,//4

        999.f, 999.f,
        999.f, 999.f,

        999.f, 999.f,
        999.f, 999.f,//5

        999.f, 999.f,
        999.f, 999.f,

        999.f, 999.f,
        999.f, 999.f,//6

        999.f, 999.f,
        999.f, 999.f,

        999.f, 999.f,
        999.f, 999.f,//7

        999.f, 999.f,
        999.f, 999.f,

        999.f, 999.f,
        999.f, 999.f,//8
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}
