// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "openvino/reference/scatter_nd_update.hpp"
#include "scatter_nd_update_inst.h"

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
    tests::random_generator rg;

    void SetUp() override {
        std::string suite_name = std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
                                 std::string(::testing::UnitTest::GetInstance()->current_test_info()->name());
        rg.set_seed(suite_name);
    }

    format get_default_format(int rank = 4) {
        if (rank <= 4)
            return cldnn::format::bfyx;
        else if (rank == 5)
            return cldnn::format::bfzyx;
        else
            return cldnn::format::bfwzyx;
    }

    template <typename T>
    std::vector<T> generate_unique_indices(const scatter_nd_update_basic_test_params& p) {
        std::set<std::vector<T>> unique_indices;
        std::vector<T> result;
        auto indices_shape = p.indices_size.sizes(get_default_format(p.indices_rank));
        auto data_shape = p.input_size.sizes(p.input_format);
        size_t last_indices_dim = indices_shape.at(p.indices_rank - 1);

        auto count = p.indices_size.count() / last_indices_dim;

        while (unique_indices.size() != count) {
            std::vector<T> indices;
            for (size_t i = 0; i < last_indices_dim; i++) {
                indices.push_back(static_cast<T>(rg.generate_random_val<int>(0, data_shape[i] - 1)));
            }

            unique_indices.insert(indices);
        }

        std::for_each(unique_indices.begin(),
                      unique_indices.end(),
                      [&](const std::vector<T>& indices) {
                          result.insert(result.end(), indices.begin(), indices.end());
                      });

        return result;
    }

    template<typename T, typename T_size>
    void execute_fp16(const scatter_nd_update_basic_test_params& params, bool is_caching_test)
    {
        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ params.input_type, params.input_format, params.input_size });
        auto input2 = engine.allocate_memory({ params.indices_type, params.indices_format, params.indices_size });
        auto input3 = engine.allocate_memory({ params.updates_type, params.updates_format, params.updates_size });

        std::vector<int> input_vec(static_cast<int>(cldnn::format::dimension(params.input_format)));
        for (size_t i = 0; i < input_vec.size(); ++i)
            input_vec[i] = static_cast<int>(params.input_size.sizes()[i]);
        std::reverse(input_vec.begin() + 2, input_vec.end());

        std::vector<int> updates_vec(static_cast<int>(cldnn::format::dimension(params.updates_format)));
        for (size_t i = 0; i < updates_vec.size(); ++i)
            updates_vec[i] = static_cast<int>(params.updates_size.sizes()[i]);
        std::reverse(updates_vec.begin() + 2, updates_vec.end());

        std::vector<int> indices_vec(static_cast<int>(cldnn::format::dimension(params.indices_format)));
        for (size_t i = 0; i < indices_vec.size(); ++i)
            indices_vec[i] = static_cast<int>(params.indices_size.sizes()[i]);
        std::reverse(indices_vec.begin() + 2, indices_vec.end());
        indices_vec.resize(params.indices_rank);

        auto input_data_fp16 = rg.generate_random_1d<T>(params.input_size.count(), -127, 127);
        auto indices_data_fp16 = generate_unique_indices<T>(params);
        auto updates_data_fp16 = rg.generate_random_1d<T>(params.updates_size.count(), -127, 127);

        std::vector<float> input_data(params.input_size.count());
        for (size_t i = 0; i < params.input_size.count(); ++i)
            input_data[i] = static_cast<float>(input_data_fp16[i]);
        std::vector<float> indices_data(params.indices_size.count());
        for (size_t i = 0; i < params.indices_size.count(); ++i)
            indices_data[i] = static_cast<float>(indices_data_fp16[i]);
        std::vector<float> updates_data(params.updates_size.count());
        for (size_t i = 0; i < params.updates_size.count(); ++i)
            updates_data[i] = static_cast<float>(updates_data_fp16[i]);

        set_values(input1, input_data_fp16);
        set_values(input2, indices_data_fp16);
        set_values(input3, updates_data_fp16);

        // execute scatter_nd_update
        topology topology(
            input_layout("InputData", input1->get_layout()),
            input_layout("InputIndices", input2->get_layout()),
            input_layout("InputUpdates", input3->get_layout()),
            reorder("reorder1", input_info("InputData"), params.input_result_format, params.input_type),
            reorder("reorder2", input_info("InputIndices"), params.indices_result_format, params.indices_type),
            reorder("reorder3", input_info("InputUpdates"), params.updates_result_format, params.updates_type),
            scatter_nd_update("scatter_nd_update", input_info("reorder1"), input_info("reorder2"), input_info("reorder3"), params.indices_rank),
            reorder("out", input_info("scatter_nd_update"), params.input_format, params.input_type)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("InputData", input1);
        network->set_input_data("InputIndices", input2);
        network->set_input_data("InputUpdates", input3);

        auto outputs = network->execute();
        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<T_size> outputs_ptr(output, get_test_stream());

        auto outputs_ref = std::vector<float>(params.input_size.count());
        ov::reference::scatterNdUpdate<float, float>(input_data.data(),
                                                     indices_data.data(),
                                                     updates_data.data(),
                                                     outputs_ref.data(),
                                                     ov::Shape(input_vec.begin(), input_vec.end()),
                                                     ov::Shape(indices_vec.begin(), indices_vec.end()),
                                                     ov::Shape(updates_vec.begin(), updates_vec.end()));

        for (size_t i = 0; i < outputs_ref.size(); ++i) {
            ASSERT_EQ(outputs_ref[i], half_to_float(outputs_ptr[i]));
        }
    }

    template<typename T>
    void execute(const scatter_nd_update_basic_test_params& params, bool is_caching_test)
    {
        // create input, indices, updates using params
        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ params.input_type, params.input_format, params.input_size });
        auto input2 = engine.allocate_memory({ params.indices_type, params.indices_format, params.indices_size });
        auto input3 = engine.allocate_memory({ params.updates_type, params.updates_format, params.updates_size });

        std::vector<int> input_vec(static_cast<int>(cldnn::format::dimension(params.input_format)));
        for (size_t i = 0; i < input_vec.size(); ++i)
            input_vec[i] = static_cast<int>(params.input_size.sizes()[i]);
        std::reverse(input_vec.begin() + 2, input_vec.end());

        std::vector<int> updates_vec(static_cast<int>(cldnn::format::dimension(params.updates_format)));
        for (size_t i = 0; i < updates_vec.size(); ++i)
            updates_vec[i] = static_cast<int>(params.updates_size.sizes()[i]);
        std::reverse(updates_vec.begin() + 2, updates_vec.end());

        std::vector<int> indices_vec(static_cast<int>(cldnn::format::dimension(params.indices_format)));
        for (size_t i = 0; i < indices_vec.size(); ++i)
            indices_vec[i] = static_cast<int>(params.indices_size.sizes()[i]);
        std::reverse(indices_vec.begin() + 2, indices_vec.end());
        indices_vec.resize(params.indices_rank);

        auto input_data = rg.generate_random_1d<T>(params.input_size.count(), -127, 127);
        auto indices_data = generate_unique_indices<T>(params);
        auto updates_data = rg.generate_random_1d<T>(params.updates_size.count(), -127, 127);

        set_values(input1, input_data);
        set_values(input2, indices_data);
        set_values(input3, updates_data);

        // execute scatter_nd_update
        topology topology(
            input_layout("InputData", input1->get_layout()),
            input_layout("InputIndices", input2->get_layout()),
            input_layout("InputUpdates", input3->get_layout()),
            reorder("reorder1", input_info("InputData"), params.input_result_format, params.input_type),
            reorder("reorder2", input_info("InputIndices"), params.indices_result_format, params.indices_type),
            reorder("reorder3", input_info("InputUpdates"), params.updates_result_format, params.updates_type),
            scatter_nd_update("scatter_nd_update", input_info("reorder1"), input_info("reorder2"), input_info("reorder3"), params.indices_rank),
            reorder("out", input_info("scatter_nd_update"), params.input_format, params.input_type)
        );

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("InputData", input1);
        network->set_input_data("InputIndices", input2);
        network->set_input_data("InputUpdates", input3);

        auto outputs = network->execute();
        auto output = outputs.at("out").get_memory();
        cldnn::mem_lock<T> outputs_ptr(output, get_test_stream());

        auto outputs_ref = std::vector<T>(params.input_size.count());
        ov::reference::scatterNdUpdate<T, T>(input_data.data(),
                                             indices_data.data(),
                                             updates_data.data(),
                                             outputs_ref.data(),
                                             ov::Shape(input_vec.begin(), input_vec.end()),
                                             ov::Shape(indices_vec.begin(), indices_vec.end()),
                                             ov::Shape(updates_vec.begin(), updates_vec.end()));

        for (size_t i = 0; i < outputs_ref.size(); ++i) {
            ASSERT_EQ(outputs_ref[i], outputs_ptr[i]);
        }
    }
};

TEST_P(scatter_nd_update_random_test, random)
{
    auto param = GetParam();
    if (param.input_type == data_types::u8)
        this->execute<uint8_t>(param, false);
    else if (param.input_type == data_types::i8)
        this->execute<int8_t>(param, false);
    else if (param.input_type == data_types::i32)
        this->execute<int32_t>(param, false);
    else if (param.input_type == data_types::i64)
        this->execute<int64_t>(param, false);
    else if (param.input_type == data_types::f16)
        this->execute_fp16<ov::float16, uint16_t>(param, false);
    else if (param.input_type == data_types::f32)
        this->execute<float>(param, false);
    else
        OPENVINO_THROW("unidentified data type");
}

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp32_bsv32_fsv16_4d_rank_1,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16,
                               { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 },
                               1 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp32_bsv32_fsv16_4d_rank_2,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16,
                               { 48, 24, 3, 3 }, { 3, 2, 1, 1 }, { 3, 3, 1, 3 },
                               2 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp32_fsv16_4d_rank_1,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16,
                               { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 },
                               1 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp32_fsv16_4d_rank_2,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16,
                               { 48, 24, 3, 3 }, { 3, 2, 1, 1 }, { 3, 3, 1, 3 },
                               2 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp32_fsv16_5d_rank_2,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32,
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv16, format::b_fs_yx_fsv16, format::b_fs_zyx_fsv16,
                               { 6, 7, 3, 3, 10 }, { 5, 2, 1, 1 }, { 5, 10, 1, 3, 3 },
                               2 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp32_fsv16_5d_rank_3,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32,
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv16, format::b_fs_yx_fsv16, format::b_fs_zyx_fsv16,
                               { 6, 7, 8, 9, 10 }, { 5, 2, 1, 2 }, { 5, 2, 8, 9, 10 },
                               3 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp32_fsv16_5d_rank_4,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f32, data_types::f32, data_types::f32,
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv16, format::b_fs_yx_fsv16, format::b_fs_zyx_fsv16,
                               { 6, 7, 8, 9, 10 }, { 5, 2, 4, 3 }, { 5, 2, 1, 8, 3 },
                               4 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp16_fsv16_4d_rank_1,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16,
                               { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 },
                               1 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp16_fsv16_4d_rank_2,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16,
                               { 48, 24, 3, 3 }, { 3, 2, 1, 1 }, { 3, 3, 1, 3 },
                               2 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp16_fsv16_5d_rank_1,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16,
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv16, format::b_fs_yx_fsv16, format::b_fs_zyx_fsv16,
                               { 6, 7, 8, 9, 10 }, { 5, 1, 1, 1 }, { 5, 7, 8, 9, 10 },
                               1 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp16_fsv16_5d_rank_2,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16,
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv16, format::b_fs_yx_fsv16, format::b_fs_zyx_fsv16,
                               { 6, 7, 8, 9, 10 }, { 5, 4, 1, 1 }, { 5, 8, 1, 1, 1 },
                               2 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp16_fsv16_5d_rank_3,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16,
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv16, format::b_fs_yx_fsv16, format::b_fs_zyx_fsv16,
                               { 6, 7, 8, 9, 10 }, { 5, 2, 1, 3 }, { 5, 2, 1, 8, 9 },
                               3 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp16_fsv16_5d_rank_4,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16,
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv16, format::b_fs_yx_fsv16, format::b_fs_zyx_fsv16,
                               { 6, 7, 8, 9, 10 }, { 5, 2, 4, 3 }, { 5, 2, 1, 8, 3 },
                               4 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp16_bsv32_fsv16_4d_rank_1,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16,
                               { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 },
                               1 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_fp16_bsv32_fsv16_4d_rank_2,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::f16, data_types::f16, data_types::f16,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16,
                               { 48, 24, 3, 3 },  {3, 2, 1, 1 }, { 3, 3, 1, 3 },
                               2 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_i8_bsv32_fsv16_4d_rank_2,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16, format::bs_fs_yx_bsv32_fsv16,
                               { 41, 23, 3, 3 }, { 3, 2, 1, 1 }, { 3, 3, 1, 3 },
                               2 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_i8_bsv32_fsv32_4d_rank_1,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv32,
                               { 6, 1, 1, 1 }, { 3, 1, 1, 1 }, { 3, 1, 1, 1 },
                               1 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_i8_fsv32_4d_rank_2,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv32, format::b_fs_yx_fsv32, format::b_fs_yx_fsv32,
                               { 41, 23, 3, 3 }, { 3, 2, 1, 1 }, { 3, 3, 1, 3 },
                               2 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_i8_fsv32_5d_rank_3,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8,
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv32, format::b_fs_yx_fsv32, format::b_fs_zyx_fsv32,
                               { 6, 7, 8, 9, 10 }, { 5, 2, 1, 2 }, { 5, 2, 8, 9, 10 },
                               3 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_i8_fsv16_4d_rank_2,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8,
                               format::bfyx, format::bfyx, format::bfyx,
                               format::b_fs_yx_fsv16, format::b_fs_yx_fsv16, format::b_fs_yx_fsv16,
                               { 41, 23, 3, 3 }, { 3, 2, 1, 1 }, { 3, 3, 1, 3 },
                               2 }
                         }));

INSTANTIATE_TEST_SUITE_P(scatter_nd_update_gpu_random_test_i8_fsv16_5d_rank_4,
                         scatter_nd_update_random_test,
                         testing::ValuesIn(
                             std::vector<scatter_nd_update_basic_test_params>{
                             { data_types::i8, data_types::i8, data_types::i8,
                               format::bfzyx, format::bfyx, format::bfzyx,
                               format::b_fs_zyx_fsv16, format::b_fs_yx_fsv16, format::b_fs_zyx_fsv16,
                               { 6, 7, 8, 9, 10 }, { 5, 2, 3, 3 }, { 5, 2, 8, 9, 3 },
                               4 }
                         }));


TEST(scatter_nd_update_gpu_fp16_test15, data5_indice3_update5) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 2, 4, 3 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx,  { 1, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 2, 4, 3, 2 } }); // updates

    set_values(input1, {
        // 0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        // 1
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
    });

    set_values(input2, {
        ov::float16(1.0f),
        ov::float16(0.0f),
    });

    set_values(input3, {
        // 0
        ov::float16(91.0f), ov::float16(2.0f),    ov::float16(83.0f), ov::float16(4.0f),      ov::float16(71.0f), ov::float16(2.0f),   ov::float16(63.0f), ov::float16(4.0f),
        ov::float16(95.0f), ov::float16(6.0f),    ov::float16(87.0f), ov::float16(8.0f),      ov::float16(75.0f), ov::float16(6.0f),   ov::float16(67.0f), ov::float16(8.0f),
        ov::float16(99.0f), ov::float16(10.0f),   ov::float16(811.0f), ov::float16(12.0f),    ov::float16(79.0f), ov::float16(10.0f),  ov::float16(611.0f), ov::float16(12.0f),

        ov::float16(91.0f), ov::float16(2.0f),    ov::float16(83.0f), ov::float16(4.0f),      ov::float16(71.0f), ov::float16(2.0f),   ov::float16(63.0f), ov::float16(4.0f),
        ov::float16(95.0f), ov::float16(6.0f),    ov::float16(87.0f), ov::float16(8.0f),      ov::float16(75.0f), ov::float16(6.0f),   ov::float16(67.0f), ov::float16(8.0f),
        ov::float16(99.0f), ov::float16(10.0f),   ov::float16(811.0f), ov::float16(12.0f),    ov::float16(79.0f), ov::float16(10.0f),  ov::float16(611.0f), ov::float16(12.0f),
        // 1
        ov::float16(91.0f), ov::float16(2.0f),    ov::float16(83.0f), ov::float16(4.0f),      ov::float16(71.0f), ov::float16(2.0f),   ov::float16(63.0f), ov::float16(4.0f),
        ov::float16(95.0f), ov::float16(6.0f),    ov::float16(87.0f), ov::float16(8.0f),      ov::float16(75.0f), ov::float16(6.0f),   ov::float16(67.0f), ov::float16(8.0f),
        ov::float16(99.0f), ov::float16(10.0f),   ov::float16(811.0f), ov::float16(12.0f),    ov::float16(79.0f), ov::float16(10.0f),  ov::float16(611.0f), ov::float16(12.0f),

        ov::float16(91.0f), ov::float16(2.0f),    ov::float16(83.0f), ov::float16(4.0f),      ov::float16(71.0f), ov::float16(2.0f),   ov::float16(63.0f), ov::float16(4.0f),
        ov::float16(95.0f), ov::float16(6.0f),    ov::float16(87.0f), ov::float16(8.0f),      ov::float16(75.0f), ov::float16(6.0f),   ov::float16(67.0f), ov::float16(8.0f),
        ov::float16(99.0f), ov::float16(10.0f),   ov::float16(811.0f), ov::float16(12.0f),    ov::float16(79.0f), ov::float16(10.0f),  ov::float16(611.0f), ov::float16(12.0f),
    });

    std::vector<float> expected_results = {
        // 0
        ov::float16(91.0f), ov::float16(2.0f),    ov::float16(83.0f), ov::float16(4.0f),      ov::float16(71.0f), ov::float16(2.0f),   ov::float16(63.0f), ov::float16(4.0f),
        ov::float16(95.0f), ov::float16(6.0f),    ov::float16(87.0f), ov::float16(8.0f),      ov::float16(75.0f), ov::float16(6.0f),   ov::float16(67.0f), ov::float16(8.0f),
        ov::float16(99.0f), ov::float16(10.0f),   ov::float16(811.0f), ov::float16(12.0f),    ov::float16(79.0f), ov::float16(10.0f),  ov::float16(611.0f), ov::float16(12.0f),

        ov::float16(91.0f), ov::float16(2.0f),    ov::float16(83.0f), ov::float16(4.0f),      ov::float16(71.0f), ov::float16(2.0f),   ov::float16(63.0f), ov::float16(4.0f),
        ov::float16(95.0f), ov::float16(6.0f),    ov::float16(87.0f), ov::float16(8.0f),      ov::float16(75.0f), ov::float16(6.0f),   ov::float16(67.0f), ov::float16(8.0f),
        ov::float16(99.0f), ov::float16(10.0f),   ov::float16(811.0f), ov::float16(12.0f),    ov::float16(79.0f), ov::float16(10.0f),  ov::float16(611.0f), ov::float16(12.0f),
        // 1
        ov::float16(91.0f), ov::float16(2.0f),    ov::float16(83.0f), ov::float16(4.0f),      ov::float16(71.0f), ov::float16(2.0f),   ov::float16(63.0f), ov::float16(4.0f),
        ov::float16(95.0f), ov::float16(6.0f),    ov::float16(87.0f), ov::float16(8.0f),      ov::float16(75.0f), ov::float16(6.0f),   ov::float16(67.0f), ov::float16(8.0f),
        ov::float16(99.0f), ov::float16(10.0f),   ov::float16(811.0f), ov::float16(12.0f),    ov::float16(79.0f), ov::float16(10.0f),  ov::float16(611.0f), ov::float16(12.0f),

        ov::float16(91.0f), ov::float16(2.0f),    ov::float16(83.0f), ov::float16(4.0f),      ov::float16(71.0f), ov::float16(2.0f),   ov::float16(63.0f), ov::float16(4.0f),
        ov::float16(95.0f), ov::float16(6.0f),    ov::float16(87.0f), ov::float16(8.0f),      ov::float16(75.0f), ov::float16(6.0f),   ov::float16(67.0f), ov::float16(8.0f),
        ov::float16(99.0f), ov::float16(10.0f),   ov::float16(811.0f), ov::float16(12.0f),    ov::float16(79.0f), ov::float16(10.0f),  ov::float16(611.0f), ov::float16(12.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 3)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test14, data5_indice2_update3) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 2, 4, 3 } }); // data 2x2x3x4x2 (bfzyx)
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx,  { 3, 3, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 4, 1, 1, 2 } }); // updates

    set_values(input1, {
        // 0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        // 1
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(1.0f), ov::float16(1.0f), ov::float16(2.0f),
        ov::float16(1.0f), ov::float16(1.0f), ov::float16(0.0f),
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(1.0f),
        });

    set_values(input3, {
        ov::float16(51.0f), ov::float16(52.0f), ov::float16(53.0f), ov::float16(54.0f), ov::float16(55.0f), ov::float16(56.0f), ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(61.0f), ov::float16(62.0f), ov::float16(63.0f), ov::float16(64.0f), ov::float16(65.0f), ov::float16(66.0f), ov::float16(67.0f), ov::float16(68.0f),
        ov::float16(71.0f), ov::float16(72.0f), ov::float16(73.0f), ov::float16(74.0f), ov::float16(75.0f), ov::float16(76.0f), ov::float16(77.0f), ov::float16(78.0f),
        });

    std::vector<float> expected_results = {
        // 0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(71.0f), ov::float16(72.0f), ov::float16(73.0f), ov::float16(74.0f),     ov::float16(75.0f), ov::float16(76.0f), ov::float16(77.0f), ov::float16(78.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        // 1
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(61.0f), ov::float16(62.0f), ov::float16(63.0f), ov::float16(64.0f),     ov::float16(65.0f), ov::float16(66.0f), ov::float16(67.0f), ov::float16(68.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(51.0f), ov::float16(52.0f), ov::float16(53.0f), ov::float16(54.0f),     ov::float16(55.0f), ov::float16(56.0f), ov::float16(57.0f), ov::float16(58.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test13, data4_indice2_update2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 4, 2 } }); // data 2x3x2x4 (bfyx)
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 1 } }); // updates

    set_values(input1, {
        ov::float16(1.0f), ov::float16(2.0f),  ov::float16(3.0f), ov::float16(4.0f),       ov::float16(1.0f), ov::float16(2.0f),  ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),  ov::float16(7.0f), ov::float16(8.0f),       ov::float16(5.0f), ov::float16(6.0f),  ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),     ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),  ov::float16(3.0f), ov::float16(4.0f),       ov::float16(1.0f), ov::float16(2.0f),  ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),  ov::float16(7.0f), ov::float16(8.0f),       ov::float16(5.0f), ov::float16(6.0f),  ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),     ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(1.0f), ov::float16(1.0f), ov::float16(0.0f),
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(1.0f),
        ov::float16(0.0f), ov::float16(2.0f), ov::float16(1.0f),
        });

    set_values(input3, {
        ov::float16(51.0f), ov::float16(52.0f), ov::float16(53.0f), ov::float16(54.0f),
        ov::float16(61.0f), ov::float16(62.0f), ov::float16(63.0f), ov::float16(64.0f),
        ov::float16(71.0f), ov::float16(72.0f), ov::float16(73.0f), ov::float16(74.0f),
        });

    std::vector<float> expected_results = {
        ov::float16(1.0f), ov::float16(2.0f),  ov::float16(3.0f), ov::float16(4.0f),       ov::float16(1.0f), ov::float16(2.0f),  ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),  ov::float16(7.0f), ov::float16(8.0f),       ov::float16(5.0f), ov::float16(6.0f),  ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),     ov::float16(71.0f), ov::float16(72.0f), ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(1.0f), ov::float16(2.0f),  ov::float16(3.0f), ov::float16(4.0f),       ov::float16(1.0f), ov::float16(2.0f),  ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(51.0f), ov::float16(52.0f), ov::float16(53.0f), ov::float16(54.0f),    ov::float16(5.0f), ov::float16(6.0f),  ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),     ov::float16(61.0f), ov::float16(62.0f), ov::float16(63.0f), ov::float16(64.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test12, data3_indice3_update1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 1, 4 } }); // data 3x3x4 (bfy)
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 3, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } }); // updates

    set_values(input1, {
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(2.0f), ov::float16(0.0f), ov::float16(0.0f),
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f),
        ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(0.0f),
        });

    set_values(input3, {
        ov::float16(51.0f), ov::float16(52.0f), ov::float16(53.0f), ov::float16(54.0f),
        });

    std::vector<float> expected_results = {
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(54.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(53.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(52.0f),

        ov::float16(51.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test11, data6_indice1_update6) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 2, 3, 4, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 2, 3, 4, 2 } }); // updates

    set_values(input1, {
        // 0, 0, 0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        // 0, 0, 1
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        // 0, 1, 0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),


        // 1, 0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        // 1, 1
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(1.0f),
        });

    set_values(input3, {
        // 0
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(50.0f), ov::float16(51.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),

        ov::float16(150.0f), ov::float16(151.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),
        });

    std::vector<float> expected_results = {
        // 0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        // 1
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(50.0f), ov::float16(51.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),

        ov::float16(150.0f), ov::float16(151.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test10, data5_indice1_update5) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 3, 4, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 2, 3, 4, 2 } }); // updates

    set_values(input1, {
        // 0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        // 1
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(1.0f), ov::float16(0.0f),
        });

    set_values(input3, {
        // 0
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(50.0f), ov::float16(51.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        // 1
        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),

        ov::float16(150.0f), ov::float16(151.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),
        });

    std::vector<float> expected_results = {
        // 0
        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),

        ov::float16(150.0f), ov::float16(151.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),

        // 1
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(50.0f), ov::float16(51.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test9, data4_indice1_update4) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 4, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 4, 2 } }); // updates

    set_values(input1, {
        // 0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        // 1
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        // 2
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(2.0f), ov::float16(0.0f),
        });

    set_values(input3, {
        // 0
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        // 1
        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),
        });

    std::vector<float> expected_results = {
        // 0
        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),
        // 1
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        // 2
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test8, data6_indice2_update5) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 2, 4, 3, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx,   { 2, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 2, 1, 2, 4, 3 } }); // updates

    set_values(input1, {
        //0,0
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        //0,1
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f),
        ov::float16(0.0f), ov::float16(0.0f)
        });

    set_values(input3, {
        // 0
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),


        // 1
        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),

        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),
        });

    std::vector<float> expected_results = {
        // 0,0
        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),

        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),

        // 0,1
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test7, data5_indice2_update4) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 1, 2, 3, 4, 2 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx,  { 2, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfzyx,  { 2, 2, 1, 3, 4 } }); // updates


    set_values(input1, {
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f),
        ov::float16(0.0f), ov::float16(0.0f)
        });

    set_values(input3, {
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),
        });

    std::vector<float> expected_results = {
        ov::float16(151.0f), ov::float16(152.0f),    ov::float16(153.0f), ov::float16(154.0f),      ov::float16(155.0f), ov::float16(156.0f),   ov::float16(157.0f), ov::float16(158.0f),
        ov::float16(159.0f), ov::float16(160.0f),    ov::float16(161.0f), ov::float16(162.0f),      ov::float16(163.0f), ov::float16(164.0f),   ov::float16(165.0f), ov::float16(166.0f),
        ov::float16(167.0f), ov::float16(168.0f),    ov::float16(169.0f), ov::float16(170.0f),      ov::float16(171.0f), ov::float16(172.0f),   ov::float16(173.0f), ov::float16(174.0f),

        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16_test6, data4_indice2_update3) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 2, 4 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 2 } }); // updates


    set_values(input1, {
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),   ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),   ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),  ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(1.0f), ov::float16(1.0f),
        ov::float16(1.0f), ov::float16(0.0f),
        ov::float16(0.0f), ov::float16(2.0f)
        });

    set_values(input3, {
        ov::float16(51.0f), ov::float16(52.0f),    ov::float16(53.0f), ov::float16(54.0f),      ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(59.0f), ov::float16(60.0f),    ov::float16(61.0f), ov::float16(62.0f),      ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(67.0f), ov::float16(68.0f),    ov::float16(69.0f), ov::float16(70.0f),      ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),
        });

    std::vector<float> expected_results = {
        ov::float16(1.0f), ov::float16(2.0f),    ov::float16(3.0f), ov::float16(4.0f),      ov::float16(1.0f), ov::float16(2.0f),     ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f),    ov::float16(7.0f), ov::float16(8.0f),      ov::float16(5.0f), ov::float16(6.0f),     ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(67.0f), ov::float16(68.0f),  ov::float16(69.0f), ov::float16(70.0f),    ov::float16(71.0f), ov::float16(72.0f),   ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(59.0f), ov::float16(60.0f),  ov::float16(61.0f), ov::float16(62.0f),    ov::float16(63.0f), ov::float16(64.0f),   ov::float16(65.0f), ov::float16(66.0f),
        ov::float16(51.0f), ov::float16(52.0f),  ov::float16(53.0f), ov::float16(54.0f),    ov::float16(55.0f), ov::float16(56.0f),   ov::float16(57.0f), ov::float16(58.0f),
        ov::float16(9.0f), ov::float16(10.0f),   ov::float16(11.0f), ov::float16(12.0f),    ov::float16(9.0f), ov::float16(10.0f),    ov::float16(11.0f), ov::float16(12.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test5, data3_indice2_update2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 1, 4 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 1 } }); // updates


    set_values(input1, {
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(1.0f), ov::float16(1.0f),
        ov::float16(1.0f), ov::float16(0.0f),
        ov::float16(0.0f), ov::float16(2.0f)
        });

    set_values(input3, {
        ov::float16(51.0f), ov::float16(52.0f), ov::float16(53.0f), ov::float16(54.0f),
        ov::float16(61.0f), ov::float16(62.0f), ov::float16(63.0f), ov::float16(64.0f),
        ov::float16(71.0f), ov::float16(72.0f), ov::float16(73.0f), ov::float16(74.0f),
        });

    std::vector<float> expected_results = {
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(71.0f), ov::float16(72.0f), ov::float16(73.0f), ov::float16(74.0f),

        ov::float16(61.0f), ov::float16(62.0f), ov::float16(63.0f), ov::float16(64.0f),
        ov::float16(51.0f), ov::float16(52.0f), ov::float16(53.0f), ov::float16(54.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test4, data2_indice2_update1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 1 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 1, 1, 1 } }); // updates


    set_values(input1, {
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
        ov::float16(2.0f), ov::float16(1.0f),
        ov::float16(0.0f), ov::float16(3.0f),
        ov::float16(0.0f), ov::float16(2.0f)
        });

    set_values(input3, {
        ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f)
        });

    std::vector<float> expected_results = {
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(23.0f), ov::float16(22.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(21.0f), ov::float16(11.0f), ov::float16(12.0f),
        };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test3, data3_indice1_update3) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 4, 1 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 4, 1 } }); // updates


    set_values(input1, {
        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),
        });

    set_values(input2, {
            ov::float16(2.0f), ov::float16(0.0f)
        });

    set_values(input3, {
        ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f), ov::float16(24.0f),
        ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f), ov::float16(28.0f),
        ov::float16(29.0f), ov::float16(30.0f), ov::float16(31.0f), ov::float16(32.0f),

        ov::float16(41.0f), ov::float16(42.0f), ov::float16(43.0f), ov::float16(44.0f),
        ov::float16(45.0f), ov::float16(46.0f), ov::float16(47.0f), ov::float16(48.0f),
        ov::float16(49.0f), ov::float16(50.0f), ov::float16(51.0f), ov::float16(52.0f),
        });

    std::vector<float> expected_results = {
        ov::float16(41.0f), ov::float16(42.0f), ov::float16(43.0f), ov::float16(44.0f),
        ov::float16(45.0f), ov::float16(46.0f), ov::float16(47.0f), ov::float16(48.0f),
        ov::float16(49.0f), ov::float16(50.0f), ov::float16(51.0f), ov::float16(52.0f),

        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f),
        ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
        ov::float16(9.0f), ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f),

        ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f), ov::float16(24.0f),
        ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f), ov::float16(28.0f),
        ov::float16(29.0f), ov::float16(30.0f), ov::float16(31.0f), ov::float16(32.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}


TEST(scatter_nd_update_gpu_fp16_test2, data2_indice1_update2) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 4, 1, 1 } }); // data
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 4, 1, 1 } }); // updates


    set_values(input1, {
        ov::float16(13.0f), ov::float16(12.0f), ov::float16(11.0f), ov::float16(10.0f),
        ov::float16(9.0f), ov::float16(8.0f), ov::float16(7.0f), ov::float16(6.0f),
        ov::float16(5.0f), ov::float16(4.0f), ov::float16(3.0f), ov::float16(2.0f)
        });

    set_values(input2, {
            ov::float16(2.0f), ov::float16(0.0f)
        });

    set_values(input3, {
            ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f),
            ov::float16(24.0f), ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f)
        });

    std::vector<float> expected_results = {
        ov::float16(24.0f), ov::float16(25.0f), ov::float16(26.0f), ov::float16(27.0f),
        ov::float16(9.0f), ov::float16(8.0f), ov::float16(7.0f), ov::float16(6.0f),
        ov::float16(20.0f), ov::float16(21.0f), ov::float16(22.0f), ov::float16(23.0f),
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16_test1, data1_indice1_update1) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 8, 1, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } }); // Indexes
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } }); // Updates


    set_values(input1, {
        ov::float16(9.0f), ov::float16(8.0f), ov::float16(7.0f), ov::float16(6.0f), ov::float16(5.0f), ov::float16(4.0f), ov::float16(3.0f), ov::float16(2.0f)
    });

    set_values(input2, {
        ov::float16(2.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(7.0f)
    });

    set_values(input3, {
        ov::float16(10.0f), ov::float16(11.0f), ov::float16(12.0f), ov::float16(13.0f)
    });

    std::vector<float> expected_results = {
        9.f, 8.f, 10.f, 6.f, 11.f, 12.f, 3.f, 13.f
    };

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f), ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f), ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),
        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f), ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f), ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),
        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f), ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),
        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f), ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        ov::float16(136.f), ov::float16(137.f), ov::float16(138.f), ov::float16(139.f), ov::float16(140.f), ov::float16(141.f),
        ov::float16(142.f), ov::float16(143.f), ov::float16(144.f), ov::float16(145.f), ov::float16(146.f), ov::float16(147.f),
        ov::float16(148.f), ov::float16(149.f), ov::float16(150.f), ov::float16(151.f), ov::float16(152.f), ov::float16(153.f),
        ov::float16(154.f), ov::float16(155.f), ov::float16(156.f), ov::float16(157.f), ov::float16(158.f), ov::float16(159.f),
        ov::float16(160.f), ov::float16(161.f), ov::float16(162.f), ov::float16(163.f), ov::float16(164.f), ov::float16(165.f),
        ov::float16(166.f), ov::float16(167.f), ov::float16(168.f), ov::float16(169.f), ov::float16(170.f), ov::float16(171.f),

        ov::float16(172.f), ov::float16(173.f), ov::float16(174.f), ov::float16(175.f), ov::float16(176.f), ov::float16(177.f),
        ov::float16(178.f), ov::float16(179.f), ov::float16(180.f), ov::float16(181.f), ov::float16(182.f), ov::float16(183.f),
        ov::float16(184.f), ov::float16(185.f), ov::float16(186.f), ov::float16(187.f), ov::float16(188.f), ov::float16(189.f),
        ov::float16(190.f), ov::float16(191.f), ov::float16(192.f), ov::float16(193.f), ov::float16(194.f), ov::float16(195.f),
        ov::float16(196.f), ov::float16(197.f), ov::float16(198.f), ov::float16(199.f), ov::float16(200.f), ov::float16(201.f),
        ov::float16(202.f), ov::float16(203.f), ov::float16(204.f), ov::float16(205.f), ov::float16(206.f), ov::float16(207.f),

        ov::float16(208.f), ov::float16(209.f), ov::float16(210.f), ov::float16(211.f), ov::float16(212.f), ov::float16(213.f),
        ov::float16(214.f), ov::float16(215.f), ov::float16(216.f), ov::float16(217.f), ov::float16(218.f), ov::float16(219.f),
        ov::float16(220.f), ov::float16(221.f), ov::float16(222.f), ov::float16(223.f), ov::float16(224.f), ov::float16(225.f),
        ov::float16(226.f), ov::float16(227.f), ov::float16(228.f), ov::float16(229.f), ov::float16(230.f), ov::float16(231.f),
        ov::float16(232.f), ov::float16(233.f), ov::float16(234.f), ov::float16(235.f), ov::float16(236.f), ov::float16(237.f),
        ov::float16(238.f), ov::float16(239.f), ov::float16(240.f), ov::float16(241.f), ov::float16(242.f), ov::float16(243.f),

        ov::float16(244.f), ov::float16(245.f), ov::float16(246.f), ov::float16(247.f), ov::float16(248.f), ov::float16(249.f),
        ov::float16(250.f), ov::float16(251.f), ov::float16(252.f), ov::float16(253.f), ov::float16(254.f), ov::float16(255.f),
        ov::float16(256.f), ov::float16(257.f), ov::float16(258.f), ov::float16(259.f), ov::float16(260.f), ov::float16(261.f),
        ov::float16(262.f), ov::float16(263.f), ov::float16(264.f), ov::float16(265.f), ov::float16(266.f), ov::float16(267.f),
        ov::float16(268.f), ov::float16(269.f), ov::float16(270.f), ov::float16(271.f), ov::float16(272.f), ov::float16(273.f),
        ov::float16(274.f), ov::float16(275.f), ov::float16(276.f), ov::float16(277.f), ov::float16(278.f), ov::float16(279.f),

        ov::float16(280.f), ov::float16(281.f), ov::float16(282.f), ov::float16(283.f), ov::float16(284.f), ov::float16(285.f),
        ov::float16(286.f), ov::float16(287.f), ov::float16(288.f), ov::float16(289.f), ov::float16(290.f), ov::float16(291.f),
        ov::float16(292.f), ov::float16(293.f), ov::float16(294.f), ov::float16(295.f), ov::float16(296.f), ov::float16(297.f),
        ov::float16(298.f), ov::float16(299.f), ov::float16(300.f), ov::float16(301.f), ov::float16(302.f), ov::float16(303.f),
        ov::float16(304.f), ov::float16(305.f), ov::float16(306.f), ov::float16(307.f), ov::float16(308.f), ov::float16(309.f),
        ov::float16(310.f), ov::float16(311.f), ov::float16(312.f), ov::float16(313.f), ov::float16(314.f), ov::float16(315.f),
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f),
        ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f)
        });

    set_values(input3, {
        ov::float16(999.0f), ov::float16(888.0f)
        });


    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f), ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f), ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),
        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f), ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f), ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),
        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f), ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),
        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f), ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        ov::float16(136.f), ov::float16(137.f), ov::float16(138.f), ov::float16(139.f), ov::float16(140.f), ov::float16(141.f),
        ov::float16(142.f), ov::float16(143.f), ov::float16(144.f), ov::float16(145.f), ov::float16(146.f), ov::float16(147.f),
        ov::float16(148.f), ov::float16(149.f), ov::float16(150.f), ov::float16(151.f), ov::float16(152.f), ov::float16(153.f),
        ov::float16(154.f), ov::float16(155.f), ov::float16(156.f), ov::float16(157.f), ov::float16(158.f), ov::float16(159.f),
        ov::float16(160.f), ov::float16(161.f), ov::float16(162.f), ov::float16(163.f), ov::float16(164.f), ov::float16(165.f),
        ov::float16(166.f), ov::float16(167.f), ov::float16(168.f), ov::float16(169.f), ov::float16(170.f), ov::float16(171.f),

        ov::float16(172.f), ov::float16(173.f), ov::float16(174.f), ov::float16(175.f), ov::float16(176.f), ov::float16(177.f),
        ov::float16(178.f), ov::float16(179.f), ov::float16(180.f), ov::float16(181.f), ov::float16(182.f), ov::float16(183.f),
        ov::float16(184.f), ov::float16(185.f), ov::float16(186.f), ov::float16(187.f), ov::float16(188.f), ov::float16(189.f),
        ov::float16(190.f), ov::float16(191.f), ov::float16(192.f), ov::float16(193.f), ov::float16(194.f), ov::float16(195.f),
        ov::float16(196.f), ov::float16(197.f), ov::float16(198.f), ov::float16(199.f), ov::float16(200.f), ov::float16(201.f),
        ov::float16(202.f), ov::float16(203.f), ov::float16(204.f), ov::float16(205.f), ov::float16(206.f), ov::float16(207.f),

        ov::float16(208.f), ov::float16(209.f), ov::float16(210.f), ov::float16(211.f), ov::float16(212.f), ov::float16(213.f),
        ov::float16(214.f), ov::float16(215.f), ov::float16(216.f), ov::float16(217.f), ov::float16(218.f), ov::float16(219.f),
        ov::float16(220.f), ov::float16(221.f), ov::float16(222.f), ov::float16(223.f), ov::float16(224.f), ov::float16(225.f),
        ov::float16(226.f), ov::float16(227.f), ov::float16(228.f), ov::float16(229.f), ov::float16(230.f), ov::float16(231.f),
        ov::float16(232.f), ov::float16(233.f), ov::float16(234.f), ov::float16(235.f), ov::float16(236.f), ov::float16(237.f),
        ov::float16(238.f), ov::float16(239.f), ov::float16(240.f), ov::float16(241.f), ov::float16(242.f), ov::float16(243.f),

        ov::float16(244.f), ov::float16(245.f), ov::float16(246.f), ov::float16(247.f), ov::float16(248.f), ov::float16(249.f),
        ov::float16(250.f), ov::float16(251.f), ov::float16(252.f), ov::float16(253.f), ov::float16(254.f), ov::float16(255.f),
        ov::float16(256.f), ov::float16(257.f), ov::float16(258.f), ov::float16(259.f), ov::float16(260.f), ov::float16(261.f),
        ov::float16(262.f), ov::float16(263.f), ov::float16(264.f), ov::float16(265.f), ov::float16(266.f), ov::float16(267.f),
        ov::float16(268.f), ov::float16(269.f), ov::float16(270.f), ov::float16(271.f), ov::float16(272.f), ov::float16(273.f),
        ov::float16(274.f), ov::float16(275.f), ov::float16(276.f), ov::float16(277.f), ov::float16(278.f), ov::float16(279.f),

        ov::float16(280.f), ov::float16(281.f), ov::float16(282.f), ov::float16(283.f), ov::float16(284.f), ov::float16(285.f),
        ov::float16(286.f), ov::float16(287.f), ov::float16(288.f), ov::float16(289.f), ov::float16(290.f), ov::float16(291.f),
        ov::float16(292.f), ov::float16(293.f), ov::float16(294.f), ov::float16(295.f), ov::float16(296.f), ov::float16(297.f),
        ov::float16(298.f), ov::float16(299.f), ov::float16(300.f), ov::float16(301.f), ov::float16(302.f), ov::float16(303.f),
        ov::float16(304.f), ov::float16(305.f), ov::float16(306.f), ov::float16(307.f), ov::float16(308.f), ov::float16(309.f),
        ov::float16(310.f), ov::float16(311.f), ov::float16(312.f), ov::float16(313.f), ov::float16(314.f), ov::float16(315.f),
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f),
        ov::float16(3.0f), ov::float16(4.0f),
        });

    set_values(input3, {
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f), ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f), ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),
        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f), ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f), ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),
        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f), ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),
        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f), ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        ov::float16(136.f), ov::float16(137.f), ov::float16(138.f), ov::float16(139.f), ov::float16(140.f), ov::float16(141.f),
        ov::float16(142.f), ov::float16(143.f), ov::float16(144.f), ov::float16(145.f), ov::float16(146.f), ov::float16(147.f),
        ov::float16(148.f), ov::float16(149.f), ov::float16(150.f), ov::float16(151.f), ov::float16(152.f), ov::float16(153.f),
        ov::float16(154.f), ov::float16(155.f), ov::float16(156.f), ov::float16(157.f), ov::float16(158.f), ov::float16(159.f),
        ov::float16(160.f), ov::float16(161.f), ov::float16(162.f), ov::float16(163.f), ov::float16(164.f), ov::float16(165.f),
        ov::float16(166.f), ov::float16(167.f), ov::float16(168.f), ov::float16(169.f), ov::float16(170.f), ov::float16(171.f),

        ov::float16(172.f), ov::float16(173.f), ov::float16(174.f), ov::float16(175.f), ov::float16(176.f), ov::float16(177.f),
        ov::float16(178.f), ov::float16(179.f), ov::float16(180.f), ov::float16(181.f), ov::float16(182.f), ov::float16(183.f),
        ov::float16(184.f), ov::float16(185.f), ov::float16(186.f), ov::float16(187.f), ov::float16(188.f), ov::float16(189.f),
        ov::float16(190.f), ov::float16(191.f), ov::float16(192.f), ov::float16(193.f), ov::float16(194.f), ov::float16(195.f),
        ov::float16(196.f), ov::float16(197.f), ov::float16(198.f), ov::float16(199.f), ov::float16(200.f), ov::float16(201.f),
        ov::float16(202.f), ov::float16(203.f), ov::float16(204.f), ov::float16(205.f), ov::float16(206.f), ov::float16(207.f),

        ov::float16(208.f), ov::float16(209.f), ov::float16(210.f), ov::float16(211.f), ov::float16(212.f), ov::float16(213.f),
        ov::float16(214.f), ov::float16(215.f), ov::float16(216.f), ov::float16(217.f), ov::float16(218.f), ov::float16(219.f),
        ov::float16(220.f), ov::float16(221.f), ov::float16(222.f), ov::float16(223.f), ov::float16(224.f), ov::float16(225.f),
        ov::float16(226.f), ov::float16(227.f), ov::float16(228.f), ov::float16(229.f), ov::float16(230.f), ov::float16(231.f),
        ov::float16(232.f), ov::float16(233.f), ov::float16(234.f), ov::float16(235.f), ov::float16(236.f), ov::float16(237.f),
        ov::float16(238.f), ov::float16(239.f), ov::float16(240.f), ov::float16(241.f), ov::float16(242.f), ov::float16(243.f),

        ov::float16(244.f), ov::float16(245.f), ov::float16(246.f), ov::float16(247.f), ov::float16(248.f), ov::float16(249.f),
        ov::float16(250.f), ov::float16(251.f), ov::float16(252.f), ov::float16(253.f), ov::float16(254.f), ov::float16(255.f),
        ov::float16(256.f), ov::float16(257.f), ov::float16(258.f), ov::float16(259.f), ov::float16(260.f), ov::float16(261.f),
        ov::float16(262.f), ov::float16(263.f), ov::float16(264.f), ov::float16(265.f), ov::float16(266.f), ov::float16(267.f),
        ov::float16(268.f), ov::float16(269.f), ov::float16(270.f), ov::float16(271.f), ov::float16(272.f), ov::float16(273.f),
        ov::float16(274.f), ov::float16(275.f), ov::float16(276.f), ov::float16(277.f), ov::float16(278.f), ov::float16(279.f),

        ov::float16(280.f), ov::float16(281.f), ov::float16(282.f), ov::float16(283.f), ov::float16(284.f), ov::float16(285.f),
        ov::float16(286.f), ov::float16(287.f), ov::float16(288.f), ov::float16(289.f), ov::float16(290.f), ov::float16(291.f),
        ov::float16(292.f), ov::float16(293.f), ov::float16(294.f), ov::float16(295.f), ov::float16(296.f), ov::float16(297.f),
        ov::float16(298.f), ov::float16(299.f), ov::float16(300.f), ov::float16(301.f), ov::float16(302.f), ov::float16(303.f),
        ov::float16(304.f), ov::float16(305.f), ov::float16(306.f), ov::float16(307.f), ov::float16(308.f), ov::float16(309.f),
        ov::float16(310.f), ov::float16(311.f), ov::float16(312.f), ov::float16(313.f), ov::float16(314.f), ov::float16(315.f),
        });

    set_values(input2, {
        ov::float16(0.0f),
        ov::float16(3.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(777.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(777.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(777.0f), ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(777.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f), ov::float16(777.0f),

        ov::float16(666.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(666.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(888.0f), ov::float16(666.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(666.0f), ov::float16(888.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(666.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f), ov::float16(666.0f),
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),
        ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f),
        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),
        ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f),
        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),
        ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f),
        ov::float16(132.f), ov::float16(133.f),
        ov::float16(134.f), ov::float16(135.f)
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(1.0f),
        ov::float16(2.0f), ov::float16(1.0f), ov::float16(2.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),
        ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f),
        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),
        ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f),
        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),
        ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f),
        ov::float16(132.f), ov::float16(133.f),
        ov::float16(134.f), ov::float16(135.f)
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(2.0f),
        ov::float16(2.0f), ov::float16(1.0f), ov::float16(2.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(777.0f), ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),
        ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f),
        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),
        ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f),
        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),
        ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f),
        ov::float16(132.f), ov::float16(133.f),
        ov::float16(134.f), ov::float16(135.f)
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f),
        ov::float16(2.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),
        ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f),
        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),
        ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f),
        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),
        ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f),
        ov::float16(132.f), ov::float16(133.f),
        ov::float16(134.f), ov::float16(135.f)
        });

    set_values(input2, {
        ov::float16(0.0f),
        ov::float16(2.0f)
        });

    set_values(input3, {
        ov::float16(666.0f), ov::float16(666.0f),
        ov::float16(666.0f), ov::float16(666.0f),
        ov::float16(666.0f), ov::float16(666.0f),

        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(888.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(888.0f),

        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 2
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 3
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f)
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(2.0f),
        ov::float16(2.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 2
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 3
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f)
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
        ov::float16(2.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 2
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 3
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f)
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(1.0f),
        ov::float16(2.0f), ov::float16(1.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 2
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 3
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f)
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f),
        ov::float16(2.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(555.0f), ov::float16(555.0f), ov::float16(555.0f),
        ov::float16(555.0f), ov::float16(555.0f), ov::float16(555.0f),

        ov::float16(666.0f), ov::float16(666.0f), ov::float16(666.0f),
        ov::float16(666.0f), ov::float16(666.0f), ov::float16(666.0f),

        ov::float16(444.0f), ov::float16(444.0f), ov::float16(444.0f),
        ov::float16(444.0f), ov::float16(444.0f), ov::float16(444.0f),

        ov::float16(777.0f), ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f),

        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 2
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f),

        // 3
        ov::float16(100.f), ov::float16(101.f), ov::float16(102.f),
        ov::float16(103.f), ov::float16(104.f), ov::float16(105.f),

        ov::float16(106.f), ov::float16(107.f), ov::float16(108.f),
        ov::float16(109.f), ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f), ov::float16(114.f),
        ov::float16(115.f), ov::float16(116.f), ov::float16(117.f),

        ov::float16(118.f), ov::float16(119.f), ov::float16(120.f),
        ov::float16(121.f), ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f), ov::float16(126.f),
        ov::float16(127.f), ov::float16(128.f), ov::float16(129.f),

        ov::float16(130.f), ov::float16(131.f), ov::float16(132.f),
        ov::float16(133.f), ov::float16(134.f), ov::float16(135.f)
        });

    set_values(input2, {
        ov::float16(0.0f),
        ov::float16(2.0f)
        });

    set_values(input3, {
        ov::float16(555.0f), ov::float16(555.0f), ov::float16(555.0f),
        ov::float16(555.0f), ov::float16(555.0f), ov::float16(555.0f),

        ov::float16(666.0f), ov::float16(666.0f), ov::float16(666.0f),
        ov::float16(666.0f), ov::float16(666.0f), ov::float16(666.0f),

        ov::float16(444.0f), ov::float16(444.0f), ov::float16(444.0f),
        ov::float16(444.0f), ov::float16(444.0f), ov::float16(444.0f),

        ov::float16(555.0f), ov::float16(555.0f), ov::float16(555.0f),
        ov::float16(555.0f), ov::float16(555.0f), ov::float16(555.0f),

        ov::float16(666.0f), ov::float16(666.0f), ov::float16(666.0f),
        ov::float16(666.0f), ov::float16(666.0f), ov::float16(666.0f),

        ov::float16(444.0f), ov::float16(444.0f), ov::float16(444.0f),
        ov::float16(444.0f), ov::float16(444.0f), ov::float16(444.0f),

        ov::float16(777.0f), ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f),

        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f),

        ov::float16(777.0f), ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f),
        ov::float16(888.0f), ov::float16(888.0f), ov::float16(888.0f),

        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),//1

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),//2

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),//3

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),

        ov::float16(128.f), ov::float16(129.f),
        ov::float16(130.f), ov::float16(131.f),//4

        ov::float16(132.f), ov::float16(133.f),
        ov::float16(134.f), ov::float16(135.f),

        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),//5

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),//6

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),//7

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),//8
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(0.0f),
        ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(0.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),//1

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),//2

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),//3

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),

        ov::float16(128.f), ov::float16(129.f),
        ov::float16(130.f), ov::float16(131.f),//4

        ov::float16(132.f), ov::float16(133.f),
        ov::float16(134.f), ov::float16(135.f),

        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),//5

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),//6

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),//7

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),//8
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
        ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),//1

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),//2

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),//3

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),

        ov::float16(128.f), ov::float16(129.f),
        ov::float16(130.f), ov::float16(131.f),//4

        ov::float16(132.f), ov::float16(133.f),
        ov::float16(134.f), ov::float16(135.f),

        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),//5

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),//6

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),//7

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),//8
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f),
        ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),//1

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),//2

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),//3

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),

        ov::float16(128.f), ov::float16(129.f),
        ov::float16(130.f), ov::float16(131.f),//4

        ov::float16(132.f), ov::float16(133.f),
        ov::float16(134.f), ov::float16(135.f),

        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),//5

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),//6

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),//7

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),//8
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f), ov::float16(1.0f),
        ov::float16(1.0f), ov::float16(1.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f),

        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
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
        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),//1

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),//2

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),//3

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),

        ov::float16(128.f), ov::float16(129.f),
        ov::float16(130.f), ov::float16(131.f),//4

        ov::float16(132.f), ov::float16(133.f),
        ov::float16(134.f), ov::float16(135.f),

        ov::float16(100.f), ov::float16(101.f),
        ov::float16(102.f), ov::float16(103.f),//5

        ov::float16(104.f), ov::float16(105.f),
        ov::float16(106.f), ov::float16(107.f),

        ov::float16(108.f), ov::float16(109.f),
        ov::float16(110.f), ov::float16(111.f),//6

        ov::float16(112.f), ov::float16(113.f),
        ov::float16(114.f), ov::float16(115.f),

        ov::float16(116.f), ov::float16(117.f),
        ov::float16(118.f), ov::float16(119.f),//7

        ov::float16(120.f), ov::float16(121.f),
        ov::float16(122.f), ov::float16(123.f),

        ov::float16(124.f), ov::float16(125.f),
        ov::float16(126.f), ov::float16(127.f),//8
        });

    set_values(input2, {
        ov::float16(0.0f), ov::float16(1.0f),
        ov::float16(1.0f), ov::float16(1.0f)
        });

    set_values(input3, {
        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(777.0f), ov::float16(777.0f),
        ov::float16(777.0f), ov::float16(777.0f),

        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f),

        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f),

        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f),

        ov::float16(999.0f), ov::float16(999.0f),
        ov::float16(999.0f), ov::float16(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    network network(engine, topology, get_test_default_config(engine));


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

template <typename T>
void test_d222222_i211111(bool is_caching_test) {
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
        T(100.f), T(101.f),
        T(102.f), T(103.f),

        T(104.f), T(105.f),
        T(106.f), T(107.f),//1

        T(108.f), T(109.f),
        T(110.f), T(111.f),

        T(112.f), T(113.f),
        T(114.f), T(115.f),//2

        T(116.f), T(117.f),
        T(118.f), T(119.f),

        T(120.f), T(121.f),
        T(122.f), T(123.f),//3

        T(124.f), T(125.f),
        T(126.f), T(127.f),

        T(128.f), T(129.f),
        T(130.f), T(131.f),//4

        T(132.f), T(133.f),
        T(134.f), T(135.f),

        T(100.f), T(101.f),
        T(102.f), T(103.f),//5

        T(104.f), T(105.f),
        T(106.f), T(107.f),

        T(108.f), T(109.f),
        T(110.f), T(111.f),//6

        T(112.f), T(113.f),
        T(114.f), T(115.f),

        T(116.f), T(117.f),
        T(118.f), T(119.f),//7

        T(120.f), T(121.f),
        T(122.f), T(123.f),

        T(124.f), T(125.f),
        T(126.f), T(127.f),//8
        });

    set_values(input2, {
        T(0.0f),
        T(1.0f)
        });

    set_values(input3, {
        T(777.0f), T(777.0f),
        T(777.0f), T(777.0f),

        T(777.0f), T(777.0f),
        T(777.0f), T(777.0f),

        T(777.0f), T(777.0f),
        T(777.0f), T(777.0f),

        T(777.0f), T(777.0f),
        T(777.0f), T(777.0f),

        T(777.0f), T(777.0f),
        T(777.0f), T(777.0f),

        T(777.0f), T(777.0f),
        T(777.0f), T(777.0f),

        T(777.0f), T(777.0f),
        T(777.0f), T(777.0f),

        T(777.0f), T(777.0f),
        T(777.0f), T(777.0f),

        T(999.0f), T(999.0f),
        T(999.0f), T(999.0f),

        T(999.0f), T(999.0f),
        T(999.0f), T(999.0f),

        T(999.0f), T(999.0f),
        T(999.0f), T(999.0f),

        T(999.0f), T(999.0f),
        T(999.0f), T(999.0f),

        T(999.0f), T(999.0f),
        T(999.0f), T(999.0f),

        T(999.0f), T(999.0f),
        T(999.0f), T(999.0f),

        T(999.0f), T(999.0f),
        T(999.0f), T(999.0f),

        T(999.0f), T(999.0f),
        T(999.0f), T(999.0f)
        });

    topology topology;
    topology.add(input_layout("InputData", input1->get_layout()));
    topology.add(input_layout("InputIndices", input2->get_layout()));
    topology.add(input_layout("InputUpdates", input3->get_layout()));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("InputData", input1);
    network->set_input_data("InputIndices", input2);
    network->set_input_data("InputUpdates", input3);

    auto outputs = network->execute();


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
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

TEST(scatter_nd_update_gpu_fp16, d222222_i211111) {
    test_d222222_i211111<ov::float16>(false);
}

TEST(scatter_nd_update_gpu, dynamic) {
    //  Dictionary : 2x1x2x8
    //  Indexes : 2x3
    //  Updates : 2x8
    //  Output : 2x1x2x8
    //  Input values in fp32
    //
    auto& engine = get_test_engine();

    auto input1_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto input2_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };
    auto input3_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };

    auto input1 = engine.allocate_memory({ { 2, 1, 2, 8 }, data_types::f32, format::bfyx }); // Dictionary
    auto input2 = engine.allocate_memory({ { 2, 3 },       data_types::f32, format::bfyx }); // Indexes
    auto input3 = engine.allocate_memory({ { 2, 8 },       data_types::f32, format::bfyx }); // Updates

    set_values(input1, {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f,
        24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f
    });

    set_values(input2, {
        0.f, 1.f, 1.f, 2.f, 2.f, 2.f
    });

    set_values(input3, {
        24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f,
        42.f, 42.f, 42.f, 42.f, 42.f, 42.f, 42.f, 42.f
    });

    topology topology;
    topology.add(input_layout("InputData", input1_layout));
    topology.add(input_layout("InputIndices", input2_layout));
    topology.add(input_layout("InputUpdates", input3_layout));
    topology.add(
        scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto inst = network.get_primitive("scatter_nd_update");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f, 24.f,
        16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f,
        42.f, 42.f, 42.f, 42.f, 42.f, 42.f, 42.f, 42.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}


TEST(scatter_nd_update_gpu, dynamic_padded_output) {
    //  Dictionary : 2x1x2x8
    //  Indexes : 0x3
    //  Updates : 0x8
    //  Output : 2x1x2x8
    //  Input values in fp32
    //
    auto& engine = get_test_engine();

    auto input1_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto input2_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };
    auto input3_layout = layout{ ov::PartialShape::dynamic(2), data_types::f32, format::bfyx };

    auto input1 = engine.allocate_memory({ { 1, 1, 2, 8 }, data_types::f32, format::bfyx }); // Dictionary
    auto input2 = engine.allocate_memory({ { 0, 3 },       data_types::f32, format::bfyx }); // Indexes
    auto input3 = engine.allocate_memory({ { 0, 8 },       data_types::f32, format::bfyx }); // Updates

    set_values(input1, {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f,
    });

    auto scatter_nd_upd = scatter_nd_update("scatter_nd_update", input_info("InputData"), input_info("InputIndices"), input_info("InputUpdates"), 2);
    scatter_nd_upd.output_paddings = { padding({0, 0, 1, 1}) };
    topology topology;
    topology.add(input_layout("InputData", input1_layout));
    topology.add(input_layout("InputIndices", input2_layout));
    topology.add(input_layout("InputUpdates", input3_layout));
    topology.add(scatter_nd_upd);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    network.set_input_data("InputData", input1);
    network.set_input_data("InputIndices", input2);
    network.set_input_data("InputUpdates", input3);

    auto inst = network.get_primitive("scatter_nd_update");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("scatter_nd_update").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 0.f,
        0.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(scatter_nd_update_gpu, dynamic_5d) {
    tests::random_generator rg(std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
                               std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));

    auto& engine = get_test_engine();

    auto input1_layout = layout{{ 8, -1, -1, 384}, data_types::f32, format::bfyx };
    auto input2_layout = layout{{-1, -1, -1, -1, -1}, data_types::i32, format::bfzyx };
    auto input3_layout = layout{{-1, -1, -1, 384}, data_types::f32, format::bfyx };

    topology topology;
    topology.add(input_layout("data", input1_layout));
    topology.add(input_layout("indices", input2_layout));
    topology.add(input_layout("updates", input3_layout));
    topology.add(scatter_nd_update("scatter_nd_update", input_info("data"), input_info("indices"), input_info("updates"), 5));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    auto get_expected_res = [](const std::vector<float>& input,
                               const std::vector<int32_t>& indices,
                               const std::vector<float>& updates,
                               ov::Shape input_shape,
                               ov::Shape indices_shape,
                               ov::Shape updates_shape) -> std::vector<float> {
        size_t count = std::accumulate(input_shape.begin(), input_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        auto outputs_ref = std::vector<float>(count);
        ov::reference::scatterNdUpdate<float, int32_t>(input.data(),
                                                       indices.data(),
                                                       updates.data(),
                                                       outputs_ref.data(),
                                                       input_shape,
                                                       indices_shape,
                                                       updates_shape);

        return outputs_ref;
    };


    auto generate_unique_indices = [&rg](ov::Shape data_shape, ov::Shape indices_shape) -> std::vector<int32_t>{
        std::set<std::vector<int32_t>> unique_indices;
        std::vector<int32_t> result;
        size_t last_indices_dim = indices_shape.at(indices_shape.size() - 1);

        size_t count = std::accumulate(indices_shape.begin(),
                                       indices_shape.end(),
                                       static_cast<size_t>(1),
                                       std::multiplies<size_t>()) / last_indices_dim;

        while (unique_indices.size() != count) {
            std::vector<int32_t> indices;
            for (size_t i = 0; i < last_indices_dim; i++) {
                const int min = 0;
                const int max = static_cast<int>(data_shape[i]) - 1;
                indices.push_back(static_cast<int32_t>(rg.generate_random_val<int>(min, max)));
            }

            unique_indices.insert(indices);
        }

        std::for_each(unique_indices.begin(),
                      unique_indices.end(),
                      [&](const std::vector<int32_t>& indices) {
                          result.insert(result.end(), indices.begin(), indices.end());
                      });

        return result;
    };

    std::vector<std::vector<ov::Shape>> test_shapes = {
        { { 8, 3, 1, 384 }, { 1, 3, 1, 384, 4 }, { 1, 3, 1, 384 } },
        { { 8, 3, 2, 384 }, { 1, 3, 1, 384, 4 }, { 1, 3, 1, 384 } },
    };

    for (auto& shapes : test_shapes) {
        ov::Shape in1_shape = shapes[0];
        ov::Shape in2_shape = shapes[1];
        ov::Shape in3_shape = shapes[2];
        auto input1 = engine.allocate_memory({ in1_shape, data_types::f32, format::bfyx });  // Dictionary
        auto input2 = engine.allocate_memory({ in2_shape, data_types::i32, format::bfzyx }); // Indexes
        auto input3 = engine.allocate_memory({ in3_shape, data_types::f32, format::bfyx });  // Updates

        std::vector<float> input_data = rg.generate_random_1d<float>(input1->count(), 1, 100);
        std::vector<int32_t> indices = generate_unique_indices(in1_shape, in2_shape);
        std::vector<float> updates = rg.generate_random_1d<float>(input3->count(), 100, 200);
        auto expected_res = get_expected_res(input_data, indices, updates, in1_shape, in2_shape, in3_shape);

        set_values<float>(input1, input_data);
        set_values<int32_t>(input2, indices);
        set_values<float>(input3, updates);

        network.set_input_data("data", input1);
        network.set_input_data("indices", input2);
        network.set_input_data("updates", input3);

        auto inst = network.get_primitive("scatter_nd_update");
        auto impl = inst->get_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto outputs = network.execute();

        auto output = outputs.at("scatter_nd_update").get_memory();
        ASSERT_EQ(output->get_layout().get_partial_shape(), input1->get_layout().get_partial_shape());
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        for (size_t i = 0; i < expected_res.size(); ++i) {
            ASSERT_EQ(expected_res[i], output_ptr[i]) << " i = " << i;
        }
    }
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(scatter_nd_update_random_test, random_cached)
{
    auto param = GetParam();
    if (param.input_type == data_types::u8)
        this->execute<uint8_t>(param, true);
    else if (param.input_type == data_types::i8)
        this->execute<int8_t>(param, true);
    else if (param.input_type == data_types::i32)
        this->execute<int32_t>(param, true);
    else if (param.input_type == data_types::i64)
        this->execute<int64_t>(param, true);
    else if (param.input_type == data_types::f16)
        this->execute_fp16<ov::float16, uint16_t>(param, true);
    else if (param.input_type == data_types::f32)
        this->execute<float>(param, true);
    else
        OPENVINO_THROW("unidentified data type");
}
#endif
TEST(scatter_nd_update_gpu_fp16, d222222_i211111_cached) {
    test_d222222_i211111<ov::float16>(true);
}
