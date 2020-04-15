/*
// Copyright (c) 2016-2019 Intel Corporation
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/convolution.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include <api/data.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <thread>
#include <type_traits>
#include <fstream>
#include <api/reorder.hpp>

using namespace cldnn;
using namespace tests;

namespace cldnn
{
    template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}


TEST(concat_gpu, mixed_input_types) {
    const auto& engine = get_test_engine();

    auto input0 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 4, 3 } });
    auto input1 = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 1, 4, 3 } });
    auto input2 = memory::allocate(engine, { data_types::i8, format::bfyx, { 1, 1, 4, 3 } });
    auto input3 = memory::allocate(engine, { data_types::f16, format::bfyx, { 1, 1, 4, 3 } });

    set_values<float>(input0, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values<int32_t>(input1, { 11, 12, 13, 14, 12, 12, 13, 14, 13, 13, 13, 15 });
    set_values<int8_t>(input2, { 21, 22, 23, 24, 22, 22, 23, 24, 23, 23, 23, 25 });
    set_values(input3, { half_t(31.f), half_t(32.f), half_t(33.f),
                         half_t(34.f), half_t(32.f), half_t(32.f),
                         half_t(33.f), half_t(34.f), half_t(33.f),
                         half_t(33.f), half_t(33.f), half_t(35.f) });

    VF<float> output_vec = {
            1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f,
            31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f };

    topology topology(
            input_layout("input0", input0.get_layout()),
            input_layout("input1", input1.get_layout()),
            input_layout("input2", input2.get_layout()),
            input_layout("input3", input3.get_layout()),
            concatenation("concat",
                          { "input0", "input1", "input2", "input3" },
                          concatenation::concatenation_axis::along_f,
                          data_types::f32,
                          padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 3);
    EXPECT_EQ(x_size, 4);
    EXPECT_EQ(f_size, 4);
    EXPECT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        EXPECT_EQ(output_vec[x], output_ptr[x]);
    }
}

TEST(concat_gpu, mixed_input_types_5d) {
    const auto& engine = get_test_engine();

    auto input0 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });
    auto input1 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });
    auto input2 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });
    auto input3 = memory::allocate(engine, { data_types::f16, format::bfzyx, { 1, 1, 1, 4, 3 } });

    set_values(input0, { half_t(1.0f), half_t(2.0f), half_t(3.0f),
                         half_t(4.0f), half_t(2.0f), half_t(2.0f),
                         half_t(3.0f), half_t(4.0f), half_t(3.0f),
                         half_t(3.0f), half_t(3.0f), half_t(5.0f) });
    set_values(input1, { half_t(11), half_t(12), half_t(13),
                         half_t(14), half_t(12), half_t(12),
                         half_t(13), half_t(14), half_t(13),
                         half_t(13), half_t(13), half_t(15) });
    set_values(input2, { half_t(21), half_t(22), half_t(23),
                         half_t(24), half_t(22), half_t(22),
                         half_t(23), half_t(24), half_t(23),
                         half_t(23), half_t(23), half_t(25) });
    set_values(input3, { half_t(31.f), half_t(32.f), half_t(33.f),
                         half_t(34.f), half_t(32.f), half_t(32.f),
                         half_t(33.f), half_t(34.f), half_t(33.f),
                         half_t(33.f), half_t(33.f), half_t(35.f) });

    VF<float> output_vec = {
            1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f,
            11.0f, 12.0f, 13.0f, 14.0f, 12.0f, 12.0f, 13.0f, 14.0f, 13.0f, 13.0f, 13.0f, 15.0f,
            21.0f, 22.0f, 23.0f, 24.0f, 22.0f, 22.0f, 23.0f, 24.0f, 23.0f, 23.0f, 23.0f, 25.0f,
            31.0f, 32.0f, 33.0f, 34.0f, 32.0f, 32.0f, 33.0f, 34.0f, 33.0f, 33.0f, 33.0f, 35.0f };

    topology topology(
            input_layout("input0", input0.get_layout()),
            input_layout("input1", input1.get_layout()),
            input_layout("input2", input2.get_layout()),
            input_layout("input3", input3.get_layout()),
            concatenation("concat",
                          { "input0", "input1", "input2", "input3" },
                          concatenation::concatenation_axis::along_f,
                          data_types::f32,
                          padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "concat");

    auto output_memory = outputs.at("concat").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int z_size = output_layout.size.spatial[2];
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfzyx);
    EXPECT_EQ(z_size, 3);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 1);
    EXPECT_EQ(f_size, 4);
    EXPECT_EQ(b_size, 1);

    for (size_t x = 0; x < output_layout.count(); ++x) {
        EXPECT_EQ(output_vec[x], output_ptr[x]);
    }
}

using TestParamType_concat = ::testing::tuple<size_t,   // 0 - Input Batch size
        std::vector<size_t>,                            // 1 - Inputs Features Sizes
        size_t,                                         // 2 - Input Y Size
        size_t>;                                        // 3 - Input X Size


struct concat_gpu : public ::testing::TestWithParam<TestParamType_concat>
{
    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_concat> param_info)
    {
        std::string in;
        for (size_t i = 0; i < testing::get<1>(param_info.param).size() - 1; i++) {
            in += std::to_string(testing::get<1>(param_info.param)[i]) + "_";
        }
        in += std::to_string(testing::get<1>(param_info.param)[testing::get<1>(param_info.param).size() - 1]);

        return "in" + std::to_string(testing::get<0>(param_info.param))
               + "x" + in + "x" + std::to_string(testing::get<2>(param_info.param))
               + 'x' + std::to_string(testing::get<3>(param_info.param));
    }
};

static const auto concat_gpu_all_params = ::testing::Values(
    // Input Batch, Input Features, Input Y, Input X
    TestParamType_concat(2, { 2, 15 }, 2, 1),
    TestParamType_concat(2, { 2, 31 }, 2, 1),
    TestParamType_concat(2, { 2, 32 }, 2, 1),
    TestParamType_concat(2, { 2, 37 }, 2, 1),
    TestParamType_concat(2, { 2, 63 }, 2, 1),
    TestParamType_concat(2, { 2, 64 }, 2, 1),
    TestParamType_concat(2, { 2, 65 }, 2, 1),
    TestParamType_concat(2, { 2, 75 }, 2, 1),
    TestParamType_concat(2, { 15, 2 }, 2, 1),
    TestParamType_concat(2, { 31, 2 }, 2, 1),
    TestParamType_concat(2, { 32, 2 }, 2, 1),
    TestParamType_concat(2, { 37, 2 }, 2, 1),
    TestParamType_concat(2, { 63, 2 }, 2, 1),
    TestParamType_concat(2, { 64, 2 }, 2, 1),
    TestParamType_concat(2, { 65, 2 }, 2, 1),
    TestParamType_concat(2, { 75, 2 }, 2, 1),
    TestParamType_concat(2, { 2, 15 }, 1, 2),
    TestParamType_concat(2, { 2, 31 }, 1, 2),
    TestParamType_concat(2, { 2, 32 }, 1, 2),
    TestParamType_concat(2, { 2, 37 }, 1, 2),
    TestParamType_concat(2, { 2, 63 }, 1, 2),
    TestParamType_concat(2, { 2, 64 }, 1, 2),
    TestParamType_concat(2, { 2, 65 }, 1, 2),
    TestParamType_concat(2, { 2, 75 }, 1, 2),
    TestParamType_concat(2, { 15, 2 }, 1, 2),
    TestParamType_concat(2, { 31, 2 }, 1, 2),
    TestParamType_concat(2, { 32, 2 }, 1, 2),
    TestParamType_concat(2, { 37, 2 }, 1, 2),
    TestParamType_concat(2, { 63, 2 }, 1, 2),
    TestParamType_concat(2, { 64, 2 }, 1, 2),
    TestParamType_concat(2, { 65, 2 }, 1, 2),
    TestParamType_concat(2, { 75, 2 }, 1, 2),
    TestParamType_concat(2, { 32, 32 }, 1, 1),
    TestParamType_concat(2, { 64, 64 }, 1, 1),
    TestParamType_concat(2, { 2, 2, 2 }, 1, 1),
    TestParamType_concat(2, { 2, 32, 2 }, 1, 1),
    TestParamType_concat(2, { 31, 32, 32 }, 1, 1),
    TestParamType_concat(2, { 32, 31, 2 }, 1, 1),
    TestParamType_concat(2, { 32, 31, 32 }, 1, 1),
    TestParamType_concat(2, { 32, 32, 32 }, 1, 1),
    TestParamType_concat(2, { 33, 32, 32 }, 1, 1),
    TestParamType_concat(2, { 33, 3, 3 }, 1, 1),
    TestParamType_concat(2, { 33, 3, 33 }, 1, 1),
    TestParamType_concat(2, { 64, 64, 64, 64 }, 1, 1)
);

template <typename Type>
struct concat_gpu_4d : public concat_gpu {
public:

    void test(format::type fmt) {
        auto data_type = type_to_data_type<Type>::value;

        const auto& engine = get_test_engine();
        const size_t batch_num = testing::get<0>(GetParam());
        const std::vector<size_t> in_features = testing::get<1>(GetParam());
        const size_t input_y = testing::get<2>(GetParam());
        const size_t input_x = testing::get<3>(GetParam());
        size_t output_f = 0;
        for (auto& f : in_features)
            output_f += f;

        topology topology;

        std::vector<VVVVF<Type>> in_data;
        std::vector<memory> in_memory;
        std::vector<primitive_id> input_ids;
        for (size_t i = 0; i < in_features.size(); i++) {
            auto size = tensor(static_cast<int32_t>(batch_num),
                               static_cast<int32_t>(in_features[i]),
                               static_cast<int32_t>(input_x),
                               static_cast<int32_t>(input_y));
            auto data = generate_random_4d<Type>(batch_num, in_features[i], input_y, input_x, -1, 1);
            auto in_lay = layout(data_type, fmt, size);
            auto data_flat = std::vector<Type>(in_lay.get_linear_size(), 0);

            for (size_t bi = 0; bi < batch_num; ++bi) {
                for (size_t fi = 0; fi < in_features[i]; ++fi) {
                    for (size_t yi = 0; yi < input_y; ++yi) {
                        for (size_t xi = 0; xi < input_x; ++xi) {
                            auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                            auto in_offset = in_lay.get_linear_offset(coords);

                            data_flat[in_offset] = data[bi][fi][yi][xi];
                        }
                    }
                }
            }

            auto in_mem = memory::allocate(engine, in_lay);
            set_values(in_mem, data_flat);
            in_memory.push_back(in_mem);

            topology.add(input_layout("input" + std::to_string(i), in_lay));
            in_data.emplace_back(std::move(data));
            input_ids.push_back("input" + std::to_string(i));
        }

        topology.add(concatenation("concat", input_ids, concatenation::concatenation_axis::along_f));

        build_options options;
        options.set_option(build_option::optimize_data(true));
        network network(engine, topology, options);

        for (size_t i = 0; i < in_features.size(); i++) {
            network.set_input_data(input_ids[i], in_memory[i]);
        }

        network.execute();

        auto out_mem = network.get_output("concat").get_memory();
        auto out_ptr = out_mem.pointer<Type>();

        for (size_t bi = 0; bi < batch_num; bi++) {
            size_t f_sum = 0;
            for (size_t in_i = 0; in_i < in_features.size(); in_i++) {
                for (size_t fi = 0; fi < in_features[in_i]; fi++) {
                    for (size_t yi = 0; yi < input_y; yi++) {
                        for (size_t xi = 0; xi < input_x; xi++) {
                            auto output_coords = tensor(batch(bi), feature(f_sum + fi), spatial(xi, yi, 0, 0));
                            auto output_offset = out_mem.get_layout().get_linear_offset(output_coords);

                            auto ref_val = in_data[in_i][bi][fi][yi][xi];
                            auto actual_val = out_ptr[output_offset];
                            EXPECT_EQ(ref_val, actual_val)
                                << " b=" << bi << ", f=" << f_sum + fi << "(input " << in_i << "), y=" << yi << ", x=" << xi;
                        }
                    }
                }
                f_sum += in_features[in_i];
            }
        }
    }
};

using concat_gpu_4d_f16 = concat_gpu_4d<FLOAT16>;
using concat_gpu_4d_i8 = concat_gpu_4d<int8_t>;
using concat_gpu_4d_u8 = concat_gpu_4d<uint8_t>;

TEST_P(concat_gpu_4d_f16, fs_b_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::fs_b_yx_fsv32));
}

INSTANTIATE_TEST_CASE_P(smoke,
                        concat_gpu_4d_f16,
                        concat_gpu_all_params,
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_gpu_4d_i8, b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

INSTANTIATE_TEST_CASE_P(smoke_low_precision,
                        concat_gpu_4d_i8,
                        concat_gpu_all_params,
                        concat_gpu::PrintToStringParamName);

TEST_P(concat_gpu_4d_u8, b_fs_yx_fsv32) {
    ASSERT_NO_FATAL_FAILURE(test(format::b_fs_yx_fsv32));
}

INSTANTIATE_TEST_CASE_P(smoke_low_precision,
                        concat_gpu_4d_u8,
                        concat_gpu_all_params,
                        concat_gpu::PrintToStringParamName);
