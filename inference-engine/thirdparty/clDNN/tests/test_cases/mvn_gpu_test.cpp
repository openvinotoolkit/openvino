/*
// Copyright (c) 2018 Intel Corporation
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
#include <api/CPP/memory.hpp>
#include <api/CPP/input_layout.hpp>
#include "api/CPP/mvn.hpp"
#include "api/CPP/reorder.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <iostream>
#include "float16.h"
#include "test_utils.h"

using namespace cldnn;

class mvn_gpu_test : public ::testing::TestWithParam<cldnn::format>
{
};

template <typename T>
void mvn_compute_mean_accross_channels_bfyx(cldnn::memory &output, bool normalize_variance)
{
    using namespace tests;

    const auto output_desc = generic_test::get_linear_memory_desc(output.get_layout());

    auto output_sizes = output.get_layout().size.sizes();

    uint32_t batch_size = output_sizes[0];
    uint32_t feature_size = output_sizes[1];
    uint32_t y_size = output_sizes[3];
    uint32_t x_size = output_sizes[2];

    auto buff = output.pointer<T>();

    float err_margin = output.get_layout().data_type == data_types::f32 ? 1e-03F : 1e-02F;

    for (uint32_t b = 0; b < batch_size; ++b)
    {
        float sum = 0.f;
        float variance = 0.f;
        for (uint32_t f = 0; f < feature_size; ++f)
        {
            for (uint32_t y = 0; y < y_size; ++y)
            {
                for (uint32_t x = 0; x < x_size; ++x)
                {
                    size_t data_index = generic_test::get_linear_index(output.get_layout(), b, f, y, x, output_desc);
                    float data = static_cast<float>(buff[data_index]);
                    sum += data;
                    if (normalize_variance)
                        variance += data*data;
                }
            }
        }
        sum /= feature_size * y_size * x_size;
        T result_sum = static_cast<T>(sum);
        EXPECT_NEAR(result_sum, 0.f, err_margin);

        if (normalize_variance)
        {
            variance /= feature_size * y_size * x_size;
            T result_variance = static_cast<T>(variance);
            EXPECT_NEAR(result_variance, 1.f, err_margin);
        }
    }
}

template <typename T>
void mvn_compute_mean_within_channels_bfyx(cldnn::memory &output, bool normalize_variance)
{
    using namespace tests;

    const auto output_desc = generic_test::get_linear_memory_desc(output.get_layout());

    auto output_sizes = output.get_layout().size.sizes();

    uint32_t batch_size = output_sizes[0];
    uint32_t feature_size = output_sizes[1];
    uint32_t y_size = output_sizes[3];
    uint32_t x_size = output_sizes[2];

    auto buff = output.pointer<T>();

    float err_margin = output.get_layout().data_type == data_types::f32 ? 1e-03F : 1e-02F;

    for (uint32_t b = 0; b < batch_size; ++b)
    {
        for (uint32_t f = 0; f < feature_size; ++f)
        {
            float sum = 0.f;
            float variance = 0.f;
            for (uint32_t y = 0; y < y_size; ++y)
            {
                for (uint32_t x = 0; x < x_size; ++x)
                {
                    size_t data_index = generic_test::get_linear_index(output.get_layout(), b, f, y, x, output_desc);
                    float data = static_cast<float>(buff[data_index]);
                    sum += data;
                    if (normalize_variance)
                        variance += data*data;
                }
            }
            sum /= y_size * x_size;
            T result_sum = static_cast<T>(sum);
            EXPECT_NEAR(result_sum, 0.f, err_margin);

            if (normalize_variance)
            {
                variance /= y_size * x_size;
                T result_variance = static_cast<T>(variance);
                EXPECT_NEAR(result_variance, 1.f, err_margin);
            }
        }
    }
}

TEST(mvn_gpu_test, mvn_test_across_channels_bfyx)
{
    //mvn accross channels fp32 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 7, 10, 17, 13 } });

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mvn("mvn", "input", true, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_accross_channels_bfyx<float>(output, false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_bfyx_fp16)
{
    //mvn accross channels fp16 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx,{ 7, 10, 17, 13 } });

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mvn("mvn", "input", true, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_accross_channels_bfyx<FLOAT16>(output, false);
}

TEST(mvn_gpu_test, mvn_test_across_channels_bfyx_normalize_variance)
{
    //mvn accross channels fp32 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 7, 10, 17, 13 } });

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mvn("mvn", "input", true, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_accross_channels_bfyx<float>(output, true);
}

TEST(mvn_gpu_test, mvn_test_across_channels_bfyx_normalize_variance_fp16)
{
    //mvn accross channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx,{ 7, 10, 17, 13 } });

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mvn("mvn", "input", true, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_accross_channels_bfyx<FLOAT16>(output, true);
}

TEST(mvn_gpu_test, mvn_test_within_channels_bfyx)
{
    //mvn within channels fp32 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 7, 10, 17, 13 } });

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mvn("mvn", "input", false, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels_bfyx<float>(output, false);
}

TEST(mvn_gpu_test, mvn_test_within_channels_bfyx_fp16)
{
    //mvn within channels fp16 test with normalize_variance set to false
    using namespace cldnn;
    using namespace tests;

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx,{ 7, 10, 17, 13 } });

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mvn("mvn", "input", false, false));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels_bfyx<FLOAT16>(output, false);
}

TEST(mvn_gpu_test, mvn_test_within_channels_bfyx_normalize_variance)
{
    //mvn within channels fp32 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 7, 10, 17, 13 } });

    tests::set_random_values<float>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mvn("mvn", "input", false, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels_bfyx<float>(output, true);
}

TEST(mvn_gpu_test, mvn_test_within_channels_bfyx_normalize_variance_fp16)
{
    //mvn within channels fp16 test with normalize_variance set to true
    using namespace cldnn;
    using namespace tests;

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f16, format::bfyx,{ 7, 10, 17, 13 } });

    tests::set_random_values<FLOAT16>(input, true, 8, 100);

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(mvn("mvn", "input", false, true));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "mvn");

    auto output = outputs.begin()->second.get_memory();
    mvn_compute_mean_within_channels_bfyx<FLOAT16>(output, true);
}