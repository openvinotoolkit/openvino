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

#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/arg_max_min.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace std;
using namespace tests;




TEST(arg_max_gpu, base) {
	//  Input  : 2x3x2x2
	static const int32_t x_size = 2, y_size = 2, feature_num = 3, batch_num = 2;
	const auto& engine = get_test_engine();

	auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(arg_max_min("arg_max", "input", arg_max_min::max));

	vector<float> input_vec = {
			//y0x0 y0x1 y1x0 y1x1
		/*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
		/*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
		/*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

		/*b1f0*/3.f,  0.5f,  7.f,   10.f,
		/*b1f1*/4.f,  0.5f,  8.f,   8.2f,
		/*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
	};
	set_values(input, input_vec);

	network network(engine, topology);

	network.set_input_data("input", input);
	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "arg_max");

	auto output = outputs.at("arg_max").get_memory();
	auto output_ptr = output.pointer<float>();
	float out_buffer[batch_num];
	for (uint32_t i = 0; i < batch_num; i++)
	{
		out_buffer[i] = get_value<float>(output_ptr, i);
	}	
	int size = x_size * y_size * feature_num;
	int index;
	float value;
	for (int i = 0; i < batch_num; i++) {
		EXPECT_GE(out_buffer[i], 0);
		EXPECT_LT(out_buffer[i], size);
		index = (int)out_buffer[i];
		value = input_vec[i*size + (int)index];
		for (int j = 0; j < size; j++)
		{
			EXPECT_LE(input_vec[i*size + j], value);
		}
	}
}

TEST(arg_max_gpu_batch_one, base) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 5, batch_num = 1, top_k = 8;
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", "input", arg_max_min::max, top_k));

    vector<float> input_vec = {
        //y0x0 y0x1 y1x0 y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f3*/0.2f, 0.2f,  -10.f, 4.2f,
        /*b0f3*/0.1f, 0.3f,  -11.f, 15.0f
    };
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "arg_max");

    auto output = outputs.at("arg_max").get_memory();
    auto output_ptr = output.pointer<float>();
    float out_buffer[batch_num * top_k];
    for (uint32_t i = 0; i < batch_num * top_k; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
     int size = x_size * y_size * feature_num;
     int index;
     float value;
     for (int i = 0; i < batch_num; i++) {
         int count = 0;
         int amount = 0;
         int same_values = 1;
         int j;
         for (j = 0; j < top_k; j++) {
             EXPECT_GE((int)out_buffer[i*top_k + j], 0);
             EXPECT_LT((int)out_buffer[i*top_k + j], size);
             if (top_k - 1 == j) {
                 if (input_vec[i*size + (int)out_buffer[i*top_k + j]] != input_vec[i*size + (int)out_buffer[i*top_k + j - 1]]) {
                     amount += j;
                 }
                 else
                     amount += same_values * (j - same_values + 1);
             }
             else if (input_vec[i*size + (int)out_buffer[i*top_k + j]] != input_vec[i*size + (int)out_buffer[i*top_k + j + 1]]) {
                 if (same_values != j + 1) {
                     amount += same_values * (j - same_values + 1);
                     same_values = 1;
                 }
             }
             else
                 same_values++;
         }
         EXPECT_GE(out_buffer[i*top_k + top_k - 1], 0);
         EXPECT_LT(out_buffer[i*top_k + top_k - 1], size);
         for (int j = 0; j < top_k; j++)
         {
             index = (int)out_buffer[i*top_k + j];
             value = input_vec[i*size + index];
             for (int k = 0; k < size; k++)
             {
                 if (input_vec[i*size + k] > value)
                     count++;
             }
         }
         EXPECT_EQ(count, amount);
     }
}


TEST(arg_max_gpu_top_k, base) {
	//  Input  : 2x3x2x2
	static const int32_t x_size = 2, y_size = 2, feature_num = 5, batch_num = 2;
	const auto& engine = get_test_engine();
	const int top_k = 8;
	auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(arg_max_min("arg_max", "input", arg_max_min::max, top_k));

	vector<float> input_vec = {
		//y0x0 y0x1 y1x0 y1x1
		/*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
		/*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
		/*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,
		/*b0f3*/0.2f, 0.2f,  -10.f, 4.2f,
		/*b0f3*/0.1f, 0.3f,  -11.f, 15.0f,

		/*b1f0*/3.f,  0.5f,  7.f,   10.f,
		/*b1f1*/4.f,  0.5f,  8.f,   8.2f,
		/*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
		/*b1f3*/4.f,  0.5f,  8.f,   8.2f,
		/*b0f3*/0.1f, 0.3f,  -11.f, 15.0f,
	};
	set_values(input, input_vec);

	network network(engine, topology);

	network.set_input_data("input", input);
	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "arg_max");

	auto output = outputs.at("arg_max").get_memory();
	auto output_ptr = output.pointer<float>();
	float out_buffer[batch_num * top_k];
	for (uint32_t i = 0; i < batch_num * top_k; i++)
	{
		out_buffer[i] = get_value<float>(output_ptr, i);
	}
	int size = x_size * y_size * feature_num;
	int index;
	float value;
	for (int i = 0; i < batch_num; i++) {
		int count = 0;
		int amount = 0;
		int same_values = 1;
		int j;
		for (j = 0; j < top_k; j++) {
			EXPECT_GE((int)out_buffer[i*top_k + j], 0);
			EXPECT_LT((int)out_buffer[i*top_k + j], size);
			if (top_k - 1 == j) {
				if (input_vec[i*size + (int)(int)out_buffer[i*top_k + j]] != input_vec[i*size + (int)(int)out_buffer[i*top_k + j - 1]]) {
					amount += j;
				}
				else
					amount += same_values * (j - same_values + 1);
			}
			else if (input_vec[i*size + (int)(int)out_buffer[i*top_k + j]] != input_vec[i*size + (int)(int)out_buffer[i*top_k + j + 1]]) {
				if (same_values != j+1) {
					amount += same_values * (j - same_values + 1);
					same_values = 1;
				}
			}
			else
				same_values++;
		}
		EXPECT_GE(out_buffer[i*top_k + top_k - 1], 0);
		EXPECT_LT(out_buffer[i*top_k + top_k - 1], size);
		for (int j = 0; j < top_k; j++)
		{
			index = (int)out_buffer[i*top_k + j];
			value = input_vec[i*size + index];
			for (int k = 0; k < size; k++)
			{
				if (input_vec[i*size + k] > value)
					count++;
			}
		}
		EXPECT_EQ(count, amount);
	}
}

TEST(arg_max_gpu_min, base) {
	//  Input  : 2x3x2x2
	static const int32_t x_size = 2, y_size = 2, feature_num = 4,
		batch_num = 2;
	const auto& engine = get_test_engine();

	auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(arg_max_min("arg_max", "input", arg_max_min::min));

	vector<float> input_vec = {
		//y0x0 y0x1 y1x0 y1x1
		/*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
		/*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
		/*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,
		/*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

		/*b1f0*/3.f,  0.5f,  7.f,   10.f,
		/*b1f1*/4.f,  0.5f,  8.f,   8.2f,
		/*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
		/*b0f2*/0.2f, 0.2f,  -10.f, 5.2f
	};
	set_values(input, input_vec);

	network network(engine, topology);

	network.set_input_data("input", input);
	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "arg_max");

	auto output = outputs.at("arg_max").get_memory();
	auto output_ptr = output.pointer<float>();
	float out_buffer[batch_num];
	for (uint32_t i = 0; i < batch_num; i++)
	{
		out_buffer[i] = get_value<float>(output_ptr, i);
	}
	int size = x_size * y_size * feature_num;
	int index;
	float value;
	for (int i = 0; i < batch_num; i++) {
		EXPECT_GE(out_buffer[i], 0);
		EXPECT_LT(out_buffer[i], size);
		index = (int)out_buffer[i];
		value = input_vec[i*size + index];
		for (int j = 0; j < size; j++)
		{
			EXPECT_GE(input_vec[i*size + j], value);
		}
	}
}

TEST(arg_max_gpu_min_top_k, base) {
	//  Input  : 2x3x2x2
	static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
	const auto& engine = get_test_engine();
	const int top_k = 3;
	auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(arg_max_min("arg_max", "input", arg_max_min::min, top_k));

	vector<float> input_vec = {
		        //f0b0 f0b1 f1b0 f1b1
		/*x0y0*/0.1f, -0.1f, 0.9f,  1.5f,
		/*x0y1*/0.2f, 0.2f,  -10.f, 5.2f,
		/*x0y2*/0.2f, 0.2f,  -10.f, 5.2f,
		/*x0f3*/0.2f, 0.2f,  -10.f, 4.2f,

		/*x1y0*/3.f,  0.5f,  7.f,   10.f,
		/*x1y1*/4.f,  0.5f,  8.f,   8.2f,
		/*x1y2*/0.2f, 0.2f,  -10.f, 5.2f,
		/*x1y3*/4.f,  0.5f,  8.f,   8.2f
	};
	set_values(input, input_vec);

	network network(engine, topology);

	network.set_input_data("input", input);
	auto outputs = network.execute();

	EXPECT_EQ(outputs.size(), size_t(1));
	EXPECT_EQ(outputs.begin()->first, "arg_max");

	auto output = outputs.at("arg_max").get_memory();
	auto output_ptr = output.pointer<float>();
	float out_buffer[batch_num * top_k];
	for (uint32_t i = 0; i < batch_num * top_k; i++)
	{
		out_buffer[i] = get_value<float>(output_ptr, i);
	}
	int size = x_size * y_size * feature_num;
	int index;
	float value;
	for (int i = 0; i < batch_num; i++) {
		int count = 0;
		int amount = 0;
		int same_values = 1;
		int j;
		for (j = 0; j < top_k; j++) {
			EXPECT_GE((int)out_buffer[i*top_k + j], 0);
			EXPECT_LT((int)out_buffer[i*top_k + j], size);
			if (top_k - 1 == j) {
				if (input_vec[i*size + (int)out_buffer[i*top_k + j]] != input_vec[i*size + (int)out_buffer[i*top_k + j - 1]]) {
					amount += j;
				}
				else
					amount += same_values * (j - same_values + 1);
			}
			else if (input_vec[i*size + (int)out_buffer[i*top_k + j]] != input_vec[i*size + (int)out_buffer[i*top_k + j + 1]]) {
				if (same_values != j + 1) {
					amount += same_values * (j - same_values + 1);
					same_values = 1;
				}
			}
			else
				same_values++;
		}
		EXPECT_GE(out_buffer[i*top_k + top_k - 1], 0);
		EXPECT_LT(out_buffer[i*top_k + top_k - 1], size);
		for (int j = 0; j < top_k; j++)
		{
			index = (int)out_buffer[i*top_k + j];
			value = input_vec[i*size + index];
			for (int k = 0; k < size; k++)
			{
				if (input_vec[i*size + k] < value)
					count++;
			}
		}
		EXPECT_EQ(count, amount);
	}
}

TEST(arg_max_gpu_min_axis_batch, base) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", "input", arg_max_min::min, top_k, arg_max_min::batch));

    vector<float> input_vec = {
        //y0x0 y0x1 y1x0 y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f3*/0.2f, 0.2f,  -10.f, 4.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   10.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f3*/4.f,  0.5f,  8.f,   8.2f
    };
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = y_size * feature_num * x_size * top_k;
    auto output = outputs.at("arg_max").get_memory();
    auto output_ptr = output.pointer<float>();
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    for (int i = 0; i < out_size; i++)
    {
        EXPECT_EQ(out_buffer[i], i % 2 == 0 ? 0 : 1);
    }
}