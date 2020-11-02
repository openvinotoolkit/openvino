
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
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/arg_max_min.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include <api/mutable_data.hpp>
#include <api/data.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

template <typename Tin, typename Tout>
void generic_arg_max_test_xyf(int input_b, int input_f, int input_y, int input_x, arg_max_min::out_type mode, bool expect_throw = false)
{
    auto axis = arg_max_min::axis_name::xyf;
    auto sort_type = arg_max_min::sort_type::sort_by_values;
    auto test_input_fmt = format::bfyx;
    const auto& engine = get_test_engine();

    tensor input_tensor(input_b, input_f, input_x, input_y);
    auto input = memory::allocate(engine, { type_to_data_type<Tin>::value, test_input_fmt, input_tensor });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, mode, 1U, axis, sort_type, false, padding(), type_to_data_type<Tout>::value));

    int min_random = -2, max_random = 2;
    VVVVF<Tin> input_rnd = generate_random_4d<Tin>(input_b, input_f, input_y, input_x, min_random, max_random);
    VF<Tin> input_rnd_vec = flatten_4d<Tin>(test_input_fmt, input_rnd);

    set_values(input, input_rnd_vec);

    if (expect_throw) {
        std::string msg_to_find = "Current output data type is unable to hold maximum index of a tensor.";
        EXPECT_ANY_THROW(check_exception_massage(engine, topology, msg_to_find));
        return;
    }
    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "arg_max");

    int out_size = input_x * input_y * input_f;

    auto output = outputs.at("arg_max").get_memory();
    auto output_ptr = output.pointer<Tout>();

    Tout index;
    Tin value;
    for (auto i = 0; i < input_b; i++) {
        index = get_value<Tout>(output_ptr, i);
        EXPECT_GE(index, (Tout)0);
        EXPECT_LT(index, (Tout)out_size);
        value = input_rnd_vec[i*out_size + (int)index];
        for (auto j = 0; j < out_size; j++) {
            if (mode == arg_max_min::out_type::max) {
                EXPECT_LE(input_rnd_vec[i*out_size + j], value);
            }
            else {
                EXPECT_GE(input_rnd_vec[i*out_size + j], value);
            }
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
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k));

    std::vector<float> input_vec = {
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
	topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k));

	std::vector<float> input_vec = {
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

TEST(arg_max_gpu_min_top_k, base) {
	//  Input  : 2x3x2x2
	static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
	const auto& engine = get_test_engine();
	const int top_k = 3;
	auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
	topology topology;
	topology.add(input_layout("input", input.get_layout()));
	topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::min, top_k));

	std::vector<float> input_vec = {
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
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::min, top_k, arg_max_min::batch));

    std::vector<float> input_vec = {
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
        EXPECT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
    }
}

TEST(arg_max_gpu, f32) {
    generic_arg_max_test_xyf<float, float>(50, 25, 25, 25, arg_max_min::out_type::max);
}

TEST(arg_max_gpu_min, f32) {
    generic_arg_max_test_xyf<float, float>(50, 25, 25, 25, arg_max_min::out_type::min);
}

TEST(arg_max_gpu, u8) {
    generic_arg_max_test_xyf<float, uint8_t>(4, 2, 2, 2, arg_max_min::out_type::max);
}

TEST(arg_max_gpu_min, u8) {
    generic_arg_max_test_xyf<float, uint8_t>(4, 2, 2, 2, arg_max_min::out_type::min);
}

TEST(arg_max_gpu, i8) {
    generic_arg_max_test_xyf<float, int8_t>(4, 2, 2, 2, arg_max_min::out_type::max);
}

TEST(arg_max_gpu_bad_sizes, i8) {
    generic_arg_max_test_xyf<float, uint8_t>(50, 25, 25, 25, arg_max_min::out_type::max, true);
}

TEST(arg_max_gpu_min, i8) {
    generic_arg_max_test_xyf<float, int8_t>(4, 2, 2, 2, arg_max_min::out_type::min);
}

TEST(arg_max_gpu, i32) {
    generic_arg_max_test_xyf<float, int32_t>(50, 25, 25, 25, arg_max_min::out_type::max);
}

TEST(arg_max_gpu_min, i32) {
    generic_arg_max_test_xyf<float, int32_t>(50, 25, 25, 25, arg_max_min::out_type::min);
}

TEST(arg_max_gpu, i64) {
    generic_arg_max_test_xyf<float, int64_t>(50, 25, 25, 25, arg_max_min::out_type::max);
}

TEST(arg_max_gpu_min, i64) {
    generic_arg_max_test_xyf<float, int64_t>(50, 25, 25, 25, arg_max_min::out_type::min);
}

TEST(arg_max_gpu_min_axis_batch, i32) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::min, top_k, arg_max_min::batch, arg_max_min::sort_by_values, false, padding(), data_types::i32));

    std::vector<float> input_vec = {
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
    auto output_ptr = output.pointer<int32_t>();
    int32_t out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = get_value<int32_t>(output_ptr, i);
    }
    for (int i = 0; i < out_size; i++)
    {
        EXPECT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
    }
}

TEST(arg_max_gpu_min_axis_batch_bfzyx, i32) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, z_size = 1, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ batch_num, feature_num, x_size , y_size, z_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::min, top_k, arg_max_min::batch, arg_max_min::sort_by_values, false, padding(), data_types::i32));

    std::vector<float> input_vec = {
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
    auto output_ptr = output.pointer<int32_t>();
    int32_t out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = get_value<int32_t>(output_ptr, i);
    }
    for (int i = 0; i < out_size; i++)
    {
        EXPECT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb, f32) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::y, arg_max_min::sort_by_values, false, padding(), data_types::f32));

    std::vector<float> input_vec = {
            0.1f, -0.1f,
            0.9f,  1.5f,
            0.2f, 0.2f,
            -10.f, 5.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            0.2f, 0.2f,
            -10.f, 4.2f,

            3.f,  0.5f,
            7.f,   10.f,
            4.f,  0.5f,
            8.f,   8.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            4.f,  0.5f,
            8.f,   8.2f
    };

    std::vector<float> ref_vec = {
            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,

            0.f, 0.f,
            0.f, 0.f,
            1.f, 1.f,
            1.f, 1.f,
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
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_batch_yxfb, f32) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::batch, arg_max_min::sort_by_values, false, padding(), data_types::f32));

    std::vector<float> input_vec = {
            0.1f, -0.1f,
            0.9f,  1.5f,
            0.2f, 0.2f,
            -10.f, 5.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            0.2f, 0.2f,
            -10.f, 4.2f,

            3.f,  0.5f,
            7.f,   10.f,
            4.f,  0.5f,
            8.f,   8.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            4.f,  0.5f,
            8.f,   8.2f
    };

    std::vector<float> ref_vec = {
            0.f, 1.f,
            0.f, 1.f,
            0.f, 1.f,
            0.f, 1.f,

            0.f, 1.f,
            0.f, 1.f,
            0.f, 1.f,
            0.f, 1.f
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
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb_topk_2, f32) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::y, arg_max_min::sort_by_values, false, padding(), data_types::f32));

    std::vector<float> input_vec = {
            0.1f, -0.1f,
            0.9f,  1.5f,
            0.2f, 0.2f,
            -10.f, 5.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            0.2f, 0.2f,
            -10.f, 4.2f,

            3.f,  0.5f,
            7.f,   10.f,
            4.f,  0.5f,
            8.f,   8.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            4.f,  0.5f,
            8.f,   8.2f
    };

    std::vector<float> ref_vec = {
            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,

            0.f, 0.f,
            0.f, 0.f,
            1.f, 1.f,
            1.f, 1.f,

            0.f, 0.f,
            0.f, 0.f,
            0.f, 0.f,
            0.f, 0.f,

            1.f, 1.f,
            1.f, 1.f,
            0.f, 0.f,
            0.f, 0.f,
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
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(top_k_layer_tests, second_output) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    auto top_k_input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1 , 1 } });
    auto second_output = memory::allocate(engine, { data_types::f32, format::bfyx, { top_k, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max", { "input", "const", "second_output" }, arg_max_min::min, top_k, arg_max_min::batch));

    std::vector<float> input_vec = {
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
    auto second_output_ptr = second_output.pointer<float>();
    float out_buffer[out_size];
    float second_out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
        second_out_buffer[i] = get_value<float>(second_output_ptr, i);
    }
    for (int i = 0; i < out_size; i++)
    {
        EXPECT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
        EXPECT_EQ(second_out_buffer[i], input_vec[i]);
    }
}

TEST(top_k_layer_tests, second_output2) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    auto top_k_input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1 , 1 } });
    auto second_output = memory::allocate(engine, { data_types::f32, format::yxfb, { top_k, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max", { "input", "const", "second_output" }, arg_max_min::max, top_k, arg_max_min::batch, arg_max_min::sort_by_values, false, padding(), data_types::f32));

    std::vector<float> input_vec = {
            0.1f, -0.1f,
            0.9f,  1.5f,
            0.2f, 0.2f,
            -10.f, 5.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            0.2f, 0.2f,
            -10.f, 4.2f,

            3.f,  0.5f,
            7.f,   10.f,
            4.f,  0.5f,
            8.f,   8.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            4.f,  0.5f,
            8.f,   8.2f
    };

    std::vector<float> ref_vec = {
            0.f, 1.f,
            0.f, 1.f,
            0.f, 1.f,
            0.f, 1.f,

            0.f, 1.f,
            0.f, 1.f,
            0.f, 1.f,
            0.f, 1.f
    };

    std::vector<float> second_ref_vec = {
            0.1f,
            1.5f,
            0.2f,
            5.2f,

            0.2f,
            5.2f,
            0.2f,
            4.2f,

            3.f,
            10.f,
            4.f,
            8.2f,

            0.2f,
            5.2f,
            4.f,
            8.2f
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
    auto second_output_ptr = second_output.pointer<float>();
    float out_buffer[out_size];
    float second_out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++)
    {
        out_buffer[i] = get_value<float>(output_ptr, i);
        second_out_buffer[i] = get_value<float>(second_output_ptr, i);
    }
    for (int i = 0; i < out_size; i++)
    {
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
        EXPECT_EQ(second_out_buffer[i], second_ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb_topk_2, sort_by_values) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::y, arg_max_min::sort_by_values, false, padding(), data_types::f32));

    std::vector<float> input_vec = {
            0.1f, -0.1f,
            0.9f,  1.5f,
            0.2f, 0.2f,
            -10.f, 5.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            0.2f, 0.2f,
            -10.f, 4.2f,

            3.f,  0.5f,
            7.f,   10.f,
            4.f,  0.5f,
            8.f,   8.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            4.f,  0.5f,
            8.f,   8.2f
    };

    std::vector<float> ref_vec = {
            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,

            0.f, 0.f,
            0.f, 0.f,
            1.f, 1.f,
            1.f, 1.f,

            0.f, 0.f,
            0.f, 0.f,
            0.f, 0.f,
            0.f, 0.f,

            1.f, 1.f,
            1.f, 1.f,
            0.f, 0.f,
            0.f, 0.f,
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
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb_topk_2, sort_by_indices) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    const auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::y, arg_max_min::sort_by_indices, false, padding(), data_types::f32));

    std::vector<float> input_vec = {
            0.1f, -0.1f,
            0.9f,  1.5f,
            0.2f, 0.2f,
            -10.f, 5.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            0.2f, 0.2f,
            -10.f, 4.2f,

            3.f,  0.5f,
            7.f,   10.f,
            4.f,  0.5f,
            8.f,   8.2f,

            0.2f, 0.2f,
            -10.f, 5.2f,
            4.f,  0.5f,
            8.f,   8.2f
    };

    std::vector<float> ref_vec = {
            0.f, 0.f,
            0.f, 0.f,
            0.f, 0.f,
            0.f, 0.f,

            0.f, 0.f,
            0.f, 0.f,
            0.f, 0.f,
            0.f, 0.f,

            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,

            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,
            1.f, 1.f,
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
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}
