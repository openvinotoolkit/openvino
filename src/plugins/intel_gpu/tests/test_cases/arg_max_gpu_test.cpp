// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/arg_max_min.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>

using namespace cldnn;
using namespace ::tests;

template <typename Tin, typename Tout>
void generic_arg_max_test_xyf(int input_b, int input_f, int input_y, int input_x, arg_max_min::out_type mode, bool expect_throw = false) {
    auto axis = arg_max_min::axis_name::xyf;
    auto sort_type = arg_max_min::sort_type::sort_by_values;
    auto test_input_fmt = format::bfyx;
    auto& engine = get_test_engine();

    tensor input_tensor(input_b, input_f, input_x, input_y);
    auto input = engine.allocate_memory({ type_to_data_type<Tin>::value, test_input_fmt, input_tensor });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, mode, 1U, axis, sort_type, false, "", padding(), type_to_data_type<Tout>::value));

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
    cldnn::mem_lock<Tout> output_ptr(output, get_test_stream());

    Tout index;
    Tin value;
    for (auto i = 0; i < input_b; i++) {
        index = get_value<Tout>(output_ptr.data(), i);
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
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[batch_num * top_k];
    for (uint32_t i = 0; i < batch_num * top_k; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
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
             } else if (input_vec[i*size + (int)out_buffer[i*top_k + j]] != input_vec[i*size + (int)out_buffer[i*top_k + j + 1]]) {
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
         for (int j = 0; j < top_k; j++) {
             index = (int)out_buffer[i*top_k + j];
             value = input_vec[i*size + index];
             for (int k = 0; k < size; k++) {
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
	auto& engine = get_test_engine();
	const int top_k = 8;
	auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
	topology topology;
	topology.add(input_layout("input", input->get_layout()));
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
	cldnn::mem_lock<float> output_ptr(output, get_test_stream());
	float out_buffer[batch_num * top_k];
	for (uint32_t i = 0; i < batch_num * top_k; i++) {
		out_buffer[i] = get_value<float>(output_ptr.data(), i);
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
			} else if (input_vec[i*size + (int)(int)out_buffer[i*top_k + j]] != input_vec[i*size + (int)(int)out_buffer[i*top_k + j + 1]]) {
				if (same_values != j+1) {
					amount += same_values * (j - same_values + 1);
					same_values = 1;
				}
			} else {
				same_values++;
            }
		}
		EXPECT_GE(out_buffer[i*top_k + top_k - 1], 0);
		EXPECT_LT(out_buffer[i*top_k + top_k - 1], size);
		for (int j = 0; j < top_k; j++) {
			index = (int)out_buffer[i*top_k + j];
			value = input_vec[i*size + index];
			for (int k = 0; k < size; k++) {
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
	auto& engine = get_test_engine();
	const int top_k = 3;
	auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
	topology topology;
	topology.add(input_layout("input", input->get_layout()));
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
	float out_buffer[batch_num * top_k];
	for (uint32_t i = 0; i < batch_num * top_k; i++) {
		out_buffer[i] = get_value<float>(output_ptr.data(), i);
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
			} else if (input_vec[i*size + (int)out_buffer[i*top_k + j]] != input_vec[i*size + (int)out_buffer[i*top_k + j + 1]]) {
				if (same_values != j + 1) {
					amount += same_values * (j - same_values + 1);
					same_values = 1;
				}
			} else {
				same_values++;
            }
		}
		EXPECT_GE(out_buffer[i*top_k + top_k - 1], 0);
		EXPECT_LT(out_buffer[i*top_k + top_k - 1], size);
		for (int j = 0; j < top_k; j++) {
			index = (int)out_buffer[i*top_k + j];
			value = input_vec[i*size + index];
			for (int k = 0; k < size; k++) {
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
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
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
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::min, top_k, arg_max_min::batch, arg_max_min::sort_by_values, false, "", padding(), data_types::i32));

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
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    int32_t out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<int32_t>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
    }
}

TEST(arg_max_gpu_min_axis_batch_bfzyx, i32) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, z_size = 1, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ batch_num, feature_num, x_size , y_size, z_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::min, top_k, arg_max_min::batch, arg_max_min::sort_by_values, false, "", padding(), data_types::i32));

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
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    int32_t out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<int32_t>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb, f32) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::y, arg_max_min::sort_by_values, false, "", padding(), data_types::f32));

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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_batch_yxfb, f32) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::batch, arg_max_min::sort_by_values, false, "", padding(), data_types::f32));

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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb_topk_2, f32) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::y, arg_max_min::sort_by_values, false, "", padding(), data_types::f32));

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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(top_k_layer_tests, second_output) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    auto top_k_input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1 , 1 } });
    auto second_output = engine.allocate_memory({ data_types::f32, format::bfyx, { top_k, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> second_output_ptr(second_output, get_test_stream());

    float out_buffer[out_size];
    float second_out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
        second_out_buffer[i] = get_value<float>(second_output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], i < (out_size / 2) ? 0 : 1);
        EXPECT_EQ(second_out_buffer[i], input_vec[i]);
    }
}

TEST(top_k_layer_tests, second_output2) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 1;
    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    auto top_k_input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1 , 1 } });
    auto second_output = engine.allocate_memory({ data_types::f32, format::yxfb, { top_k, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(cldnn::data("const", top_k_input));
    topology.add(mutable_data("second_output", second_output));
    topology.add(arg_max_min("arg_max", { "input", "const", "second_output" }, arg_max_min::max, top_k, arg_max_min::batch, arg_max_min::sort_by_values, false, "", padding(), data_types::f32));

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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> second_output_ptr(second_output, get_test_stream());
    float out_buffer[out_size];
    float second_out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
        second_out_buffer[i] = get_value<float>(second_output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
        EXPECT_EQ(second_out_buffer[i], second_ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb_topk_2, sort_by_values) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::y, arg_max_min::sort_by_values, false, "", padding(), data_types::f32));

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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}

TEST(arg_max_gpu_min_axis_y_yxfb_topk_2, sort_by_indices) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { "input" }, arg_max_min::max, top_k, arg_max_min::y, arg_max_min::sort_by_indices, false, "", padding(), data_types::f32));

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
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<float>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}


TEST(top_k_layer_tests, sort_probabilities_by_indices) {
    static const int32_t x_size = 10, y_size = 1, feature_num = 1, batch_num = 1;
    auto& engine = get_test_engine();
    const int top_k = 5;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(arg_max_min("arg_max", { "input"}, arg_max_min::max, top_k, arg_max_min::x,
                             arg_max_min::sort_by_values, false, "", padding(), data_types::i32));

    std::vector<float> input_vec = {
           0.9f,
           0.1f,
           0.2f,
           0.8f,
           0.5f,
           0.6f,
           0.3f,
           0.4f,
           0.7f,
           0.95f
    };

    std::vector<int> ref_vec = {
           9, 0, 3, 8, 5
    };

    set_values(input, input_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "arg_max");
    const int out_size = top_k;
    auto output = outputs.at("arg_max").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());
    int out_buffer[out_size];
    for (uint32_t i = 0; i < out_size; i++) {
        out_buffer[i] = get_value<int>(output_ptr.data(), i);
    }
    for (int i = 0; i < out_size; i++) {
        EXPECT_EQ(out_buffer[i], ref_vec[i]);
    }
}

const std::vector<float> input_vec1 = {
0.000109,
0.000282,
0.000112,
0.000108,
0.000154,
0.000026,
0.000103,
0.000072,
0.000138,
0.000098,
0.001701,
0.000206,
0.000554,
0.000135,
0.000058,
0.000190,
0.000051,
0.000043,
0.000062,
0.000262,
0.000232,
0.000112,
0.000105,
0.000107,
0.000227,
0.000104,
0.000075,
0.000076,
0.000076,
0.000143,
0.000135,
0.000073,
0.000126,
0.000120,
0.000100,
0.000249,
0.000144,
0.000507,
0.000185,
0.000293,
0.000198,
0.000033,
0.000240,
0.000064,
0.000128,
0.000094,
0.000114,
0.000058,
0.000148,
0.000103,
0.000262,
0.000278,
0.000123,
0.000051,
0.000058,
0.000148,
0.000052,
0.000454,
0.000090,
0.000071,
0.000100,
0.000143,
0.000148,
0.000123,
0.000257,
0.000060,
0.000157,
0.000159,
0.000135,
0.000155,
0.000495,
0.000246,
0.000234,
0.000387,
0.000449,
0.000261,
0.000059,
0.000155,
0.000086,
0.000230,
0.000197,
0.000221,
0.000034,
0.000093,
0.000499,
0.000103,
0.000165,
0.000102,
0.000435,
0.000239,
0.000061,
0.000039,
0.000047,
0.000036,
0.000224,
0.000055,
0.000041,
0.000082,
0.000129,
0.000346,
0.000640,
0.000288,
0.000231,
0.000184,
0.000104,
0.000156,
0.000225,
0.000493,
0.000147,
0.000101,
0.000032,
0.000222,
0.000117,
0.000246,
0.000110,
0.000106,
0.000130,
0.000127,
0.000154,
0.000136,
0.000228,
0.000177,
0.000239,
0.000209,
0.000113,
0.000076,
0.000151,
0.000260,
0.000123,
0.000150,
0.000034,
0.000180,
0.000111,
0.000052,
0.000096,
0.000345,
0.000095,
0.000221,
0.000171,
0.000461,
0.000080,
0.000103,
0.000081,
0.000132,
0.000138,
0.000161,
0.000159,
0.000109,
0.000140,
0.000481,
0.000172,
0.000067,
0.000157,
0.000491,
0.000117,
0.000070,
0.000270,
0.000156,
0.000229,
0.000184,
0.000130,
0.000049,
0.000157,
0.000144,
0.000143,
0.000203,
0.000134,
0.000512,
0.000172,
0.000187,
0.000102,
0.000202,
0.000100,
0.000107,
0.000060,
0.000196,
0.000109,
0.000137,
0.000270,
0.000180,
0.000124,
0.000144,
0.000098,
0.000095,
0.000272,
0.000169,
0.000269,
0.000370,
0.000212,
0.000323,
0.000391,
0.000055,
0.000114,
0.000338,
0.000208,
0.000067,
0.000457,
0.000129,
0.000175,
0.000295,
0.000185,
0.000269,
0.000090,
0.000094,
0.000312,
0.000112,
0.000266,
0.000493,
0.000186,
0.000127,
0.000070,
0.000265,
0.000160,
0.000243,
0.000067,
0.000118,
0.000295,
0.000083,
0.000239,
0.000134,
0.000161,
0.000127,
0.000179,
0.000120,
0.000180,
0.000163,
0.000043,
0.000135,
0.000170,
0.000132,
0.000291,
0.000234,
0.000513,
0.000800,
0.000257,
0.000140,
0.000106,
0.000049,
0.000075,
0.000070,
0.000239,
0.000187,
0.000118,
0.000056,
0.000088,
0.000152,
0.000224,
0.000124,
0.000092,
0.000218,
0.000194,
0.000259,
0.000409,
0.000207,
0.000191,
0.000085,
0.000047,
0.000518,
0.000088,
0.000367,
0.000203,
0.000388,
0.000197,
0.000145,
0.000207,
0.000108,
0.000150,
0.000142,
0.000332,
0.000276,
0.000434,
0.000240,
0.000139,
0.000435,
0.000170,
0.000331,
0.000363,
0.002229,
0.000178,
0.000104,
0.000049,
0.000390,
0.000655,
0.001461,
0.000378,
0.000060,
0.000893,
0.000110,
0.000231,
0.001369,
0.000158,
0.001266,
0.000297,
0.001158,
0.001414,
0.000334,
0.000410,
0.000143,
0.000275,
0.000303,
0.000066,
0.000120,
0.000220,
0.000095,
0.000101,
0.000184,
0.000053,
0.000086,
0.000212,
0.000118,
0.000215,
0.000104,
0.000072,
0.000139,
0.000076,
0.000152,
0.000083,
0.000105,
0.000329,
0.000192,
0.000149,
0.000170,
0.000066,
0.000097,
0.000285,
0.000146,
0.000236,
0.000129,
0.000091,
0.000076,
0.000100,
0.000134,
0.000079,
0.000125,
0.000272,
0.000185,
0.000187,
0.000086,
0.000149,
0.000045,
0.000561,
1.813452,
0.000385,
0.000380,
0.001857,
0.000571,
0.000130,
0.000650,
0.000133,
0.000147,
0.000210,
0.000342,
0.002329,
0.000712,
0.001396,
0.000610,
0.000405,
0.000096,
0.000120,
0.000102,
0.000091,
0.000078,
0.001877,
0.000361,
0.000724,
0.000161,
0.000082,
0.000243,
0.000173,
0.000075,
0.000182,
0.000085,
0.000205,
0.000199,
0.000085,
0.000040,
0.000737,
0.000237,
0.000108,
0.000219,
0.000099,
0.000156,
0.000038,
0.000059,
0.000474,
0.000527,
0.000265,
0.000683,
0.000070,
0.000165,
0.000362,
0.000083,
0.000138,
0.000213,
0.000085,
0.000118,
0.000165,
0.000186,
0.000181,
0.000112,
0.000119,
0.000249,
0.000402,
0.000347,
0.000110,
0.000122,
0.000293,
0.000054,
0.000112,
0.000148,
0.000167,
0.000226,
0.000188,
0.000097,
0.000127,
0.000172,
0.000047,
0.000054,
0.000195,
0.000239,
0.000254,
0.000175,
0.000108,
0.000123,
0.000131,
0.000102,
0.000200,
0.000088,
0.000090,
0.000083,
0.000150,
0.000169,
0.000225,
0.000212,
0.000077,
0.000267,
0.000259,
0.000106,
0.000487,
0.000287,
0.000262,
0.000070,
0.000187,
0.000147,
0.000272,
0.000179,
0.000127,
0.000130,
0.000079,
0.000289,
0.000094,
0.000049,
0.000197,
0.000131,
0.000145,
0.000047,
0.000075,
0.000105,
0.000344,
0.000033,
0.000107,
0.000126,
0.000068,
0.000123,
0.000103,
0.000120,
0.000141,
0.000078,
0.000083,
0.000079,
0.000094,
0.000096,
0.000105,
0.000115,
0.000348,
0.000072,
0.000102,
0.000246,
0.000105,
0.000089,
0.000425,
0.000387,
0.000077,
0.000201,
0.000121,
0.000083,
0.000234,
0.000351,
0.000328,
0.000135,
0.000080,
0.000155,
0.000061,
0.000041,
0.000289,
0.000071,
0.000066,
0.000377,
0.000077,
0.000114,
0.000133,
0.000090,
0.000213,
0.000088,
0.000156,
0.000153,
0.000079,
0.000155,
0.000123,
0.000268,
0.000173,
0.000050,
0.000136,
0.000153,
0.000074,
0.000106,
0.000173,
0.000111,
0.000196,
0.000285,
0.000066,
0.000190,
0.000094,
0.000306,
0.000327,
0.000085,
0.000082,
0.000200,
0.000602,
0.000138,
0.000207,
0.000178,
0.000101,
0.000190,
0.000152,
0.000153,
0.000088,
0.000051,
0.000141,
0.000128,
0.000220,
0.000095,
0.000148,
0.000300,
0.000171,
0.000053,
0.000212,
0.000282,
0.000142,
0.000175,
0.000151,
0.000084,
0.000118,
0.000205,
0.000429,
0.000044,
0.000112,
0.000107,
0.000397,
0.000087,
0.000208,
0.000116,
0.000069,
0.000037,
0.000178,
0.000060,
0.000107,
0.000124,
0.000208,
0.000115,
0.000051,
0.000093,
0.000150,
0.000152,
0.000104,
0.000165,
0.000189,
0.000417,
0.000081,
0.000052,
0.000027,
0.000075,
0.000158,
0.000073,
0.000067,
0.000159,
0.000062,
0.000112,
0.000058,
0.000116,
0.000100,
0.000167,
0.000314,
0.000089,
0.000095,
0.000126,
0.000112,
0.000074,
0.000106,
0.000129,
0.000253,
0.000252,
0.000136,
0.000107,
0.000110,
0.000183,
0.000096,
0.000092,
0.000148,
0.000138,
0.000098,
0.000107,
0.000202,
0.000180,
0.000111,
0.000053,
0.000145,
0.000096,
0.000113,
0.000215,
0.000124,
0.000059,
0.000093,
0.000382,
0.000133,
0.000079,
0.000097,
0.000284,
0.000105,
0.000098,
0.000180,
0.000071,
0.000104,
0.000472,
0.000068,
0.000041,
0.000063,
0.000179,
0.000128,
0.000169,
0.000219,
0.000110,
0.000294,
0.000199,
0.000403,
0.000189,
0.000126,
0.000209,
0.000230,
0.000108,
0.000192,
0.000344,
0.000156,
0.000112,
0.000101,
0.000207,
0.000125,
0.000233,
0.000114,
0.000258,
0.000174,
0.000207,
0.000112,
0.000242,
0.000272,
0.000151,
0.000107,
0.000134,
0.000147,
0.000346,
0.000040,
0.000102,
0.000191,
0.000082,
0.000267,
0.000172,
0.000063,
0.000180,
0.000115,
0.000233,
0.000098,
0.000264,
0.000071,
0.000120,
0.000140,
0.000160,
0.000288,
0.000028,
0.000080,
0.000084,
0.000327,
0.000091,
0.000100,
0.000209,
0.000087,
0.000150,
0.000064,
0.000110,
0.000096,
0.000198,
0.000246,
0.000290,
0.000130,
0.000143,
0.000130,
0.000120,
0.000283,
0.000092,
0.000186,
0.000159,
0.000181,
0.000114,
0.000058,
0.000165,
0.000153,
0.000260,
0.000079,
0.000302,
0.000222,
0.000173,
0.000091,
0.000081,
0.000133,
0.000163,
0.000115,
0.000156,
0.000188,
0.000049,
0.000109,
0.000159,
0.000088,
0.000163,
0.000103,
0.000203,
0.000199,
0.000098,
0.000258,
0.000138,
0.000080,
0.000079,
0.000199,
0.000084,
0.000308,
0.000166,
0.000169,
0.000065,
0.000102,
0.000189,
0.000249,
0.000067,
0.000069,
0.000241,
0.000155,
0.000109,
0.000095,
0.000172,
0.000131,
0.000081,
0.000221,
0.000046,
0.000338,
0.000135,
0.000207,
0.000094,
0.000026,
0.000055,
0.000297,
0.000107,
0.000113,
0.000105,
0.000069,
0.000150,
0.000179,
0.000161,
0.000041,
0.000205,
0.000193,
0.000265,
0.000274,
0.000057,
0.000157,
0.000120,
0.000186,
0.000141,
0.000261,
0.000086,
0.000289,
0.000050,
0.000069,
0.000103,
0.000087,
0.000087,
0.000050,
0.000066,
0.000188,
0.000152,
0.000162,
0.000308,
0.000102,
0.000146,
0.000096,
0.000158,
0.000085,
0.000110,
0.000046,
0.000342,
0.000231,
0.000333,
0.000071,
0.000199,
0.000209,
0.000126,
0.000264,
0.000124,
0.000190,
0.000139,
0.000209,
0.000018,
0.000317,
0.000111,
0.000054,
0.000078,
0.000089,
0.000132,
0.000053,
0.000218,
0.000126,
0.000243,
0.000105,
0.000128,
0.000120,
0.000070,
0.000054,
0.000077,
0.000140,
0.000170,
0.000091,
0.000212,
0.000179,
0.000159,
0.000112,
0.000098,
0.000242,
0.000292,
0.000365,
0.000311,
0.000046,
0.000080,
0.000084,
0.000197,
0.000068,
0.000157,
0.000181,
0.000150,
0.000084,
0.000095,
0.000114,
0.000118,
0.000149,
0.000299,
0.000061,
0.000122,
0.000091,
0.000083,
0.000277,
0.000335,
0.000104,
0.000153,
0.000088,
0.000094,
0.000128,
0.000088,
0.000208,
0.000364,
0.000202,
0.000116,
0.000168,
0.000117,
0.000110,
0.000149,
0.000128,
0.000160,
0.000126,
0.000089,
0.000322,
0.000112,
0.000253,
0.000218,
0.000100,
0.000244,
0.000260,
0.000055,
0.000059,
0.000198,
0.000236,
0.000606,
0.000110,
0.000184,
0.000123,
0.000149,
0.000169,
0.000147,
0.000131,
0.000146,
0.000078,
0.000317,
0.000326,
0.000411,
0.000113,
0.000093,
0.000054,
0.000219,
0.000119,
0.000203,
0.000210,
0.000099,
0.000101,
0.000047,
0.000059,
0.000102,
0.000128,
0.000176,
0.000043,
0.000072,
0.000189,
0.000180,
0.000448,
0.000198,
0.000117,
0.000060,
0.000153,
0.000137,
0.000069,
0.000362,
0.000150,
0.000144,
0.000163,
0.000116,
0.000171,
0.000128,
0.000124,
0.000295,
0.000078,
0.000265,
0.000072,
0.000096,
0.000156,
0.000205,
0.000154,
0.000072,
0.000069,
0.000279,
0.000141,
0.000117,
0.000078,
0.000178,
0.000106,
0.000118,
0.000204,
0.000286,
0.000362,
0.000089,
0.000102,
0.000223,
0.000187,
0.000269,
0.000413,
0.000165,
0.000059,
0.000104,
0.000264,
0.000212,
0.000096,
0.000148,
0.000066,
0.000120,
0.000097,
0.000161,
0.000140,
0.000266,
0.000106,
0.000300,
0.000202,
0.000033,
0.000050,
0.000136,
0.000161,
0.000142,
0.000299,
0.000088,
0.000233,
0.000149,
0.000104,
0.000190,
0.000320,
0.000101,
0.000199,
0.000110,
0.000070,
0.000264,
0.000069
};
const int output_ref = 341;

TEST(top_k_layer_tests, md_sync) {
    static const int32_t x_size = 1, y_size = 1, feature_num = 1001, batch_num = 1;
    const int top_k = 1;
    layout inp_l = { data_types::f32, format::yxfb, { batch_num, feature_num, x_size, y_size } };
    layout mutableLayout = { data_types::i32, format::bfyx, { 1, 1, 1, 1 } };

    auto& engine = get_test_engine();
    auto input1 = engine.allocate_memory(inp_l);
    set_values(input1, input_vec1);

    auto shared_memory = engine.allocate_memory(mutableLayout);
    const std::vector<int> topk_vec = {1};
    auto top_k_input = engine.allocate_memory(mutableLayout);
    set_values(top_k_input, topk_vec);

    topology topology;
    topology.add(input_layout("input1", input1->get_layout()));
    topology.add(mutable_data("arg_max_md_write", shared_memory));
    topology.add(data("const", top_k_input));
    topology.add(arg_max_min("arg_max.0", { "input1", "const", "arg_max_md_write" },
        arg_max_min::max, top_k, arg_max_min::feature, arg_max_min::sort_by_indices, true));
    topology.add(mutable_data("arg_max.1", { "arg_max.0" }, shared_memory));

    network network(engine, topology);
    network.set_input_data("input1", input1);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(2));
    auto output = outputs.at("arg_max.1").get_memory();
    mem_lock<int> output_ptr(output, get_test_stream());

    EXPECT_EQ(output_ptr[0], output_ref);
}
