// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "test_utils.h"
#include "random_generator.hpp"
#include "fusions/fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include "crop_inst.h"

using namespace cldnn;
using namespace ::tests;

TEST(crop_gpu, basic_in2x3x2x2_crop_all) {
    //  Reference  : 1x2x2x2
    //  Input      : 2x3x4x5
    //  Output     : 1x2x2x3

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 5;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 2;

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<float> input_vec = rg.generate_random_1d<float>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x2x2x3_crop_all) {
    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<float> input_vec;
    for (int i = 0; i < batch_num * feature_num * y_size * x_size; i++)
        input_vec.push_back(static_cast<float>(i));
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i32_in2x3x2x2_crop_all) {
    //  Reference  : 1x2x2x2
    //  Input      : 2x3x4x5
    //  Output     : 1x2x2x3

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 5;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 2;

    auto input = engine.allocate_memory({ data_types::i32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int32_t> input_vec = rg.generate_random_1d<int32_t>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i64_in2x3x2x2_crop_all) {
    //  Reference  : 1x2x2x2
    //  Input      : 2x3x4x5
    //  Output     : 1x2x2x3

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 5;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 2;

    auto input = engine.allocate_memory({ data_types::i64, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int64_t> input_vec = rg.generate_random_1d<int64_t>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all_bfyx) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::f32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, {0, 0, 0, 0} ));

    std::vector<float> input_vec = rg.generate_random_1d<float>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    std::vector<float> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i32_in2x3x2x2_crop_all_bfyx) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::i32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int32_t> input_vec = rg.generate_random_1d<int32_t>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    std::vector<int32_t> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i64_in2x3x2x2_crop_all_bfyx) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::i64,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int64_t> input_vec = rg.generate_random_1d<int64_t>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());
    std::vector<int64_t> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all_fyxb) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::f32,format::fyxb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, {0, 0, 0, 0} ));

    std::vector<float> input_vec = rg.generate_random_1d<float>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (x + x_size * (y + y_size * f));
                    int output_linear_id = b + crop_batch_num * (x + crop_x_size * (y + crop_y_size * f));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i32_in2x3x2x2_crop_all_fyxb) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::i32,format::fyxb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int32_t> input_vec = rg.generate_random_1d<int32_t>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (x + x_size * (y + y_size * f));
                    int output_linear_id = b + crop_batch_num * (x + crop_x_size * (y + crop_y_size * f));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i64_in2x3x2x2_crop_all_fyxb) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::i64,format::fyxb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int64_t> input_vec = rg.generate_random_1d<int64_t>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (x + x_size * (y + y_size * f));
                    int output_linear_id = b + crop_batch_num * (x + crop_x_size * (y + crop_y_size * f));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_offsets) {
    //  Reference  : 1x2x2x1
    //  Offsets    : 1x0x1x1
    //  Input      : 2x2x3x2
    //  Output     : 1x2x2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13
    //  f1: b0:  7    8  -16   b1:   12   8    -17

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) + batch_num * ((f + feature_offset) + feature_num * ((x + x_offset) + x_size * (y + y_offset)));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i32_in2x3x2x2_crop_offsets) {
    //  Reference  : 1x2x2x1
    //  Offsets    : 1x0x1x1
    //  Input      : 2x2x3x2
    //  Output     : 1x2x2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   50   -5   -15
    //  f1: b0:  5    6  -12   b1:   15   52   -13
    //  f1: b0:  7    8  -16   b1:   12   8    -17

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input = engine.allocate_memory({ data_types::i32, format::yxfb,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<int32_t> input_vec = { 1, 0, 5, 15,
        2, 0, 6, 52,
        -10, -11, -12, -13,
        3, 50, 7, 12,
        4, -5, 8, 8,
        -14, -15, -16, -17 };
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) + batch_num * ((f + feature_offset) + feature_num * ((x + x_offset) + x_size * (y + y_offset)));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i64_in2x3x2x2_crop_offsets) {
    //  Reference  : 1x2x2x1
    //  Offsets    : 1x0x1x1
    //  Input      : 2x2x3x2
    //  Output     : 1x2x2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   50   -5   -15
    //  f1: b0:  5    6  -12   b1:   15   52   -13
    //  f1: b0:  7    8  -16   b1:   12   8    -17

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input = engine.allocate_memory({ data_types::i64, format::yxfb,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<int64_t> input_vec = { 1, 0, 5, 15,
        2, 0, 6, 52,
        -10, -11, -12, -13,
        3, 50, 7, 12,
        4, -5, 8, 8,
        -14, -15, -16, -17 };
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) + batch_num * ((f + feature_offset) + feature_num * ((x + x_offset) + x_size * (y + y_offset)));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in1x4x1x1_split) {
    // Tests split with crop implementation
    //                 _CROP_1(1x3x1x1,offset(0x0x0x0))
    //                |
    //  INPUT(1x4x1x1)
    //                |_
    //                  CROP_2(1x1x1x1,offset(0x3x0x0))
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1.0
    //  f1:  2.0
    //  f2: -3.0
    //  f3:  4.0

    //  Out1:
    //  f0: -1.0
    //  f1:  2.0
    //  f2: -3.0

    //  Out2:
    //  f0: 4.0
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop1", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    std::vector<float> out1 = { -1.f, 2.f,-3.f };
    std::vector<float> out2 = { 4.f, };
    set_values(input, input_vec);
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(topology.get_primitives_ids()));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size();i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    std::cout << std::endl;
    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size();i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_in1x4x1x1_crop_pad) {
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    padding in_pad({0, 0, 1, 1}, {0, 0, 1, 1});
    auto padded_layout = input->get_layout().with_padding(in_pad);
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("input_reorder", input_info("input"), padded_layout));
    topology.add(crop("crop1", input_info("input_reorder"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(reorder("out_reorder", input_info("crop1"), format::bfyx, data_types::f32));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    std::vector<float> out1 = { -1.f, 2.f,-3.f };
    set_values(input, input_vec);
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("out_reorder").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size();i++)
        ASSERT_EQ(output_ptr[i], out1[i]);
}

TEST(crop_gpu, basic_i32_in1x4x1x1_split) {
    // Tests split with crop implementation
    //                 _CROP_1(1x3x1x1,offset(0x0x0x0))
    //                |
    //  INPUT(1x4x1x1)
    //                |_
    //                  CROP_2(1x1x1x1,offset(0x3x0x0))
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1
    //  f1:  2
    //  f2: -3
    //  f3:  4

    //  Out1:
    //  f0: -1
    //  f1:  2
    //  f2: -3

    //  Out2:
    //  f0: 4
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop1", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));

    std::vector<int32_t> input_vec = { -1, 2, -3, 4 };
    std::vector<int32_t> out1 = { -1, 2,-3 };
    std::vector<int32_t> out2 = { 4, };
    set_values(input, input_vec);
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(topology.get_primitives_ids()));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<int32_t> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_i64_in1x4x1x1_split) {
    // Tests split with crop implementation
    //                 _CROP_1(1x3x1x1,offset(0x0x0x0))
    //                |
    //  INPUT(1x4x1x1)
    //                |_
    //                  CROP_2(1x1x1x1,offset(0x3x0x0))
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1
    //  f1:  2
    //  f2: -3
    //  f3:  4

    //  Out1:
    //  f0: -1
    //  f1:  2
    //  f2: -3

    //  Out2:
    //  f0: 4
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop1", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));

    std::vector<int64_t> input_vec = { -1, 2, -3, 4 };
    std::vector<int64_t> out1 = { -1, 2,-3 };
    std::vector<int64_t> out2 = { 4, };
    set_values(input, input_vec);
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(topology.get_primitives_ids()));

    network network(engine, topology, config);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<int64_t> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_in1x4x1x1_split_w_relu) {
    // Tests split with crop implementation
    //                        _ CROP_1(1x3x1x1,offset(0x0x0x0)) --> RELU
    //                       |
    //  INPUT(1x4x1x1)--RELU
    //                       |_
    //                          CROP_2(1x1x1x1,offset(0x3x0x0)) --> RELU
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1.0
    //  f1:  2.0
    //  f2: -3.0
    //  f3:  4.0

    //  Out1:
    //  f0: 0.0
    //  f1: 2.0
    //  f2: 0.0

    //  Out2:
    //  f0: 4.0
    // disable memory pool when we want to check optimized out internal results
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl);
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;
    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = engine->allocate_memory({ data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(activation("relu", input_info("input"), activation_func::relu));
    topology.add(crop("crop1", input_info("relu"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", input_info("relu"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));
    topology.add(activation("relu1", input_info("crop1"), activation_func::relu));
    topology.add(activation("relu2", input_info("crop2"), activation_func::relu));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    std::vector<float> out1 = { 0.f, 2.f,0.f };
    std::vector<float> out2 = { 4.f, };
    set_values(input, input_vec);

    ExecutionConfig cfg = get_test_default_config(*engine);
    cfg.set_property(ov::intel_gpu::enable_memory_pool(false));
    cfg.set_property(ov::intel_gpu::optimize_data(true));

    network network(*engine, topology, cfg);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("relu1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // check if crop has been executed in place
    auto in_place = engine->is_the_same_buffer(*network.get_output_memory("crop1"), *network.get_output_memory("relu"));
    ASSERT_TRUE(in_place);

    for (size_t i = 0; i < out1.size();i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("relu2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size();i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_in3x1x2x2x1_crop_all_bfzyx) {
    //  Reference  : 3x1x2x2x1
    //  Input      : 6x2x4x3x2
    //  Output     : 3x1x2x2x1

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;
    auto z_size = 2;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;
    auto crop_z_size = z_size - 1;

    auto input = engine.allocate_memory({ data_types::f32,format::bfzyx,{ batch_num, feature_num, x_size, y_size, z_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size, crop_z_size }, { 0, 0, 0, 0, 0 }));

    std::vector<float> input_vec = rg.generate_random_1d<float>(input->count(), -10, 10);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int z = 0; z < crop_z_size; ++z) { //Z
                for (int y = 0; y < crop_y_size; ++y) { //Y
                    for (int x = 0; x < crop_x_size; ++x) { //X
                        int linear_id = x + x_size * (y + y_size * (z + z_size * (f + feature_num * b)));
                        int output_linear_id = x + crop_x_size * (y + crop_y_size * (z + crop_z_size * (f + crop_feature_num * b)));
                        ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                    }
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in3x1x3x2x2x1_crop_all_bfwzyx) {
    //  Reference  : 3x1x3x2x2x1
    //  Input      : 6x2x6x4x3x2
    //  Output     : 3x1x3x2x2x1

    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;
    auto z_size = 2;
    auto w_size = 6;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;
    auto crop_z_size = z_size - 1;
    auto crop_w_size = w_size - 3;

    tensor in_size = tensor(format::bfwzyx, { batch_num, feature_num, w_size, z_size, y_size, x_size });
    tensor crop_size = tensor(format::bfwzyx, { crop_batch_num, crop_feature_num, crop_w_size, crop_z_size, crop_y_size, crop_x_size });
    auto input = engine.allocate_memory({ data_types::f32,format::bfwzyx, in_size });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), crop_size, tensor{ 0 }));

    VVVVVVF<float> input_rnd = rg.generate_random_6d<float>(batch_num, feature_num, w_size, z_size, y_size, x_size, -10, 10);
    VF<float> input_vec = flatten_6d<float>(format::bfwzyx, input_rnd);
    set_values(input, input_vec);

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int w = 0; w < crop_w_size; ++w) { //W
                for (int z = 0; z < crop_z_size; ++z) { //Z
                    for (int y = 0; y < crop_y_size; ++y) { //Y
                        for (int x = 0; x < crop_x_size; ++x) { //X
                            int linear_id = x + x_size * (y + y_size * (z + z_size * (w + w_size * (f + feature_num * b))));
                            int output_linear_id = x + crop_x_size * (y + crop_y_size * (z + crop_z_size * (w + crop_w_size * (f + crop_feature_num * b))));
                            ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                        }
                    }
                }
            }
        }
    }
}

// batch size, input feature, crop out feature, (in_out format, crop format)
using crop_test_params = std::tuple<size_t, size_t, size_t, std::pair<cldnn::format,cldnn::format>, impl_types, bool>;

class crop_gpu : public ::testing::TestWithParam<crop_test_params> {};

TEST_P(crop_gpu, pad_test) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto batch_num = std::get<0>(p);
    auto feature_num = std::get<1>(p);
    auto x_size = 1;
    auto y_size = 1;
    auto z_size = 1;

    auto crop_batch_num = batch_num;
    auto crop_feature_num_1 = std::get<2>(p);
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto crop_z_size = 1;
    auto feature_offset_1 = feature_num - crop_feature_num_1;

    auto in_out_format = std::get<3>(p).first;
    auto crop_format = std::get<3>(p).second;
    auto impl_type = std::get<4>(p);
    bool is_caching_test = std::get<5>(p);

    auto input = engine.allocate_memory({ data_types::f32, in_out_format, { tensor(spatial(x_size, y_size, z_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", input_info("input"), crop_format, data_types::f32));
    topology.add(crop("crop1", input_info("reorder"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size, crop_z_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0,0), batch(0)) }));
    topology.add(reorder("out", input_info("crop1"), in_out_format, data_types::f32));

    std::vector<float> input_vec;
    std::vector<float> res;
    std::vector<float> input_data;
    std::vector<float> res_data;
    for (size_t i = 0; i < feature_num; i++) {
        input_data.push_back(static_cast<float>(i));
    }
    for (size_t i = 0; i < crop_feature_num_1; i++) {
        res_data.push_back(input_data[feature_offset_1 + i]);
    }
    for (size_t i = 0; i < batch_num; i++) {
        input_vec.insert(input_vec.end(), input_data.begin(), input_data.end());
        res.insert(res.end(), res_data.begin(), res_data.end());
    }
    set_values(input, input_vec);
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    if (impl_type != impl_types::any) {
        auto forcing_map = ov::intel_gpu::ImplForcingMap{{"crop1", {crop_format, "", impl_type}}};
        config.set_property(ov::intel_gpu::force_implementations(forcing_map));
    }

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);
    auto outputs = network->execute();

    auto output = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < res.size(); i++)
        ASSERT_EQ(output_ptr[i], res[i]);
}

static std::vector<std::pair<cldnn::format,cldnn::format>> formats = {
    std::make_pair<cldnn::format, cldnn::format>(format::bfyx, format::b_fs_yx_fsv16),
    std::make_pair<cldnn::format, cldnn::format>(format::bfzyx, format::b_fs_zyx_fsv16),
    std::make_pair<cldnn::format, cldnn::format>(format::bfyx, format::bs_fs_yx_bsv16_fsv16),
    std::make_pair<cldnn::format, cldnn::format>(format::bfzyx, format::bs_fs_zyx_bsv16_fsv16),
    };
static std::vector<size_t> batches = {1, 8, 16, 17};
static std::vector<size_t> in_features = {18, 24, 32};
static std::vector<size_t> crop_features = {4, 8, 12, 17};

INSTANTIATE_TEST_SUITE_P(crop_test, crop_gpu,
                        ::testing::Combine(
                                ::testing::ValuesIn(batches),
                                ::testing::ValuesIn(in_features),
                                ::testing::ValuesIn(crop_features),
                                ::testing::ValuesIn(formats),
                                ::testing::Values(impl_types::any),
                                ::testing::Values(false)
                                ));

INSTANTIATE_TEST_SUITE_P(export_import_crop_test, crop_gpu,
                        ::testing::Combine(
                                ::testing::Values(batches[0]),
                                ::testing::Values(in_features[0]),
                                ::testing::Values(crop_features[0]),
                                ::testing::Values(formats[0]),
                                ::testing::Values(impl_types::any),
                                ::testing::Values(true)
                                ));

class crop_gpu_dynamic : public ::testing::TestWithParam<std::tuple<impl_types>> {};
TEST_P(crop_gpu_dynamic, i32_in2x3x2x2_crop_offsets) {
    auto test_params = GetParam();
    impl_types impl_type = std::get<0>(test_params);

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input_dyn_layout    = layout{ ov::PartialShape{ov::Dimension(1, 10), feature_num, y_size, x_size}, data_types::f32, format::bfyx };
    auto input_actual_layout = layout{ ov::PartialShape{batch_num, feature_num, y_size, x_size}, data_types::f32, format::bfyx };

    auto input = engine.allocate_memory(input_actual_layout);

    topology topology;
    topology.add(input_layout("input", input_dyn_layout));
    topology.add(crop("crop", input_info("input"), tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<float> input_vec = {1.f, 0.f,  5.f, 15.f, 2.f, 0.f,  6.f, 52.f, -10.f, -11.f, -12.f, -13.f,
                                    3.f, 50.f, 7.f, 12.f, 4.f, -5.f, 8.f, 8.f,  -14.f, -15.f, -16.f, -17.f};
    set_values(input, input_vec);
    ExecutionConfig config1 = get_test_default_config(engine);
    config1.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ExecutionConfig config2 = config1;

    if (impl_type != impl_types::any) {
        auto forcing_map = ov::intel_gpu::ImplForcingMap{{"crop", {format::bfyx, "", impl_type}}};
        config1.set_property(ov::intel_gpu::force_implementations(forcing_map));
    }

    network network1(engine, topology, config1); // run with shape agnostic kernel
    network1.set_input_data("input", input);
    auto outputs1 = network1.execute();
    auto output1 = outputs1.at("crop").get_memory();
    cldnn::mem_lock<float> output1_ptr(output1, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) * (feature_num * y_size * x_size) + (f + feature_offset) * (y_size * x_size) + (y + y_offset) * x_size + (x + x_offset);
                    int output_linear_id = b * (crop_feature_num * crop_y_size * crop_x_size) + f * (crop_y_size * crop_x_size) + y * crop_x_size + x;
                    ASSERT_EQ(output1_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
    network network2(engine, topology, config2); // run with static kernel
    network2.set_input_data("input", input);
    auto outputs2 = network2.execute();
    auto output2 = outputs2.at("crop").get_memory();
    cldnn::mem_lock<float> output2_ptr(output2, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) * (feature_num * y_size * x_size) + (f + feature_offset) * (y_size * x_size) + (y + y_offset) * x_size + (x + x_offset);
                    int output_linear_id = b * (crop_feature_num * crop_y_size * crop_x_size) + f * (crop_y_size * crop_x_size) + y * crop_x_size + x;
                    ASSERT_EQ(output2_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

static std::vector<impl_types> impls = { impl_types::any, impl_types::cpu };
INSTANTIATE_TEST_SUITE_P(crop_test, crop_gpu_dynamic,
                        ::testing::Combine(
                                ::testing::ValuesIn(impls)
                                ));

TEST(crop_cpu, basic_in2x3x2x2_crop_all_bfyx_disable_usm) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    tests::random_generator rg(GET_SUITE_NAME);
    auto engine = create_test_engine(engine_types::ocl, runtime_types::ocl, false);

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine->allocate_memory({ data_types::f32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", input_info("input"), { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, {0, 0, 0, 0} ));

    std::vector<float> input_vec = rg.generate_random_1d<float>(input->count(), -10, 10);
    set_values(input, input_vec);

    ExecutionConfig config = get_test_default_config(*engine);
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"crop", {format::bfyx, "", impl_types::cpu}} }));

    network network(*engine, topology, config);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    std::vector<float> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    ASSERT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, dynamic_in1x4x1x1_split) {
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 2;
    auto crop_feature_num_2 = 2;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 2;
    auto axis = 1;

    auto input_dyn_layout    = layout{ ov::PartialShape{ov::Dimension(1, 10), feature_num, y_size, x_size}, data_types::f32, format::bfyx };
    auto input_actual_layout = layout{ ov::PartialShape{batch_num, feature_num, y_size, x_size}, data_types::f32, format::bfyx };

    auto input_mem = engine.allocate_memory(input_actual_layout);
    auto data_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    set_values<int64_t>(data_mem, {axis});

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::split;
    size_t num_splits = 2;
    topology topology;
    topology.add(input_layout("input", input_dyn_layout));
    topology.add(data("data", data_mem));
    topology.add(crop("crop1", { input_info("input"), input_info("data") }, tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }, op_mode, 0, axis, num_splits));
    topology.add(crop("crop2", { input_info("input"), input_info("data") }, tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }, op_mode, 1, axis, num_splits));

    std::vector<float> input_vec = { -1.0f, 2.0f, -3.0f, 4.0f };
    std::vector<float> out1 = { -1.0f, 2.0f };
    std::vector<float> out2 = { -3.0f, 4.0f };
    set_values(input_mem, input_vec);
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(topology.get_primitives_ids()));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    auto impl1 = network.get_primitive("crop1")->get_impl();
    ASSERT_TRUE(impl1 != nullptr);
    ASSERT_TRUE(impl1->is_dynamic());
    auto impl2 = network.get_primitive("crop2")->get_impl();
    ASSERT_TRUE(impl2 != nullptr);
    ASSERT_TRUE(impl2->is_dynamic());

    auto output1 = outputs.at("crop1").get_memory();
    cldnn::mem_lock<float> output_ptr_1(output1, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr_1[i], out1[i]);

    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, dynamic_in1x4x1x1_varaidic_split) {
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto axis = 1;

    auto input_dyn_layout    = layout{ ov::PartialShape{ov::Dimension(1, 10), feature_num, y_size, x_size}, data_types::f32, format::bfyx };
    auto input_actual_layout = layout{ ov::PartialShape{batch_num, feature_num, y_size, x_size}, data_types::f32, format::bfyx };

    auto input_mem = engine.allocate_memory(input_actual_layout);
    auto axis_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    auto splits_length_mem = engine.allocate_memory({ {2}, data_types::i64, format::bfyx });

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology;
    topology.add(input_layout("input", input_dyn_layout));
    topology.add(data("axis", axis_mem));
    topology.add(data("splits_length", splits_length_mem));
    topology.add(crop("crop1", { input_info("input"), input_info("axis"), input_info("splits_length") }, tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }, op_mode, 0, axis));
    topology.add(crop("crop2", { input_info("input"), input_info("axis"), input_info("splits_length") }, tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }, op_mode, 1, axis));

    std::vector<float> input_vec = { -1.0f, 2.0f, -3.0f, 4.0f };
    std::vector<float> out1 = { -1.0f, 2.0f, -3.0f };
    std::vector<float> out2 = { 4.0f };
    std::vector<int64_t> splits_vec = {3, 1};

    set_values(input_mem, input_vec);
    set_values<int64_t>(axis_mem, {axis});
    set_values(splits_length_mem, splits_vec);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(topology.get_primitives_ids()));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    auto impl1 = network.get_primitive("crop1")->get_impl();
    ASSERT_TRUE(impl1 != nullptr);
    ASSERT_TRUE(impl1->is_dynamic());
    auto impl2 = network.get_primitive("crop2")->get_impl();
    ASSERT_TRUE(impl2 != nullptr);
    ASSERT_TRUE(impl2->is_dynamic());

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, dynamic_input_padding_varaidic_split) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto y_size = 128;
    auto x_size = 4;

    auto split_axis = 2;
    auto data_y_pad_axis = 2;
    auto data_x_pad_axis = 3;
    auto input_y_pad_before = 64;
    auto input_y_pad_after = 32;
    auto input_x_pad_before = 8;
    auto input_x_pad_after = 2;

    auto input_dyn_layout = layout{ ov::PartialShape{-1, feature_num, y_size, x_size}, data_types::f32, format::bfyx };
    input_dyn_layout.data_padding._dynamic_dims_mask[data_y_pad_axis] = 1;
    input_dyn_layout.data_padding._lower_size[data_x_pad_axis] = input_x_pad_before;
    input_dyn_layout.data_padding._upper_size[data_x_pad_axis] = input_x_pad_after;

    auto input_actual_layout = layout{ ov::PartialShape{batch_num, feature_num, y_size, x_size}, data_types::f32, format::bfyx };
    input_actual_layout.data_padding._lower_size[data_y_pad_axis] = input_y_pad_before;
    input_actual_layout.data_padding._upper_size[data_y_pad_axis] = input_y_pad_after;
    input_actual_layout.data_padding._lower_size[data_x_pad_axis] = input_x_pad_before;
    input_actual_layout.data_padding._upper_size[data_x_pad_axis] = input_x_pad_after;

    auto input_mem = engine.allocate_memory(input_actual_layout);
    auto axis_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    auto splits_length_mem = engine.allocate_memory({ {2}, data_types::i64, format::bfyx });

    auto elements_count = input_mem->size() / sizeof(float);
    auto input_data = rg.generate_random_1d<float>(elements_count, -10, 10);

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology;
    topology.add(input_layout("input", input_dyn_layout));
    topology.add(data("split_axis", axis_mem));
    topology.add(data("splits_length", splits_length_mem));
    topology.add(crop("variadic_split.out0", { input_info("input"), input_info("split_axis"), input_info("splits_length") }, tensor(1), tensor(0), op_mode, 0, split_axis));
    topology.add(crop("variadic_split.out1", { input_info("input"), input_info("split_axis"), input_info("splits_length") }, tensor(1), tensor(0), op_mode, 1, split_axis));

    std::vector<int64_t> splits_vec = { 64, 64 };

    set_values(input_mem, input_data);
    set_values(splits_length_mem, splits_vec);
    set_values<int64_t>(axis_mem, {split_axis});

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(topology.get_primitives_ids()));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto check_output = [&](size_t output_idx, cldnn::network_output output) {
        auto y_start = std::accumulate(splits_vec.begin(), splits_vec.begin() + output_idx, 0);
        auto y_size_output = splits_vec[output_idx];

        auto output_layout = output.get_layout();
        auto output_mem = output.get_memory();
        cldnn::mem_lock<float> output_ptr(output_mem, get_test_stream());
        for (size_t b = 0; b < static_cast<size_t>(batch_num); b++) {
            for (size_t f = 0; f < static_cast<size_t>(feature_num); f++) {
                for (size_t y = 0; y < static_cast<size_t>(y_size_output); y++) {
                    for (size_t x = 0; x < static_cast<size_t>(x_size); x++) {
                        auto input_offset = input_actual_layout.get_linear_offset(cldnn::tensor(static_cast<int32_t>(b),
                                                                                                static_cast<int32_t>(f),
                                                                                                static_cast<int32_t>(x),
                                                                                                static_cast<int32_t>(y + y_start), 0, 0));
                        auto output_offset = output_layout.get_linear_offset(cldnn::tensor(static_cast<int32_t>(b),
                                                                                           static_cast<int32_t>(f),
                                                                                           static_cast<int32_t>(x),
                                                                                           static_cast<int32_t>(y), 0, 0));

                        ASSERT_EQ(input_data[input_offset], output_ptr[output_offset]);
                    }
                }
            }
        }
    };

    auto outputs = network.execute();

    check_output(0, outputs.at("variadic_split.out0"));
    check_output(1, outputs.at("variadic_split.out1"));
}

TEST(crop_gpu, static_split_batch) {
    auto& engine = get_test_engine();

    auto input_actual_layout = layout{ ov::PartialShape{3, 4, 1, 1}, data_types::f32, format::bfyx };
    auto input_mem = engine.allocate_memory(input_actual_layout);

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::none;
    topology topology;
    topology.add(input_layout("input", input_actual_layout));
    topology.add(crop("crop1", { input_info("input") }, tensor(1, 4, 1, 1), { tensor(0, 0, 0, 0) }, op_mode, 0));
    topology.add(crop("crop2", { input_info("input") }, tensor(1, 4, 1, 1), { tensor(1, 0, 0, 0) }, op_mode, 1));
    topology.add(crop("crop3", { input_info("input") }, tensor(1, 4, 1, 1), { tensor(2, 0, 0, 0) }, op_mode, 2));

    std::vector<int32_t> input_vec(12);
    for (int32_t i = 0; i < 12; i++) {
        input_vec[i] = i;
    }

    std::vector<int32_t> out1 = { 0, 1, 2, 3 };
    std::vector<int32_t> out2 = { 4, 5, 6, 7 };
    std::vector<int32_t> out3 = { 8, 9, 10, 11 };

    set_values(input_mem, input_vec);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(topology.get_primitives_ids()));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<int32_t> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);

    auto output_3 = outputs.at("crop3").get_memory();
    cldnn::mem_lock<int32_t> output_ptr_3(output_3, get_test_stream());

    for (size_t i = 0; i < out3.size(); i++)
        ASSERT_EQ(output_ptr_3[i], out3[i]);
}

TEST(crop_gpu, optimized_out_crop) {
    auto& engine = get_test_engine();

    auto input_actual_layout = layout{ ov::PartialShape{5, 6, 1, 1}, data_types::f32, format::bfyx };
    auto input_mem = engine.allocate_memory(input_actual_layout);
    std::vector<int32_t> input_vec = {
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
    };
    set_values(input_mem, input_vec);

    std::vector<int32_t> out_vec = {
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
    };

    topology topology;
    topology.add(input_layout("input", input_actual_layout));
    topology.add(crop("crop1", { input_info("input") }, tensor(5, 5, 1, 1), { tensor(0, 0, 0, 0) }));
    topology.add(crop("crop2", { input_info("crop1") }, tensor(5, 4, 1, 1), { tensor(0, 0, 0, 0) }));
    topology.add(reorder("reorder_out", input_info("crop2"), layout{ ov::PartialShape{5, 4, 1, 1}, data_types::f32, format::bfyx }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    auto output = outputs.at("reorder_out").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out_vec.size(); i++)
        ASSERT_EQ(output_ptr[i], out_vec[i]);

    ASSERT_TRUE(network.get_primitive("crop1")->can_be_optimized());
    ASSERT_TRUE(network.get_primitive("crop2")->can_be_optimized());
}

TEST(crop_single_axis, simple_Baxis) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 1, 2 } });

    set_values(input1, {
        1.f, 2.f,  3.f,  4.f,
        5.f, 6.f,  7.f,  8.f,
        9.f, 10.f, 11.f, 12.f
    });

    topology topology;
    topology.add(input_layout("Input", input1->get_layout()));
    topology.add(crop("crop", input_info("Input"), tensor{1, 2, 1, 2}, tensor(1, 0, 0, 0)));
    topology.add(reorder("reorder", input_info("crop"), format::bfyx, data_types::i8));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("Input", input1);


    auto outputs = network.execute();
    auto output = outputs.at("reorder").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> expected_results = {
        5, 6, 7, 8
    };

    int crop_batch_num = 1;
    int crop_feature_num = 2;
    int crop_y_size = 2;
    int crop_x_size = 1;
    for (int b = 0; b < crop_batch_num; ++b) {
        for (int f = 0; f < crop_feature_num; ++f) {
            for (int y = 0; y < crop_y_size; ++y) {
                for (int x = 0; x < crop_x_size; ++x) {
                    int linear_id = x + y + 2 * f;
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    ASSERT_EQ(output_ptr[output_linear_id], expected_results[linear_id]);
                }
            }
        }
    }

    auto crop_prim = network.get_primitive("crop");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);
}

TEST(crop_single_axis, simple_Xaxis) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 3, 1 } });

    set_values(input0, {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
        7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f, 17.f, 18.f,
    });

    topology topology;
    topology.add(input_layout("Input", input0->get_layout()));
    topology.add(crop("crop", input_info("Input"), tensor{3, 2, 1, 1}, tensor(0, 0, 1, 0)));
    topology.add(reorder("reorder", input_info("crop"), format::bfyx, data_types::i32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("Input", input0);

    auto outputs = network.execute();
    auto output = outputs.at("reorder").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
        2, 5, 8, 11, 14, 17,
    };

    for (size_t i = 0; i < expected_results.size(); i++) {
        ASSERT_EQ(output_ptr[i], expected_results[i]);
    }

    auto crop_prim = network.get_primitive("crop");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);
}

TEST(crop_single_axis, simple_all_axis) {
    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 3, 3, 3 } });

    std::vector<float> input0_vals;
    for (uint32_t i = 0; i < 81; ++i)
        input0_vals.push_back(i);

    set_values(input0, input0_vals);

    topology topology;
    topology.add(input_layout("Input", input0->get_layout()));
    topology.add(crop("crop", input_info("Input"), tensor{1, 1, 1, 1}, tensor(1, 1, 1, 1)));
    topology.add(reorder("reorder", input_info("crop"), format::bfyx, data_types::i32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("Input", input0);

    auto outputs = network.execute();
    auto output = outputs.at("reorder").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
        40,
    };

    for (size_t i = 0; i < expected_results.size(); i++) {
        ASSERT_EQ(output_ptr[i], expected_results[i]);
    }

    auto crop_prim = network.get_primitive("crop");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);
}

struct crop_input_test_params {
    data_types  input_type;
    tensor      input_size;
    tensor      output_size;

    format::type input_format;
};

// Use BaseFusingTest simplifying testing logic
class CropBaseTest : public ::BaseFusingTest<crop_input_test_params> {
public:
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);

        auto data = rg.generate_random_4d<T>(b, f, y, x, min, max, k);
        mem_lock<T> ptr{mem, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<ov::float16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }
};

class crop_batching_input_test : public CropBaseTest {
public:
    // Comapre Crop result of a given 'params.input_format' with its default formats' result
    void execute(crop_input_test_params& params, bool is_checking) {
        auto& engine = get_test_engine();

        auto dims = format::dimension(params.input_format);
        auto in_layout = layout(params.input_type, format::get_default_format(dims), params.input_size);
        auto in_mem = engine.allocate_memory(in_layout);
        fill_random(in_mem);

        const int before_pad = 1;
        tensor offset(feature(0), spatial(0,0,0,0), batch(before_pad));

        cldnn::topology topo;
        topo.add(input_layout("input", in_layout));
        topo.add(crop("crop1", input_info("input"), params.output_size, offset));
        topo.add(reorder("out_reorder", input_info("crop1"), format::get_default_format(dims), data_types::f32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"out_reorder"}));
        config.set_property(ov::intel_gpu::optimize_data(false));

        cldnn::network net(engine, topo, config);
        net.set_input_data("input", in_mem);
        auto result = net.execute();
        auto output = result.at("out_reorder").get_memory();
        cldnn::mem_lock<float_t> output_ptr(output, get_test_stream());

        // blocked format
        cldnn::topology topo_blocked;
        topo_blocked.add(input_layout("input_blk", in_layout));
        topo_blocked.add(reorder("input_blk_reorder", input_info("input_blk"), params.input_format, params.input_type));
        topo_blocked.add(crop("crop2", input_info("input_blk_reorder"), params.output_size, offset));
        topo_blocked.add(reorder("out_blk_reorder", input_info("crop2"), format::get_default_format(dims), data_types::f32));

        ExecutionConfig config_blk = get_test_default_config(engine);
        config_blk.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"out_blk_reorder"}));
        ov::intel_gpu::ImplementationDesc reorder_ref = { params.input_format, "reorder_data" };
        config_blk.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"input_blk_reorder", reorder_ref} }));

        cldnn::network net_blk(engine, topo_blocked, config_blk);
        net_blk.set_input_data("input_blk", in_mem);
        auto result_blk = net_blk.execute();
        auto output_blk = result_blk.at("out_blk_reorder").get_memory();
        cldnn::mem_lock<float_t> output_blk_ptr(output_blk, get_test_stream());
        if (is_checking) {
            for (size_t i = 0; i < output_ptr.size(); ++i) {
                ASSERT_EQ(output_ptr[i], output_blk_ptr[i]);
            }
        }
    }
};

TEST_P(crop_batching_input_test, blocked_formats) {
    // To test accuracy issue of batching operation of blocked formats
    auto param = GetParam();
    execute(param, true);
}

INSTANTIATE_TEST_SUITE_P(batching_test,
                        crop_batching_input_test,
                        ::testing::ValuesIn(std::vector<crop_input_test_params>{
                            crop_input_test_params{ data_types::f16, {3, 4, 2, 2},     {1, 4, 2, 2},     format::b_fs_yx_fsv4 },
                            crop_input_test_params{ data_types::f16, {3, 16, 2, 2},    {1, 16, 2, 2},    format::b_fs_yx_fsv16 },
                            crop_input_test_params{ data_types::f16, {3, 20, 2, 2},    {1, 20, 2, 2},    format::b_fs_yx_fsv16 },
                            crop_input_test_params{ data_types::i8,  {3, 8, 2, 2},     {1, 8, 2, 2},     format::b_fs_yx_fsv32 },
                            crop_input_test_params{ data_types::f16, {3, 4, 2, 3, 2},  {1, 4, 2, 3, 2},  format::b_fs_zyx_fsv4 },
                            crop_input_test_params{ data_types::f16, {3, 16, 3, 2, 2}, {1, 16, 3, 2, 2}, format::b_fs_zyx_fsv16 },
                            crop_input_test_params{ data_types::u8,  {3, 32, 1, 2, 2}, {1, 32, 1, 2, 2}, format::b_fs_zyx_fsv32 },
                            crop_input_test_params{ data_types::f16, {3, 20, 3, 2, 2}, {1, 16, 3, 2, 2}, format::b_fs_zyx_fsv16 },
                            crop_input_test_params{ data_types::f16, {3, 4, 4, 2, 2},  {1, 4, 4, 2, 2},  format::b_fs_zyx_fsv32 },
                            crop_input_test_params{ data_types::f16, {64, 16, 2, 2},   {32, 16, 2, 2},   format::bs_fs_yx_bsv32_fsv16 },
                            crop_input_test_params{ data_types::f16, {3, 16, 2, 2},    {1, 16, 2, 2},    format::bs_fs_yx_bsv32_fsv16 },
                            crop_input_test_params{ data_types::f16, {3, 32, 2, 2},    {1, 32, 2, 2},    format::bs_fs_yx_bsv16_fsv16 },
                            crop_input_test_params{ data_types::f16, {3, 16, 3, 2, 2}, {1, 16, 3, 2, 2}, format::bs_fs_zyx_bsv32_fsv16 },
                            crop_input_test_params{ data_types::i8,  {3, 32, 1, 2, 2}, {1, 32, 1, 2, 2}, format::bs_fs_zyx_bsv16_fsv32 },
                        }));
