/*
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include <api/input_layout.hpp>
#include "api/reduce.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/data.hpp>
#include "test_utils/float16.h"

using namespace cldnn;
using namespace tests;

TEST(reduce_gpu, common_bfyx) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, 1, 1}});

    set_values(input, {1.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum, {cldnn::reduce::along_b}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfyx_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 3, 4, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum, {cldnn::reduce::along_x, cldnn::reduce::along_y}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {6.0f, 22.0f, 38.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, regr_bfyx_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, {1, 3, 2, 2} });

    set_values(input, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum, { cldnn::reduce::along_b, cldnn::reduce::along_x }, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = { 1.0f, 5.0f, 9.0f, 13.0f, 17.0f, 21.0f };

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfzyx) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfzyx, {1, 1, 1, 1, 1}});

    set_values(input, {1.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum, {cldnn::reduce::along_b}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfzyx_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfzyx, {1, 1, 1, 1, 1}});

    set_values(input, {1.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum, {cldnn::reduce::along_b}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, tensor(format::bfwzyx, {1, 3, 4, 1, 1, 1})});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum, {cldnn::reduce::along_w, cldnn::reduce::along_z, cldnn::reduce::along_y, cldnn::reduce::along_x}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {6.0f, 22.0f, 38.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, tensor(format::bfwzyx, {1, 3, 4, 1, 1, 1})});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum, {cldnn::reduce::along_f, cldnn::reduce::along_w, cldnn::reduce::along_z}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {66.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_max_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 4, 1, 1, 1}});

    set_values(input, {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                       12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::max, {cldnn::reduce::along_b, cldnn::reduce::along_f}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {20.0f, 21.0f, 22.0f, 23.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_min) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::min, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {0.0f, 3.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_min_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::min, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {0.0f, 3.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_mean) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::mean, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f, 4.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_mean_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::mean, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f, 4.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_prod) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::prod, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {0.0f, 60.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_prod_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::prod, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {0.0f, 60.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_sum_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 4, 1, 1, 1}});

    set_values(input, {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                       12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum, {cldnn::reduce::along_b, cldnn::reduce::along_f}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {60.0f, 66.0f, 72.0f, 78.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_logical_and) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::logical_and, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<char> ref_data = {0, 1};

    auto output_ptr = output.pointer<char>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_logical_and_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::logical_and, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<char> ref_data = {0, 1};

    auto output_ptr = output.pointer<char>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_logical_or) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::logical_or, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<char> ref_data = {1, 1};

    auto output_ptr = output.pointer<char>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_logical_or_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::logical_or, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<char> ref_data = {1, 1};

    auto output_ptr = output.pointer<char>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_sum_square) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum_square, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {5.0f, 50.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_sum_square_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::sum_square, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {5.0f, 50.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_l1) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, -2.0f, 3.0f, 4.0f, -5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::l1, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {3.0f, 12.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_l1_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, -2.0f, 3.0f, 4.0f, -5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::l1, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {3.0f, 12.0f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_l2) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, -2.0f, 3.0f, 4.0f, -5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::l2, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {2.236067977f, 7.071067812f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_l2_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, -2.0f, 3.0f, 4.0f, -5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::l2, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {2.236067977f, 7.071067812f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_log_sum) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::log_sum, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0986122887f, 2.4849066498f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_log_sum_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::log_sum, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0986122887f, 2.4849066498f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_log_sum_exp) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::log_sum_exp, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 0));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {2.407605964f, 5.407605964f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_log_sum_exp_keepdims) {
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, {data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reduce("reduce", "input", reduce_mode::log_sum_exp, {cldnn::reduce::along_f, cldnn::reduce::along_w}, 1));

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {2.407605964f, 5.407605964f};

    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < ref_data.size(); ++i) {
        EXPECT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}
