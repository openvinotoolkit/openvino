// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/non_zero.hpp>
#include "openvino/reference/non_zero.hpp"

#include "non_zero_inst.h"
#include "test_utils.h"

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

template<typename T>
void make_min_num_zero(std::vector<T>& vec, std::default_random_engine& generator, int min_num_zero) {
    auto num_zero = std::count_if(vec.begin(), vec.end(), [](T val) {
        return (val == 0);
    });
    min_num_zero = std::max(0, std::min((static_cast<int>(vec.size()) - 1), min_num_zero));

    if (num_zero < min_num_zero) {
        std::uniform_int_distribution<size_t> index_dist(0, (vec.size()-1));
        for (auto idx = num_zero; idx < min_num_zero; idx++) {
            while(true) {
                auto index = index_dist(generator);
                if (vec[index] != 0) {
                    vec[index] = 0;
                    break;
                }
            }
        }
    }
}

template<typename T>
void test_count_non_zero(layout in_layout, std::vector<T> in_data) {
    auto& engine = get_test_engine();
    auto input_mem = engine.allocate_memory(in_layout);
    auto count_non_zero = ov::reference::non_zero_get_count<T>(in_data.data(), in_layout.get_shape());

    set_values(input_mem, in_data);

    topology topology;
    topology.add(input_layout("InputData", in_layout));
    topology.add(count_nonzero("count_nonzero", input_info("InputData"))
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("InputData", input_mem);
    auto outputs = network.execute();
    auto output = outputs.at("count_nonzero").get_memory();

    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    ASSERT_EQ(count_non_zero, output_ptr[0]);
}

TEST(test_count_non_zero, 4d_fp32_1_2_1_5) {
    std::vector<float> in_data = {
        0.3f, 0.2f   , 0.3f, 0.0f, 0.0f,
        0.4f, 0.0001f, 0.1f, 0.9f, 0.10f
    };
    test_count_non_zero<float>(layout{ov::PartialShape{1, 2, 1, 5}, data_types::f32, format::bfyx}, in_data);
}

TEST(test_count_non_zero, 5d_fp16_1_3_2_1_2) {
    std::vector<ov::float16> in_data = {
        0.1f, 0.2f, 0.3f, 0.0f, 12.1f, 11.1f,
        0.0f, 0.0f, 0.1f, 0.9f, 0.10f, 0.001f
    };
    test_count_non_zero<ov::float16>(layout{ov::PartialShape{1, 3, 2, 1, 2}, data_types::f16, format::bfzyx}, in_data);
}

TEST(test_count_non_zero, 2d_int32_1_256) {
    tests::random_generator rg(GET_SUITE_NAME);
    layout in_layout = {ov::PartialShape{1, 256}, data_types::i32, format::bfyx};
    auto in_data = rg.generate_random_1d<int32_t>(in_layout.count(), -2, 2);
    make_min_num_zero(in_data, rg.get_generator(), 100);
    test_count_non_zero<int32_t>(in_layout, in_data);
}

TEST(test_count_non_zero, 2d_f32_1_513) {
    tests::random_generator rg(GET_SUITE_NAME);
    layout in_layout = {ov::PartialShape{1, 513}, data_types::f32, format::bfyx};
    auto in_data = rg.generate_random_1d<float>(in_layout.count(), -2, 2);
    make_min_num_zero(in_data, rg.get_generator(), 40);
    test_count_non_zero<float>(in_layout, in_data);
}

TEST(test_count_non_zero, 6d_f32_21_18_1_5_3_2) {
    tests::random_generator rg(GET_SUITE_NAME);
    layout in_layout = {ov::PartialShape{21, 18, 1, 5, 3, 2}, data_types::f32, format::bfwzyx};
    auto in_data = rg.generate_random_1d<float>(in_layout.count(), -2, 2);
    make_min_num_zero(in_data, rg.get_generator(), 172);
    test_count_non_zero<float>(in_layout, in_data);
}

TEST(test_count_non_zero, 5d_f32_1_16_4_2_24) {
    tests::random_generator rg(GET_SUITE_NAME);
    layout in_layout = {ov::PartialShape{1, 16, 4, 2, 24}, data_types::f32, format::bfzyx};
    auto in_data = rg.generate_random_1d<float>(in_layout.count(), -2, 2);
    make_min_num_zero(in_data, rg.get_generator(), 128);
    test_count_non_zero<float>(in_layout, in_data);
}

class dyn_nonzero_count_net {
public:
    void create_network(cldnn::engine& engine, cldnn::topology& topology, ExecutionConfig& config) {
        _network = std::make_shared<cldnn::network>(engine, topology, config);
    }

    std::map<primitive_id, network_output> execute(memory::ptr input_mem) {
        _network->set_input_data("InputData", input_mem);
        return _network->execute();
    }

private:
    network::ptr _network;
};

TEST(test_count_non_zero, dynamic_2d_f32_bfyx) {
    auto& engine = get_test_engine();
    auto in_dyn_layout = layout(ov::PartialShape::dynamic(2), data_types::f32, format::bfyx);

    topology topology;
    topology.add(input_layout("InputData", in_dyn_layout));
    topology.add(count_nonzero("count_nonzero", input_info("InputData")));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    std::vector<size_t> input_shapes = {171, 531, 168, 169, 174, 172, 168, 167, 1169, 16, 677};
    dyn_nonzero_count_net _test;
    _test.create_network(engine, topology, config);

    tests::random_generator rg(GET_SUITE_NAME);

    for (size_t& input_length : input_shapes) {
        auto in_layout = layout({1, static_cast<long int>(input_length)}, data_types::f32, format::bfyx);
        auto input_mem = engine.allocate_memory(in_layout);

        auto in_data = rg.generate_random_1d<float>(in_layout.count(), -2, 2);
        make_min_num_zero(in_data, rg.get_generator(), 50);

        set_values(input_mem, in_data);
        auto outputs = _test.execute(input_mem);

        ASSERT_EQ(outputs.size(), size_t(1));
        auto output = outputs.at("count_nonzero").get_memory();
        cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

        auto count_non_zero = ov::reference::non_zero_get_count<float>(in_data.data(), in_layout.get_shape());
        ASSERT_EQ(count_non_zero, output_ptr[0]);
    }
}

template<typename T>
void test_gather_non_zero(layout in_layout, std::vector<T> in_data) {
    auto& engine = get_test_engine();
    auto input_mem = engine.allocate_memory(in_layout);
    auto count_non_zero = ov::reference::non_zero_get_count<T>(in_data.data(), in_layout.get_shape());
    auto in_rank = in_layout.get_shape().size();
    std::vector<int32_t> expected_results(count_non_zero * in_rank);
    ov::reference::non_zero<T, int32_t>(in_data.data(), expected_results.data(), in_layout.get_shape());

    auto output_shape_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto output_shape_mem = engine.allocate_memory(output_shape_layout);
    set_values(input_mem, in_data);

    std::vector<int32_t> output_shape_data = {(int32_t)count_non_zero};

    set_values(output_shape_mem, output_shape_data);

    topology topology;
    topology.add(input_layout("InputData", in_layout));
    topology.add(data("OutputShape", output_shape_mem));
    topology.add(
        gather_nonzero("gather_nonzero", input_info("InputData"), input_info("OutputShape"))
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("InputData", input_mem);
    auto outputs = network.execute();
    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    cldnn::mem_lock<int32_t> shape_ptr(output_shape_mem, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(test_gather_non_zero, 4d_fp32_1_3_3_1) {
    std::vector<float> in_data = {
        0.1f, 0.2f, 0.3f, 0.0f,
        0.0f, 0.4f, 0.1f, 0.9f, 0.10f
    };
    test_gather_non_zero<float>(layout{ov::PartialShape{1, 3, 3, 1}, data_types::f32, format::bfyx}, in_data);
}

TEST(test_gather_non_zero, 4d_fp32_2_4_3_2) {
    std::vector<float> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_gather_non_zero<float>(layout{ov::PartialShape{2, 4, 3, 2}, data_types::f32, format::bfyx}, in_data);
}
TEST(test_gather_non_zero, 4d_fp16_2_4_3_2) {
    std::vector<ov::float16> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_gather_non_zero<ov::float16>(layout{ov::PartialShape{2, 4, 3, 2}, data_types::f16, format::bfyx}, in_data);
}

TEST(test_gather_non_zero, 5d_fp32_1_3_3_2_2) {
    std::vector<float> in_data = {
        0.1f, 0.2f, 0.3f, 0.0f, 12.1f, 11.1f,
        0.0f, 0.0f, 0.1f, 0.9f, 0.10f, 0.001f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        0.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
    };
    test_gather_non_zero<float>(layout{ov::PartialShape{1, 3, 4, 2, 2}, data_types::f32, format::bfzyx}, in_data);
}

TEST(test_gather_non_zero, 6d_fp16_2_3_1_3_2_4) {
   std::vector<float> in_data = {
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        1.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        19.0f, 0.0f, 0.1f, 0.9f,  0.10f, -0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        13.0f, 1.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        11.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.0001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
        0.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        -13.0f, 1.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
        0.1f,  1.2f, 0.3f, 99.0f,  12.1f, 11.1f,
       100.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        13.0f, 1.0f, 0.1f, 0.9f,  -0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.0001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
    };
    test_gather_non_zero<float>(layout{ov::PartialShape{2, 3, 1, 3, 2, 4}, data_types::f32, format::bfwzyx}, in_data);
}

TEST(non_zero_gpu, dynamic) {
    auto& engine = get_test_engine();
    ov::Shape in_shape = { 3, 3 };
    auto in_layout = layout{ov::PartialShape::dynamic(in_shape.size()), data_types::f32, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape(in_shape), data_types::f32, format::bfyx});

    std::vector<float> input_data = {
        3.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        5.f, 6.f, 0.f,
    };
    set_values(input, input_data);

    std::vector<int32_t> out_data = {
        0, 1, 2, 2, 0, 1, 0, 1
    };

    topology topology;
    topology.add(input_layout("InputData", in_layout));
    topology.add(count_nonzero("count_nonzero", input_info("InputData")));
    topology.add(gather_nonzero("gather_nonzero", input_info("InputData"), input_info("count_nonzero")));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("InputData", input);

    auto count_nonzero_inst = network.get_primitive("count_nonzero");
    auto count_nonzero_impl = count_nonzero_inst->get_impl();
    ASSERT_TRUE(count_nonzero_impl != nullptr);
    ASSERT_TRUE(count_nonzero_impl->is_dynamic());

    auto gather_nonzero_inst = network.get_primitive("gather_nonzero");
    auto gather_nonzero_impl = gather_nonzero_inst->get_impl();
    ASSERT_TRUE(gather_nonzero_impl != nullptr);
    ASSERT_TRUE(gather_nonzero_impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), (uint32_t)8);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
        ASSERT_FLOAT_EQ(output_ptr[i], out_data[i]);
    }
}

template<typename T>
void test_non_zero(layout in_layout, std::vector<T> in_data) {
    auto& engine = get_test_engine();
    auto input_mem = engine.allocate_memory(in_layout);
    auto count_non_zero = ov::reference::non_zero_get_count<T>(in_data.data(), in_layout.get_shape());
    auto in_rank = in_layout.get_shape().size();
    std::vector<int32_t> expected_results(count_non_zero * in_rank);
    ov::reference::non_zero<T, int32_t>(in_data.data(), expected_results.data(), in_layout.get_shape());

    set_values(input_mem, in_data);

    std::vector<int32_t> output_shape_data = {(int32_t)in_rank, (int32_t)count_non_zero, 1, 1};

    topology topology;
    topology.add(input_layout("InputData", in_layout));
    topology.add(count_nonzero("count_nonzero", input_info("InputData")));
    topology.add(gather_nonzero("gather_nonzero", input_info("InputData"), input_info("count_nonzero")));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("InputData", input_mem);
    auto outputs = network.execute();
    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(test_non_zero, 1d_fp16_48) {
    std::vector<ov::float16> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_non_zero<ov::float16>(layout{ov::PartialShape{48}, data_types::f16, format::bfyx}, in_data);
}

TEST(test_non_zero, 2d_fp32_2_34) {
    std::vector<float> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_non_zero<float>(layout{ov::PartialShape{2, 24}, data_types::f32, format::bfyx}, in_data);
}

TEST(test_non_zero, 3d_fp16_4_3_4) {
    std::vector<float> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_non_zero<float>(layout{ov::PartialShape{4, 3, 4}, data_types::f32, format::bfyx}, in_data);
}

TEST(test_non_zero, 4d_fp16_2_4_3_2) {
    std::vector<ov::float16> in_data = {
        0.1f,   0.2f,  0.3f, 0.0f, 12.0f, 2.0f,   0.4f,  0.1f,
        1.9f,   0.10f, 1.0f, 0.0f, 0.1f,  0.2f,   0.0f,  100.0f,
        0.0001f,   0.0f,  2.9f, 0.2f, 4.0f,  0.0f,   9.1f,  0.9f,
        100.0f, 0.4f,  0.1f, 0.3f, 0.0f,  24.2f,  1.23f, 0.0f,
        4.0f,   0.0f,  3.1f, 0.9f, 0.10f, 49.2f,  0.0f,  0.3f,
        100.0f, 0.4f,  0.1f, 0.9f, 0.1f,  33.12f, 12.1f, 0.0001f
    };
    test_non_zero<ov::float16>(layout{ov::PartialShape{2, 4, 3, 2}, data_types::f16, format::bfyx}, in_data);
}

TEST(test_non_zero, 5d_fp32_1_3_3_2_2) {
    std::vector<float> in_data = {
        0.1f, 0.2f, 0.3f, 0.0f, 12.1f, 11.1f,
        0.0f, 0.0f, 0.1f, 0.9f, 0.10f, 0.001f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        0.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
    };
    test_non_zero<float>(layout{ov::PartialShape{1, 3, 4, 2, 2}, data_types::f32, format::bfzyx}, in_data);
}

TEST(test_non_zero, 6d_fp16_2_3_1_3_2_4) {
    std::vector<float> in_data = {
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        1.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        19.0f, 0.0f, 0.1f, 0.9f,  0.10f, -0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        8.0f,  3.0f, 0.1f, 0.00001f,  0.10f, 0.001f,
        0.1f, -0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        13.0f, 1.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        11.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.0001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
        0.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        -13.0f, 1.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
        0.1f,  1.2f, 0.3f, 99.0f,  12.1f, 11.1f,
       100.0f,  0.0f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 0.0f,  12.1f, 11.1f,
        13.0f, 1.0f, 0.1f, 0.9f,  -0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 66.0f, 12.1f, 11.1f,
        0.0f,  0.0001f, 0.1f, 0.9f,  0.10f, 0.001f,
        0.1f,  0.2f, 0.3f, 2.0f,  12.1f, 11.1f,
    };
    test_non_zero<float>(layout{ov::PartialShape{2, 3, 1, 3, 2, 4}, data_types::f32, format::bfwzyx}, in_data);
}

TEST(test_non_zero, 6d_fp16_2_2_2_1_5_1) {
    std::vector<int32_t> in_data = {
        10, 12, 23, 1232, 11, 9, 10, 23, 0, 1,
        0,  12, 23, 0, 11, 9, 10, 23, 0, 1,
        10, 0,  2,  32, 11, 9, 10, 23, 0, 1,
        10, 12, 3, 12, 11, 9, 0, 23, 0, 1
    };
    test_non_zero<int32_t>(layout{ov::PartialShape{2, 2, 2, 1, 5, 1}, data_types::i32, format::bfwzyx}, in_data);
}

TEST(test_gather_non_zero, not_use_local_mem) {
    auto& engine = get_test_engine();
    auto max_local_mem_size = engine.get_device_info().max_local_mem_size;

    auto in_layout = layout{ov::PartialShape{ov::Dimension(max_local_mem_size)}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory(in_layout);
    auto in_data = std::vector<float>(max_local_mem_size, 1.f);
    set_values(input_mem, in_data);

    auto output_shape_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto output_shape_mem = engine.allocate_memory(output_shape_layout);
    set_values(output_shape_mem, {static_cast<int32_t>(max_local_mem_size)});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("output_shape", output_shape_mem));
    topology.add(gather_nonzero("gather_nonzero", input_info("input"), input_info("output_shape")));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input_mem);

    auto outputs = network.execute();
    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    std::vector<int32_t> expected_results(max_local_mem_size);
    ov::reference::non_zero<float, int32_t>(in_data.data(), expected_results.data(), in_layout.get_shape());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(non_zero_gpu, const_input) {
    auto& engine = get_test_engine();
    ov::Shape in_shape = { 3, 3 };
    auto input_data_mem = engine.allocate_memory(layout{ov::PartialShape(in_shape), data_types::f32, format::bfyx});

    std::vector<float> input_data = {
        3.f, 0.f, 0.f,
        0.f, 4.f, 0.f,
        5.f, 6.f, 0.f,
    };
    set_values(input_data_mem, input_data);

    std::vector<int32_t> out_data = {
        0, 1, 2, 2, 0, 1, 0, 1
    };

    topology topology;
    topology.add(data("InputData", input_data_mem));
    topology.add(count_nonzero("count_nonzero", input_info("InputData")));
    topology.add(gather_nonzero("gather_nonzero", input_info("InputData"), input_info("count_nonzero")));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    auto count_nonzero_inst = network.get_primitive("count_nonzero");
    auto count_nonzero_impl = count_nonzero_inst->get_impl();
    ASSERT_TRUE(count_nonzero_impl == nullptr);
    // count nonzero should be optimized out from propagate_constant pass
    ASSERT_TRUE(count_nonzero_inst->get_node().is_type<data>());

    auto gather_nonzero_inst = network.get_primitive("gather_nonzero");
    auto gather_nonzero_impl = gather_nonzero_inst->get_impl();
    ASSERT_TRUE(gather_nonzero_impl != nullptr);
    ASSERT_TRUE(gather_nonzero_impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), (uint32_t)8);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
        ASSERT_FLOAT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(non_zero_gpu, empty_input) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape{1, -1}, data_types::f32, format::bfyx};
    auto in_data_layout = layout{ov::PartialShape{1, 0}, data_types::f32, format::bfyx};
    auto input_data_mem = engine.allocate_memory(in_data_layout);

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(count_nonzero("count_nonzero", input_info("input")));
    topology.add(gather_nonzero("gather_nonzero", input_info("input"), input_info("count_nonzero")));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network net(engine, topology, config);

    net.set_input_data("input", input_data_mem);

    auto count_nonzero_inst = net.get_primitive("count_nonzero");

    // Put some value into out buffer to ensure that it's non empty
    // That is needed to ensure that implementation correctly handles the cases when input tensor is empty and set count non zero to 0
    count_nonzero_inst->output_memory(0).fill(engine.get_service_stream(), 1, true);
    engine.get_service_stream().finish();

    auto count_nonzero_impl = count_nonzero_inst->get_impl();
    ASSERT_TRUE(count_nonzero_impl != nullptr);

    auto gather_nonzero_inst = net.get_primitive("gather_nonzero");
    auto gather_nonzero_impl = gather_nonzero_inst->get_impl();
    ASSERT_TRUE(gather_nonzero_impl != nullptr);
    ASSERT_TRUE(gather_nonzero_impl->is_dynamic());

    auto outputs = net.execute();

    auto output = outputs.at("gather_nonzero").get_memory();
    ASSERT_EQ(output, nullptr);
}
