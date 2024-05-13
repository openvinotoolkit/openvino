// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/attr_types.hpp"
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/extract_image_patches.hpp>
#include <intel_gpu/primitives/data.hpp>


using namespace cldnn;
using namespace ::tests;

TEST(extract_image_patches_gpu, basic) {
    //  Input  : 1x1x10x10
    //  Output : 1x9x2x2

    tensor output_shape = {1, 9, 2, 2};
    auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    ov::Shape sizes = {3, 3};
    ov::Strides strides = {5, 5};
    ov::Shape rates = {1, 1};
    ov::op::PadType auto_pad = ov::op::PadType::VALID;

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(extract_image_patches("extract_image_patches", input_info("Input0"), sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         1,  6,
        51, 56,

         2,  7,
        52, 57,

         3,  8,
        53, 58,

        11, 16,
        61, 66,

        12, 17,
        62, 67,

        13, 18,
        63, 68,

        21, 26,
        71, 76,

        22, 27,
        72, 77,

        23, 28,
        73, 78
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic2) {
    //  Input  : 1x1x10x10
    //  Output : 1x16x1x1

    auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    ov::Shape sizes = {4, 4};
    ov::Strides strides = {8, 8};
    ov::Shape rates = {1, 1};
    ov::op::PadType auto_pad = ov::op::PadType::VALID;
    tensor output_shape = {1, 16, 1, 1};

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(extract_image_patches("extract_image_patches", input_info("Input0"), sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         1,
         2,
         3,
         4,
        11,
        12,
        13,
        14,
        21,
        22,
        23,
        24,
        31,
        32,
        33,
        34
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic3) {
    //  Input  : 1x1x10x10
    //  Output : 1x16x2x2

    auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    ov::Shape sizes = {4, 4};
    ov::Strides strides = {9, 9};
    ov::Shape rates = {1, 1};
    ov::op::PadType auto_pad = ov::op::PadType::SAME_UPPER;
    tensor output_shape = {1, 16, 2, 2};

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(extract_image_patches("extract_image_patches", input_info("Input0"), sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0,   0,
         0,  89,

         0,   0,
        81,  90,

         0,   0,
        82,   0,

         0,   0,
        83,   0,

         0,   9,
         0,  99,

         1,  10,
        91, 100,

         2,   0,
        92,   0,

         3,   0,
        93,   0,

         0,  19,
         0,   0,

        11,  20,
         0,   0,

        12,   0,
         0,   0,

        13,   0,
         0,   0,

         0,  29,
         0,   0,

        21,  30,
         0,   0,

        22,   0,
         0,   0,

        23,   0,
         0,   0,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic3_same_lower) {
    //  Input  : 1x1x10x10
    //  Output : 1x16x2x2

    auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    ov::Shape sizes = {4, 4};
    ov::Strides strides = {9, 9};
    ov::Shape rates = {1, 1};
    ov::op::PadType auto_pad = ov::op::PadType::SAME_LOWER;
    tensor output_shape = {1, 16, 2, 2};

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(extract_image_patches("extract_image_patches", input_info("Input0"), sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         0,   0,
         0,  78,

         0,   0,
         0,  79,

         0,   0,
        71,  80,

         0,   0,
        72,   0,

         0,   0,
         0,  88,

         0,   0,
         0,  89,

         0,   0,
        81,  90,

         0,   0,
        82,   0,

         0,   8,
         0,  98,

         0,   9,
         0,  99,

         1,  10,
        91, 100,

         2,   0,
        92,   0,

         0,  18,
         0,   0,

         0,  19,
         0,   0,

        11,  20,
         0,   0,

        12,   0,
         0,   0,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic3_enough_space) {
    //  Input  : 1x1x10x10
    //  Output : 1x9x2x2

    auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    ov::Shape sizes = {3, 3};
    ov::Strides strides = {7, 7};
    ov::Shape rates = {1, 1};
    ov::op::PadType auto_pad = ov::op::PadType::SAME_UPPER;
    tensor output_shape = {1, 9, 2, 2};

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(extract_image_patches("extract_image_patches", input_info("Input0"), sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         1,   8,
        71,  78,

         2,   9,
        72,  79,

         3,  10,
        73,  80,

        11,  18,
        81,  88,

        12,  19,
        82,  89,

        13,  20,
        83,  90,

        21,  28,
        91,  98,

        22,  29,
        92,  99,

        23,  30,
        93, 100,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic4) {
    //  Input  : 1x1x10x10
    //  Output : 1x9x2x2

    auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    ov::Shape sizes = {3, 3};
    ov::Strides strides = {5, 5};
    ov::Shape rates = {2, 2};
    ov::op::PadType auto_pad = ov::op::PadType::VALID;
    tensor output_shape = {1, 9, 2, 2};

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(extract_image_patches("extract_image_patches", input_info("Input0"), sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> answers = {
         1,   6,
        51,  56,

         3,   8,
        53,  58,

         5,  10,
        55,  60,

        21,  26,
        71,  76,

        23,  28,
        73,  78,

        25,  30,
        75,  80,

        41,  46,
        91,  96,

        43,  48,
        93,  98,

        45,  50,
        95, 100
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

template <typename T>
void test_extract_image_patches_gpu_basic5(bool is_caching_test) {
    //  Input  : 1x2x5x5
    //  Output : 1x8x2x2

    auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 2;
    auto in_rows = 5;
    auto in_cols = 5;
    ov::Shape sizes = {2, 2};
    ov::Strides strides = {3, 3};
    ov::Shape rates = {1, 1};
    ov::op::PadType auto_pad = ov::op::PadType::VALID;
    tensor output_shape = {1, 8, 2, 2};

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<T> inputVals(batch * depth * in_rows * in_cols);
    float n = 1;
    for (auto& val : inputVals) {
        val = n++;
    }

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input->get_layout()));
    topology.add(extract_image_patches("extract_image_patches", input_info("Input0"), sizes, strides, rates, auto_pad, output_shape));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("Input0", input);
    auto outputs = network->execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    std::vector<T> answers = {
         1,  4,
        16, 19,

        26, 29,
        41, 44,

         2,  5,
        17, 20,

        27, 30,
        42, 45,

         6,  9,
        21, 24,

        31, 34,
        46, 49,

         7, 10,
        22, 25,

        32, 35,
        47, 50
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        ASSERT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic5) {
    test_extract_image_patches_gpu_basic5<float>(false);
}

TEST(extract_image_patches_gpu, export_import) {
    test_extract_image_patches_gpu_basic5<float>(true);
}
