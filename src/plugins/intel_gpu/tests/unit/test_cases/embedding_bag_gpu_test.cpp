// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/embedding_bag.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

TEST(embedding_bag_fp16_gpu, packed_sum_basic) {
    //  emb_table : 5x2
    //  indices : 3x2
    //  per_sample_weights : 3x2
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 2, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2,
            1, 2,
            3, 4
    });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f),
            ov::float16(0.5f), ov::float16(0.5f),
            ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::packed_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(data("Input2", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2") }, type, output_shape)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -1.05f, -1.2f,
            -1.f, -1.1f,
            -0.1f, 0.4f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, packed_sum_basic_without_weights) {
    //  emb_table : 5x2
    //  indices : 3x2
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 2, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2,
            1, 2,
            3, 4
    });

    auto type = embedding_bag::packed_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1") }, type, output_shape)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -2.1f, -2.4f,
            -2.f, -2.2f,
            -0.2f, 0.8f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, packed_sum_dim2) {
    //  emb_table : 5x2x2
    //  indices : 3x2
    //  per_sample_weights : 3x2
    //  Output : 3x2x2
    //  Input values in fp16
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 2, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 2, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } });
    tensor output_shape = {3, 2, 2, 1};

    /*
     * [ 5
     *   [ 2
     *       [ -0.2, 1.3 ], [ 0.5, -0.3 ]
     *   ],
     *   [ 2
     *       [ 2.3, 1.3 ], [ -0.4, -0.7 ]
     *   ],
     *   [ 2
     *       [ 3.3, -4.1 ], [ 2.1, 0.8 ]
     *   ],
     *   [ 2
     *       [ 3.5, -5.7 ], [ -0.1, 0.3 ]
     *   ],
     *   [ 2
     *       [ 0.3, 1.0 ], [ 2.3, -4.1 ]
     *   ]
     * ]
     */
    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16( 1.3f), ov::float16( 0.5f), ov::float16(-0.3f),
            ov::float16( 2.3f), ov::float16( 1.3f), ov::float16(-0.4f), ov::float16(-0.7f),
            ov::float16( 3.3f), ov::float16(-4.1f), ov::float16( 2.1f), ov::float16( 0.8f),
            ov::float16( 3.5f), ov::float16(-5.7f), ov::float16(-0.1f), ov::float16( 0.3f),
            ov::float16( 0.3f), ov::float16( 1.0f), ov::float16( 2.3f), ov::float16(-4.1f)
    });
    set_values<int32_t>(indices, {
            0, 2,
            1, 2,
            3, 4
    });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f),
            ov::float16(0.5f), ov::float16(0.5f),
            ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::packed_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(data("Input2", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2") }, type, output_shape)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    /*
     * [ 3
     *   [ 2
     *       [ 1.55, -1.4 ], [ 1.3, 0.25 ]
     *   ],
     *   [ 2
     *       [ 2.8, -1.4 ], [ 0.85, 0.05 ]
     *   ],
     *   [ 2
     *       [ 1.9, -2.35 ], [ 1.1, -1.9 ]
     *   ],
     * ]
     */
    std::vector<float> expected_results = {
            1.55f,  -1.4f,  1.3f,  0.25f,
             2.8f,  -1.4f, 0.85f,  0.05f,
             1.9f, -2.35f,  1.1f,  -1.9f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]), static_cast<float>(1e-2))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, packed_sum_dim3) {
    //  emb_table : 5x2x3x2
    //  indices : 3x2
    //  per_sample_weights : 3x2
    //  Output : 3x2x3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 3, 2 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 2, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } });
    tensor output_shape = {3, 2, 3, 2};

    /*
     * [ 5
     *   [ 2
     *     [ 3
     *       [ -0.2, 1.3 ], [ 0.5, -0.3 ], [ 0.4, -0.4 ]
     *     ],
     *     [ 3
     *       [ -0.1, 1.0 ], [ 2.1, 0.7 ], [ -0.2, -0.7 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.9, -2.4 ], [ 3.4, -0.7 ], [ -0.4, 0.5 ]
     *     ],
     *     [ 3
     *       [ 2.3, 1.3 ], [ -0.4, -0.7 ], [ 1.8, -0.9 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.5, -2.4 ], [ 4.2, 3.2 ], [ -0.6, 0.9 ]
     *     ],
     *     [ 3
     *       [ 3.3, -4.1 ], [ 2.1, 0.8 ], [ 5.2, -2.5 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 0.8, -1.9 ], [ 0.7, 3.4 ], [ -3.3, 0.1 ]
     *     ],
     *     [ 3
     *       [ 3.5, -5.7 ], [ -0.1, 0.3 ], [ 0.4, 3.3 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 6.1, 8.3 ], [ 0.4, -4.4 ], [ -5.2, 0.9 ]
     *     ],
     *     [ 3
     *       [ 0.3, 1.0 ], [ 2.3, -4.1 ], [ 2.0, -5.7 ]
     *     ],
     *   ]
     * ]
     */
    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16( 1.3f), ov::float16( 0.5f), ov::float16(-0.3f), ov::float16( 0.4f), ov::float16(-0.4f),
            ov::float16(-0.1f), ov::float16( 1.0f), ov::float16( 2.1f), ov::float16( 0.7f), ov::float16(-0.2f), ov::float16(-0.7f),
            ov::float16( 1.9f), ov::float16(-2.4f), ov::float16( 3.4f), ov::float16(-0.7f), ov::float16(-0.4f), ov::float16( 0.5f),
            ov::float16( 2.3f), ov::float16( 1.3f), ov::float16(-0.4f), ov::float16(-0.7f), ov::float16( 1.8f), ov::float16(-0.9f),
            ov::float16( 1.5f), ov::float16(-2.4f), ov::float16( 4.2f), ov::float16( 3.2f), ov::float16(-0.6f), ov::float16( 0.9f),
            ov::float16( 3.3f), ov::float16(-4.1f), ov::float16( 2.1f), ov::float16( 0.8f), ov::float16( 5.2f), ov::float16(-2.5f),
            ov::float16( 0.8f), ov::float16(-1.9f), ov::float16( 0.7f), ov::float16( 3.4f), ov::float16(-3.3f), ov::float16( 0.1f),
            ov::float16( 3.5f), ov::float16(-5.7f), ov::float16(-0.1f), ov::float16( 0.3f), ov::float16( 0.4f), ov::float16( 3.3f),
            ov::float16( 6.1f), ov::float16( 8.3f), ov::float16( 0.4f), ov::float16(-4.4f), ov::float16(-5.2f), ov::float16( 0.9f),
            ov::float16( 0.3f), ov::float16( 1.0f), ov::float16( 2.3f), ov::float16(-4.1f), ov::float16( 2.0f), ov::float16(-5.7f)
    });
    set_values<int32_t>(indices, {
            0, 2,
            1, 2,
            3, 4
    });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f),
            ov::float16(0.5f), ov::float16(0.5f),
            ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::packed_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(data("Input2", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2") }, type, output_shape)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    /*
     * [ 3
     *   [ 2
     *     [ 3
     *       [ 0.65, -0.55 ], [ 2.35, 1.45 ], [ -0.1, 0.25 ]
     *     ],
     *     [ 3
     *       [ 1.6, -1.55 ], [ 2.1, 0.75 ], [ 2.5, -1.6 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.7, -2.4 ], [ 3.8, 1.25 ], [ -0.5, 0.7 ]
     *     ],
     *     [ 3
     *       [ 2.8, -1.4 ], [ 0.85, 0.05 ], [ 3.5, -1.7 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 3.45, 3.2 ], [ 0.55, -0.5 ], [ -4.25, 0.5 ]
     *     ],
     *     [ 3
     *       [ 1.9, -2.35 ], [ 1.1, -1.9 ], [ 1.2, -1.2 ]
     *     ],
     *   ]
     * ]
     */
    std::vector<float> expected_results = {
        0.65f, -0.55f, 2.35f, 1.45f,  -0.1f, 0.25f,
         1.6f, -1.55f,  2.1f, 0.75f,   2.5f, -1.6f,
         1.7f,  -2.4f,  3.8f, 1.25f,  -0.5f,  0.7f,
         2.8f,  -1.4f, 0.85f, 0.05f,   3.5f, -1.7f,
        3.45f,   3.2f, 0.55f, -0.5f, -4.25f,  0.5f,
         1.9f, -2.35f,  1.1f, -1.9f,   1.2f, -1.2f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]), static_cast<float>(1e-2))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, offsets_sum_basic) {
    //  emb_table : 5x2
    //  indices : 4x1
    //  offsets : 3x1
    //  per_sample_weights : 4x1
    //  default_index : 1x1
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto offsets = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 1, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(offsets, {
            0, 2, 2
    });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::offsets_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", offsets->get_layout()));
    topology.add(data("Input3", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2"), input_info("Input3") }, type, output_shape, 0)
    );
    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", offsets);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -1.05f, -1.2f,
            -0.2f, -0.6f,
            -0.1f, 0.4f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, offsets_sum_basic_first_empty) {
    //  emb_table : 5x2
    //  indices : 4x1
    //  offsets : 3x1
    //  per_sample_weights : 4x1
    //  default_index : 1x1
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto offsets = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 1, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(offsets, {
            0, 0, 2
    });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::offsets_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", offsets->get_layout()));
    topology.add(data("Input3", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2"), input_info("Input3") }, type, output_shape, 2)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", offsets);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -1.9f, -1.8f,
            -1.05f, -1.2f,
            -0.1f, 0.4f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, offsets_sum_basic_last_empty) {
    //  emb_table : 5x2
    //  indices : 4x1
    //  offsets : 3x1
    //  per_sample_weights : 4x1
    //  default_index : 1x1
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto offsets = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 1, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(offsets, {
            0, 2, 4
    });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::offsets_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", offsets->get_layout()));
    topology.add(data("Input3", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2"), input_info("Input3") }, type, output_shape, 2)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", offsets);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -1.05f, -1.2f,
            -0.1f, 0.4f,
            -1.9f, -1.8f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, offsets_sum_without_weights_and_def_index) {
    //  emb_table : 5x2
    //  indices : 4x1
    //  offsets : 3x1
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto offsets = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 1, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(offsets, {
            0, 2, 2
    });

    auto type = embedding_bag::offsets_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", offsets->get_layout()));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2") }, type, output_shape)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", offsets);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -2.1f, -2.4f,
                0,     0,
            -0.2f,  0.8f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, offsets_sum_dim3) {
    //  emb_table : 5x2x3x2
    //  indices : 4x1
    //  offsets : 3x1
    //  per_sample_weights : 4x1
    //  default_index : 1x1
    //  Output : 3x2x3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 3, 2 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto offsets = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 1, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } });
    tensor output_shape = {3, 2, 3, 2};

    /*
     * [ 5
     *   [ 2
     *     [ 3
     *       [ -0.2, 1.3 ], [ 0.5, -0.3 ], [ 0.4, -0.4 ]
     *     ],
     *     [ 3
     *       [ -0.1, 1.0 ], [ 2.1, 0.7 ], [ -0.2, -0.7 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.9, -2.4 ], [ 3.4, -0.7 ], [ -0.4, 0.5 ]
     *     ],
     *     [ 3
     *       [ 2.3, 1.3 ], [ -0.4, -0.7 ], [ 1.8, -0.9 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.5, -2.4 ], [ 4.2, 3.2 ], [ -0.6, 0.9 ]
     *     ],
     *     [ 3
     *       [ 3.3, -4.1 ], [ 2.1, 0.8 ], [ 5.2, -2.5 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 0.8, -1.9 ], [ 0.7, 3.4 ], [ -3.3, 0.1 ]
     *     ],
     *     [ 3
     *       [ 3.5, -5.7 ], [ -0.1, 0.3 ], [ 0.4, 3.3 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 6.1, 8.3 ], [ 0.4, -4.4 ], [ -5.2, 0.9 ]
     *     ],
     *     [ 3
     *       [ 0.3, 1.0 ], [ 2.3, -4.1 ], [ 2.0, -5.7 ]
     *     ],
     *   ]
     * ]
     */
    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16( 1.3f), ov::float16( 0.5f), ov::float16(-0.3f), ov::float16( 0.4f), ov::float16(-0.4f),
            ov::float16(-0.1f), ov::float16( 1.0f), ov::float16( 2.1f), ov::float16( 0.7f), ov::float16(-0.2f), ov::float16(-0.7f),
            ov::float16( 1.9f), ov::float16(-2.4f), ov::float16( 3.4f), ov::float16(-0.7f), ov::float16(-0.4f), ov::float16( 0.5f),
            ov::float16( 2.3f), ov::float16( 1.3f), ov::float16(-0.4f), ov::float16(-0.7f), ov::float16( 1.8f), ov::float16(-0.9f),
            ov::float16( 1.5f), ov::float16(-2.4f), ov::float16( 4.2f), ov::float16( 3.2f), ov::float16(-0.6f), ov::float16( 0.9f),
            ov::float16( 3.3f), ov::float16(-4.1f), ov::float16( 2.1f), ov::float16( 0.8f), ov::float16( 5.2f), ov::float16(-2.5f),
            ov::float16( 0.8f), ov::float16(-1.9f), ov::float16( 0.7f), ov::float16( 3.4f), ov::float16(-3.3f), ov::float16( 0.1f),
            ov::float16( 3.5f), ov::float16(-5.7f), ov::float16(-0.1f), ov::float16( 0.3f), ov::float16( 0.4f), ov::float16( 3.3f),
            ov::float16( 6.1f), ov::float16( 8.3f), ov::float16( 0.4f), ov::float16(-4.4f), ov::float16(-5.2f), ov::float16( 0.9f),
            ov::float16( 0.3f), ov::float16( 1.0f), ov::float16( 2.3f), ov::float16(-4.1f), ov::float16( 2.0f), ov::float16(-5.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(offsets, {
            0, 2, 2
    });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f),
            ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::offsets_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", offsets->get_layout()));
    topology.add(data("Input3", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2"), input_info("Input3") }, type, output_shape, 0)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", offsets);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    /*
     * [ 3
     *   [ 2
     *     [ 3
     *       [ 0.65, -0.55 ], [ 2.35, 1.45 ], [ -0.1, 0.25 ]
     *     ],
     *     [ 3
     *       [ 1.6, -1.55 ], [ 2.1, 0.75 ], [ 2.5, -1.6 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ -0.2, 1.3 ], [ 0.5, -0.3 ], [ 0.4, -0.4 ]
     *     ],
     *     [ 3
     *       [ -0.1, 1.0 ], [ 2.1, 0.7 ], [ -0.2, -0.7 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 3.45, 3.2 ], [ 0.55, -0.5 ], [ -4.25, 0.5 ]
     *     ],
     *     [ 3
     *       [ 1.9, -2.35 ], [ 1.1, -1.9 ], [ 1.2, -1.2 ]
     *     ],
     *   ]
     * ]
     */
    std::vector<float> expected_results = {
        0.65f, -0.55f, 2.35f, 1.45f,  -0.1f, 0.25f,
         1.6f, -1.55f,  2.1f, 0.75f,   2.5f, -1.6f,
        -0.2f,   1.3f,  0.5f, -0.3f,   0.4f, -0.4f,
        -0.1f,   1.0f,  2.1f,  0.7f,  -0.2f, -0.7f,
        3.45f,   3.2f, 0.55f, -0.5f, -4.25f,  0.5f,
         1.9f, -2.35f,  1.1f, -1.9f,   1.2f, -1.2f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]), static_cast<float>(1e-2))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, segments_sum_basic) {
    //  emb_table : 5x2
    //  indices : 4x1
    //  segment_ids : 4x1
    //  per_sample_weights : 4x1
    //  default_index : 1x1
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segment_ids = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segments_num = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(segment_ids, {
            0, 0, 2, 2
    });
    set_values<int32_t>(segments_num, { 4 });

    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::segments_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", segment_ids->get_layout()));
    topology.add(input_layout("Input3", segments_num->get_layout()));
    topology.add(data("Input4", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2"), input_info("Input3"), input_info("Input4") }, type, output_shape, 0)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", segment_ids);
    network.set_input_data("Input3", segments_num);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -1.05f, -1.2f,
            -0.2f, -0.6f,
            -0.1f, 0.4f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, segments_sum_basic_first_empty) {
    //  emb_table : 5x2
    //  indices : 4x1
    //  segment_ids : 4x1
    //  per_sample_weights : 4x1
    //  default_index : 1x1
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segment_ids = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segments_num = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(segment_ids, {
            1, 1, 2, 2
    });
    set_values<int32_t>(segments_num, { 4 });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::segments_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", segment_ids->get_layout()));
    topology.add(input_layout("Input3", segments_num->get_layout()));
    topology.add(data("Input4", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2"), input_info("Input3"), input_info("Input4") }, type, output_shape, 2)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", segment_ids);
    network.set_input_data("Input3", segments_num);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -1.9f, -1.8f,
            -1.05f, -1.2f,
            -0.1f, 0.4f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, segments_sum_basic_last_empty) {
    //  emb_table : 5x2
    //  indices : 4x1
    //  segment_ids : 4x1
    //  per_sample_weights : 4x1
    //  default_index : 1x1
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segment_ids = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segments_num = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(segment_ids, {
            0, 0, 1, 1
    });
    set_values<int32_t>(segments_num, { 4 });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::segments_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", segment_ids->get_layout()));
    topology.add(input_layout("Input3", segments_num->get_layout()));
    topology.add(data("Input4", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2"), input_info("Input3"), input_info("Input4") }, type, output_shape, 2)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", segment_ids);
    network.set_input_data("Input3", segments_num);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -1.05f, -1.2f,
            -0.1f, 0.4f,
            -1.9f, -1.8f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, segments_sum_without_weights_and_def_index) {
    //  emb_table : 5x2
    //  indices : 4x1
    //  segment_ids : 4x1
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segment_ids = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segments_num = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16(-0.6f),
            ov::float16(-0.1f), ov::float16(-0.4f),
            ov::float16(-1.9f), ov::float16(-1.8f),
            ov::float16(-1.0f), ov::float16(1.5f),
            ov::float16(0.8f), ov::float16(-0.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(segment_ids, {
            0, 0, 2, 2
    });
    set_values<int32_t>(segments_num, { 4 });

    auto type = embedding_bag::segments_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", segment_ids->get_layout()));
    topology.add(input_layout("Input3", segments_num->get_layout()));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2"), input_info("Input3") }, type, output_shape)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", segment_ids);
    network.set_input_data("Input3", segments_num);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -2.1f, -2.4f,
                0,     0,
            -0.2f,  0.8f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]))) << i;
    }
}

TEST(embedding_bag_fp16_gpu, segments_sum_dim3) {
    //  emb_table : 5x2x3x2
    //  indices : 4x1
    //  segment_ids : 4x1
    //  per_sample_weights : 4x1
    //  default_index : 1x1
    //  Output : 3x2x3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f16, format::bfyx, { 5, 2, 3, 2 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segment_ids = engine.allocate_memory({ data_types::i32, format::bfyx, { 4, 1, 1, 1 } });
    auto segments_num = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 1, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 4, 1, 1, 1 } });
    tensor output_shape = {3, 2, 3, 2};

    /*
     * [ 5
     *   [ 2
     *     [ 3
     *       [ -0.2, 1.3 ], [ 0.5, -0.3 ], [ 0.4, -0.4 ]
     *     ],
     *     [ 3
     *       [ -0.1, 1.0 ], [ 2.1, 0.7 ], [ -0.2, -0.7 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.9, -2.4 ], [ 3.4, -0.7 ], [ -0.4, 0.5 ]
     *     ],
     *     [ 3
     *       [ 2.3, 1.3 ], [ -0.4, -0.7 ], [ 1.8, -0.9 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.5, -2.4 ], [ 4.2, 3.2 ], [ -0.6, 0.9 ]
     *     ],
     *     [ 3
     *       [ 3.3, -4.1 ], [ 2.1, 0.8 ], [ 5.2, -2.5 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 0.8, -1.9 ], [ 0.7, 3.4 ], [ -3.3, 0.1 ]
     *     ],
     *     [ 3
     *       [ 3.5, -5.7 ], [ -0.1, 0.3 ], [ 0.4, 3.3 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 6.1, 8.3 ], [ 0.4, -4.4 ], [ -5.2, 0.9 ]
     *     ],
     *     [ 3
     *       [ 0.3, 1.0 ], [ 2.3, -4.1 ], [ 2.0, -5.7 ]
     *     ],
     *   ]
     * ]
     */
    set_values(emb_table, {
            ov::float16(-0.2f), ov::float16( 1.3f), ov::float16( 0.5f), ov::float16(-0.3f), ov::float16( 0.4f), ov::float16(-0.4f),
            ov::float16(-0.1f), ov::float16( 1.0f), ov::float16( 2.1f), ov::float16( 0.7f), ov::float16(-0.2f), ov::float16(-0.7f),
            ov::float16( 1.9f), ov::float16(-2.4f), ov::float16( 3.4f), ov::float16(-0.7f), ov::float16(-0.4f), ov::float16( 0.5f),
            ov::float16( 2.3f), ov::float16( 1.3f), ov::float16(-0.4f), ov::float16(-0.7f), ov::float16( 1.8f), ov::float16(-0.9f),
            ov::float16( 1.5f), ov::float16(-2.4f), ov::float16( 4.2f), ov::float16( 3.2f), ov::float16(-0.6f), ov::float16( 0.9f),
            ov::float16( 3.3f), ov::float16(-4.1f), ov::float16( 2.1f), ov::float16( 0.8f), ov::float16( 5.2f), ov::float16(-2.5f),
            ov::float16( 0.8f), ov::float16(-1.9f), ov::float16( 0.7f), ov::float16( 3.4f), ov::float16(-3.3f), ov::float16( 0.1f),
            ov::float16( 3.5f), ov::float16(-5.7f), ov::float16(-0.1f), ov::float16( 0.3f), ov::float16( 0.4f), ov::float16( 3.3f),
            ov::float16( 6.1f), ov::float16( 8.3f), ov::float16( 0.4f), ov::float16(-4.4f), ov::float16(-5.2f), ov::float16( 0.9f),
            ov::float16( 0.3f), ov::float16( 1.0f), ov::float16( 2.3f), ov::float16(-4.1f), ov::float16( 2.0f), ov::float16(-5.7f)
    });
    set_values<int32_t>(indices, {
            0, 2, 3, 4
    });
    set_values<int32_t>(segment_ids, {
            0, 0, 2, 2
    });
    set_values<int32_t>(segments_num, { 4 });
    set_values(per_sample_weights, {
            ov::float16(0.5f), ov::float16(0.5f),
            ov::float16(0.5f), ov::float16(0.5f)
    });

    auto type = embedding_bag::segments_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", segment_ids->get_layout()));
    topology.add(input_layout("Input3", segments_num->get_layout()));
    topology.add(data("Input4", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2"), input_info("Input3"), input_info("Input4") }, type, output_shape, 0)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);
    network.set_input_data("Input2", segment_ids);
    network.set_input_data("Input3", segments_num);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    /*
     * [ 3
     *   [ 2
     *     [ 3
     *       [ 0.65, -0.55 ], [ 2.35, 1.45 ], [ -0.1, 0.25 ]
     *     ],
     *     [ 3
     *       [ 1.6, -1.55 ], [ 2.1, 0.75 ], [ 2.5, -1.6 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ -0.2, 1.3 ], [ 0.5, -0.3 ], [ 0.4, -0.4 ]
     *     ],
     *     [ 3
     *       [ -0.1, 1.0 ], [ 2.1, 0.7 ], [ -0.2, -0.7 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 3.45, 3.2 ], [ 0.55, -0.5 ], [ -4.25, 0.5 ]
     *     ],
     *     [ 3
     *       [ 1.9, -2.35 ], [ 1.1, -1.9 ], [ 1.2, -1.2 ]
     *     ],
     *   ]
     * ]
     */
    std::vector<float> expected_results = {
        0.65f, -0.55f, 2.35f, 1.45f,  -0.1f, 0.25f,
         1.6f, -1.55f,  2.1f, 0.75f,   2.5f, -1.6f,
        -0.2f,   1.3f,  0.5f, -0.3f,   0.4f, -0.4f,
        -0.1f,   1.0f,  2.1f,  0.7f,  -0.2f, -0.7f,
        3.45f,   3.2f, 0.55f, -0.5f, -4.25f,  0.5f,
         1.9f, -2.35f,  1.1f, -1.9f,   1.2f, -1.2f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], half_to_float(output_ptr[i]), static_cast<float>(1e-2))) << i;
    }
}

TEST(embedding_bag_fp32_gpu, packed_sum_basic) {
    //  emb_table : 5x2
    //  indices : 3x2
    //  per_sample_weights : 3x2
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f32, format::bfyx, { 5, 2, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 2, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 2, 1, 1 } });
    tensor output_shape = {3, 2, 1, 1};

    set_values(emb_table, {
            -0.2f, -0.6f,
            -0.1f, -0.4f,
            -1.9f, -1.8f,
            -1.0f, 1.5f,
            0.8f, -0.7f
    });
    set_values<int32_t>(indices, {
            0, 2,
            1, 2,
            3, 4
    });
    set_values(per_sample_weights, {
            0.5f, 0.5f,
            0.5f, 0.5f,
            0.5f, 0.5f
    });

    auto type = embedding_bag::packed_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(data("Input2", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2") }, type, output_shape)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            -1.05f, -1.2f,
            -1.f, -1.1f,
            -0.1f, 0.4f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i])) << i;
    }
}

TEST(embedding_bag_fp32_gpu, packed_sum_dim3) {
    //  emb_table : 5x2x3x2
    //  indices : 3x2
    //  per_sample_weights : 3x2
    //  Output : 3x2x3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f32, format::bfyx, { 5, 2, 3, 2 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 3, 2, 1, 1 } });
    auto per_sample_weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 3, 2, 1, 1 } });
    tensor output_shape = {3, 2, 3, 2};

    /*
     * [ 5
     *   [ 2
     *     [ 3
     *       [ -0.2, 1.3 ], [ 0.5, -0.3 ], [ 0.4, -0.4 ]
     *     ],
     *     [ 3
     *       [ -0.1, 1.0 ], [ 2.1, 0.7 ], [ -0.2, -0.7 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.9, -2.4 ], [ 3.4, -0.7 ], [ -0.4, 0.5 ]
     *     ],
     *     [ 3
     *       [ 2.3, 1.3 ], [ -0.4, -0.7 ], [ 1.8, -0.9 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.5, -2.4 ], [ 4.2, 3.2 ], [ -0.6, 0.9 ]
     *     ],
     *     [ 3
     *       [ 3.3, -4.1 ], [ 2.1, 0.8 ], [ 5.2, -2.5 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 0.8, -1.9 ], [ 0.7, 3.4 ], [ -3.3, 0.1 ]
     *     ],
     *     [ 3
     *       [ 3.5, -5.7 ], [ -0.1, 0.3 ], [ 0.4, 3.3 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 6.1, 8.3 ], [ 0.4, -4.4 ], [ -5.2, 0.9 ]
     *     ],
     *     [ 3
     *       [ 0.3, 1.0 ], [ 2.3, -4.1 ], [ 2.0, -5.7 ]
     *     ],
     *   ]
     * ]
     */
    set_values(emb_table, {
            -0.2f,  1.3f,  0.5f, -0.3f,  0.4f, -0.4f,
            -0.1f,  1.0f,  2.1f,  0.7f, -0.2f, -0.7f,
             1.9f, -2.4f,  3.4f, -0.7f, -0.4f,  0.5f,
             2.3f,  1.3f, -0.4f, -0.7f,  1.8f, -0.9f,
             1.5f, -2.4f,  4.2f,  3.2f, -0.6f,  0.9f,
             3.3f, -4.1f,  2.1f,  0.8f,  5.2f, -2.5f,
             0.8f, -1.9f,  0.7f,  3.4f, -3.3f,  0.1f,
             3.5f, -5.7f, -0.1f,  0.3f,  0.4f,  3.3f,
             6.1f,  8.3f,  0.4f, -4.4f, -5.2f,  0.9f,
             0.3f,  1.0f,  2.3f, -4.1f,  2.0f, -5.7f
    });
    set_values<int32_t>(indices, {
            0, 2,
            1, 2,
            3, 4
    });
    set_values(per_sample_weights, {
            0.5f, 0.5f,
            0.5f, 0.5f,
            0.5f, 0.5f
    });

    auto type = embedding_bag::packed_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(data("Input2", per_sample_weights));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2") }, type, output_shape)
    );

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("Input0", emb_table);
    network.set_input_data("Input1", indices);

    auto outputs = network.execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    /*
     * [ 3
     *   [ 2
     *     [ 3
     *       [ 0.65, -0.55 ], [ 2.35, 1.45 ], [ -0.1, 0.25 ]
     *     ],
     *     [ 3
     *       [ 1.6, -1.55 ], [ 2.1, 0.75 ], [ 2.5, -1.6 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 1.7, -2.4 ], [ 3.8, 1.25 ], [ -0.5, 0.7 ]
     *     ],
     *     [ 3
     *       [ 2.8, -1.4 ], [ 0.85, 0.05 ], [ 3.5, -1.7 ]
     *     ],
     *   ],
     *   [ 2
     *     [ 3
     *       [ 3.45, 3.2 ], [ 0.55, -0.5 ], [ -4.25, 0.5 ]
     *     ],
     *     [ 3
     *       [ 1.9, -2.35 ], [ 1.1, -1.9 ], [ 1.2, -1.2 ]
     *     ],
     *   ]
     * ]
     */
    std::vector<float> expected_results = {
        0.65f, -0.55f, 2.35f, 1.45f,  -0.1f, 0.25f,
         1.6f, -1.55f,  2.1f, 0.75f,   2.5f, -1.6f,
         1.7f,  -2.4f,  3.8f, 1.25f,  -0.5f,  0.7f,
         2.8f,  -1.4f, 0.85f, 0.05f,   3.5f, -1.7f,
        3.45f,   3.2f, 0.55f, -0.5f, -4.25f,  0.5f,
         1.9f, -2.35f,  1.1f, -1.9f,   1.2f, -1.2f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i])) << i;
    }
}

template <typename T>
void test_embedding_bag_fp32_gpu_extended5_6(bool is_caching_test) {
    //  emb_table : 5x2
    //  indices : 3x2
    //  per_sample_weights : 3x2
    //  Output : 3x2
    //  Input values in fp16
    auto& engine = get_test_engine();

    auto emb_table = engine.allocate_memory({ data_types::f32, format::bfyx, { 5, 6, 1, 1 } });
    auto indices = engine.allocate_memory({ data_types::i32, format::bfyx, { 5, 1, 1, 1 } });
    auto segment_ids = engine.allocate_memory({ data_types::i32, format::bfyx, { 5, 1, 1, 1 } });
    tensor output_shape = {5, 6, 1, 1};

    set_values(emb_table, {
            0.f, 1.f, 8.f,  5.f, 5.f,  2.f,
            0.f, 7.f, 7.f, 10.f, 4.f,  5.f,
            9.f, 0.f, 0.f,  5.f, 7.f,  0.f,
            4.f, 0.f, 4.f,  7.f, 6.f, 10.f,
            9.f, 5.f, 1.f,  7.f, 4.f,  7.f
    });
    set_values<int32_t>(indices, { 0, 1, 2, 2, 3 });
    set_values<int32_t>(segment_ids, { 0, 0, 2, 2, 4 });

    auto type = embedding_bag::segments_sum;
    topology topology;
    topology.add(input_layout("Input0", emb_table->get_layout()));
    topology.add(input_layout("Input1", indices->get_layout()));
    topology.add(input_layout("Input2", segment_ids->get_layout()));
    topology.add(
            embedding_bag("embedding_bag", { input_info("Input0"), input_info("Input1"), input_info("Input2") }, type, output_shape)
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("Input0", emb_table);
    network->set_input_data("Input1", indices);
    network->set_input_data("Input2", segment_ids);

    auto outputs = network->execute();

    auto output = outputs.at("embedding_bag").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    std::vector<T> expected_results = {
            0, 8, 15,  15, 9,  7,
            0, 0, 0, 0, 0,  0,
            18, 0, 0,  10, 14,  0,
            0, 0, 0,  0, 0,  0,
            4, 0, 4,  7, 6, 10,
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i])) << i;
    }
}

TEST(embedding_bag_fp32_gpu, extended5_6) {
    test_embedding_bag_fp32_gpu_extended5_6<float>(false);
}

TEST(export_import_embedding_bag_fp32_gpu, extended5_6) {
    test_embedding_bag_fp32_gpu_extended5_6<float>(true);
}
