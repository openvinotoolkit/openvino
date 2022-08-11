// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/tile.hpp>

#include <iostream>

using namespace cldnn;
using namespace ::tests;

template<typename data_t>
void tile_ref(const memory::ptr input, memory::ptr output, int64_t axis, int num_tiles) {
    auto get_sizes = [](const layout& l, int64_t axis, size_t rank) -> std::pair<int, int> {
        switch (axis) {
            case 0: return std::make_pair(1, l.batch() * l.feature() * l.spatial(2) * l.spatial(1) * l.spatial(0));
            case 1: return std::make_pair(l.batch(), l.feature() * l.spatial(2) * l.spatial(1) * l.spatial(0));
            case 2:
                if (rank > 4)
                    return std::make_pair(l.batch() * l.feature(), l.spatial(2) * l.spatial(1) * l.spatial(0));
                else
                    return std::make_pair(l.batch() * l.feature() * l.spatial(2), l.spatial(1) * l.spatial(0));
            case 3:
                if (rank > 4)
                    return std::make_pair(l.batch() * l.feature() * l.spatial(2), l.spatial(1) * l.spatial(0));
                else
                    return std::make_pair(l.batch() * l.feature() * l.spatial(2) * l.spatial(1), l.spatial(0));
            case 4: return std::make_pair(l.batch() * l.feature() * l.spatial(2) * l.spatial(1), l.spatial(0));
            default: throw std::invalid_argument("Invalid axis(" + std::to_string(static_cast<int>(axis)) + ") in tile ref version");
        }
    };

    cldnn::mem_lock<data_t> src(input, get_test_stream());
    cldnn::mem_lock<data_t> dst(output, get_test_stream());

    const data_t* psrc = src.data();
    data_t* pdst = dst.data();

    auto sizes = get_sizes(input->get_layout(), axis, input->get_layout().get_rank());
    int outer_dim = sizes.first;
    int inner_dim = sizes.second;

    for (int i = 0; i < outer_dim; i++) {
        for (int t = 0; t < num_tiles; t++) {
            for (int j = 0; j < inner_dim; j++) {
                pdst[j] = psrc[j];
            }
            pdst += inner_dim;
        }
        psrc += inner_dim;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_b) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", { 2, 1, 1, 1 }));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
                                     2.f, 0.f, 6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, 0, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_f) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", { 1, 2, 1, 1 }));

    std::vector<float> input_vec = { 1.f, 0.f,
                                     5.f, 1.5f,

                                     2.f, 0.f,
                                     6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, 1, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_y) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 3, 4 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 3, 8 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", { 1, 1, 2, 1 }));

    std::vector<float> input_vec = { 0.f, 1.f, 2.f,
                                     3.f, 4.f, 5.f,
                                     6.f, 7.f, 8.f,
                                     9.f, 10.f, 11.f,

                                     12.f, 13.f, 14.f,
                                     15.f, 16.f, 17.f,
                                     18.f, 19.f, 20.f,
                                     21.f, 22.f, 23.f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, 2, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_x) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", { 1, 1, 1, 2 }));

    std::vector<float> input_vec = { 0.f, 1.f,
                                     2.f, 3.f,

                                     4.f, 5.f,
                                     6.f, 7.f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, 3, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_x_dense) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 1, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", { 1, 1, 1, 4 }));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, 3, 4);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_z) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 2, 2, 2, 2 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::bfzyx,{ 1, 2, 2, 2, 4 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(tile("tile", "input", { 1, 1, 2, 1, 1 }));

    std::vector<float> input_vec = {
        1.f, 0.f,
        5.f, 1.5f,
        2.f, 0.f,
        6.f, 5.2f,
        1.f, 0.f,
        5.f, 1.5f,
        2.f, 0.f,
        6.f, 5.2f
    };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, 2, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}
