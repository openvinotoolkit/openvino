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
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/tile.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

#include <iostream>

using namespace cldnn;
using namespace tests;

template<typename data_t>
void tile_ref(const memory& input, memory& output, tile::tile_axis axis, int num_tiles)
{
    auto get_sizes = [](const tensor& size, tile::tile_axis axis) -> std::pair<int, int>
    {
        switch (axis)
        {
            case tile::along_b: return std::make_pair(1, size.batch[0]*size.feature[0]*size.spatial[2]*size.spatial[1]*size.spatial[0]);
            case tile::along_f: return std::make_pair(size.batch[0], size.feature[0]*size.spatial[2]*size.spatial[1]*size.spatial[0]);
            case tile::along_z: return std::make_pair(size.batch[0]*size.feature[0], size.spatial[2]*size.spatial[1]*size.spatial[0]);
            case tile::along_y: return std::make_pair(size.batch[0]*size.feature[0]*size.spatial[2], size.spatial[1]*size.spatial[0]);
            case tile::along_x: return std::make_pair(size.batch[0]*size.feature[0]*size.spatial[2]*size.spatial[1], size.spatial[0]);
            default: throw std::invalid_argument("Invalid axis(" + std::to_string(static_cast<int>(axis)) + ") in tile ref version");
        }
    };

    const pointer<data_t> src = input.pointer<data_t>();
    pointer<data_t> dst = output.pointer<data_t>();

    const data_t* psrc = src.data();
    data_t* pdst = dst.data();

    auto sizes = get_sizes(input.get_layout().size, axis);
    int outer_dim = sizes.first;
    int inner_dim = sizes.second;

    for (int i = 0; i < outer_dim; i++)
    {
        for (int t = 0; t < num_tiles; t++)
        {
            for (int j = 0; j < inner_dim; j++)
            {
                pdst[j] = psrc[j];
            }
            pdst += inner_dim;
        }
        psrc += inner_dim;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_b) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(tile("tile", "input", tile::along_b, 2));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
                                     2.f, 0.f, 6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_b, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_ref_ptr = output_ref.pointer<float>();

    for (unsigned int i = 0; i < output_ref.count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_f) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 2, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(tile("tile", "input", tile::along_f, 2));

    std::vector<float> input_vec = { 1.f, 0.f,
                                     5.f, 1.5f,

                                     2.f, 0.f,
                                     6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_f, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_ref_ptr = output_ref.pointer<float>();

    for (unsigned int i = 0; i < output_ref.count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_y) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 4, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(tile("tile", "input", tile::along_y, 2));

    std::vector<float> input_vec = { 1.f, 0.f,
                                     5.f, 1.5f,

                                     2.f, 0.f,
                                     6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_y, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_ref_ptr = output_ref.pointer<float>();

    for (unsigned int i = 0; i < output_ref.count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_x) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 2 } });
    auto output_ref = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 4 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(tile("tile", "input", tile::along_x, 2));

    std::vector<float> input_vec = { 1.f, 0.f,
                                     5.f, 1.5f,

                                     2.f, 0.f,
                                     6.f, 5.2f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_x, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_ref_ptr = output_ref.pointer<float>();

    for (unsigned int i = 0; i < output_ref.count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_x_dense) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 1 } });
    auto output_ref = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 2, 4 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(tile("tile", "input", tile::along_x, 4));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f};
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_x, 4);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_ref_ptr = output_ref.pointer<float>();

    for (unsigned int i = 0; i < output_ref.count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

TEST(tile_gpu, basic_in1x2x2x2_axis_z) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 2, 2, 2, 2 } });
    auto output_ref = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 2, 2, 2, 4 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(tile("tile", "input", tile::along_z, 2));

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
    tile_ref<float>(input, output_ref, tile::along_z, 2);

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_ref_ptr = output_ref.pointer<float>();

    for (unsigned int i = 0; i < output_ref.count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]) << "Index=" << i;
    }
}

