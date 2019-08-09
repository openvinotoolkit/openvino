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
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/reorder.hpp>
#include <api/CPP/data.hpp>
#include <api/CPP/activation.hpp>
#include <api/CPP/mutable_data.hpp>
#include <api/CPP/layout.hpp>
#include <api/CPP/tile.hpp>
#include <api/CPP/reshape.hpp>

#include <api/CPP/batch_norm.hpp>
#include <api/CPP/concatenation.hpp>

using namespace cldnn;
using namespace tests;

/*
These tests are inteded to check if additional reorders are being added  properly during 
add_reorders optimization pass.
*/

//Input has incompatible format
TEST(add_reorders_gpu, basic1) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::fyxb,{ 2, 2, 3, 2 } }); //format unsupported by batch_norm!
    auto mean = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });
    auto variance = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 1, 1 } });

    set_values(input, {
        1.f, 2.f, -10.f,
        3.f, 4.f, -14.f,
        5.f, 6.f, -12.f,
        7.f, 8.f, -16.f,
        0.f, 0.f, -11.f,
        0.5f, -0.5f, -15.f,
        1.5f, 5.2f, -13.f,
        12.f, 9.f, -17.f
    });

    set_values(mean, { 0.1f, 0.2f });
    set_values(variance, { 0.4f, 0.5f });

    float epsilon = 1e-3f;
    float expected_out[] = {
        1.42125f,  3.00042f,
       -0.28256f, -0.28256f,
      -15.94960f,  4.57958f,
      -15.82340f,  0.42384f,
        6.15875f,-22.26620f,
       -0.98896f,-21.47460f,
        7.73791f,  9.31708f,
        1.83664f,  7.06401f,
       -19.1079f,  10.8962f,
       -18.6490f,  16.6711f,
        12.4754f, -25.4246f,
        12.4327f, -24.3002f};
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("mean", mean));
    topology.add(data("variance", variance));
    topology.add(batch_norm("batch_norm", "input", "mean", "variance", epsilon));

    network network(engine, topology); // without additional reorders we would get an exception here
    network.set_input_data("input", input);

    EXPECT_EQ(network.get_all_primitive_org_ids().size(), size_t(5));

    auto outputs = network.execute();

    auto output = outputs.at("batch_norm").get_memory().pointer<float>();
    for (int i = 0; i < 2 * 2 * 3 * 2; i++)
    {
        EXPECT_NEAR(expected_out[i], output[i], epsilon);
    }
}

//concatenation of incompatible convolutions
TEST(add_reorders_gpu, two_convolutions_and_concatenation) {
    const auto& engine = get_test_engine();
    build_options build_opt;
    build_opt.set_option(build_option::optimize_data(false));

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 1, 2 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 1, 1, 2 } });

    set_values(input, { 1.1f, 1.2f, 1.3f, 1.4f });
    set_values(weights1, { 2.1f, 3.1f});
    set_values(weights2, { 1.1f, 0.1f});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));

    topology.add(cldnn::convolution("conv1", { "input" }, { "weights1" }));
    topology.add(cldnn::reorder("reorder", "input", cldnn::layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(cldnn::convolution("conv2", { "reorder" }, { "weights2" }));

    topology.add(cldnn::concatenation("concat", { "conv1", "conv2" }, cldnn::concatenation::along_f));

    network network(engine, topology, build_opt);
    network.set_input_data("input", input);

    //concatenation accepts inputs in different formats, so no reorders should be added here
    EXPECT_EQ(network.get_all_primitive_org_ids().size(), size_t(7));
    auto outputs = network.execute();

    float expected_out[] = { 6.34f, 1.34f, 6.86f, 1.46f };
    float epsilon = 1e-3f;

    for (auto& it : outputs)
    {
        auto output = it.second.get_memory().pointer<float>();
        for (size_t cntr = 0; cntr < 2 * 2; cntr++)
        {
            EXPECT_NEAR(expected_out[cntr], output[cntr], epsilon);
        }
    }
}

template<typename data_t>
void tile_ref(const memory& input, memory& output, tile::tile_axis axis, int num_tiles)
{
    auto get_sizes = [](const tensor& size, tile::tile_axis axis) -> std::pair<int, int>
    {
        switch (axis)
        {
        case tile::along_b: return std::make_pair(1, size.batch[0] * size.feature[0] * size.spatial[2] * size.spatial[1] * size.spatial[0]);
        case tile::along_f: return std::make_pair(size.batch[0], size.feature[0] * size.spatial[2] * size.spatial[1] * size.spatial[0]);
        case tile::along_z: return std::make_pair(size.batch[0] * size.feature[0], size.spatial[2] * size.spatial[1] * size.spatial[0]);
        case tile::along_y: return std::make_pair(size.batch[0] * size.feature[0] * size.spatial[2], size.spatial[1] * size.spatial[0]);
        case tile::along_x: return std::make_pair(size.batch[0] * size.feature[0] * size.spatial[2] * size.spatial[1], size.spatial[0]);
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

TEST(add_reorders_gpu, basic_reshape_and_tile) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::byxf,{ 1, 2, 2, 1 } });
    auto output_ref = memory::allocate(engine, { data_types::f32, format::byxf,{ 2, 1, 4, 2 } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reshape("reshape", "input", tensor(2, 1, 2, 1)));
    topology.add(tile("tile", "reshape", tile::along_y, 4));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_y, 4);

    network network(engine, topology);
    network.set_input_data("input", input);

    //reorder is required as tile accepts only bfyx format
    EXPECT_EQ(network.get_all_primitive_org_ids().size(), size_t(4));
    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    auto output_ptr = output.pointer<float>();
    auto output_ref_ptr = output_ref.pointer<float>();

    for (unsigned int i = 0; i < output_ref.count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]);
    }
}
