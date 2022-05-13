// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/tile.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/concatenation.hpp>

using namespace cldnn;
using namespace ::tests;

/*
These tests are inteded to check if additional reorders are being added  properly during
add_reorders optimization pass.
*/

//concatenation of incompatible convolutions
TEST(add_reorders_gpu, two_convolutions_and_concatenation) {
    auto& engine = get_test_engine();
    build_options build_opt;
    build_opt.set_option(build_option::optimize_data(false));

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights1 = engine.allocate_memory({ data_types::f32, format::yxio,{ 1, 1, 1, 2 } });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::oiyx,{ 1, 1, 1, 2 } });

    set_values(input, { 1.1f, 1.2f, 1.3f, 1.4f });
    set_values(weights1, { 2.1f, 3.1f});
    set_values(weights2, { 1.1f, 0.1f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));

    topology.add(cldnn::convolution("conv1", { "input" }, { "weights1" }));
    topology.add(cldnn::reorder("reorder", "input", cldnn::layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(cldnn::convolution("conv2", { "reorder" }, { "weights2" }));

    topology.add(cldnn::concatenation("concat", { "conv1", "conv2" }, 1));

    network network(engine, topology, build_opt);
    network.set_input_data("input", input);

    //concatenation accepts inputs in different formats, so no reorders should be added here
    EXPECT_EQ(network.get_all_primitive_org_ids().size(), size_t(7));
    auto outputs = network.execute();

    float expected_out[] = { 6.34f, 1.34f, 6.86f, 1.46f };
    float epsilon = 1e-3f;

    for (auto& it : outputs) {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        for (size_t cntr = 0; cntr < 2 * 2; cntr++) {
            EXPECT_NEAR(expected_out[cntr], output[cntr], epsilon);
        }
    }
}

template<typename data_t>
void tile_ref(const memory::ptr input, memory::ptr output, tile::tile_axis axis, int num_tiles) {
    auto get_sizes = [](const tensor& size, tile::tile_axis axis) -> std::pair<int, int> {
        switch (axis) {
        case tile::along_b: return std::make_pair(1, size.batch[0] * size.feature[0] * size.spatial[2] * size.spatial[1] * size.spatial[0]);
        case tile::along_f: return std::make_pair(size.batch[0], size.feature[0] * size.spatial[2] * size.spatial[1] * size.spatial[0]);
        case tile::along_z: return std::make_pair(size.batch[0] * size.feature[0], size.spatial[2] * size.spatial[1] * size.spatial[0]);
        case tile::along_y: return std::make_pair(size.batch[0] * size.feature[0] * size.spatial[2], size.spatial[1] * size.spatial[0]);
        case tile::along_x: return std::make_pair(size.batch[0] * size.feature[0] * size.spatial[2] * size.spatial[1], size.spatial[0]);
        default: throw std::invalid_argument("Invalid axis(" + std::to_string(static_cast<int>(axis)) + ") in tile ref version");
        }
    };

    cldnn::mem_lock<data_t> src(input, get_test_stream());
    cldnn::mem_lock<data_t> dst(output, get_test_stream());

    const data_t* psrc = src.data();
    data_t* pdst = dst.data();

    auto sizes = get_sizes(input->get_layout().size, axis);
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

TEST(add_reorders_gpu, basic_reshape_and_tile) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 2, 2, 1 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::byxf,{ 2, 1, 4, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reshape("reshape", "input", tensor(2, 1, 2, 1), 4));
    topology.add(tile("tile", "reshape", tensor(2, 1, 2, 4)));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f };
    set_values(input, input_vec);
    tile_ref<float>(input, output_ref, tile::along_y, 4);

    network network(engine, topology);
    network.set_input_data("input", input);

    //reorder is required as tile accepts only bfyx format
    EXPECT_EQ(network.get_all_primitive_org_ids().size(), size_t(4));
    auto outputs = network.execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    cldnn::mem_lock<float> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        EXPECT_EQ(output_ptr[i], output_ref_ptr[i]);
    }
}
