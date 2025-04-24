// Copyright (C) 2018-2025 Intel Corporation
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
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));

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

    topology.add(cldnn::convolution("conv1", input_info("input"), "weights1", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(cldnn::reorder("reorder", input_info("input"), cldnn::layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(cldnn::convolution("conv2", input_info("reorder"), "weights2", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    topology.add(cldnn::concatenation("concat", { input_info("conv1"), input_info("conv2") }, 1));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    //concatenation accepts inputs in different formats, so no reorders should be added here
    ASSERT_EQ(network.get_all_primitive_org_ids().size(), size_t(7));
    auto outputs = network.execute();

    float expected_out[] = { 6.34f, 1.34f, 6.86f, 1.46f };
    float epsilon = 1e-3f;

    for (auto& it : outputs) {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        for (size_t cntr = 0; cntr < 2 * 2; cntr++) {
            ASSERT_NEAR(expected_out[cntr], output[cntr], epsilon);
        }
    }
}

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
    const auto& input_layout = input->get_layout();

    auto sizes = get_sizes(input_layout, axis, input_layout.get_rank());
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

template <typename T>
void test_add_reorders_gpu_basic_reshape_and_tile(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 2, 2, 1 } });
    auto output_ref = engine.allocate_memory({ data_types::f32, format::byxf,{ 2, 1, 4, 2 } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reshape("reshape", input_info("input"), tensor(2, 1, 2, 1)));
    topology.add(tile("tile", input_info("reshape"), std::vector<int64_t>{ 1, 1, 4, 1 }));

    std::vector<T> input_vec = { 1.f, 0.f, 5.f, 1.5f };
    set_values(input, input_vec);
    tile_ref<T>(input, output_ref, 2, 4);

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    //reorder is required as tile accepts only bfyx format
    ASSERT_EQ(network->get_all_primitive_org_ids().size(), size_t(4));
    auto outputs = network->execute();

    auto output = outputs.at("tile").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());
    cldnn::mem_lock<T> output_ref_ptr(output_ref, get_test_stream());

    for (unsigned int i = 0; i < output_ref->count(); ++i) {
        ASSERT_EQ(output_ptr[i], output_ref_ptr[i]);
    }
}

TEST(add_reorders_gpu, basic_reshape_and_tile) {
    test_add_reorders_gpu_basic_reshape_and_tile<float>(false);
}

TEST(export_import_add_reorders_gpu, basic_reshape_and_tile) {
    test_add_reorders_gpu_basic_reshape_and_tile<float>(true);
}
