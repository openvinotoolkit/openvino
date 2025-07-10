// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reshape.hpp>

using namespace cldnn;
using namespace ::tests;

//We expect additional reorder to be added in between "weights1" and "reshape1".
//This situation should be handled properly by propagate constants optimization phase
template <typename T>
void test_copy_dependecies_from_nodes(bool is_caching_test) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto input = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 2 } });
    auto weights1 = engine.allocate_memory({ data_types::f16, format::yxfb,{ 1, 1, 2, 1 } });
    auto weights2 = engine.allocate_memory({ data_types::f32, format::byxf,{ 1, 1, 1, 2 } });

    set_values(input, { T(1.1f), T(1.2f), T(1.3f), T(1.4f) });
    set_values(weights1, { T(2.1f), T(3.1f) });
    set_values(weights2, { 1.1f, 0.1f });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("weights1", weights1));
    topology.add(data("weights2", weights2));
    topology.add(reshape("reshape1", input_info("weights1"), tensor(spatial(1, 2))));
    topology.add(reorder("reorder2", input_info("input"), layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(reorder("reorder1", input_info("reshape1"), layout(data_types::f32, format::byxf, tensor(4))));
    topology.add(concatenation("concat", { input_info("reorder1"), input_info("weights2") }, 3));
    topology.add(convolution("conv2", input_info("reorder2"), "concat", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);

    auto outputs = network->execute();

    float epsilon = 1e-2f;
    for (auto& it : outputs) {
        cldnn::mem_lock<float> output(it.second.get_memory(), get_test_stream());
        ASSERT_NEAR(7.8f, output[0], epsilon);
    }
}

TEST(propagate_constants, copy_dependecies_from_nodes) {
    test_copy_dependecies_from_nodes<ov::float16>(false);
}

TEST(propagate_constants, copy_dependecies_from_nodes_cached) {
    test_copy_dependecies_from_nodes<ov::float16>(true);
}

TEST(propagate_constants, permute_1_0_reorder_fc) {
    auto& engine = get_test_engine();

    auto input2_layout_dyn = layout{ ov::PartialShape{ -1, 32 }, data_types::f16, format::bfyx };

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto input2 = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 32, 2 }, data_types::f32, format::bfyx });

    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_2d<ov::float16>(2, 32, -1, 1);
    auto input2_data = rg.generate_random_2d<ov::float16>(2, 32, -1, -1);
    auto weights_data = rg.generate_random_2d<float>(32, 2, -1, 1);

    set_values(input, flatten_2d(format::bfyx, input_data));
    set_values(input2, input2_data);
    set_values(weights, flatten_2d(format::bfyx, weights_data));

    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("input2", input2_layout_dyn),
        data("weights", weights),
        permute("permute_test", input_info("weights"), {1, 0}),
        reorder("reorder_dt", input_info("permute_test"), format::any, data_types::f16, std::vector<float>()),
        fully_connected("fc1", input_info("input"), { "reorder_dt" }, "", data_types::f16),
        fully_connected("fc2", input_info("input2"), { "reorder_dt" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc1", fc_impl} }));
    }

    cldnn::network network(engine, topology, config);
    network.set_input_data("input", input);
    network.set_input_data("input2", input2);

    auto outputs = network.execute();
    auto output = outputs.at("fc1").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());

    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(false));
    config_ref.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
        config_ref.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc1", fc_impl} }));
    }

    cldnn::network network_ref(engine, topology, config_ref);
    network_ref.set_input_data("input", input);
    network_ref.set_input_data("input2", input2);

    auto outputs_ref = network_ref.execute();
    auto output_ref = outputs_ref.at("fc1").get_memory();
    cldnn::mem_lock<ov::float16> output_ref_ptr(output_ref, get_test_stream());

    for (size_t i = 0; i < output_ref_ptr.size(); ++i) {
        ASSERT_EQ(output_ptr[i], output_ref_ptr[i]);
    }
}
