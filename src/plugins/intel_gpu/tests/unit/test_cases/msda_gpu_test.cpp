// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/msda.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include "msda_inst.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

TEST(msda_gpu, dynamic) {
    auto& engine = get_test_engine();

    ov::PartialShape dyn_shape_4d{1, 4, 2, 2};
    layout dyn_layout{ov::PartialShape::dynamic(dyn_shape_4d.size()), data_types::f32, format::bfyx};

    // 1. data_value (1, 4, 2, 2)
    auto data_value = engine.allocate_memory({dyn_shape_4d, data_types::f32, format::bfyx});
    set_values(data_value, std::vector<float>{
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,
        1.3f, 1.4f, 1.5f, 1.6f
    });

    // 2. data_spatial_shapes (1,1,1,4) → shape=[4,4,2,2] flat
    layout shapes_layout{ov::PartialShape{1, 1, 1, 4}, data_types::i32, format::bfyx};
    auto spatial_shapes = engine.allocate_memory(shapes_layout);
    set_values(spatial_shapes, {4, 4, 2, 2});

    // 3. data_level_start_idx (1,1,1,2)
    layout start_idx_layout{ov::PartialShape{1, 1, 1, 2}, data_types::i32, format::bfyx};
    auto level_start = engine.allocate_memory(start_idx_layout);
    set_values(level_start, {0, 16});

    // 4. data_sampling_loc (1,1,1,16)
    layout sampling_loc_layout{ov::PartialShape{1, 1, 1, 16}, data_types::f32, format::bfyx};
    auto sampling_loc = engine.allocate_memory(sampling_loc_layout);
    set_values(sampling_loc, std::vector<float>(16, 0.5f));

    // 5. data_attn_weight (1,1,1,8)
    layout attn_weight_layout{ov::PartialShape{1, 1, 1, 8}, data_types::f32, format::bfyx};
    auto attn_weight = engine.allocate_memory(attn_weight_layout);
    set_values(attn_weight, std::vector<float>(8, 1.0f));

    // --------------------------- Topology & Primitive ---------------------------
    topology topology;
    topology.add(input_layout("data_value", dyn_layout));
    topology.add(input_layout("data_spatial_shapes", shapes_layout));
    topology.add(input_layout("data_level_start_idx", start_idx_layout));
    topology.add(input_layout("data_sampling_loc", sampling_loc_layout));
    topology.add(input_layout("data_attn_weight", attn_weight_layout));

    topology.add(msda("msda", {
        input_info("data_value"),
        input_info("data_spatial_shapes"),
        input_info("data_level_start_idx"),
        input_info("data_sampling_loc"),
        input_info("data_attn_weight")
    }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = cldnn::network(engine, topology, config);

    network.set_input_data("data_value", data_value);
    network.set_input_data("data_spatial_shapes", spatial_shapes);
    network.set_input_data("data_level_start_idx", level_start);
    network.set_input_data("data_sampling_loc", sampling_loc);
    network.set_input_data("data_attn_weight", attn_weight);

    auto inst = network.get_primitive("msda");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    // auto outputs = network.execute();
    // ASSERT_EQ(outputs.size(), size_t(1));
    // ASSERT_EQ(outputs.begin()->first, "msda");

    // auto output_memory = outputs.at("msda").get_memory();
    // auto output_layout = output_memory->get_layout();

    // cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    // ASSERT_EQ(output_layout.format, format::bfyx);
    // EXPECT_GT(output_layout.get_linear_size(), 0u);
}