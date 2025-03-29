// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/col2im.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

static void test_col2im_output(bool is_caching_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 12, 9, 1 } });

    set_values(input, {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
        4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
        6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f,
        7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f,
        8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f,
        9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f,
    });
    ov::Shape output_size({4, 4});
    ov::Shape kernel_size({2, 2});

    topology topology;
    topology.add(cldnn::input_layout("input", input->get_layout()));
    topology.add(cldnn::reorder("reorder_input", input_info("input"), cldnn::layout(data_types::f16, format::byxf, { 1, 12, 9, 1 })));
    topology.add(cldnn::col2im("col2im", input_info("reorder_input"), {1, 1}, {1, 1}, {0, 0}, {0, 0}, output_size, kernel_size));
    topology.add(cldnn::activation("activate", input_info("col2im"), cldnn::activation_func::relu_negative_slope, {0.25f, 0.f}));
    topology.add(cldnn::reorder("convert:output", input_info("activate"), format::any, data_types::f32, {}, reorder_mean_mode::subtract, padding(), true));
    topology.add(cldnn::reorder("result:output/sink_port_0", input_info("convert:output"), format::bfyx, data_types::f32, {}, reorder_mean_mode::subtract, padding(), false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();

    auto output = outputs.at("result:output/sink_port_0").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.0f, 2.0f, 2.0f, 1.0f,
        3.0f, 8.0f, 8.0f, 5.0f,
        3.0f, 8.0f, 8.0f, 5.0f,
        3.0f, 6.0f, 6.0f, 3.0f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(col2im_gpu_simple, fp32_input_fp32_output) {
    test_col2im_output(false);
}
