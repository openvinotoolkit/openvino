// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/activation.hpp"

#include <memory>
#include <vector>

using namespace cldnn;
using namespace ::tests;

TEST(record_replay_gpu_test, returns_correct_results) {
    auto& engine = get_test_engine();
    if (engine.type() != engine_types::ze) {
        GTEST_SKIP() << "Record and replay requires a Level Zero engine";
    }

    layout in_layout{{1, 1, 2, 2}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory(in_layout);
    set_values(input_mem, std::vector<float>{-1, -0.1, 0.1, 1});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(activation("clamp", input_info("input"), activation_func::clamp, {-5.f, 5.f}));
    topology.add(activation("relu", input_info("clamp"), activation_func::relu));

    auto exec_and_read_out = [](network& net) {
        auto output = net.execute().at("relu").get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> lock(output, net.get_stream());
        return std::vector<float>(lock.data(), lock.data() + output->count());
    };

    network ref_net(engine, topology, get_test_default_config(engine));
    ref_net.set_input_data("input", input_mem);
    const auto reference = exec_and_read_out(ref_net);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::record_replay(true));
    network net(engine, topology, config);
    net.set_input_data("input", input_mem);

    // First iteration records, subsequent iterations replay
    constexpr size_t iterations = 4;
    for (size_t iter = 0; iter < iterations; ++iter) {
        const auto out = exec_and_read_out(net);
        ASSERT_EQ(out.size(), reference.size());
        for (size_t i = 0; i < reference.size(); ++i)
            ASSERT_FLOAT_EQ(out[i], reference[i]);
    }
}
