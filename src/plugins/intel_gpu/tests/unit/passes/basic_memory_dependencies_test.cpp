// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "convolution_inst.h"
#include "reshape_inst.h"
#include "reorder_inst.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

// Verify that end-to-end inference produces correct results when a fused
// eltwise-sum residual passes through an optimized-out reshape (in-place chain).
// Compares GPU output with memory pool enabled vs disabled.
TEST(basic_memory_dependencies, inplace_chain_eltwise_sum_correctness) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto in_layout = layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx};
    auto weight_layout = layout{ov::PartialShape({16, 16, 1, 1}), data_types::f16, format::bfyx};
    auto weight_mem = engine.allocate_memory(weight_layout);

    tests::random_generator rg;
    rg.set_seed("basic_memory_dependencies_correctness");
    {
        auto rnd = rg.generate_random_1d<ov::float16>(weight_layout.count(), -1, 1);
        set_values(weight_mem, rnd);
    }

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight_mem));
    topology.add(convolution("conv1", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reshape("reshape", input_info("conv1"), tensor(1, 16, 32, 32)));
    topology.add(convolution("conv2", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("conv3", input_info("conv2"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltwise", input_info("conv3"), input_info("reshape"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltwise"), format::bfyx, data_types::f32));

    auto input_mem = engine.allocate_memory(in_layout);
    {
        auto rnd = rg.generate_random_1d<ov::float16>(in_layout.count(), -1, 1);
        set_values(input_mem, rnd);
    }

    // Reference: optimized but with memory pool disabled
    ExecutionConfig config_ref = get_test_default_config(engine);
    config_ref.set_property(ov::intel_gpu::optimize_data(true));
    config_ref.set_property(ov::intel_gpu::enable_memory_pool(false));
    network net_ref(engine, topology, config_ref);
    net_ref.set_input_data("input", input_mem);
    auto outputs_ref = net_ref.execute();

    // Test: optimized with memory pool enabled
    ExecutionConfig config_opt = get_test_default_config(engine);
    config_opt.set_property(ov::intel_gpu::optimize_data(true));
    config_opt.set_property(ov::intel_gpu::enable_memory_pool(true));
    network net_opt(engine, topology, config_opt);
    net_opt.set_input_data("input", input_mem);
    auto outputs_opt = net_opt.execute();

    auto out_ref = outputs_ref.at("reorder").get_memory();
    auto out_opt = outputs_opt.at("reorder").get_memory();

    ASSERT_NE(out_ref, nullptr);
    ASSERT_NE(out_opt, nullptr);
    ASSERT_EQ(out_ref->count(), out_opt->count());

    cldnn::mem_lock<float> ref_ptr(out_ref, get_test_stream());
    cldnn::mem_lock<float> opt_ptr(out_opt, get_test_stream());

    const float tolerance = 1e-3f;
    for (size_t i = 0; i < out_ref->count(); i++) {
        ASSERT_NEAR(ref_ptr[i], opt_ptr[i], tolerance)
            << "Mismatch at index " << i
            << ": ref=" << ref_ptr[i] << " opt=" << opt_ptr[i]
            << "\nThis may indicate memory pool corruption from buffer reuse "
               "through an in-place chain";
    }
}
