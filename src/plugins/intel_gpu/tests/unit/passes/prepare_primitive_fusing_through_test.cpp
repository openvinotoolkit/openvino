// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "quantize_inst.h"
#include "reshape_inst.h"
#include "fully_connected_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

// Positive test: per-channel quantize should fuse through reshape chain when shapes match (round-trip).
// Topology: Input[4,3,1,1] -> FC[4,3,1,1] -> Reshape1[1,3,2,2] -> Reshape2[4,3,1,1] -> Quantize(per-ch) -> Reorder
// After pass: Quantize should be moved next to FC (quantize's dep 0 == "fc").
TEST(prepare_primitive_fusing_through, fuse_quantize_per_channel_through_reshape_chain) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape{4, 3, 1, 1}, data_types::f32, format::bfyx};
    auto weights_mem = engine.allocate_memory({ov::PartialShape{3, 3}, data_types::f32, format::bfyx});
    auto in_lo_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});
    auto in_hi_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});
    auto out_lo_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});
    auto out_hi_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});

    set_values<float>(in_lo_mem, {0.f, 0.f, 0.f});
    set_values<float>(in_hi_mem, {1.f, 2.f, 3.f});
    set_values<float>(out_lo_mem, {0.f, 0.f, 0.f});
    set_values<float>(out_hi_mem, {255.f, 255.f, 255.f});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weights", weights_mem));
    topology.add(fully_connected("fc", input_info("input"), "weights"));
    topology.add(reshape("reshape1", input_info("fc"), tensor(1, 3, 2, 2)));
    topology.add(reshape("reshape2", input_info("reshape1"), tensor(4, 3, 1, 1)));
    topology.add(data("in_lo", in_lo_mem));
    topology.add(data("in_hi", in_hi_mem));
    topology.add(data("out_lo", out_lo_mem));
    topology.add(data("out_hi", out_hi_mem));
    topology.add(quantize("quantize", input_info("reshape2"),
                          input_info("in_lo"), input_info("in_hi"),
                          input_info("out_lo"), input_info("out_hi"), 256, data_types::u8));
    topology.add(reorder("output", input_info("quantize"), format::bfyx, data_types::f32));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_quantization>(*prog);
    program_wrapper::apply_opt_pass<prepare_primitive_fusing_through>(*prog);

    ASSERT_NE(prog, nullptr);
    auto& quant_node = prog->get_node("quantize");
    // After the pass, quantize should have been moved next to FC
    ASSERT_EQ(quant_node.get_dependency(0).id(), "fc");
}

// Negative test: per-channel quantize should NOT fuse through reshape when shapes don't match.
// Topology: Input[4,3,1,1] -> FC[4,3,1,1] -> Reshape1[1,3,2,2] -> Quantize(per-ch) -> Reorder
// FC output is {4,3,1,1} but quantize input is {1,3,2,2} — mismatch blocks fusing.
TEST(prepare_primitive_fusing_through, dont_fuse_quantize_per_channel_shape_mismatch) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape{4, 3, 1, 1}, data_types::f32, format::bfyx};
    auto weights_mem = engine.allocate_memory({ov::PartialShape{3, 3}, data_types::f32, format::bfyx});
    auto in_lo_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});
    auto in_hi_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});
    auto out_lo_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});
    auto out_hi_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});

    set_values<float>(in_lo_mem, {0.f, 0.f, 0.f});
    set_values<float>(in_hi_mem, {1.f, 2.f, 3.f});
    set_values<float>(out_lo_mem, {0.f, 0.f, 0.f});
    set_values<float>(out_hi_mem, {255.f, 255.f, 255.f});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weights", weights_mem));
    topology.add(fully_connected("fc", input_info("input"), "weights"));
    topology.add(reshape("reshape1", input_info("fc"), tensor(1, 3, 2, 2)));
    topology.add(data("in_lo", in_lo_mem));
    topology.add(data("in_hi", in_hi_mem));
    topology.add(data("out_lo", out_lo_mem));
    topology.add(data("out_hi", out_hi_mem));
    topology.add(quantize("quantize", input_info("reshape1"),
                          input_info("in_lo"), input_info("in_hi"),
                          input_info("out_lo"), input_info("out_hi"), 256, data_types::u8));
    topology.add(reorder("output", input_info("quantize"), format::bfyx, data_types::f32));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_quantization>(*prog);
    program_wrapper::apply_opt_pass<prepare_primitive_fusing_through>(*prog);

    ASSERT_NE(prog, nullptr);
    auto& quant_node = prog->get_node("quantize");
    // Quantize should NOT have moved — its input should still be reshape1
    ASSERT_EQ(quant_node.get_dependency(0).id(), "reshape1");
}

// Negative test: per-channel quantize should NOT fuse through when shapes are dynamic.
// Topology: Input[-1,3,1,1] -> FC[-1,3,1,1] -> Reshape[-1,3,1,1] -> Quantize(per-ch) -> Reorder
// FC output and quantize input have the same dynamic shape,
// but fusing is conservatively blocked since shape equality cannot be guaranteed at graph compilation time.
TEST(prepare_primitive_fusing_through, dont_fuse_quantize_per_channel_dynamic_shapes) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape{-1, 3, 1, 1}, data_types::f32, format::bfyx};
    auto weights_mem = engine.allocate_memory({ov::PartialShape{1, 1}, data_types::f32, format::bfyx});
    auto in_lo_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});
    auto in_hi_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});
    auto out_lo_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});
    auto out_hi_mem = engine.allocate_memory({ov::PartialShape{1, 3, 1, 1}, data_types::f32, format::bfyx});

    set_values<float>(in_lo_mem, {0.f, 0.f, 0.f});
    set_values<float>(in_hi_mem, {1.f, 2.f, 3.f});
    set_values<float>(out_lo_mem, {0.f, 0.f, 0.f});
    set_values<float>(out_hi_mem, {255.f, 255.f, 255.f});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weights", weights_mem));
    topology.add(fully_connected("fc", input_info("input"), "weights"));
    topology.add(reshape("reshape1", input_info("fc"), true, std::vector<int64_t>{0, 0, 0, 0}, ov::PartialShape{-1, 3, 1, 1}));
    topology.add(data("in_lo", in_lo_mem));
    topology.add(data("in_hi", in_hi_mem));
    topology.add(data("out_lo", out_lo_mem));
    topology.add(data("out_hi", out_hi_mem));
    topology.add(quantize("quantize",
                          input_info("reshape1"),
                          input_info("in_lo"),
                          input_info("in_hi"),
                          input_info("out_lo"),
                          input_info("out_hi"),
                          256,
                          data_types::u8));
    topology.add(reorder("output", input_info("quantize"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_quantization>(*prog);
    program_wrapper::apply_opt_pass<prepare_primitive_fusing_through>(*prog);

    ASSERT_NE(prog, nullptr);
    auto& quant_node = prog->get_node("quantize");
    // With dynamic shapes, quantize should NOT have moved — stays after reshape
    ASSERT_EQ(quant_node.get_dependency(0).id(), "reshape1");
}
