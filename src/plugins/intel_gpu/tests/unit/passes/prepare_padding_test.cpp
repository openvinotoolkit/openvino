// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "data_inst.h"
#include "mvn_inst.h"
#include "col2im_inst.h"
#include "convolution_inst.h"
#include "pass_manager.h"
#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(prepare_padding, groupconv_with_output) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto in_layout = layout{{1, 18, 76, 135}, data_types::f16, format::bfyx};
    auto weights_data = rg.generate_random_5d<ov::float16>(1, 18, 1, 3, 3, -1, 1);
    auto weights_mem = engine.allocate_memory({ {18, 1, 1, 3, 3}, data_types::f16, format::bfzyx});
    set_values(weights_mem, weights_data);

    topology topo;
    topo.add(input_layout("input", in_layout));
    topo.add(data("weight", weights_mem));
    topo.add(convolution("conv", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {2, 2}, true));
    topo.add(reorder("reorder", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topo, config, false, true);
    reorder_factory rf;
    program_wrapper::apply_opt_pass<prepare_padding>(*prog, true);
    const auto& node = prog->get_node("reorder_input_conv");
    auto params = node.get_kernel_impl_params();
    ASSERT_EQ(params->get_output_layout().data_padding._upper_size[2 + 0], 2);
}

TEST(prepare_padding, onednn_int4_fc_aligns_rows_to_4_bytes) {
    auto& engine = get_test_engine();
    auto input_mem = engine.allocate_memory({{32, 9}, data_types::f16, format::bfyx});
    auto weights_mem = engine.allocate_memory({{4, 9}, data_types::u4, format::bfyx});

    topology topo;
    topo.add(input_layout("input", input_mem->get_layout()));
    topo.add(data("weights", weights_mem));
    topo.add(fully_connected("fc", input_info("input"), "weights", "", 2, 2));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{{"fc", {format::bfyx, "", impl_types::onednn}}}));
    auto prog = program::build_program(engine, topo, config, false, true);

    prog->get_node("fc").set_preferred_impl_type(impl_types::onednn);
    program_wrapper::apply_opt_pass<prepare_padding>(*prog, false);

    const auto& input_reorder = prog->get_node("padding_reorder_for_input");
    const auto& weights_reorder = prog->get_node("padding_reorder_for_weights");
    EXPECT_EQ(input_reorder.get_output_layout(false).data_padding._upper_size[1], 1);
    EXPECT_EQ(weights_reorder.get_output_layout(false).data_padding._upper_size[1], 7);
}

TEST(prepare_padding, onednn_int4_fc_does_not_pad_f32_activation) {
    auto& engine = get_test_engine();
    auto weights_mem = engine.allocate_memory({{4, 9}, data_types::u4, format::bfyx});

    topology topo;
    topo.add(input_layout("input", layout{{32, 9}, data_types::f32, format::bfyx}));
    topo.add(data("weights", weights_mem));
    topo.add(fully_connected("fc", input_info("input"), "weights", "", 2, 2));

    auto prog = program::build_program(engine, topo, get_test_default_config(engine), false, true);
    prog->get_node("fc").set_preferred_impl_type(impl_types::onednn);
    program_wrapper::apply_opt_pass<prepare_padding>(*prog, false);

    const auto& fc = prog->get_node("fc");
    EXPECT_EQ(fc.get_dependency(0).id(), "input");
    EXPECT_FALSE(fc.get_input_layout(0).data_padding);
    EXPECT_EQ(fc.get_input_layout(1).data_padding._upper_size[1], 7);
}

TEST(prepare_padding, onednn_int4_fc_does_not_pad_even_k) {
    auto& engine = get_test_engine();
    for (const int64_t k : {10, 12, 14, 16}) {
        SCOPED_TRACE(k);
        auto weights_mem = engine.allocate_memory({{4, k}, data_types::u4, format::bfyx});

        topology topo;
        topo.add(input_layout("input", layout{{32, k}, data_types::f16, format::bfyx}));
        topo.add(data("weights", weights_mem));
        topo.add(fully_connected("fc", input_info("input"), "weights", "", 2, 2));

        auto prog = program::build_program(engine, topo, get_test_default_config(engine), false, true);
        prog->get_node("fc").set_preferred_impl_type(impl_types::onednn);
        program_wrapper::apply_opt_pass<prepare_padding>(*prog, false);

        const auto& fc = prog->get_node("fc");
        EXPECT_EQ(fc.get_dependency(0).id(), "input");
        EXPECT_EQ(fc.get_dependency(1).id(), "weights");
        EXPECT_FALSE(fc.get_input_layout(0).data_padding);
        EXPECT_FALSE(fc.get_input_layout(1).data_padding);
    }
}

TEST(prepare_padding, onednn_int4_fc_aligns_existing_activation_padding) {
    auto& engine = get_test_engine();
    auto weights_mem = engine.allocate_memory({{4, 9}, data_types::u4, format::bfyx});
    for (const int64_t upper_pad : {1, 2, 3}) {
        SCOPED_TRACE(upper_pad);
        const layout in_layout{{32, 9}, data_types::f16, format::bfyx, padding({0}, {0, upper_pad})};

        topology topo;
        topo.add(input_layout("input", in_layout));
        topo.add(data("weights", weights_mem));
        topo.add(fully_connected("fc", input_info("input"), "weights", "", 2, 2));

        auto prog = program::build_program(engine, topo, get_test_default_config(engine), false, true);
        prog->get_node("fc").set_preferred_impl_type(impl_types::onednn);
        program_wrapper::apply_opt_pass<prepare_padding>(*prog, false);

        const auto& fc = prog->get_node("fc");
        if (upper_pad % 2 != 0) {
            EXPECT_EQ(fc.get_dependency(0).id(), "input");
            EXPECT_EQ(fc.get_input_layout(0), in_layout);
        } else {
            EXPECT_EQ(fc.get_dependency(0).id(), "padding_reorder_for_input");
            EXPECT_EQ(fc.get_input_layout(0).data_padding._upper_size[1], upper_pad + 1);
        }
        EXPECT_EQ(fc.get_input_layout(1).data_padding._upper_size[1], 7);
    }
}

TEST(prepare_padding, mvn_conv) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto in_layout = layout{{1, 3, 512, 512}, data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(in_layout);
    auto weights_data = rg.generate_random_4d<ov::float16>(3, 3, 3, 3, -1, 1);
    auto weights_mem = engine.allocate_memory({ {3, 3, 3, 3}, data_types::f16, format::bfyx});
    set_values(weights_mem, weights_data);

    topology topo;
    topo.add(input_layout("input", in_layout));
    topo.add(mvn("mvn", input_info("input"), true, 1e-10f, true, { 2 }));
    topo.add(data("weight", weights_mem));
    topo.add(convolution("conv", input_info("mvn"), "weight", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
    topo.add(reorder("reorder_output", input_info("conv"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topo, config);
    network.set_input_data("input", input);
    EXPECT_NO_THROW(network.execute());

    for (auto& item : network.get_executed_primitives()) {
        auto prim = network.get_primitive(item.first);
        if (prim->get_node().is_type<convolution>() && !prim->get_impl()->is_onednn()) {
            ASSERT_TRUE(has_node(*network.get_program(), "conv_padding_reorder_for_mvn"));
        }
    }
}

TEST(prepare_padding, col2im_conv) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto in_layout = layout{{20, 6272, 301, 1}, data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(in_layout);
    auto weights_data = rg.generate_random_4d<ov::float16>(128, 128, 3, 3, -1, 1);
    auto weights_mem = engine.allocate_memory({ {128, 128, 3, 3}, data_types::f16, format::bfyx});
    set_values(weights_mem, weights_data);

    topology topo;
    topo.add(input_layout("input", in_layout));
    topo.add(col2im("col2im", input_info("input"), {3, 3}, {1, 1}, {3, 3}, {3, 3}, {20, 128, 60, 108}, {7, 7}));
    topo.add(data("weight", weights_mem));
    topo.add(convolution("conv", input_info("col2im"), "weight", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false));
    topo.add(reorder("reorder_output", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topo, config);
    network.set_input_data("input", input);
    EXPECT_NO_THROW(network.execute());

    for (auto& item : network.get_executed_primitives()) {
        auto prim = network.get_primitive(item.first);
        if (prim->get_node().is_type<convolution>() && !prim->get_impl()->is_onednn()) {
            ASSERT_TRUE(has_node(*network.get_program(), "conv_padding_reorder_for_col2im"));
        }
    }
}