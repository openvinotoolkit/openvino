// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "permute_inst.h"
#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace skip_permute_tests {

struct skip_permute_params {
    layout input_layout_static;
    std::vector<uint16_t> permute_order;
    bool expected_result1;
    bool expected_result2;
};

class skip_permute_at_runtime_test : public testing::TestWithParam<skip_permute_params> {};

TEST_P(skip_permute_at_runtime_test, runtime_skip) {
    auto p = GetParam();
    auto& engine = get_test_engine();
    auto rank = p.input_layout_static.get_partial_shape().size();
    auto input_layout_dynamic = layout {ov::PartialShape::dynamic(rank), data_types::f16, format::get_default_format(rank)};
    topology topology(input_layout("input", input_layout_dynamic),
                      permute("permute", input_info("input"), p.permute_order),
                      reorder("reorder", input_info("permute"), format::get_default_format(rank), data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto permute_inst = network.get_primitive("permute");
    ASSERT_EQ(permute_inst->can_be_optimized(), p.expected_result1);

    auto input_mem = engine.allocate_memory(p.input_layout_static);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();
    outputs.begin()->second.get_memory();

    ASSERT_EQ(permute_inst->can_be_optimized(), p.expected_result2);
}

INSTANTIATE_TEST_SUITE_P(smoke, skip_permute_at_runtime_test,
    testing::ValuesIn(std::vector<skip_permute_params> {
        { layout{ov::PartialShape{8,8}, data_types::f16, format::bfyx}, {1, 0}, true, false },
        { layout{ov::PartialShape{8,1}, data_types::f16, format::bfyx}, {1, 0}, true, true },
        { layout{ov::PartialShape{8, 2, 8}, data_types::f16, format::bfyx}, {2, 1, 0}, true, false },
        { layout{ov::PartialShape{8, 2, 1}, data_types::f16, format::bfyx}, {2, 1, 0}, true, false },
        { layout{ov::PartialShape{1, 12, 1}, data_types::f16, format::bfyx}, {2, 1, 0}, true, true },
        { layout{ov::PartialShape{2, 3, 1, 14}, data_types::f16, format::bfyx}, {1, 0, 2, 3}, true, false },
        { layout{ov::PartialShape{1, 3, 1, 14}, data_types::f16, format::bfyx}, {1, 0, 2, 3}, true, true },
        { layout{ov::PartialShape{20, 1, 30, 1}, data_types::f16, format::bfyx}, {1, 0, 3, 2}, true, true},
        { layout{ov::PartialShape{12, 3, 1, 14}, data_types::f16, format::bfyx}, {0, 2, 1, 3}, true, true },
        { layout{ov::PartialShape{12, 3, 2, 14}, data_types::f16, format::bfyx}, {0, 2, 1, 3}, true, false },
        { layout{ov::PartialShape{12, 1, 1, 14}, data_types::f16, format::bfyx}, {0, 3, 1, 2}, true, true },
        { layout{ov::PartialShape{12, 1, 1, 14}, data_types::f16, format::bfyx}, {0, 3, 1, 2}, true, true },
        { layout{ov::PartialShape{1, 1, 1, 14}, data_types::f16, format::bfyx}, {0, 3, 1, 2}, true, true },
        { layout{ov::PartialShape{1, 3, 2, 14}, data_types::f16, format::bfyx}, {0, 3, 1, 2}, true, false },
        { layout{ov::PartialShape{1, 1, 4, 14}, data_types::f16, format::bfyx}, {2, 0, 1, 3}, true, true },
        { layout{ov::PartialShape{1, 4, 4, 1}, data_types::f16, format::bfyx}, {2, 0, 1, 3}, true, false },
        { layout{ov::PartialShape{1, 10, 1, 1, 11}, data_types::f16, format::bfzyx}, {3, 2, 1, 0, 4}, true, true },
        { layout{ov::PartialShape{1, 10, 2, 1, 10}, data_types::f16, format::bfzyx}, {3, 2, 1, 0, 4}, true, false },
        { layout{ov::PartialShape{1, 4, 1, 3, 4, 1}, data_types::f16, format::bfwzyx}, {0, 2, 1, 3, 5, 4}, true, true },
        { layout{ov::PartialShape{1, 4, 2, 3, 4, 1}, data_types::f16, format::bfwzyx}, {0, 2, 1, 3, 5, 4}, true, false },
        { layout{ov::PartialShape{1, 1, 1, 1, 4, 1}, data_types::f16, format::bfwzyx}, {4, 5, 2, 3, 0, 1}, true, true },
        { layout{ov::PartialShape{1, 1, 1, 1, 4, 2}, data_types::f16, format::bfwzyx}, {4, 5, 2, 3, 0, 1}, true, true },
        { layout{ov::PartialShape{1, 1, 3, 1, 4, 2}, data_types::f16, format::bfwzyx}, {4, 5, 2, 3, 0, 1}, true, false },
    }));

TEST(skip_permute_at_runtime, dynamic_remote_output_does_not_alias_producer) {
    auto& engine = get_test_engine();
    const auto dynamic_layout = layout{ov::PartialShape::dynamic(3), data_types::f32, format::bfyx};
    topology topology(input_layout("input", dynamic_layout),
                      activation("producer", input_info("input"), activation_func::relu),
                      permute("permute", input_info("producer"), {0, 2, 1}),
                      reorder("output", input_info("permute"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto permute_inst = network.get_primitive("permute");
    ASSERT_TRUE(permute_inst->can_be_optimized());
    ASSERT_TRUE(permute_inst->get_node().is_runtime_skippable());

    auto input_mem = engine.allocate_memory({{1, 2, 3}, data_types::f32, format::bfyx});
    set_values(input_mem, std::vector<float>{0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    auto output_remote_mem = engine.allocate_memory({{1, 3, 2}, data_types::f32, format::bfyx});

    network.set_input_data("input", input_mem);
    network.set_output_memory("output", output_remote_mem, true);
    auto outputs = network.execute();
    auto output_mem = outputs.at("output").get_memory();

    ASSERT_FALSE(permute_inst->can_be_optimized());
    ASSERT_EQ(output_mem->buffer_ptr(), output_remote_mem->buffer_ptr());
    ASSERT_NE(network.get_primitive("producer")->output_memory_ptr()->buffer_ptr(), output_remote_mem->buffer_ptr());

    const std::vector<float> expected{0.f, 3.f, 1.f, 4.f, 2.f, 5.f};
    mem_lock<float> output_ptr(output_mem, get_test_stream());
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(output_ptr[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(skip_permute_at_runtime, dynamic_non_remote_output_preserves_optimized_chain) {
    auto& engine = get_test_engine();
    const auto dynamic_layout = layout{ov::PartialShape{1, ov::Dimension(1, 3), ov::Dimension(1, 3)}, data_types::f32, format::bfyx};
    topology topology(input_layout("input", dynamic_layout),
                      activation("producer", input_info("input"), activation_func::relu),
                      permute("permute", input_info("producer"), {0, 2, 1}),
                      reorder("output", input_info("permute"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto output_mem = engine.allocate_memory({{1, 3, 1}, data_types::f32, format::bfyx});
    network.set_output_memory("output", output_mem);

    ASSERT_EQ(network.get_primitive("producer")->output_memory_ptr()->buffer_ptr(), output_mem->buffer_ptr());

    auto input_mem = engine.allocate_memory({{1, 1, 3}, data_types::f32, format::bfyx});
    set_values(input_mem, std::vector<float>{0.f, 1.f, 2.f});
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    ASSERT_TRUE(network.get_primitive("permute")->can_be_optimized());
    ASSERT_EQ(outputs.at("output").get_memory()->buffer_ptr(), output_mem->buffer_ptr());

    mem_lock<float> output_ptr(output_mem, get_test_stream());
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(output_ptr[i], static_cast<float>(i)) << "Mismatch at index " << i;
    }
}

TEST(skip_permute_at_runtime, dynamic_output_chain_caches_are_isolated_by_remote_binding) {
    auto& engine = get_test_engine();
    const auto dynamic_layout = layout{ov::PartialShape{1, ov::Dimension(1, 3), ov::Dimension(1, 3)}, data_types::f32, format::bfyx};
    topology topology(input_layout("input", dynamic_layout),
                      activation("producer", input_info("input"), activation_func::relu),
                      permute("permute", input_info("producer"), {0, 2, 1}),
                      reorder("output", input_info("permute"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto first_output = engine.allocate_memory({{1, 3, 1}, data_types::f32, format::bfyx});
    network.set_output_memory("output", first_output);
    ASSERT_EQ(network.get_primitive("producer")->output_memory_ptr()->buffer_ptr(), first_output->buffer_ptr());

    auto remote_output = engine.allocate_memory({{1, 3, 1}, data_types::f32, format::bfyx});
    network.set_output_memory("output", remote_output, true);
    ASSERT_NE(network.get_primitive("producer")->output_memory_ptr()->buffer_ptr(), remote_output->buffer_ptr());

    auto final_output = engine.allocate_memory({{1, 3, 1}, data_types::f32, format::bfyx});
    network.set_output_memory("output", final_output);
    ASSERT_EQ(network.get_primitive("producer")->output_memory_ptr()->buffer_ptr(), final_output->buffer_ptr());

    auto input_mem = engine.allocate_memory({{1, 1, 3}, data_types::f32, format::bfyx});
    set_values(input_mem, std::vector<float>{6.f, 7.f, 8.f});
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    ASSERT_TRUE(network.get_primitive("permute")->can_be_optimized());
    ASSERT_EQ(outputs.at("output").get_memory()->buffer_ptr(), final_output->buffer_ptr());

    mem_lock<float> output_ptr(final_output, get_test_stream());
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(output_ptr[i], static_cast<float>(i + 6)) << "Mismatch at index " << i;
    }
}
}  // namespace skip_permute_tests
