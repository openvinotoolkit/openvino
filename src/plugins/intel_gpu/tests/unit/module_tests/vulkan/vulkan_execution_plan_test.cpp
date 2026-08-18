// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common_utils/gpu_execution_plan.hpp"
#include "foundation_stage_local_spirv.hpp"
#include "foundation_stage_output_spirv.hpp"
#include "intel_gpu/runtime/kernel_builder.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace cldnn::vulkan;
using namespace tests;

namespace {

kernel::ptr build_spirv_kernel(engine& target_engine, const uint32_t* spirv, size_t spirv_size) {
    kernel_artifact artifact;
    artifact.payload = spirv;
    artifact.payload_size = spirv_size;
    artifact.format = KernelFormat::SPIRV;
    artifact.entry_point = "main";
    artifact.serialization.portable = true;

    std::vector<kernel::ptr> kernels;
    target_engine.create_kernel_builder()->build_kernels(artifact, kernels);
    OPENVINO_ASSERT(kernels.size() == 1, "[GPU][Vulkan] Synthetic foundation shader must contain one entry point");
    return kernels.front();
}

}  // namespace

TEST(vulkan_execution_plan, two_stage_dispatch_uses_internal_and_local_memory) {
    constexpr size_t element_count = 256;
    constexpr size_t local_size = 64;
    constexpr uint64_t local_memory_bytes = local_size * sizeof(float);

    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    auto command_stream = target_engine->create_stream(get_test_default_config(*target_engine));
    const layout buffer_layout(ov::PartialShape{element_count}, ov::element::f32, format::bfyx);
    auto input = target_engine->allocate_memory(buffer_layout);
    auto intermediate = target_engine->allocate_memory(buffer_layout);
    auto output = target_engine->allocate_memory(buffer_layout);

    std::vector<float> input_values(element_count);
    for (size_t index = 0; index < element_count; ++index) {
        input_values[index] = static_cast<float>(index) * 0.25f;
    }
    input->copy_from(*command_stream, input_values.data(), true);

    gpu_kernel_lifecycle lifecycle;
    lifecycle.emplace_back(build_spirv_kernel(*target_engine, foundation_stage_local_spirv, sizeof(foundation_stage_local_spirv)));
    lifecycle.emplace_back(build_spirv_kernel(*target_engine, foundation_stage_output_spirv, sizeof(foundation_stage_output_spirv)));

    gpu_execution_plan plan(2, {true, false});
    plan[0].local_memory.add({local_memory_bytes, sizeof(float), gpu_local_memory_mapping::specialization_constant, 1, false});

    kernel_arguments_desc local_stage;
    local_stage.workGroups.global = {element_count, 1, 1};
    local_stage.workGroups.local = {local_size, 1, 1};
    local_stage.specialize_local_size_x = true;
    local_stage.arguments = {{argument_desc::Types::INPUT, 0}, {argument_desc::Types::INTERNAL_BUFFER, 0}};
    local_memory_args_desc local_stage_backend_arguments;
    plan[0].local_memory.materialize(local_stage, local_stage_backend_arguments, target_engine->get_device_info().max_local_mem_size);

    kernel_arguments_desc output_stage;
    output_stage.workGroups.global = {element_count, 1, 1};
    output_stage.workGroups.local = {local_size, 1, 1};
    output_stage.specialize_local_size_x = true;
    output_stage.arguments = {{argument_desc::Types::INTERNAL_BUFFER, 0}, {argument_desc::Types::OUTPUT, 0}};

    kernel_arguments_data local_stage_data;
    local_stage_data.inputs = {input};
    local_stage_data.intermediates = {intermediate};
    local_stage_data.local_memory_args = &local_stage_backend_arguments;

    kernel_arguments_data output_stage_data;
    output_stage_data.intermediates = {intermediate};
    output_stage_data.outputs = {output};

    const std::vector<gpu_dispatch_binding> bindings = {{&local_stage, std::move(local_stage_data)}, {&output_stage, std::move(output_stage_data)}};
    auto completion = plan.execute(*command_stream, lifecycle, {}, true, [&](size_t dispatch_index) {
        return bindings.at(dispatch_index);
    });
    ASSERT_NE(completion, nullptr);
    completion->wait();

    std::vector<float> actual(element_count);
    output->copy_to(*command_stream, actual.data(), true);
    for (size_t index = 0; index < element_count; ++index) {
        const auto group_base = (index / local_size) * local_size;
        const auto adjacent_index = group_base + (index + 1) % local_size;
        const auto expected = (input_values[adjacent_index] + 1.0f) * 2.0f;
        EXPECT_FLOAT_EQ(actual[index], expected) << "Mismatch at element " << index;
    }
}

TEST(vulkan_execution_plan, transient_slots_survive_pool_reset_tuning_transitions) {
    constexpr size_t local_size = 64;
    constexpr uint64_t local_memory_bytes = local_size * sizeof(float);

    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    auto command_stream = target_engine->create_stream(get_test_default_config(*target_engine));

    gpu_kernel_lifecycle lifecycle;
    lifecycle.emplace_back(build_spirv_kernel(*target_engine, foundation_stage_local_spirv, sizeof(foundation_stage_local_spirv)));
    lifecycle.emplace_back(build_spirv_kernel(*target_engine, foundation_stage_output_spirv, sizeof(foundation_stage_output_spirv)));

    gpu_execution_plan plan(2, {true, false});
    plan[0].local_memory.add({local_memory_bytes, sizeof(float), gpu_local_memory_mapping::specialization_constant, 1, false});

    struct workload {
        size_t element_count;
        memory::ptr input;
        memory::ptr intermediate;
        memory::ptr output;
        std::vector<float> input_values;
    };

    std::vector<workload> workloads;
    for (const auto element_count : std::array<size_t, 2>{256, 384}) {
        const layout buffer_layout(ov::PartialShape{static_cast<int64_t>(element_count)}, ov::element::f32, format::bfyx);
        workload current{element_count,
                         target_engine->allocate_memory(buffer_layout),
                         target_engine->allocate_memory(buffer_layout),
                         target_engine->allocate_memory(buffer_layout),
                         std::vector<float>(element_count)};
        for (size_t index = 0; index < element_count; ++index) {
            current.input_values[index] = static_cast<float>(index) * 0.25f;
        }
        workloads.push_back(std::move(current));
    }

    for (size_t iteration = 0; iteration < 100; ++iteration) {
        auto& current = workloads[iteration % workloads.size()];
        current.input->copy_from(*command_stream, current.input_values.data(), true);

        kernel_arguments_desc local_stage;
        local_stage.workGroups.global = {current.element_count, 1, 1};
        local_stage.workGroups.local = {local_size, 1, 1};
        local_stage.specialize_local_size_x = true;
        local_stage.arguments = {{argument_desc::Types::INPUT, 0}, {argument_desc::Types::INTERNAL_BUFFER, 0}};
        local_memory_args_desc local_stage_backend_arguments;
        plan[0].local_memory.materialize(local_stage, local_stage_backend_arguments, target_engine->get_device_info().max_local_mem_size);

        kernel_arguments_desc output_stage;
        output_stage.workGroups.global = {current.element_count, 1, 1};
        output_stage.workGroups.local = {local_size, 1, 1};
        output_stage.specialize_local_size_x = true;
        output_stage.arguments = {{argument_desc::Types::INTERNAL_BUFFER, 0}, {argument_desc::Types::OUTPUT, 0}};

        kernel_arguments_data local_stage_data;
        local_stage_data.inputs = {current.input};
        local_stage_data.intermediates = {current.intermediate};
        local_stage_data.local_memory_args = &local_stage_backend_arguments;

        kernel_arguments_data output_stage_data;
        output_stage_data.intermediates = {current.intermediate};
        output_stage_data.outputs = {current.output};

        const std::vector<gpu_dispatch_binding> bindings = {{&local_stage, std::move(local_stage_data)}, {&output_stage, std::move(output_stage_data)}};
        auto completion = plan.execute(*command_stream, lifecycle, {}, true, [&](size_t dispatch_index) {
            return bindings.at(dispatch_index);
        });
        ASSERT_NE(completion, nullptr);
        completion->wait();

        std::vector<float> actual(current.element_count);
        current.output->copy_to(*command_stream, actual.data(), true);
        for (size_t index = 0; index < current.element_count; ++index) {
            const auto group_base = (index / local_size) * local_size;
            const auto adjacent_index = group_base + (index + 1) % local_size;
            const auto expected = (current.input_values[adjacent_index] + 1.0f) * 2.0f;
            EXPECT_FLOAT_EQ(actual[index], expected) << "Mismatch at iteration " << iteration << ", index " << index;
        }
    }
}
