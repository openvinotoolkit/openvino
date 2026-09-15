// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common_utils/gpu_execution_plan.hpp"
#include "intel_gpu/runtime/kernel_builder.hpp"
#include "slang_bootstrap_spirv.hpp"
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
    artifact.entry_point = "slang_bootstrap";

    std::vector<kernel::ptr> kernels;
    target_engine.create_kernel_builder()->build_kernels(artifact, kernels);
    OPENVINO_ASSERT(kernels.size() == 1, "[GPU][Vulkan] Slang bootstrap must contain one entry point");
    return kernels.front();
}

scalar_desc make_float_scalar(float value) {
    scalar_desc result;
    result.t = scalar_desc::Types::FLOAT32;
    result.v.f32 = value;
    return result;
}

event::ptr execute_foundation_stages(stream& command_stream,
                                     const gpu_kernel_lifecycle& lifecycle,
                                     const kernel_arguments_desc& local_stage,
                                     const kernel_arguments_data& local_stage_data,
                                     const kernel_arguments_desc& output_stage,
                                     const kernel_arguments_data& output_stage_data) {
    gpu_execution_plan plan(2, {true, false});
    return plan.execute(command_stream, lifecycle, {}, true, [&](size_t dispatch_index) {
        return dispatch_index == 0 ? gpu_dispatch_binding{&local_stage, local_stage_data} : gpu_dispatch_binding{&output_stage, output_stage_data};
    });
}

}  // namespace

TEST(vulkan_execution_plan, two_stage_dispatch_uses_internal_and_local_memory) {
    constexpr size_t element_count = 256;
    constexpr size_t local_size = 64;

    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    auto profiling_config = get_test_default_config(*target_engine);
    profiling_config.set_property(ov::enable_profiling(true));
    auto command_stream = target_engine->create_stream(profiling_config);
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
    lifecycle.emplace_back(build_spirv_kernel(*target_engine, slang_bootstrap_spirv, sizeof(slang_bootstrap_spirv)));
    lifecycle.emplace_back(build_spirv_kernel(*target_engine, slang_bootstrap_spirv, sizeof(slang_bootstrap_spirv)));

    kernel_arguments_desc local_stage;
    local_stage.workGroups.global = {element_count, 1, 1};
    local_stage.workGroups.local = {local_size, 1, 1};
    local_stage.arguments = {
        {argument_desc::Types::INPUT, 0},
        {argument_desc::Types::INTERNAL_BUFFER, 0},
        {argument_desc::Types::SCALAR, 0},
    };
    local_stage.scalars = {make_float_scalar(1.0f)};

    kernel_arguments_desc output_stage;
    output_stage.workGroups.global = {element_count, 1, 1};
    output_stage.workGroups.local = {local_size, 1, 1};
    output_stage.arguments = {
        {argument_desc::Types::INTERNAL_BUFFER, 0},
        {argument_desc::Types::OUTPUT, 0},
        {argument_desc::Types::SCALAR, 0},
    };
    output_stage.scalars = {make_float_scalar(2.0f)};

    kernel_arguments_data local_stage_data;
    local_stage_data.inputs = {input};
    local_stage_data.intermediates = {intermediate};

    kernel_arguments_data output_stage_data;
    output_stage_data.intermediates = {intermediate};
    output_stage_data.outputs = {output};

    auto completion = execute_foundation_stages(*command_stream, lifecycle, local_stage, local_stage_data, output_stage, output_stage_data);
    ASSERT_NE(completion, nullptr);
    completion->wait();

    const auto profiling = completion->get_profiling_info();
    ASSERT_EQ(profiling.size(), 1);
    EXPECT_EQ(profiling.front().stage, instrumentation::profiling_stage::executing);
    EXPECT_TRUE(profiling.front().is_valid_start);
    EXPECT_GT(profiling.front().value->value(), std::chrono::nanoseconds::zero());

    std::vector<float> actual(element_count);
    output->copy_to(*command_stream, actual.data(), true);
    for (size_t index = 0; index < element_count; ++index) {
        const auto group_base = (index / local_size) * local_size;
        const auto adjacent_index = group_base + (index + 2) % local_size;
        const auto expected = input_values[adjacent_index] + 3.0f;
        EXPECT_FLOAT_EQ(actual[index], expected) << "Mismatch at element " << index;
    }
}

TEST(vulkan_execution_plan, transient_slots_survive_pool_reset_tuning_transitions) {
    constexpr size_t local_size = 64;

    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    auto command_stream = target_engine->create_stream(get_test_default_config(*target_engine));

    gpu_kernel_lifecycle lifecycle;
    lifecycle.emplace_back(build_spirv_kernel(*target_engine, slang_bootstrap_spirv, sizeof(slang_bootstrap_spirv)));
    lifecycle.emplace_back(build_spirv_kernel(*target_engine, slang_bootstrap_spirv, sizeof(slang_bootstrap_spirv)));

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
        local_stage.arguments = {
            {argument_desc::Types::INPUT, 0},
            {argument_desc::Types::INTERNAL_BUFFER, 0},
            {argument_desc::Types::SCALAR, 0},
        };
        local_stage.scalars = {make_float_scalar(1.0f)};

        kernel_arguments_desc output_stage;
        output_stage.workGroups.global = {current.element_count, 1, 1};
        output_stage.workGroups.local = {local_size, 1, 1};
        output_stage.arguments = {
            {argument_desc::Types::INTERNAL_BUFFER, 0},
            {argument_desc::Types::OUTPUT, 0},
            {argument_desc::Types::SCALAR, 0},
        };
        output_stage.scalars = {make_float_scalar(2.0f)};

        kernel_arguments_data local_stage_data;
        local_stage_data.inputs = {current.input};
        local_stage_data.intermediates = {current.intermediate};

        kernel_arguments_data output_stage_data;
        output_stage_data.intermediates = {current.intermediate};
        output_stage_data.outputs = {current.output};

        auto completion = execute_foundation_stages(*command_stream, lifecycle, local_stage, local_stage_data, output_stage, output_stage_data);
        ASSERT_NE(completion, nullptr);
        completion->wait();

        std::vector<float> actual(current.element_count);
        current.output->copy_to(*command_stream, actual.data(), true);
        for (size_t index = 0; index < current.element_count; ++index) {
            const auto group_base = (index / local_size) * local_size;
            const auto adjacent_index = group_base + (index + 2) % local_size;
            const auto expected = current.input_values[adjacent_index] + 3.0f;
            EXPECT_FLOAT_EQ(actual[index], expected) << "Mismatch at iteration " << iteration << ", index " << index;
        }
    }
}
