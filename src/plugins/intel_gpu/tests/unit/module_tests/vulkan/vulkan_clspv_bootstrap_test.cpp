// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "clspv_bootstrap_spirv.hpp"
#include "intel_gpu/runtime/kernel_builder.hpp"
#include "test_utils.h"
#include "vulkan/vulkan_kernel_interface.hpp"
#include "vulkan/vulkan_pipeline_cache.hpp"
#include "vulkan/vulkan_stream.hpp"

using namespace cldnn;
using namespace cldnn::vulkan;
using namespace tests;

namespace {

constexpr uint32_t workgroup_size_x_spec_id = 0;
constexpr uint32_t workgroup_size_y_spec_id = 1;
constexpr uint32_t workgroup_size_z_spec_id = 2;

}  // namespace

TEST(vulkan_clspv_bootstrap, reflects_canonical_compute_interface) {
    std::vector<uint8_t> spirv(sizeof(clspv_bootstrap_spirv));
    std::memcpy(spirv.data(), clspv_bootstrap_spirv, spirv.size());

    EXPECT_EQ(vulkan_kernel_interface::get_single_entry_point(spirv), "clspv_bootstrap");
    const auto interface = vulkan_kernel_interface::reflect(spirv, "clspv_bootstrap");
    ASSERT_EQ(interface.descriptor_bindings.size(), 2);
    EXPECT_EQ(interface.descriptor_bindings[0], (vulkan_descriptor_binding{0, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}));
    EXPECT_EQ(interface.descriptor_bindings[1], (vulkan_descriptor_binding{0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}));
    EXPECT_EQ(interface.push_constant_size, sizeof(float));
    EXPECT_EQ(interface.local_size_defaults, (std::array<uint32_t, 3>{1, 1, 1}));
    ASSERT_TRUE(interface.local_size_specialization_ids[0].has_value());
    ASSERT_TRUE(interface.local_size_specialization_ids[1].has_value());
    ASSERT_TRUE(interface.local_size_specialization_ids[2].has_value());
    EXPECT_EQ(*interface.local_size_specialization_ids[0], workgroup_size_x_spec_id);
    EXPECT_EQ(*interface.local_size_specialization_ids[1], workgroup_size_y_spec_id);
    EXPECT_EQ(*interface.local_size_specialization_ids[2], workgroup_size_z_spec_id);
    EXPECT_EQ(interface.specialization_ids, (std::vector<uint32_t>{workgroup_size_x_spec_id, workgroup_size_y_spec_id, workgroup_size_z_spec_id}));
}

TEST(vulkan_clspv_bootstrap, executes_host_compiled_opencl_c_through_vulkan_runtime) {
    constexpr size_t element_count = 256;
    constexpr size_t local_size = 64;
    constexpr float increment = 1.25f;

    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    auto command_stream = target_engine->create_stream(get_test_default_config(*target_engine));
    const layout buffer_layout(ov::PartialShape{element_count}, ov::element::f32, format::bfyx);
    auto input = target_engine->allocate_memory(buffer_layout);
    auto output = target_engine->allocate_memory(buffer_layout);

    std::vector<float> input_values(element_count);
    for (size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(index) * 0.25f;
    }
    input->copy_from(*command_stream, input_values.data(), true);

    std::vector<kernel::ptr> kernels;
    target_engine->create_kernel_builder()->build_kernels(clspv_bootstrap_spirv, sizeof(clspv_bootstrap_spirv), KernelFormat::SPIRV, {}, kernels);
    ASSERT_EQ(kernels.size(), 1);
    EXPECT_EQ(kernels.front()->get_id(), "clspv_bootstrap");

    kernel_arguments_desc descriptor;
    descriptor.workGroups.global = {element_count, 1, 1};
    descriptor.workGroups.local = {local_size, 1, 1};
    descriptor.arguments = {
        {argument_desc::Types::INPUT, 0},
        {argument_desc::Types::OUTPUT, 0},
        {argument_desc::Types::SCALAR, 0},
    };
    scalar_desc increment_argument;
    increment_argument.t = scalar_desc::Types::FLOAT32;
    increment_argument.v.f32 = increment;
    descriptor.scalars = {increment_argument};

    kernel_arguments_data arguments;
    arguments.inputs = {input};
    arguments.outputs = {output};

    const vulkan_specialization_constants specialization = {
        {workgroup_size_x_spec_id, static_cast<uint32_t>(local_size)},
        {workgroup_size_y_spec_id, 1},
        {workgroup_size_z_spec_id, 1},
    };
    auto& vulkan_command_stream = dynamic_cast<vulkan_stream&>(*command_stream);
    auto completion = vulkan_command_stream.enqueue_kernel(*kernels.front(), descriptor, arguments, specialization, {}, true);
    ASSERT_NE(completion, nullptr);
    completion->wait();

    std::vector<float> actual(element_count);
    output->copy_to(*command_stream, actual.data(), true);
    for (size_t index = 0; index < actual.size(); ++index) {
        EXPECT_FLOAT_EQ(actual[index], input_values[index] + increment) << "Mismatch at element " << index;
    }
}
