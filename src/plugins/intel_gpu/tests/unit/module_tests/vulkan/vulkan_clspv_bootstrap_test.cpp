// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "clspv_bootstrap_spirv.hpp"
#include "intel_gpu/runtime/kernel_builder.hpp"
#include "openvino/core/except.hpp"
#include "test_utils.h"
#include "vulkan/vulkan_clspv_compiler.hpp"
#include "vulkan/vulkan_device.hpp"
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

TEST(vulkan_clspv_bootstrap, compiles_source_with_storage_buffers_and_scalar_arguments) {
    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    const auto& device = dynamic_cast<const vulkan_device&>(*target_engine->get_device());
    constexpr char source[] = R"(
        kernel void compiler_contract(global const float* input, global float* output, float increment) {
            const size_t index = get_global_id(0);
            output[index] = input[index] + increment;
        }
    )";
    const auto compilation = vulkan_clspv_compiler{}.compile(source, {}, "compiler_contract", device);
    const auto interface = vulkan_kernel_interface::reflect(compilation.spirv, "compiler_contract");
    ASSERT_EQ(interface.descriptor_bindings.size(), 2);
    EXPECT_EQ(interface.descriptor_bindings[0], (vulkan_descriptor_binding{0, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}));
    EXPECT_EQ(interface.descriptor_bindings[1], (vulkan_descriptor_binding{0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}));
    EXPECT_EQ(interface.push_constant_size, sizeof(float));
    for (const auto& specialization : interface.local_size_specialization_ids) {
        EXPECT_TRUE(specialization.has_value());
    }
}

TEST(vulkan_clspv_bootstrap, propagates_source_compilation_diagnostics) {
    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    const auto& device = dynamic_cast<const vulkan_device&>(*target_engine->get_device());
    try {
        (void)vulkan_clspv_compiler{}.compile("kernel void invalid_source( {", {}, "invalid_source", device);
        FAIL() << "Malformed OpenCL C must not compile";
    } catch (const ov::Exception& exception) {
        const std::string message = exception.what();
        EXPECT_NE(message.find("invalid_source"), std::string::npos);
        EXPECT_NE(message.find("CLSPV failed to compile"), std::string::npos);
        EXPECT_NE(message.find("error:"), std::string::npos);
    }
}

TEST(vulkan_clspv_bootstrap, rejects_image_descriptor_abi) {
    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    const auto& device = dynamic_cast<const vulkan_device&>(*target_engine->get_device());
    constexpr char source[] = R"(
        kernel void image_contract(write_only image2d_t output) {
            write_imagef(output, (int2)(0, 0), (float4)(1.0f));
        }
    )";
    try {
        (void)vulkan_clspv_compiler{}.compile(source, {}, "image_contract", device);
        FAIL() << "Images are outside the canonical buffer-only ABI";
    } catch (const ov::Exception& exception) {
        EXPECT_NE(std::string(exception.what()).find("storage-buffer descriptors only"), std::string::npos);
    }
}

TEST(vulkan_clspv_bootstrap, rejects_empty_compilation_inputs) {
    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    const auto& device = dynamic_cast<const vulkan_device&>(*target_engine->get_device());
    const vulkan_clspv_compiler compiler;
    EXPECT_THROW((void)compiler.compile({}, {}, "empty_source", device), ov::Exception);
    EXPECT_THROW((void)compiler.compile("kernel void empty() {}", {}, {}, device), ov::Exception);
}

TEST(vulkan_clspv_bootstrap, preserves_semantic_options_without_intel_driver_flags) {
    const auto options = vulkan_clspv_compiler::canonical_options(
        "-cl-mad-enable -cl-intel-256-GRF-per-thread -cl-intel-greater-than-4GB-buffer-required -DTYPE=float");
    EXPECT_NE(options.find("-cl-mad-enable"), std::string::npos);
    EXPECT_NE(options.find("-DTYPE=float"), std::string::npos);
    EXPECT_EQ(options.find("-cl-intel-"), std::string::npos);
    EXPECT_EQ(options, vulkan_clspv_compiler::canonical_options("-cl-mad-enable -DTYPE=float"));
}

TEST(vulkan_clspv_bootstrap, executes_interleaved_buffer_and_pod_arguments) {
    constexpr size_t element_count = 64;
    constexpr char source[] = R"(
        kernel void interleaved_pod(global const float* input, float increment,
                                    global float* output, uint count, int bias) {
            const size_t index = get_global_id(0);
            if (index < count) output[index] = input[index] + increment + (float)bias;
        }
    )";
    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    auto command_stream = target_engine->create_stream(get_test_default_config(*target_engine));
    const layout buffer_layout(ov::PartialShape{element_count}, ov::element::f32, format::bfyx);
    auto input = target_engine->allocate_memory(buffer_layout);
    auto output = target_engine->allocate_memory(buffer_layout);
    const std::vector<float> input_values(element_count, 2.0f);
    input->copy_from(*command_stream, input_values.data(), true);

    kernel_artifact artifact;
    artifact.payload = source;
    artifact.payload_size = sizeof(source) - 1;
    artifact.format = KernelFormat::SOURCE;
    artifact.entry_point = "interleaved_pod";
    std::vector<kernel::ptr> kernels;
    target_engine->create_kernel_builder()->build_kernels(artifact, kernels);
    ASSERT_EQ(kernels.size(), 1);

    kernel_arguments_desc descriptor;
    descriptor.workGroups.global = {element_count, 1, 1};
    descriptor.workGroups.local = {element_count, 1, 1};
    descriptor.arguments = {{argument_desc::Types::INPUT, 0},
                            {argument_desc::Types::SCALAR, 0},
                            {argument_desc::Types::OUTPUT, 0},
                            {argument_desc::Types::SCALAR, 1},
                            {argument_desc::Types::SCALAR, 2}};
    descriptor.scalars.resize(3);
    descriptor.scalars[0].t = scalar_desc::Types::FLOAT32;
    descriptor.scalars[0].v.f32 = 1.25f;
    descriptor.scalars[1].t = scalar_desc::Types::UINT32;
    descriptor.scalars[1].v.u32 = element_count;
    descriptor.scalars[2].t = scalar_desc::Types::INT32;
    descriptor.scalars[2].v.s32 = -3;
    kernel_arguments_data arguments;
    arguments.inputs = {input};
    arguments.outputs = {output};
    const vulkan_specialization_constants specialization = {{workgroup_size_x_spec_id, element_count},
                                                            {workgroup_size_y_spec_id, 1},
                                                            {workgroup_size_z_spec_id, 1}};
    auto& vulkan_command_stream = dynamic_cast<vulkan_stream&>(*command_stream);
    const auto completion =
        vulkan_command_stream.enqueue_kernel(*kernels.front(), descriptor, arguments, specialization, {}, true);
    ASSERT_NE(completion, nullptr);
    completion->wait();
    std::vector<float> actual(element_count);
    output->copy_to(*command_stream, actual.data(), true);
    for (const auto value : actual) {
        EXPECT_FLOAT_EQ(value, 0.25f);
    }
}

TEST(vulkan_clspv_bootstrap, rejects_non_scalar_pod_and_local_memory_arguments) {
    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    const auto& device = dynamic_cast<const vulkan_device&>(*target_engine->get_device());
    for (const auto& contract : std::array<std::pair<const char*, const char*>, 2>{
             {{R"(kernel void unsupported_pod(global float* output, float2 value) {
                    output[get_global_id(0)] = value.x + value.y;
                })",
               "POD arguments must be consecutive 32-bit scalars"},
              {R"(kernel void unsupported_pod(global float* output, local float* scratch) {
                    scratch[get_local_id(0)] = (float)get_local_id(0);
                    barrier(CLK_LOCAL_MEM_FENCE);
                    output[get_global_id(0)] = scratch[0];
                })",
               "local-memory arguments are outside the canonical ABI"}}}) {
        SCOPED_TRACE(contract.second);
        try {
            (void)vulkan_clspv_compiler{}.compile(contract.first, {}, "unsupported_pod", device);
            FAIL() << "Unsupported argument mapping must be rejected before pipeline creation";
        } catch (const ov::Exception& exception) {
            EXPECT_NE(std::string(exception.what()).find(contract.second), std::string::npos);
        }
    }
}

TEST(vulkan_clspv_bootstrap, reflects_required_workgroup_size) {
    auto target_engine = create_test_engine(engine_types::vulkan, runtime_types::vulkan);
    const auto& device = dynamic_cast<const vulkan_device&>(*target_engine->get_device());
    constexpr char source[] = R"(
        __attribute__((reqd_work_group_size(8, 2, 1)))
        kernel void fixed_workgroup(global float* output) {
            output[get_global_id(0)] = 1.0f;
        }
    )";
    const auto compilation = vulkan_clspv_compiler{}.compile(source, {}, "fixed_workgroup", device);
    const auto interface = vulkan_kernel_interface::reflect(compilation.spirv, "fixed_workgroup");
    EXPECT_EQ(interface.local_size_defaults, (std::array<uint32_t, 3>{8, 2, 1}));
    for (const auto& specialization : interface.local_size_specialization_ids) {
        EXPECT_FALSE(specialization.has_value());
    }
}
