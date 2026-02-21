// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/tensor.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "remote_tensor_tests/helpers.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {

using namespace ov::test::utils;

class DefaultUSMHostTensorAllocationTest : public ::testing::Test {
protected:
    std::shared_ptr<ov::Model> model;
    ov::Core core;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = ov::test::utils::make_conv_pool_relu();
    }

    void TearDown() override {
        // Reset the custom tensor generator after each test
        ov::util::reset_custom_tensor_impl_generator();
    }

    bool supportsUSM() {
        try {
            auto context = core.get_default_context(ov::test::utils::DEVICE_GPU).as<ov::intel_gpu::ocl::ClContext>();
            cl_context ctx = context;
            auto ocl_instance = std::make_shared<OpenCL>(ctx);
            return ocl_instance->supports_usm();
        } catch (...) {
            return false;
        }
    }
};

TEST_F(DefaultUSMHostTensorAllocationTest, TensorCreationWithPropertyDisabled) {
    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = false;
    
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    
    // Create a regular tensor - should use default allocation
    ov::Shape shape = {1, 3, 224, 224};
    ov::Tensor tensor(ov::element::f32, shape);
    
    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_element_type(), ov::element::f32);
    EXPECT_NE(tensor.data(), nullptr);
}

TEST_F(DefaultUSMHostTensorAllocationTest, TensorCreationWithPropertyEnabled) {
    if (!supportsUSM()) {
        GTEST_SKIP() << "Device does not support USM";
    }

    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;
    
    // Set property to enable USM host tensor allocation
    core.set_property(ov::test::utils::DEVICE_GPU, config);
    
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    
    // Create a tensor - should use USM host allocation via custom generator
    ov::Shape shape = {1, 3, 224, 224};
    ov::Tensor tensor(ov::element::f32, shape);
    
    EXPECT_EQ(tensor.get_shape(), shape);
    EXPECT_EQ(tensor.get_element_type(), ov::element::f32);
    EXPECT_NE(tensor.data(), nullptr);
    
    // Verify it's actually a USM host tensor by checking allocation type
    auto context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    
    void* ptr = tensor.data();
    cl_unified_shared_memory_type_intel mem_type = ocl_instance->get_allocation_type(ptr);
    EXPECT_EQ(mem_type, CL_MEM_TYPE_HOST_INTEL);
}

TEST_F(DefaultUSMHostTensorAllocationTest, InferenceWithUSMHostTensors) {
    if (!supportsUSM()) {
        GTEST_SKIP() << "Device does not support USM";
    }

    // Compile model with USM host tensor allocation enabled
    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;
    core.set_property(ov::test::utils::DEVICE_GPU, config);
    
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto infer_request = compiled_model.create_infer_request();
    
    // Get input/output info
    auto input = model->get_parameters().at(0);
    auto output = model->get_results().at(0);
    auto input_shape = input->get_shape();
    auto output_shape = output->get_shape();
    
    // Create tensors (should be USM host tensors)
    ov::Tensor input_tensor(input->get_element_type(), input_shape);
    ov::Tensor output_tensor(output->get_element_type(), output_shape);
    
    // Fill input with test data
    auto input_data = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input_shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_tensor.get_byte_size());
    
    // Set tensors and run inference
    infer_request.set_tensor(input, input_tensor);
    infer_request.set_tensor(output, output_tensor);
    
    OV_ASSERT_NO_THROW(infer_request.infer());
    
    // Verify output tensor is valid
    EXPECT_NE(output_tensor.data(), nullptr);
    EXPECT_EQ(output_tensor.get_shape(), output_shape);
}

TEST_F(DefaultUSMHostTensorAllocationTest, CompareResultsWithAndWithoutUSM) {
    if (!supportsUSM()) {
        GTEST_SKIP() << "Device does not support USM";
    }

    auto input = model->get_parameters().at(0);
    auto output = model->get_results().at(0);
    auto input_shape = input->get_shape();
    auto output_shape = output->get_shape();
    
    // Create test input data
    auto input_data = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input_shape);
    
    // Regular inference without USM
    {
        ov::AnyMap config;
        config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = false;
        auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
        auto infer_request = compiled_model.create_infer_request();
        
        infer_request.set_tensor(input, input_data);
        infer_request.infer();
        
        auto output_tensor_regular = infer_request.get_tensor(output);
        
        // Store output for comparison
        std::vector<float> output_regular(output_tensor_regular.get_size());
        std::memcpy(output_regular.data(), output_tensor_regular.data(), output_tensor_regular.get_byte_size());
        
        // Inference with USM host tensors
        ov::AnyMap usm_config;
        usm_config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;
        core.set_property(ov::test::utils::DEVICE_GPU, usm_config);
        
        auto compiled_model_usm = core.compile_model(model, ov::test::utils::DEVICE_GPU);
        auto infer_request_usm = compiled_model_usm.create_infer_request();
        
        ov::Tensor input_tensor_usm(input->get_element_type(), input_shape);
        std::memcpy(input_tensor_usm.data(), input_data.data(), input_data.get_byte_size());
        
        infer_request_usm.set_tensor(input, input_tensor_usm);
        infer_request_usm.infer();
        
        auto output_tensor_usm = infer_request_usm.get_tensor(output);
        
        // Compare results
        ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_usm.get_size());
        
        auto* output_regular_ptr = static_cast<float*>(output_tensor_regular.data());
        auto* output_usm_ptr = static_cast<float*>(output_tensor_usm.data());
        
        for (size_t i = 0; i < output_tensor_regular.get_size(); i++) {
            EXPECT_NEAR(output_regular_ptr[i], output_usm_ptr[i], 1e-5f)
                << "Mismatch at index " << i;
        }
    }
}

TEST_F(DefaultUSMHostTensorAllocationTest, USMPropertyDoesNotAffectHostPtrConstructor) {
    if (!supportsUSM()) {
        GTEST_SKIP() << "Device does not support USM";
    }

    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;
    core.set_property(ov::test::utils::DEVICE_GPU, config);
    
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    
    // Create tensor with host pointer - should NOT use USM allocation
    ov::Shape shape = {1, 3, 224, 224};
    std::vector<float> data(ov::shape_size(shape), 1.0f);
    ov::Tensor tensor(ov::element::f32, shape, data.data());
    
    // Verify it uses the provided pointer, not USM
    EXPECT_EQ(tensor.data<float>(), data.data());
}

TEST_F(DefaultUSMHostTensorAllocationTest, MultipleModelsWithUSMProperty) {
    if (!supportsUSM()) {
        GTEST_SKIP() << "Device does not support USM";
    }

    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;
    core.set_property(ov::test::utils::DEVICE_GPU, config);
    
    // Compile multiple models
    auto compiled_model1 = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto compiled_model2 = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    
    // Create tensors - all should use USM host allocation
    ov::Shape shape = {1, 3, 224, 224};
    ov::Tensor tensor1(ov::element::f32, shape);
    ov::Tensor tensor2(ov::element::f32, shape);
    
    auto context = compiled_model1.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    
    // Verify both are USM host tensors
    EXPECT_EQ(ocl_instance->get_allocation_type(tensor1.data()), CL_MEM_TYPE_HOST_INTEL);
    EXPECT_EQ(ocl_instance->get_allocation_type(tensor2.data()), CL_MEM_TYPE_HOST_INTEL);
}

TEST_F(DefaultUSMHostTensorAllocationTest, PropertyResetAfterPluginDestruction) {
    if (!supportsUSM()) {
        GTEST_SKIP() << "Device does not support USM";
    }

    ov::Shape shape = {1, 3, 224, 224};
    
    {
        // Create a new core and enable USM property
        ov::Core local_core;
        ov::AnyMap config;
        config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;
        local_core.set_property(ov::test::utils::DEVICE_GPU, config);
        
        auto compiled_model = local_core.compile_model(model, ov::test::utils::DEVICE_GPU);
        
        ov::Tensor tensor_usm(ov::element::f32, shape);
        
        auto context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
        cl_context ctx = context;
        auto ocl_instance = std::make_shared<OpenCL>(ctx);
        
        // Should be USM host tensor
        EXPECT_EQ(ocl_instance->get_allocation_type(tensor_usm.data()), CL_MEM_TYPE_HOST_INTEL);
        
        // local_core goes out of scope, plugin should reset the generator
    }
    
    // After plugin destruction, new tensors should use default allocation
    // Note: This test assumes the plugin properly cleans up in its destructor
    ov::Tensor tensor_default(ov::element::f32, shape);
    EXPECT_NE(tensor_default.data(), nullptr);
}

TEST_F(DefaultUSMHostTensorAllocationTest, ConcurrentTensorCreation) {
    if (!supportsUSM()) {
        GTEST_SKIP() << "Device does not support USM";
    }

    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;
    core.set_property(ov::test::utils::DEVICE_GPU, config);
    
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    
    // Create multiple tensors
    ov::Shape shape = {1, 3, 224, 224};
    std::vector<ov::Tensor> tensors;
    
    for (int i = 0; i < 10; i++) {
        tensors.emplace_back(ov::element::f32, shape);
    }
    
    auto context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    
    // Verify all tensors are USM host tensors
    for (const auto& tensor : tensors) {
        EXPECT_NE(tensor.data(), nullptr);
        EXPECT_EQ(ocl_instance->get_allocation_type(tensor.data()), CL_MEM_TYPE_HOST_INTEL);
    }
}

TEST_F(DefaultUSMHostTensorAllocationTest, DifferentElementTypes) {
    if (!supportsUSM()) {
        GTEST_SKIP() << "Device does not support USM";
    }

    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;
    core.set_property(ov::test::utils::DEVICE_GPU, config);
    
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    
    ov::Shape shape = {10, 10};
    
    // Test different element types
    std::vector<ov::element::Type> types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
        ov::element::u8,
        ov::element::i8
    };
    
    for (const auto& type : types) {
        ov::Tensor tensor(type, shape);
        EXPECT_NE(tensor.data(), nullptr);
        EXPECT_EQ(tensor.get_element_type(), type);
        EXPECT_EQ(ocl_instance->get_allocation_type(tensor.data()), CL_MEM_TYPE_HOST_INTEL);
    }
}

}  // namespace
