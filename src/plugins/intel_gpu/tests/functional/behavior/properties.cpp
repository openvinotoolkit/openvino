// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"

namespace {

class TestPropertiesGPU : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> model;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = ov::test::utils::make_conv_pool_relu();
    }
};

TEST_F(TestPropertiesGPU, NoRTInfo) {
    ov::Core core;
    ov::Any type;
    ov::Any size;
    ov::Any scale;
    ov::CompiledModel compiled_model;

    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));
    OV_ASSERT_NO_THROW(type = compiled_model.get_property(ov::hint::kv_cache_precision));
    OV_ASSERT_NO_THROW(size = compiled_model.get_property(ov::hint::dynamic_quantization_group_size));
    OV_ASSERT_NO_THROW(scale = compiled_model.get_property(ov::hint::activations_scale_factor));
}

TEST_F(TestPropertiesGPU, RTInfoPropertiesWithDefault) {
    ov::Core core;
    ov::Any type;
    ov::Any size;
    ov::Any scale;
    ov::CompiledModel compiled_model;
    model->set_rt_info("f16", "runtime_options", ov::hint::kv_cache_precision.name());
    model->set_rt_info("0", "runtime_options", ov::hint::dynamic_quantization_group_size.name());
    model->set_rt_info("8.0", "runtime_options", ov::hint::activations_scale_factor.name());

    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));
    OV_ASSERT_NO_THROW(size = compiled_model.get_property(ov::hint::dynamic_quantization_group_size));
    ASSERT_EQ(size.as<uint64_t>(), 0);

    // GPU with systolic does not support some of rt_info
    auto capabilities = core.get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
    if (find(capabilities.cbegin(), capabilities.cend(), ov::intel_gpu::capability::HW_MATMUL) != capabilities.cend())
        return;

    OV_ASSERT_NO_THROW(type = compiled_model.get_property(ov::hint::kv_cache_precision));
    OV_ASSERT_NO_THROW(scale = compiled_model.get_property(ov::hint::activations_scale_factor));
    ASSERT_EQ(type.as<ov::element::Type>(), ov::element::f16);
    ASSERT_EQ(scale.as<float>(), 8.0f);
}

TEST_F(TestPropertiesGPU, RTInfoPropertiesWithUserValuesFromCore) {
    ov::Core core;
    ov::Any type;
    ov::Any size;
    ov::Any scale;
    ov::CompiledModel compiled_model;
    model->set_rt_info("f16", "runtime_options", ov::hint::kv_cache_precision.name());
    model->set_rt_info("0", "runtime_options", ov::hint::dynamic_quantization_group_size.name());
    model->set_rt_info("8.0", "runtime_options", ov::hint::activations_scale_factor.name());
    core.set_property(ov::hint::kv_cache_precision(ov::element::i8));
    core.set_property(ov::hint::dynamic_quantization_group_size(16));
    core.set_property(ov::hint::activations_scale_factor(4.0f));

    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));
    OV_ASSERT_NO_THROW(type = compiled_model.get_property(ov::hint::kv_cache_precision));
    OV_ASSERT_NO_THROW(size = compiled_model.get_property(ov::hint::dynamic_quantization_group_size));
    OV_ASSERT_NO_THROW(scale = compiled_model.get_property(ov::hint::activations_scale_factor));
    ASSERT_EQ(type.as<ov::element::Type>(), ov::element::i8);
    ASSERT_EQ(size.as<uint64_t>(), 16);
    ASSERT_EQ(scale.as<float>(), 4.0f);
}

TEST_F(TestPropertiesGPU, RTInfoPropertiesWithUserValuesFromCompileModel) {
    ov::Core core;
    ov::Any type;
    ov::Any size;
    ov::Any scale;
    ov::CompiledModel compiled_model;
    model->set_rt_info("f16", "runtime_options", ov::hint::kv_cache_precision.name());
    model->set_rt_info("0", "runtime_options", ov::hint::dynamic_quantization_group_size.name());
    model->set_rt_info("8.0", "runtime_options", ov::hint::activations_scale_factor.name());
    ov::AnyMap config;
    config[ov::hint::kv_cache_precision.name()] = "i8";
    config[ov::hint::dynamic_quantization_group_size.name()] = "16";
    config[ov::hint::activations_scale_factor.name()] = "4.0";

    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config));
    OV_ASSERT_NO_THROW(type = compiled_model.get_property(ov::hint::kv_cache_precision));
    OV_ASSERT_NO_THROW(size = compiled_model.get_property(ov::hint::dynamic_quantization_group_size));
    OV_ASSERT_NO_THROW(scale = compiled_model.get_property(ov::hint::activations_scale_factor));
    ASSERT_EQ(type.as<ov::element::Type>(), ov::element::i8);
    ASSERT_EQ(size.as<uint64_t>(), 16);
    ASSERT_EQ(scale.as<float>(), 4.0f);
}

TEST_F(TestPropertiesGPU, DefaultUSMHostTensorAllocationProperty) {
    ov::Core core;
    ov::CompiledModel compiled_model;

    // Test default value is false
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));
    ov::Any default_value;
    OV_ASSERT_NO_THROW(default_value = compiled_model.get_property(ov::intel_gpu::default_usm_host_tensor_allocation));
    ASSERT_FALSE(default_value.as<bool>());
}

TEST_F(TestPropertiesGPU, SetDefaultUSMHostTensorAllocationTrue) {
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;

    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config));
    ov::Any value;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::intel_gpu::default_usm_host_tensor_allocation));
    ASSERT_TRUE(value.as<bool>());
}

TEST_F(TestPropertiesGPU, SetDefaultUSMHostTensorAllocationFalse) {
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = false;

    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config));
    ov::Any value;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::intel_gpu::default_usm_host_tensor_allocation));
    ASSERT_FALSE(value.as<bool>());
}

TEST_F(TestPropertiesGPU, DefaultUSMHostTensorAllocationViaSetProperty) {
    ov::Core core;
    
    // Set property via core.set_property
    ov::AnyMap config;
    config[ov::intel_gpu::default_usm_host_tensor_allocation.name()] = true;
    OV_ASSERT_NO_THROW(core.set_property(ov::test::utils::DEVICE_GPU, config));
    
    // Compile model and verify property is set
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));
    ov::Any value;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::intel_gpu::default_usm_host_tensor_allocation));
    ASSERT_TRUE(value.as<bool>());
}

} // namespace
