// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "base/ov_behavior_test_utils.hpp"
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
    core.set_property(ov::hint::kv_cache_precision(ov::element::u8));
    core.set_property(ov::hint::dynamic_quantization_group_size(16));
    core.set_property(ov::hint::activations_scale_factor(4.0f));

    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));
    OV_ASSERT_NO_THROW(type = compiled_model.get_property(ov::hint::kv_cache_precision));
    OV_ASSERT_NO_THROW(size = compiled_model.get_property(ov::hint::dynamic_quantization_group_size));
    OV_ASSERT_NO_THROW(scale = compiled_model.get_property(ov::hint::activations_scale_factor));
    ASSERT_EQ(type.as<ov::element::Type>(), ov::element::u8);
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
    config[ov::hint::kv_cache_precision.name()] = "u8";
    config[ov::hint::dynamic_quantization_group_size.name()] = "16";
    config[ov::hint::activations_scale_factor.name()] = "4.0";

    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config));
    OV_ASSERT_NO_THROW(type = compiled_model.get_property(ov::hint::kv_cache_precision));
    OV_ASSERT_NO_THROW(size = compiled_model.get_property(ov::hint::dynamic_quantization_group_size));
    OV_ASSERT_NO_THROW(scale = compiled_model.get_property(ov::hint::activations_scale_factor));
    ASSERT_EQ(type.as<ov::element::Type>(), ov::element::u8);
    ASSERT_EQ(size.as<uint64_t>(), 16);
    ASSERT_EQ(scale.as<float>(), 4.0f);
}

} // namespace
