// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <random>
#include <thread>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/zero/zero_host_tensor.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "remote_context.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_backend.hpp"
#include "zero_tensor.hpp"
#include "zero_variable_state.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {
class ZeroVariableStateTests : public ov::test::behavior::OVPluginTestBase,
                               public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config npu_config = ::intel_npu::Config(options);

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        targetDevice = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();

        init_struct = ::intel_npu::ZeroInitStructsHolder::getInstance();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        init_struct = nullptr;
        APIBaseTest::TearDown();
    }
};

TEST_P(ZeroVariableStateTests, CreateZeroStateAndCheckAgainstZeroTensorState) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};
    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, true);
    auto zero_state =
        std::make_shared<::intel_npu::ZeroVariableState>(init_struct, "state", zero_tensor, 1, 1, npu_config);
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_state()->data()));

    EXPECT_EQ(zero_state->get_state()->data(), zero_tensor->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), zero_tensor->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), zero_tensor->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), zero_tensor->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), zero_tensor->get_element_type());

    EXPECT_EQ(zero_state->get_state()->data(), zero_state->get_zero_state()->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), zero_state->get_zero_state()->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), zero_state->get_zero_state()->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), zero_state->get_zero_state()->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), zero_state->get_zero_state()->get_element_type());

    OV_ASSERT_NO_THROW(zero_state->reset());
}

TEST_P(ZeroVariableStateTests, CreateZeroStateAndUseSetStateWithZeroTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};
    auto zero_tensor0 = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, true);
    auto zero_state =
        std::make_shared<::intel_npu::ZeroVariableState>(init_struct, "state", zero_tensor0, 1, 1, npu_config);
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_state()->data()));

    auto zero_tensor1 = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, true);
    zero_state->set_state(zero_tensor1);
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_state()->data()));

    ASSERT_TRUE(zero_state->state_update_pending());
    ASSERT_TRUE(zero_state->zero_state_update_pending());

    EXPECT_NE(zero_state->get_state()->data(), zero_tensor0->data());

    EXPECT_EQ(zero_state->get_state()->data(), zero_tensor1->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), zero_tensor1->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), zero_tensor1->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), zero_tensor1->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), zero_tensor1->get_element_type());

    EXPECT_EQ(zero_state->get_state()->data(), zero_state->get_zero_state()->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), zero_state->get_zero_state()->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), zero_state->get_zero_state()->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), zero_state->get_zero_state()->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), zero_state->get_zero_state()->get_element_type());

    OV_ASSERT_NO_THROW(zero_state->reset());
}

TEST_P(ZeroVariableStateTests, CreateZeroStateAndUseSetStateWithNormalTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};
    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, true);
    auto zero_state =
        std::make_shared<::intel_npu::ZeroVariableState>(init_struct, "state", zero_tensor, 1, 1, npu_config);
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_user_state()->data()));

    // shape size is unaligned to standard page size, expect to fail
    auto data = static_cast<float*>(::operator new(ov::shape_size(shape) * sizeof(ov::element::f32)));
    auto tensor = make_tensor(ov::element::f32, shape, data);
    zero_state->set_state(tensor);
    ASSERT_FALSE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                             zero_state->get_state()->data()));
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_zero_state()->data()));

    ASSERT_TRUE(zero_state->state_update_pending());
    ASSERT_FALSE(zero_state->zero_state_update_pending());

    EXPECT_NE(zero_state->get_state()->data(), zero_tensor->data());
    EXPECT_EQ(zero_state->get_zero_state()->data(), zero_tensor->data());

    EXPECT_EQ(zero_state->get_state()->data(), tensor->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), tensor->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), tensor->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), tensor->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), tensor->get_element_type());

    EXPECT_NE(zero_state->get_state()->data(), zero_state->get_zero_state()->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), zero_state->get_zero_state()->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), zero_state->get_zero_state()->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), zero_state->get_zero_state()->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), zero_state->get_zero_state()->get_element_type());

    OV_ASSERT_NO_THROW(zero_state->reset());
}

TEST_P(ZeroVariableStateTests, CreateZeroStateAndUseSetStateWithNormalTensorAfterGetState) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};
    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, true);
    auto zero_state =
        std::make_shared<::intel_npu::ZeroVariableState>(init_struct, "state", zero_tensor, 1, 1, npu_config);
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_state()->data()));

    // shape size is unaligned to standard page size, expect to fail
    auto data = static_cast<float*>(::operator new(ov::shape_size(shape) * sizeof(ov::element::f32)));
    auto tensor = make_tensor(ov::element::f32, shape, data);
    zero_state->set_state(tensor);
    ASSERT_FALSE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                             zero_state->get_state()->data()));
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_zero_state()->data()));

    ASSERT_TRUE(zero_state->state_update_pending());
    ASSERT_TRUE(zero_state->zero_state_update_pending());

    EXPECT_NE(zero_state->get_state()->data(), zero_tensor->data());
    EXPECT_NE(zero_state->get_zero_state()->data(), zero_tensor->data());

    EXPECT_EQ(zero_state->get_state()->data(), tensor->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), tensor->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), tensor->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), tensor->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), tensor->get_element_type());

    EXPECT_NE(zero_state->get_state()->data(), zero_state->get_zero_state()->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), zero_state->get_zero_state()->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), zero_state->get_zero_state()->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), zero_state->get_zero_state()->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), zero_state->get_zero_state()->get_element_type());

    OV_ASSERT_NO_THROW(zero_state->reset());
}

TEST_P(ZeroVariableStateTests, CreateZeroStateAndUseSetStateWithHostTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    auto shape = Shape{1, 2, 2, 2};

    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, true);
    auto zero_state =
        std::make_shared<::intel_npu::ZeroVariableState>(init_struct, "state", zero_tensor, 1, 1, npu_config);

    // shape size is unaligned to standard page size, expect to fail
    auto host_tensor =
        std::make_shared<::intel_npu::ZeroHostTensor>(zero_context, init_struct, ov::element::f32, shape);
    zero_state->set_state(host_tensor);
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_state()->data()));
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_zero_state()->data()));

    ASSERT_TRUE(zero_state->state_update_pending());
    ASSERT_TRUE(zero_state->zero_state_update_pending());

    EXPECT_NE(zero_state->get_state()->data(), zero_tensor->data());
    EXPECT_NE(zero_state->get_zero_state()->data(), zero_tensor->data());

    EXPECT_EQ(zero_state->get_state()->data(), host_tensor->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), host_tensor->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), host_tensor->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), host_tensor->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), host_tensor->get_element_type());

    EXPECT_EQ(zero_state->get_state()->data(), zero_state->get_zero_state()->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), zero_state->get_zero_state()->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), zero_state->get_zero_state()->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), zero_state->get_zero_state()->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), zero_state->get_zero_state()->get_element_type());

    OV_ASSERT_NO_THROW(zero_state->reset());
}

TEST_P(ZeroVariableStateTests, CreateZeroStateAndUseSetStateWithRemoteTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    auto shape = Shape{1, 2, 2, 2};

    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, true);
    auto zero_state =
        std::make_shared<::intel_npu::ZeroVariableState>(init_struct, "state", zero_tensor, 1, 1, npu_config);

    // shape size is unaligned to standard page size, expect to fail
    auto remote_tensor =
        std::make_shared<::intel_npu::ZeroRemoteTensor>(zero_context, init_struct, ov::element::f32, shape);
    zero_state->set_state(remote_tensor);
    auto zero_remote_tensor = std::dynamic_pointer_cast<::intel_npu::ZeroRemoteTensor>(zero_state->get_state()._ptr);

    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_remote_tensor->get_original_memory()));
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            zero_state->get_zero_state()->data()));

    ASSERT_TRUE(zero_state->state_update_pending());
    ASSERT_TRUE(zero_state->zero_state_update_pending());

    EXPECT_NE(zero_remote_tensor->get_original_memory(), zero_tensor->data());
    EXPECT_NE(zero_state->get_zero_state()->data(), zero_tensor->data());

    EXPECT_EQ(zero_remote_tensor->get_original_memory(), remote_tensor->get_original_memory());
    EXPECT_EQ(zero_state->get_state()->get_shape(), remote_tensor->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), remote_tensor->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), remote_tensor->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), remote_tensor->get_element_type());

    EXPECT_EQ(zero_remote_tensor->get_original_memory(), zero_state->get_zero_state()->data());
    EXPECT_EQ(zero_state->get_state()->get_shape(), zero_state->get_zero_state()->get_shape());
    EXPECT_EQ(zero_state->get_state()->get_size(), zero_state->get_zero_state()->get_size());
    EXPECT_EQ(zero_state->get_state()->get_byte_size(), zero_state->get_zero_state()->get_byte_size());
    EXPECT_EQ(zero_state->get_state()->get_element_type(), zero_state->get_zero_state()->get_element_type());

    OV_ASSERT_NO_THROW(zero_state->reset());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
