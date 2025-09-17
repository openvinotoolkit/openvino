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

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {
class ZeroTensorRunTests : public ov::test::behavior::OVPluginTestBase,
                           public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config npu_config = ::intel_npu::Config(options);

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
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

        APIBaseTest::TearDown();
    }
};

TEST_P(ZeroTensorRunTests, AllocateDeleteAllocateZeroTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};
    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, true);
    ASSERT_TRUE(::intel_npu::zeroUtils::memory_was_allocated_in_the_same_l0_context(init_struct->getContext(),
                                                                                    zero_tensor->data()));

    zero_tensor = {};
    zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, false);
    ASSERT_TRUE(::intel_npu::zeroUtils::memory_was_allocated_in_the_same_l0_context(init_struct->getContext(),
                                                                                    zero_tensor->data()));

    void* address = zero_tensor->data();
    zero_tensor = {};
    ASSERT_FALSE(
        ::intel_npu::zeroUtils::memory_was_allocated_in_the_same_l0_context(init_struct->getContext(), address));
}

TEST_P(ZeroTensorRunTests, CheckSetSmallerShape) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 20, 20, 20};
    auto shape_size = ov::shape_size(shape);
    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, true);
    EXPECT_EQ(shape, zero_tensor->get_shape());
    EXPECT_EQ(shape_size, zero_tensor->get_size());
    EXPECT_EQ(shape_size * sizeof(ov::element::f32), zero_tensor->get_byte_size());

    auto data = zero_tensor->data();

    auto new_shape = Shape{1, 10, 10, 10};
    auto new_shape_size = ov::shape_size(new_shape);
    // Reallocation is not required.
    zero_tensor->set_shape(new_shape);
    EXPECT_EQ(new_shape, zero_tensor->get_shape());
    EXPECT_EQ(new_shape_size, zero_tensor->get_size());
    EXPECT_EQ(new_shape_size * sizeof(ov::element::f32), zero_tensor->get_byte_size());
    EXPECT_EQ(data, zero_tensor->data());
    ASSERT_TRUE(::intel_npu::zeroUtils::memory_was_allocated_in_the_same_l0_context(init_struct->getContext(),
                                                                                    zero_tensor->data()));
}

TEST_P(ZeroTensorRunTests, CheckSetBiggerShape) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 20, 20, 20};
    auto shape_size = ov::shape_size(shape);
    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element::f32, shape, false);
    EXPECT_EQ(shape, zero_tensor->get_shape());
    EXPECT_EQ(shape_size, zero_tensor->get_size());
    EXPECT_EQ(shape_size * sizeof(ov::element::f32), zero_tensor->get_byte_size());

    auto new_shape = Shape{1, 50, 50, 50};
    auto new_shape_size = ov::shape_size(new_shape);
    // set_shape() will force tensor reallocation for a larger shape. The new data pointer must also be a valid level
    // zero address.
    zero_tensor->set_shape(new_shape);
    EXPECT_EQ(new_shape, zero_tensor->get_shape());
    EXPECT_EQ(new_shape_size, zero_tensor->get_size());
    EXPECT_EQ(new_shape_size * sizeof(ov::element::f32), zero_tensor->get_byte_size());
    ASSERT_TRUE(zero_tensor->memory_address_changed());
    ASSERT_TRUE(::intel_npu::zeroUtils::memory_was_allocated_in_the_same_l0_context(init_struct->getContext(),
                                                                                    zero_tensor->data()));
}

TEST_P(ZeroTensorRunTests, CheckIsContinuousZeroTensorScalar) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor =
        std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, ov::element::f32, Shape{}, true);
    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, ov::Shape{}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(ZeroTensorRunTests, CheckIsContinuousHostTensor1Dimension) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor =
        std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, ov::element::f32, Shape{128}, true);

    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, ov::Shape{128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, ov::Shape{16}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(ZeroTensorRunTests, CheckIsContinuousZeroTensor2Dimensions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor =
        std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, ov::element::f32, Shape{32, 128}, true);
    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, Shape{16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 16}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 16}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);
}

TEST_P(ZeroTensorRunTests, CheckIsContinuousZeroTensor3Dimensions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor =
        std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, ov::element::f32, Shape{5, 32, 128}, true);
    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 64}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(ZeroTensorRunTests, CheckIsContinuousZeroTensor4Dimensions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct,
                                                                 npu_config,
                                                                 ov::element::f32,
                                                                 Shape{3, 5, 32, 128},
                                                                 true);
    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 2, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 5, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 2, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 2, 5, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{3, 5, 32, 64}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{2, 1, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 1, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(ov::element::f32, Shape{1, 1, 1, 32}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(ZeroTensorRunTests, CopyDefaultTensorExpectedThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};

    // shape size is unaligned to standard page size, expect to fail
    auto data = static_cast<float*>(::operator new(ov::shape_size(shape) * sizeof(ov::element::f32)));
    auto default_tensor = make_tensor(ov::element::f32, shape, data);
    ASSERT_THROW(auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, default_tensor, npu_config),
                 ::intel_npu::ZeroTensorException);

    ::operator delete(data);
}

TEST_P(ZeroTensorRunTests, CopyZeroTensorAndKeepAlive) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};

    auto zero_tensor =
        std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, ov::element::f32, shape, true);

    auto copy_zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, zero_tensor, npu_config);

    EXPECT_EQ(zero_tensor->get_shape(), copy_zero_tensor->get_shape());
    EXPECT_EQ(zero_tensor->data(), copy_zero_tensor->data());
    EXPECT_EQ(zero_tensor->get_byte_size(), copy_zero_tensor->get_byte_size());
    EXPECT_EQ(zero_tensor->get_size(), copy_zero_tensor->get_size());

    zero_tensor = {};
    ASSERT_TRUE(::intel_npu::zeroUtils::memory_was_allocated_in_the_same_l0_context(init_struct->getContext(),
                                                                                    copy_zero_tensor->data()));
    ASSERT_THROW(copy_zero_tensor->set_shape({1, 20, 20, 20}), ov::Exception);
}

TEST_P(ZeroTensorRunTests, CopyHostTensorAndKeepAlive) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    auto shape = Shape{1, 2, 2, 2};

    auto host_tensor =
        std::make_shared<::intel_npu::ZeroHostTensor>(zero_context, init_struct, ov::element::f32, shape);

    auto copy_zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, host_tensor, npu_config);

    EXPECT_EQ(host_tensor->get_shape(), copy_zero_tensor->get_shape());
    EXPECT_EQ(host_tensor->data(), copy_zero_tensor->data());
    EXPECT_EQ(host_tensor->get_byte_size(), copy_zero_tensor->get_byte_size());
    EXPECT_EQ(host_tensor->get_size(), copy_zero_tensor->get_size());

    host_tensor = {};
    ASSERT_TRUE(::intel_npu::zeroUtils::memory_was_allocated_in_the_same_l0_context(init_struct->getContext(),
                                                                                    copy_zero_tensor->data()));
    ASSERT_THROW(copy_zero_tensor->set_shape({1, 20, 20, 20}), ov::Exception);
}

TEST_P(ZeroTensorRunTests, CopyRemoteTensorAndKeepAlive) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    auto shape = Shape{1, 2, 2, 2};

    auto remote_tensor =
        std::make_shared<::intel_npu::ZeroRemoteTensor>(zero_context, init_struct, ov::element::f32, shape);

    auto copy_zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, remote_tensor, npu_config);

    EXPECT_EQ(remote_tensor->get_shape(), copy_zero_tensor->get_shape());
    EXPECT_EQ(remote_tensor->get_original_memory(), copy_zero_tensor->data());
    EXPECT_EQ(remote_tensor->get_byte_size(), copy_zero_tensor->get_byte_size());
    EXPECT_EQ(remote_tensor->get_size(), copy_zero_tensor->get_size());

    remote_tensor = {};
    ASSERT_TRUE(::intel_npu::zeroUtils::memory_was_allocated_in_the_same_l0_context(init_struct->getContext(),
                                                                                    copy_zero_tensor->data()));
    ASSERT_THROW(copy_zero_tensor->set_shape({1, 20, 20, 20}), ov::Exception);
}

TEST_P(ZeroTensorRunTests, CopyRemoteTensorFromAnotherContextThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    auto shape = Shape{1, 2, 2, 2};

    ov::AnyMap params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

    auto context = core->create_context(target_device, params);
    auto remote_tensor = context.create_tensor(ov::element::f32, shape);
    auto remote_tensor_impl = get_tensor_impl(remote_tensor);

    ASSERT_THROW(
        auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, remote_tensor_impl, npu_config),
        ::intel_npu::ZeroTensorException);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
