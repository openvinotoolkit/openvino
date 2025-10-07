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

using CompilationParamsAndTensorDataType = std::tuple<std::string,       // Device name
                                                      ov::AnyMap,        // Config
                                                      ov::element::Type  // Tensor data type
                                                      >;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {
class ZeroTensorTests : public ov::test::behavior::OVPluginTestBase,
                        public testing::WithParamInterface<CompilationParamsAndTensorDataType> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    ov::element::Type element_type;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config npu_config = ::intel_npu::Config(options);

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParamsAndTensorDataType>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        ov::element::Type type;
        std::tie(targetDevice, configuration, type) = obj.param;
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
        if (!type.get_type_name().empty()) {
            result << "tensorDataType=" << type.get_type_name() << "_";
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration, element_type) = this->GetParam();

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

TEST_P(ZeroTensorTests, AllocateDeleteAllocateZeroTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};
    
    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, shape, true);
    ASSERT_TRUE(
        ::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), zero_tensor->data()));

    zero_tensor = {};
    zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, shape, false);
    ASSERT_TRUE(
        ::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), zero_tensor->data()));

    void* address = zero_tensor->data();
    zero_tensor = {};
    ASSERT_FALSE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), address));
}

TEST_P(ZeroTensorTests, CheckSetSmallerShape) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 20, 20, 20};
    auto shape_size = ov::shape_size(shape);
    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, shape, true);
    EXPECT_EQ(shape, zero_tensor->get_shape());
    EXPECT_EQ(shape_size, zero_tensor->get_size());
    EXPECT_EQ(shape_size * element_type.size(), zero_tensor->get_byte_size());

    auto data = zero_tensor->data();

    auto new_shape = Shape{1, 10, 10, 10};
    auto new_shape_size = ov::shape_size(new_shape);
    // Reallocation is not required.
    zero_tensor->set_shape(new_shape);
    EXPECT_EQ(new_shape, zero_tensor->get_shape());
    EXPECT_EQ(new_shape_size, zero_tensor->get_size());
    EXPECT_EQ(new_shape_size * element_type.size(), zero_tensor->get_byte_size());
    EXPECT_EQ(data, zero_tensor->data());
    ASSERT_TRUE(
        ::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), zero_tensor->data()));
}

TEST_P(ZeroTensorTests, CheckSetBiggerShape) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 20, 20, 20};
    auto shape_size = ov::shape_size(shape);
    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, shape, false);
    EXPECT_EQ(shape, zero_tensor->get_shape());
    EXPECT_EQ(shape_size, zero_tensor->get_size());
    EXPECT_EQ(shape_size * element_type.size(), zero_tensor->get_byte_size());

    auto new_shape = Shape{1, 50, 50, 50};
    auto new_shape_size = ov::shape_size(new_shape);
    // set_shape() will force tensor reallocation for a larger shape. The new data pointer must also be a valid level
    // zero address.
    if (init_struct->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
        zero_tensor->set_shape(new_shape);
        EXPECT_EQ(new_shape, zero_tensor->get_shape());
        EXPECT_EQ(new_shape_size, zero_tensor->get_size());
        EXPECT_EQ(new_shape_size * element_type.size(), zero_tensor->get_byte_size());
        ASSERT_TRUE(zero_tensor->memory_address_changed());
        ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                                zero_tensor->data()));
    } else {
        ASSERT_THROW(zero_tensor->set_shape(new_shape), ov::Exception);
    }
}

TEST_P(ZeroTensorTests, CheckIsContinuousZeroTensorScalar) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, Shape{}, true);
    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(element_type, ov::Shape{}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(ZeroTensorTests, CheckIsContinuousHostTensor1Dimension) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor =
        std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, Shape{128}, true);

    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(element_type, ov::Shape{128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, ov::Shape{16}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(ZeroTensorTests, CheckIsContinuousZeroTensor2Dimensions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor =
        std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, Shape{32, 128}, true);
    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(element_type, Shape{16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{1, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{1, 16}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{2, 16}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);
}

TEST_P(ZeroTensorTests, CheckIsContinuousZeroTensor3Dimensions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor =
        std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, Shape{5, 32, 128}, true);
    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(element_type, Shape{2, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{2, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(element_type, Shape{1, 1, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{1, 1, 64}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{1, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(ZeroTensorTests, CheckIsContinuousZeroTensor4Dimensions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_tensor =
        std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, Shape{3, 5, 32, 128}, true);
    auto data = zero_tensor->data();
    auto strides = zero_tensor->get_strides();

    ov::Tensor view_tensor;

    view_tensor = ov::Tensor(element_type, Shape{1, 2, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{2, 5, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{2, 2, 32, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(element_type, Shape{1, 2, 5, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(element_type, Shape{3, 5, 32, 64}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(element_type, Shape{1, 1, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{2, 1, 16, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), false);

    view_tensor = ov::Tensor(element_type, Shape{1, 1, 1, 128}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);

    view_tensor = ov::Tensor(element_type, Shape{1, 1, 1, 32}, data, strides);
    EXPECT_EQ(view_tensor.is_continuous(), true);
}

TEST_P(ZeroTensorTests, CopyDefaultTensorExpectedThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};

    // shape size is unaligned to standard page size, expect to fail
    auto data = static_cast<float*>(::operator new(ov::shape_size(shape) * element_type.size()));
    auto default_tensor = make_tensor(element_type, shape, data);
    ASSERT_THROW(auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, default_tensor),
                 ::intel_npu::ZeroMemException);

    ::operator delete(data);
}

TEST_P(ZeroTensorTests, CopyZeroTensorAndKeepAlive) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};

    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, shape, true);

    auto copy_zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, zero_tensor);

    EXPECT_EQ(zero_tensor->get_shape(), copy_zero_tensor->get_shape());
    EXPECT_EQ(zero_tensor->data(), copy_zero_tensor->data());
    EXPECT_EQ(zero_tensor->get_byte_size(), copy_zero_tensor->get_byte_size());
    EXPECT_EQ(zero_tensor->get_size(), copy_zero_tensor->get_size());

    zero_tensor = {};
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            copy_zero_tensor->data()));
    ASSERT_THROW(copy_zero_tensor->set_shape({1, 20, 20, 20}), ov::Exception);
}

TEST_P(ZeroTensorTests, CopyHostTensorAndKeepAlive) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    auto shape = Shape{1, 2, 2, 2};

    auto host_tensor = std::make_shared<::intel_npu::ZeroHostTensor>(zero_context, init_struct, element_type, shape);

    auto copy_zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, host_tensor);

    EXPECT_EQ(host_tensor->get_shape(), copy_zero_tensor->get_shape());
    EXPECT_EQ(host_tensor->data(), copy_zero_tensor->data());
    EXPECT_EQ(host_tensor->get_byte_size(), copy_zero_tensor->get_byte_size());
    EXPECT_EQ(host_tensor->get_size(), copy_zero_tensor->get_size());

    host_tensor = {};
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            copy_zero_tensor->data()));
    ASSERT_THROW(copy_zero_tensor->set_shape({1, 20, 20, 20}), ov::Exception);
}

TEST_P(ZeroTensorTests, CopyRemoteTensorAndKeepAlive) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    auto shape = Shape{1, 2, 2, 2};

    auto remote_tensor =
        std::make_shared<::intel_npu::ZeroRemoteTensor>(zero_context, init_struct, element_type, shape);

    auto copy_zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, remote_tensor);

    EXPECT_EQ(remote_tensor->get_shape(), copy_zero_tensor->get_shape());
    EXPECT_EQ(remote_tensor->get_original_memory(), copy_zero_tensor->data());
    EXPECT_EQ(remote_tensor->get_byte_size(), copy_zero_tensor->get_byte_size());
    EXPECT_EQ(remote_tensor->get_size(), copy_zero_tensor->get_size());

    remote_tensor = {};
    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                            copy_zero_tensor->data()));
    ASSERT_THROW(copy_zero_tensor->set_shape({1, 20, 20, 20}), ov::Exception);
}

TEST_P(ZeroTensorTests, CopyRemoteTensorFromAnotherContextThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    auto shape = Shape{1, 2, 2, 2};

    ov::AnyMap params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}}};

    auto context = core->create_context(target_device, params);
    auto remote_tensor = context.create_tensor(element_type, shape);
    auto remote_tensor_impl = get_tensor_impl(remote_tensor);

    ASSERT_THROW(
        auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, remote_tensor_impl),
        ::intel_npu::ZeroMemException);
}

TEST_P(ZeroTensorTests, CheckStandardAllocation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 64, 64, 64};

    auto logger = ::intel_npu::Logger("CheckStandardAllocation", ov::log::Level::INFO);

    // shape size is aligned to standard page size, align address as well
    auto data =
        static_cast<float*>(::operator new(ov::shape_size(shape) * element_type.size(), std::align_val_t(4096)));
    auto default_tensor = make_tensor(element_type, shape, data);
    std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor;
    if (init_struct->isExternalMemoryStandardAllocationSupported()) {
        OV_ASSERT_NO_THROW(zero_tensor =
                               std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, default_tensor));
        logger.info("Import memory");
    } else {
        ASSERT_THROW(zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, default_tensor),
                     ::intel_npu::ZeroMemException);
        logger.error("Import memory is not supported, throw is expected.");
    }

    ::operator delete(data, std::align_val_t(4096));
}

TEST_P(ZeroTensorTests, UseSameStandardAllocationMultipleTimesCompatible) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (init_struct->isExternalMemoryStandardAllocationSupported()) {
        auto shape = Shape{1, 64, 64, 64};

        // shape size is aligned to standard page size, align address as well
        auto data =
            static_cast<float*>(::operator new(ov::shape_size(shape) * element_type.size(), std::align_val_t(4096)));
        auto default_tensor = make_tensor(element_type, shape, data);
        std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor0;
        OV_ASSERT_NO_THROW(zero_tensor0 =
                               std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, default_tensor));
        ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                                default_tensor->data()));

        std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor1;
        OV_ASSERT_NO_THROW(zero_tensor1 =
                               std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, default_tensor));
        ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                                default_tensor->data()));

        zero_tensor0 = {};
        ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                                default_tensor->data()));

        std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor2;
        OV_ASSERT_NO_THROW(zero_tensor2 =
                               std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, zero_tensor1));
        ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                                default_tensor->data()));

        zero_tensor1 = {};
        ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                                default_tensor->data()));

        zero_tensor2 = {};
        ASSERT_FALSE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                                 default_tensor->data()));

        ::operator delete(data, std::align_val_t(4096));
    }
}

TEST_P(ZeroTensorTests, UseRangePointerFromAlreadyImportedStandardAllocation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (init_struct->isExternalMemoryStandardAllocationSupported()) {
        auto shape = Shape{1, 64, 64, 64};

        // shape size is aligned to standard page size, align address as well
        auto data =
            static_cast<float*>(::operator new(ov::shape_size(shape) * element_type.size(), std::align_val_t(4096)));
        auto default_tensor = make_tensor(element_type, shape, data);
        std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor0;
        OV_ASSERT_NO_THROW(zero_tensor0 =
                               std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, default_tensor));

        auto smaller_shape = Shape{1, 2, 2, 2};
        auto smaller_tensor = make_tensor(element_type, smaller_shape, data + 16);
        std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor1;
        OV_ASSERT_NO_THROW(zero_tensor1 =
                               std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, smaller_tensor));

        EXPECT_EQ(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                              zero_tensor0->data()),
                  ::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                              zero_tensor1->data()));
        EXPECT_EQ(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                              default_tensor->data()),
                  ::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(),
                                                                              smaller_tensor->data()));
        EXPECT_EQ(default_tensor->data(), zero_tensor0->data());
        EXPECT_EQ(smaller_tensor->data(), zero_tensor1->data());
        EXPECT_NE(zero_tensor0->data(), zero_tensor1->data());
        EXPECT_EQ(static_cast<void*>(static_cast<float*>(zero_tensor0->data()) + 16), zero_tensor1->data());

        ::operator delete(data, std::align_val_t(4096));
    }
}

TEST_P(ZeroTensorTests, UseRangeOutOfBoundsPointerFromAlreadyImportedStandardAllocation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (init_struct->isExternalMemoryStandardAllocationSupported()) {
        auto shape = Shape{1, 64, 64, 64};

        // shape size is aligned to standard page size, align address as well
        auto data =
            static_cast<float*>(::operator new(ov::shape_size(shape) * element_type.size(), std::align_val_t(4096)));
        auto default_tensor = make_tensor(element_type, shape, data);
        std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor0;
        OV_ASSERT_NO_THROW(zero_tensor0 =
                               std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, default_tensor));

        auto different_shape = Shape{1, 64, 64, 64};
        auto different_tensor = make_tensor(element_type, different_shape, data + 16);
        std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor1;
        if (init_struct->isExternalMemoryStandardAllocationSupported()) {
            ASSERT_THROW(
                zero_tensor1 = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, different_tensor),
                ::intel_npu::ZeroMemException);
        } else {
            OV_ASSERT_NO_THROW(
                zero_tensor1 = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, different_tensor));
        }

        ::operator delete(data, std::align_val_t(4096));
    }
}

TEST_P(ZeroTensorTests, UseBiggerMemoryFromAlreadyImportedStandardAllocation) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (init_struct->isExternalMemoryStandardAllocationSupported()) {
        auto shape = Shape{1, 64, 64, 64};

        // shape size is aligned to standard page size, align address as well
        auto data =
            static_cast<float*>(::operator new(ov::shape_size(shape) * element_type.size(), std::align_val_t(4096)));

        auto smaller_shape = Shape{1, 16, 16, 16};
        auto smaller_tensor = make_tensor(element_type, smaller_shape, data);
        std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor0;
        OV_ASSERT_NO_THROW(zero_tensor0 =
                               std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, smaller_tensor));

        auto bigger_tensor = make_tensor(element_type, shape, data);
        std::shared_ptr<::intel_npu::ZeroTensor> zero_tensor1;
        ASSERT_THROW(zero_tensor1 = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, bigger_tensor),
                     ::intel_npu::ZeroMemException);

        ::operator delete(data, std::align_val_t(4096));
    }
}

TEST_P(ZeroTensorTests, CopySmallerHostTensorFromBiggerOne) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::shared_ptr<::intel_npu::IEngineBackend> engine_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto zero_context = std::make_shared<::intel_npu::RemoteContextImpl>(engine_backend);
    auto shape = Shape{1, 64, 64, 64};

    auto host_tensor = std::make_shared<::intel_npu::ZeroHostTensor>(zero_context, init_struct, element_type, shape);
    void* data = host_tensor->data();

    auto smaller_shape0 = Shape{1, 32, 32, 32};
    auto smaller_tensor0 = make_tensor(element_type, smaller_shape0, host_tensor->data());
    auto zero_tensor0 = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, smaller_tensor0);
    EXPECT_EQ(smaller_shape0, zero_tensor0->get_shape());
    EXPECT_EQ(host_tensor->data(), smaller_tensor0->data());

    auto smaller_shape1 = Shape{1, 16, 16, 16};
    auto smaller_tensor1 = make_tensor(element_type, smaller_shape1, smaller_tensor0->data());
    auto zero_tensor1 = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, smaller_tensor1);
    EXPECT_EQ(smaller_shape1, zero_tensor1->get_shape());
    EXPECT_EQ(host_tensor->data(), smaller_tensor1->data());

    host_tensor = {};
    ASSERT_TRUE(
        ::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), zero_tensor0->data()));

    zero_tensor0 = {};
    smaller_tensor0 = {};
    ASSERT_TRUE(
        ::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), zero_tensor1->data()));

    smaller_tensor1 = {};
    zero_tensor1 = {};
    ASSERT_FALSE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), data));
}

using ZeroTensorTestsCheckDataType = ZeroTensorTests;

TEST_P(ZeroTensorTestsCheckDataType, CopyZeroTensorAndCheckTensorDataType) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto shape = Shape{1, 2, 2, 2};

    auto zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, element_type, shape, true);
    EXPECT_EQ(element_type, zero_tensor->get_element_type());

    auto copy_zero_tensor = std::make_shared<::intel_npu::ZeroTensor>(init_struct, npu_config, zero_tensor);
    EXPECT_EQ(zero_tensor->get_element_type(), copy_zero_tensor->get_element_type());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
