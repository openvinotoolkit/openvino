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
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_mem.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

#ifdef _WIN32
#    include <windows.h>
#    define SLEEP_MS(x) Sleep(x)
#else
#    include <unistd.h>
#    define SLEEP_MS(x) usleep((x) * 1000)
#endif

using CompilationParams = std::string;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {
class ZeroMemPoolTests : public ov::test::behavior::OVPluginTestBase,
                         public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config npu_config = ::intel_npu::Config(options);

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string target_device;
        ov::AnyMap configuration;
        target_device = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '_');
        target_device = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        std::ostringstream result;
        result << "targetDevice=" << target_device << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(target_device) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        target_device = this->GetParam();

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

TEST_P(ZeroMemPoolTests, GetZeroMemoryData) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    void* data = ::operator new(4096, std::align_val_t(4096));

    std::shared_ptr<::intel_npu::ZeroMem> get_zero_mem;

    if (init_struct->isExternalMemoryStandardAllocationSupported()) {
        OV_ASSERT_NO_THROW(
            get_zero_mem =
                ::intel_npu::ZeroMemPool::get_instance().import_standard_allocation_memory(init_struct, data, 4096));
        ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), data));
    } else {
        ASSERT_THROW(
            get_zero_mem =
                ::intel_npu::ZeroMemPool::get_instance().import_standard_allocation_memory(init_struct, data, 4096),
            ::intel_npu::ZeroMemException);
        ASSERT_FALSE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), data));
    }

    get_zero_mem = {};
    ASSERT_FALSE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), data));

    ::operator delete(data, std::align_val_t(4096));
}

TEST_P(ZeroMemPoolTests, MultiThreadingReUseAlreadyAllocatedImportedMemory) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (init_struct->isExternalMemoryStandardAllocationSupported()) {
        const int threads_no = 256;
        std::array<std::thread, threads_no> threads;
        std::array<std::shared_ptr<::intel_npu::ZeroMem>, 5> zero_mem;
        std::array<void*, 4> data;

        for (int i = 0; i < 4; i++) {
            data[i] = ::operator new(4096, std::align_val_t(4096));
            zero_mem[i] =
                ::intel_npu::ZeroMemPool::get_instance().import_standard_allocation_memory(init_struct, data[i], 4096);
        }

        zero_mem[4] = ::intel_npu::ZeroMemPool::get_instance().allocate_zero_memory(init_struct, 4096, 4096);

        for (int i = 0; i < threads_no; ++i) {
            threads[i] = std::thread([this, &zero_mem, i]() -> void {
                for (int j = 0; j < 256; j++) {
                    auto get_zero_mem = ::intel_npu::ZeroMemPool::get_instance().import_standard_allocation_memory(
                        init_struct,
                        zero_mem[i % 5]->data(),
                        4096);
                    SLEEP_MS(0);
                    get_zero_mem = {};
                }
            });
        }

        for (int i = 0; i < threads_no; ++i) {
            threads[i].join();
        }

        for (int i = 0; i < 4; i++) {
            ::operator delete(data[i], std::align_val_t(4096));
        }
    }
}

TEST_P(ZeroMemPoolTests, MultiThreadingImportMemoryReUseAndDestroyIt) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const int threads_no = 128;
    const int no_of_buffers = 5;
    std::array<std::thread, threads_no> threads;
    std::array<void*, no_of_buffers> data;

    for (int i = 0; i < no_of_buffers; i++) {
        data[i] = ::operator new(4096, std::align_val_t(4096));
    }

    for (int i = 0; i < threads_no; ++i) {
        threads[i] = std::thread([this, &data, &threads_no, &no_of_buffers]() -> void {
            for (int j = 0; j < threads_no; j++) {
                std::shared_ptr<::intel_npu::ZeroMem> zero_mem;
                try {
                    zero_mem = ::intel_npu::ZeroMemPool::get_instance().import_standard_allocation_memory(
                        init_struct,
                        data[j % no_of_buffers],
                        4096);
                } catch (::intel_npu::ZeroMemException&) {
                    zero_mem = ::intel_npu::ZeroMemPool::get_instance().allocate_zero_memory(init_struct, 4096, 4096);
                }

                SLEEP_MS(0);
                if (j % 2 == 0) {
                    zero_mem = {};
                }
            }
        });
    }

    for (int i = 0; i < threads_no; ++i) {
        threads[i].join();
    }

    for (int i = 0; i < no_of_buffers; i++) {
        ::operator delete(data[i], std::align_val_t(4096));
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
