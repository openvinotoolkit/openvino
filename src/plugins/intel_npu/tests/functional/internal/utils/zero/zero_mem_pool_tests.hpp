// Copyright (C) 2018-2026 Intel Corporation
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
#include "internal/compiler_adapter/zero_init_mock.hpp"
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
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
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

        init_struct = nullptr;
        APIBaseTest::TearDown();
    }
};

TEST_P(ZeroMemPoolTests, GetZeroMemoryData) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    void* data = ::operator new(4096, std::align_val_t(4096));

    std::shared_ptr<::intel_npu::ZeroMem> get_zero_mem;

    if (init_struct->isExternalMemoryStandardAllocationSupported()) {
        OV_ASSERT_NO_THROW(get_zero_mem =
                               ::intel_npu::zero_mem::import_standard_allocation_memory(init_struct, data, 4096));
        ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(init_struct->getContext(), data));
    } else {
        ASSERT_THROW(get_zero_mem = ::intel_npu::zero_mem::import_standard_allocation_memory(init_struct, data, 4096),
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
        std::array<void*, 3> data;

        // Prep: add few entries in the pool
        for (int i = 0; i < 3; i++) {
            data[i] = ::operator new(4096, std::align_val_t(4096));
            zero_mem[i] = ::intel_npu::zero_mem::import_standard_allocation_memory(init_struct, data[i], 4096);
        }
        zero_mem[3] = ::intel_npu::zero_mem::allocate_memory(init_struct, 4096, 4096);
        zero_mem[4] = ::intel_npu::zero_mem::allocate_memory(init_struct, 4096, 4096);

        for (int i = 0; i < threads_no; ++i) {
            threads[i] = std::thread([this, &zero_mem, i]() -> void {
                for (int j = 0; j < 256; j++) {
                    std::shared_ptr<::intel_npu::ZeroMem> get_zero_mem;
                    OV_ASSERT_NO_THROW(
                        get_zero_mem = ::intel_npu::zero_mem::import_standard_allocation_memory(init_struct,
                                                                                                zero_mem[i % 5]->data(),
                                                                                                4096));
                    SLEEP_MS(0);
                }
            });
        }

        for (int i = 0; i < threads_no; ++i) {
            threads[i].join();
        }

        for (int i = 0; i < 3; i++) {
            zero_mem[i] = {};
            ::operator delete(data[i], std::align_val_t(4096));
        }
    }
}

TEST_P(ZeroMemPoolTests, MultiThreadingImportMemoryReUseAndDestroyIt) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (init_struct->isExternalMemoryStandardAllocationSupported()) {
        constexpr int threads_no = 256;
        constexpr int no_of_buffers = 5;
        std::array<std::thread, threads_no> threads;
        std::array<void*, no_of_buffers> data;

        for (int i = 0; i < no_of_buffers; i++) {
            data[i] = ::operator new(4096, std::align_val_t(4096));
        }

        for (int i = 0; i < threads_no; ++i) {
            threads[i] = std::thread([&]() -> void {
                std::shared_ptr<::intel_npu::ZeroMem> zero_mem;
                for (int j = 0; j < threads_no; j++) {
                    OV_ASSERT_NO_THROW(
                        zero_mem = ::intel_npu::zero_mem::import_standard_allocation_memory(init_struct,
                                                                                            data[j % no_of_buffers],
                                                                                            4096));
                    SLEEP_MS(0);
                    if (j % 2 == 0) {
                        zero_mem = {};
                    }
                }

                zero_mem = {};
            });
        }

        for (int i = 0; i < threads_no; ++i) {
            threads[i].join();
        }

        for (int i = 0; i < no_of_buffers; i++) {
            ::operator delete(data[i], std::align_val_t(4096));
        }
    }
}

TEST_P(ZeroMemPoolTests, GetPoolReturnsSamePoolForSameInitStruct) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_init_mock = std::make_shared<::intel_npu::ZeroInitStructsMock>();
    auto local_init_struct_0 = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock);
    auto local_init_struct_1 = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock);

    auto mem0 = ::intel_npu::zero_mem::allocate_memory(local_init_struct_0, 4096, 4096);
    auto mem1 = ::intel_npu::zero_mem::allocate_memory(local_init_struct_1, 4096, 4096);
    ASSERT_EQ(zero_init_mock->getZeroMemPool().mem_pool.size(), 2u);

    mem0 = {};
    mem1 = {};
    ASSERT_EQ(zero_init_mock->getZeroMemPool().mem_pool.size(), 0u);
}

TEST_P(ZeroMemPoolTests, CheckDifferentZeroStructsHolderPools) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto zero_init_mock_0 = std::make_shared<::intel_npu::ZeroInitStructsMock>();
    auto zero_init_mock_1 = std::make_shared<::intel_npu::ZeroInitStructsMock>();
    auto zero_init_mock_2 = std::make_shared<::intel_npu::ZeroInitStructsMock>();

    auto local_init_struct_0 = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock_0);
    auto local_init_struct_1 = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock_1);
    auto local_init_struct_2 = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock_2);

    auto& pool_init_0 = zero_init_mock_0->getZeroMemPool().mem_pool;
    auto& pool_init_1 = zero_init_mock_1->getZeroMemPool().mem_pool;
    auto& pool_init_2 = zero_init_mock_2->getZeroMemPool().mem_pool;

    auto mem0_struct_global = ::intel_npu::zero_mem::allocate_memory(local_init_struct_0, 4096, 4096);
    auto mem0_struct_1 = ::intel_npu::zero_mem::allocate_memory(local_init_struct_1, 4096, 4096);
    auto mem0_struct_2 = ::intel_npu::zero_mem::allocate_memory(local_init_struct_2, 4096, 4096);
    auto mem1_struct_2 = ::intel_npu::zero_mem::allocate_memory(local_init_struct_2, 4096, 4096);

    ASSERT_TRUE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(local_init_struct_0->getContext(),
                                                                            mem0_struct_global->data()));
    ASSERT_FALSE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(local_init_struct_2->getContext(),
                                                                             mem0_struct_1->data()));
    ASSERT_FALSE(::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(local_init_struct_0->getContext(),
                                                                             mem1_struct_2->data()));

    ASSERT_EQ(pool_init_0.size(), 1u);
    ASSERT_EQ(pool_init_1.size(), 1u);
    ASSERT_EQ(pool_init_2.size(), 2u);

    mem0_struct_2 = {};
    ASSERT_EQ(pool_init_2.size(), 1u);
    local_init_struct_2 = {};
    zero_init_mock_2 = {};
    ASSERT_EQ(pool_init_2.size(), 1u);
}

TEST_P(ZeroMemPoolTests, MultiThreadingDifferentZeroInitStructsHaveIsolatedPools) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    constexpr size_t threads_per_pool = 32;
    constexpr size_t total_threads = threads_per_pool * 3;

    auto zero_init_mock_0 = std::make_shared<::intel_npu::ZeroInitStructsMock>();
    auto zero_init_mock_1 = std::make_shared<::intel_npu::ZeroInitStructsMock>();
    auto zero_init_mock_2 = std::make_shared<::intel_npu::ZeroInitStructsMock>();

    auto local_init_struct_0 = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock_0);
    auto local_init_struct_1 = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock_1);
    auto local_init_struct_2 = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock_2);

    std::array<std::thread, total_threads> threads;
    std::array<std::shared_ptr<::intel_npu::ZeroMem>, threads_per_pool> mem_pool_0;
    std::array<std::shared_ptr<::intel_npu::ZeroMem>, threads_per_pool> mem_pool_1;
    std::array<std::shared_ptr<::intel_npu::ZeroMem>, threads_per_pool> mem_pool_2;

    for (size_t i = 0; i < threads_per_pool; ++i) {
        threads[i] = std::thread([&, i]() -> void {
            OV_ASSERT_NO_THROW(mem_pool_0[i] = ::intel_npu::zero_mem::allocate_memory(local_init_struct_0, 4096, 4096));
        });
        threads[i + threads_per_pool] = std::thread([&, i]() -> void {
            OV_ASSERT_NO_THROW(mem_pool_1[i] = ::intel_npu::zero_mem::allocate_memory(local_init_struct_1, 4096, 4096));
        });
        threads[i + (threads_per_pool * 2)] = std::thread([&, i]() -> void {
            OV_ASSERT_NO_THROW(mem_pool_2[i] = ::intel_npu::zero_mem::allocate_memory(local_init_struct_2, 4096, 4096));
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    ASSERT_EQ(zero_init_mock_0->getZeroMemPool().mem_pool.size(), threads_per_pool);
    ASSERT_EQ(zero_init_mock_1->getZeroMemPool().mem_pool.size(), threads_per_pool);
    ASSERT_EQ(zero_init_mock_2->getZeroMemPool().mem_pool.size(), threads_per_pool);

    for (size_t i = 0; i < threads_per_pool; ++i) {
        mem_pool_0[i] = {};
        mem_pool_1[i] = {};
        mem_pool_2[i] = {};
    }

    ASSERT_EQ(zero_init_mock_0->getZeroMemPool().mem_pool.size(), 0u);
    ASSERT_EQ(zero_init_mock_1->getZeroMemPool().mem_pool.size(), 0u);
    ASSERT_EQ(zero_init_mock_2->getZeroMemPool().mem_pool.size(), 0u);
}

TEST_P(ZeroMemPoolTests, MultiThreadingImportMemoryReUseAndDestroyItWithMultipleInitStructs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    if (!init_struct->isExternalMemoryStandardAllocationSupported()) {
        GTEST_SKIP() << "Test requires support for importing standard allocated memory as external memory, which is "
                        "not available on this platform.";
    }

    constexpr size_t init_structs_no = 3;
    constexpr size_t threads_per_pool = 32;
    constexpr size_t no_of_buffers = 5;

    auto zero_init_mock_0 = std::make_shared<::intel_npu::ZeroInitStructsMock>();
    auto zero_init_mock_1 = std::make_shared<::intel_npu::ZeroInitStructsMock>();
    auto zero_init_mock_2 = std::make_shared<::intel_npu::ZeroInitStructsMock>();

    std::array<std::shared_ptr<::intel_npu::ZeroInitStructsHolder>, init_structs_no> init_structs = {
        std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock_0),
        std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock_1),
        std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zero_init_mock_2)};

    std::array<std::array<void*, no_of_buffers>, init_structs_no> data = {};
    for (size_t s = 0; s < init_structs_no; ++s) {
        for (size_t i = 0; i < no_of_buffers; ++i) {
            data[s][i] = ::operator new(4096, std::align_val_t(4096));
        }
    }

    std::array<std::thread, threads_per_pool * init_structs_no> threads;
    for (size_t s = 0; s < init_structs_no; ++s) {
        for (size_t i = 0; i < threads_per_pool; ++i) {
            threads[s * threads_per_pool + i] = std::thread([&, s]() -> void {
                std::shared_ptr<::intel_npu::ZeroMem> zero_mem;
                for (size_t j = 0; j < threads_per_pool; ++j) {
                    OV_ASSERT_NO_THROW(
                        zero_mem = ::intel_npu::zero_mem::import_standard_allocation_memory(init_structs[s],
                                                                                            data[s][j % no_of_buffers],
                                                                                            4096));
                    SLEEP_MS(0);
                    if (j % 2 == 0) {
                        zero_mem = {};
                    }
                }
                zero_mem = {};
            });
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    ASSERT_EQ(zero_init_mock_0->getZeroMemPool().mem_pool.size(), 0u);
    ASSERT_EQ(zero_init_mock_1->getZeroMemPool().mem_pool.size(), 0u);
    ASSERT_EQ(zero_init_mock_2->getZeroMemPool().mem_pool.size(), 0u);

    for (size_t s = 0; s < init_structs_no; ++s) {
        for (size_t i = 0; i < no_of_buffers; ++i) {
            ::operator delete(data[s][i], std::align_val_t(4096));
        }
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
