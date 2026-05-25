// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/vcl/vcl_allocator.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "model_serializer.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov::test::behavior {

class VclAllocatorFuncTests : public ::testing::TestWithParam<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
        auto targetDevice = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        std::ostringstream result;
        result << "targetDevice=" << ov::test::utils::getDeviceNameTestCase(targetDevice) << "_";
        result << "targetPlatform=" + ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
        return result.str();
    }

protected:
    std::string targetDevice;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<::intel_npu::vcl_allocator_3> allocator;

    void SetUp() override {
        targetDevice = GetParam();
        model = ov::test::utils::make_conv_pool_relu();
        allocator = std::make_shared<::intel_npu::vcl_allocator_3>();

        try {
            std::string ov_lib_dir = ov::test::utils::getOpenvinoLibDirectory();
            ::intel_npu::VCLApi::getInstance(ov_lib_dir);
        } catch (const std::exception&) {
            GTEST_SKIP() << "Couldn't load compiler library";
        }
    }

    // Helper struct and function to reduce code duplication
    struct CompilerSetupState {
        std::string buildFlags;
        ::intel_npu::SerializedIR serializedIR;
        vcl_compiler_handle_t compiler = nullptr;
        vcl_log_handle_t logHandle = nullptr;

        CompilerSetupState() = default;
        CompilerSetupState(const CompilerSetupState&) = delete;
        CompilerSetupState& operator=(const CompilerSetupState&) = delete;
        CompilerSetupState(CompilerSetupState&& other) noexcept
            : buildFlags(std::move(other.buildFlags)),
              serializedIR(std::move(other.serializedIR)),
              compiler(other.compiler),
              logHandle(other.logHandle) {
            other.compiler = nullptr;
            other.logHandle = nullptr;
        }
        CompilerSetupState& operator=(CompilerSetupState&& other) noexcept {
            if (this != &other) {
                release();
                buildFlags = std::move(other.buildFlags);
                serializedIR = std::move(other.serializedIR);
                compiler = other.compiler;
                logHandle = other.logHandle;
                other.compiler = nullptr;
                other.logHandle = nullptr;
            }
            return *this;
        }

        ~CompilerSetupState() {
            release();
        }

    private:
        void release() {
            if (compiler != nullptr) {
                ::intel_npu::vclCompilerDestroy(compiler);
                compiler = nullptr;
            }
        }
    };

    CompilerSetupState createCompilerAndDescriptor() {
        CompilerSetupState state;

        state.buildFlags = ::intel_npu::compiler_utils::serializeIOInfo(model, true);

        auto platform = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
        if (platform.find("NPU") == 0) {
            platform = platform.substr(3);  // Remove "NPU" prefix if present
        }

        state.buildFlags +=
            " --config NPU_PLATFORM=\"" + platform + "\" NPU_COMPILATION_MODE_PARAMS=\"optimization-level=0\"";

        vcl_version_info_t vclVersion = {};
        vcl_version_info_t vclProfilingVersion = {};
        ::intel_npu::vclGetVersion(&vclVersion, &vclProfilingVersion);

        vcl_compiler_desc_t compilerDesc = {};
        compilerDesc.version = vclVersion;
        compilerDesc.debugLevel =
            static_cast<__vcl_log_level_t>(static_cast<int>(::intel_npu::Logger::global().level()) + 1);

        uint32_t defaultTileCount = std::numeric_limits<uint32_t>::max();
        if (vclVersion.major == 7 && vclVersion.minor < 6) {
            defaultTileCount = std::numeric_limits<uint16_t>::max();
        }

        vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t),
                                        0x00,
                                        std::numeric_limits<uint16_t>::max(),
                                        defaultTileCount};

        ::intel_npu::vclCompilerCreate(&compilerDesc, &deviceDesc, &state.compiler, &state.logHandle);

        if (state.compiler == nullptr) {
            ADD_FAILURE() << "vclCompilerCreate failed";
            return state;
        }

        ze_graph_compiler_version_info_t vclVersionInfo = {0, 0};
        vcl_compiler_properties_t compilerProp = {};
        ::intel_npu::vclCompilerGetProperties(state.compiler, &compilerProp);
        vclVersionInfo.major = compilerProp.version.major;
        vclVersionInfo.minor = compilerProp.version.minor;

        auto isOptionValueSupportedByCompiler = [](const std::string&, const std::optional<std::string>&) {
            return true;
        };

        state.serializedIR =
            ::intel_npu::compiler_utils::serializeIR(model,
                                                     vclVersionInfo,
                                                     10,
                                                     ::intel_npu::MODEL_SERIALIZER_VERSION::defaultValue(),
                                                     isOptionValueSupportedByCompiler);

        state.buildFlags += " NPU_MODEL_SERIALIZER_VERSION=\"" +
                            ::intel_npu::MODEL_SERIALIZER_VERSION::toString(state.serializedIR.serializerVersion) +
                            "\"";

        return state;
    }
};

TEST_P(VclAllocatorFuncTests, VerifyAllocation) {
    uint8_t* blobBuffer = nullptr;
    uint64_t blobSize = 0;
    uint8_t* compatibilityReqBuffer = nullptr;
    uint64_t compatibilityReqSize = 0;

    auto setup = createCompilerAndDescriptor();
    ASSERT_NE(setup.compiler, nullptr);

    vcl_executable_desc_t desc = {setup.serializedIR.buffer.get(),
                                  setup.serializedIR.size,
                                  setup.buildFlags.c_str(),
                                  setup.buildFlags.size()};

    auto result = ::intel_npu::vclAllocatedExecutableCreate3(setup.compiler,
                                                             desc,
                                                             allocator.get(),
                                                             &blobBuffer,
                                                             &blobSize,
                                                             &compatibilityReqBuffer,
                                                             &compatibilityReqSize);

    EXPECT_EQ(result, VCL_RESULT_SUCCESS);
    EXPECT_NE(blobBuffer, nullptr);
    EXPECT_NE(compatibilityReqBuffer, nullptr);

    // Verify it tracked exactly these two buffers
    // 2 allocations: blob & compatibility string
    EXPECT_EQ(allocator->m_info.size(), 2);

    auto blobIt = std::find_if(allocator->m_info.begin(),
                               allocator->m_info.end(),
                               [blobBuffer](const std::pair<uint8_t*, size_t>& item) {
                                   return item.first == blobBuffer;
                               });
    EXPECT_NE(blobIt, allocator->m_info.end());

    auto compatIt = std::find_if(allocator->m_info.begin(),
                                 allocator->m_info.end(),
                                 [compatibilityReqBuffer](const std::pair<uint8_t*, size_t>& item) {
                                     return item.first == compatibilityReqBuffer;
                                 });
    EXPECT_NE(compatIt, allocator->m_info.end());
}

TEST_P(VclAllocatorFuncTests, VerifyDeallocation) {
    uint8_t* blobBuffer = nullptr;
    uint64_t blobSize = 0;
    uint8_t* compatibilityReqBuffer = nullptr;
    uint64_t compatibilityReqSize = 0;

    auto setup = createCompilerAndDescriptor();
    ASSERT_NE(setup.compiler, nullptr);

    vcl_executable_desc_t desc = {setup.serializedIR.buffer.get(),
                                  setup.serializedIR.size,
                                  setup.buildFlags.c_str(),
                                  setup.buildFlags.size()};

    auto result = ::intel_npu::vclAllocatedExecutableCreate3(setup.compiler,
                                                             desc,
                                                             allocator.get(),
                                                             &blobBuffer,
                                                             &blobSize,
                                                             &compatibilityReqBuffer,
                                                             &compatibilityReqSize);

    EXPECT_EQ(result, VCL_RESULT_SUCCESS);

    // Clean up compatibility string
    if (compatibilityReqBuffer != nullptr) {
        allocator->deallocate(allocator.get(), compatibilityReqBuffer);
    }

    // Verify compatibility buffer was erased from tracking vector
    auto compatIt = std::find_if(allocator->m_info.begin(),
                                 allocator->m_info.end(),
                                 [compatibilityReqBuffer](const std::pair<uint8_t*, size_t>& item) {
                                     return item.first == compatibilityReqBuffer;
                                 });
    EXPECT_EQ(compatIt, allocator->m_info.end());
    EXPECT_EQ(allocator->m_info.size(), 1);
}

TEST_P(VclAllocatorFuncTests, VerifyOwnershipTransferToTensor) {
    uint8_t* blobBuffer = nullptr;
    uint64_t blobSize = 0;
    uint8_t* compatibilityReqBuffer = nullptr;
    uint64_t compatibilityReqSize = 0;

    auto setup = createCompilerAndDescriptor();
    ASSERT_NE(setup.compiler, nullptr);

    vcl_executable_desc_t desc = {setup.serializedIR.buffer.get(),
                                  setup.serializedIR.size,
                                  setup.buildFlags.c_str(),
                                  setup.buildFlags.size()};

    auto result = ::intel_npu::vclAllocatedExecutableCreate3(setup.compiler,
                                                             desc,
                                                             allocator.get(),
                                                             &blobBuffer,
                                                             &blobSize,
                                                             &compatibilityReqBuffer,
                                                             &compatibilityReqSize);

    EXPECT_EQ(result, VCL_RESULT_SUCCESS);

    // Retrieve the real allocated size for the blob from the allocator (mirroring compiler_impl.cpp)
    auto it = std::find_if(allocator->m_info.begin(),
                           allocator->m_info.end(),
                           [blobBuffer](const std::pair<uint8_t*, size_t>& item) {
                               return item.first == blobBuffer;
                           });
    ASSERT_NE(it, allocator->m_info.end());
    size_t alignedBlobSize = it->second;

    // Wrap Blob in ov::Tensor to test ownership transfer
    {
        ov::Tensor tensor = ::intel_npu::make_tensor_from_aligned_addr(blobBuffer, alignedBlobSize, allocator);
        EXPECT_EQ(tensor.data(), blobBuffer);
        EXPECT_EQ(tensor.get_byte_size(), alignedBlobSize);

        // Remove only the transferred blob, leaving any possible temporary allocations
        // to be safely freed by ~vcl_allocator_3 (mirroring compiler_impl.cpp)
        allocator->m_info.erase(it);
    }  // tensor goes out of scope here: custom deleter correctly handles the physical memory free

    // Clean up compatibility string manually
    if (compatibilityReqBuffer != nullptr) {
        allocator->deallocate(allocator.get(), compatibilityReqBuffer);
    }

    // After erasing the blob and deallocating the compatibility string,
    // the tracking info in the allocator should be empty.
    EXPECT_EQ(allocator->m_info.size(), 0);
}

TEST_P(VclAllocatorFuncTests, VerifyOrphanMemoryCleanupOnDestruction) {
    uint8_t* blobBuffer = nullptr;
    uint64_t blobSize = 0;
    uint8_t* compatibilityReqBuffer = nullptr;
    uint64_t compatibilityReqSize = 0;

    auto setup = createCompilerAndDescriptor();
    ASSERT_NE(setup.compiler, nullptr);

    vcl_executable_desc_t desc = {setup.serializedIR.buffer.get(),
                                  setup.serializedIR.size,
                                  setup.buildFlags.c_str(),
                                  setup.buildFlags.size()};

    auto result = ::intel_npu::vclAllocatedExecutableCreate3(setup.compiler,
                                                             desc,
                                                             allocator.get(),
                                                             &blobBuffer,
                                                             &blobSize,
                                                             &compatibilityReqBuffer,
                                                             &compatibilityReqSize);

    EXPECT_EQ(result, VCL_RESULT_SUCCESS);
    EXPECT_EQ(allocator->m_info.size(), 2);  // 2 allocations: blob & compatibility string

    // Purpose: Simulate VCL crash or early return where the memory is temporarily tracked as an orphan leak
    // When the allocator goes out of scope and gets destroyed, it should cleanly free all tracked memory

    allocator.reset();

    EXPECT_EQ(allocator, nullptr);
}

}  // namespace ov::test::behavior
