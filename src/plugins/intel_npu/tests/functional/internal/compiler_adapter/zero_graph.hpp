// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <stdlib.h>

#include <common_test_utils/test_assertions.hpp>

#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_constants.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_mem.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "vcl_serializer.hpp"
#include "ze_graph_ext_wrappers.hpp"
#include "zero_init_mock.hpp"

using namespace intel_npu;

using CompilationParamsAndExtensionVersion = std::tuple<std::string,  // Device name
                                                        ov::AnyMap,   // Config
                                                        int>;         // Extension Version
namespace {
size_t calculate_size_with_alignment_padding(size_t size, size_t alignment) {
    return size + alignment - (size % alignment);
}

size_t get_file_size(std::ifstream& file) {
    auto size = file.tellg();
    file.seekg(0, std::ios::end);
    size = file.tellg() - size;
    file.seekg(0, std::ios::beg);

    return size;
}
}  // namespace

namespace ov::test::behavior {
class ZeroGraphTest : public ::testing::TestWithParam<CompilationParamsAndExtensionVersion> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParamsAndExtensionVersion>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        int graphExtVersion;
        std::tie(targetDevice, configuration, graphExtVersion) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        result << "graphExtVersion=" + std::to_string(ZE_MAJOR_VERSION(graphExtVersion)) + "." +
                      std::to_string(ZE_MINOR_VERSION(graphExtVersion));

        return result.str();
    }

protected:
    void SetUp() override {
        using namespace ::driver_compiler_utils;

        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(targetDevice, configuration, graphExtVersion) = this->GetParam();

        const std::string BLOB_NAME = "blob_compat_dummy_model_MTL_ov_2025_4_0_driver_2020509.blob";
        blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + BLOB_NAME;

        model = ov::test::utils::make_multi_single_conv();

        std::shared_ptr<ZeroInitStructsMock> zeroInitMock = std::make_shared<ZeroInitStructsMock>(graphExtVersion);
        zeroInitStruct = std::reinterpret_pointer_cast<ZeroInitStructsHolder>(zeroInitMock);
        zeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);
    }

    void TearDown() override {
        zeGraphExt->destroyGraph(graphDescriptor);
    }

    void serializeIR() {
        auto compilerProperties = zeroInitStruct->getCompilerProperties();
        const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
        serializedIR =
            driver_compiler_utils::serializeIR(model, compilerProperties.compilerVersion, maxOpsetVersion, true);
    }

    bool bypassUmdCache() {
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                if (configItem.first == ov::cache_dir.name()) {
                    const auto set_cache_dir = configItem.second;
                    if (!set_cache_dir.empty()) {
                        return true;
                    }
                }
                if (configItem.first == ov::intel_npu::bypass_umd_caching.name()) {
                    if (configItem.second.as<bool>()) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct;
    std::shared_ptr<ZeGraphExtWrappers> zeGraphExt;
    ov::AnyMap configuration;

    SerializedIR serializedIR;
    GraphDescriptor graphDescriptor;

    std::shared_ptr<ov::Model> model;

    std::string targetDevice;
    std::string blobPath;
    int graphExtVersion;
};

using ZeroGraphCompilationTests = ZeroGraphTest;

TEST_P(ZeroGraphCompilationTests, GetGraphInitIR) {
    serializeIR();
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));

    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));
}

TEST_P(ZeroGraphCompilationTests, GetInitSetArgsDestroyGraphAlignedMemoryIR) {
    serializeIR();
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));

    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));

    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    std::shared_ptr<ZeroMem> buffer;
    OV_ASSERT_NO_THROW(buffer = ZeroMemPool::get_instance().allocate_zero_memory(zeroInitStruct,
                                                                                 totalSize,
                                                                                 ::utils::STANDARD_PAGE_SIZE,
                                                                                 false));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer->data()));
}

TEST_P(ZeroGraphTest, GetGraphInitBlob) {
    std::string platform = ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice);
    size_t pos = platform.find("3720");
    if (pos != std::string::npos) {
        std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
        ASSERT_TRUE(blobStream.is_open());
        size_t size = get_file_size(blobStream);

        std::vector<uint8_t> blob(size);
        blobStream.read(reinterpret_cast<char*>(blob.data()), size);
        blobStream.close();

        OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob.data(), blob.size()));

        uint32_t initCommandQueueOrdinal = 0;
        OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                               zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                       ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
        OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));
    } else {
        GTEST_SKIP() << "Skip due to incompatible blob format on this platform.";
    }
}

TEST_P(ZeroGraphTest, GetNetworkMeta) {
    serializeIR();
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));

    OV_ASSERT_NO_THROW(NetworkMetadata meta = zeGraphExt->getNetworkMeta(graphDescriptor));
}

TEST_P(ZeroGraphTest, QueryGraph) {
    serializeIR();
    OV_ASSERT_NO_THROW(zeGraphExt->queryGraph(std::move(serializedIR), ""));
}

TEST_P(ZeroGraphTest, GetGraphBinary) {
    std::string platform = ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice);
    size_t pos = platform.find("3720");
    if (pos != std::string::npos) {
        std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
        ASSERT_TRUE(blobStream.is_open());
        size_t size = get_file_size(blobStream);

        std::vector<uint8_t> blob(size);
        blobStream.read(reinterpret_cast<char*>(blob.data()), size);
        blobStream.close();

        OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob.data(), blob.size()));

        uint32_t initCommandQueueOrdinal = 0;
        OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                               zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                       ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
        OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));

        const uint8_t* blobPtr = nullptr;
        OV_ASSERT_NO_THROW(zeGraphExt->getGraphBinary(graphDescriptor, blob, blobPtr, size));
    } else {
        GTEST_SKIP() << "Skip due to incompatible blob format on this platform.";
    }
}

TEST_P(ZeroGraphTest, SetGraphArgOnNullBuffer) {
    serializeIR();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));

    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));

    ASSERT_ANY_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, nullptr));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphAlignedMemoryMallocBlob) {
    std::string platform = ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice);
    size_t pos = platform.find("3720");
    if (pos != std::string::npos) {
        std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
        ASSERT_TRUE(blobStream.is_open());
        size_t size = get_file_size(blobStream);
        size = calculate_size_with_alignment_padding(size, ::utils::STANDARD_PAGE_SIZE);

        uint8_t* blob = static_cast<uint8_t*>(::operator new(size, std::align_val_t(::utils::STANDARD_PAGE_SIZE)));
        blobStream.read(reinterpret_cast<char*>(blob), size);
        blobStream.close();

        OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

        uint32_t initCommandQueueOrdinal = 0;
        OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                               zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                       ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
        OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));

        std::shared_ptr<ZeroMem> buffer;
        OV_ASSERT_NO_THROW(buffer = ZeroMemPool::get_instance().allocate_zero_memory(zeroInitStruct,
                                                                                     size,
                                                                                     ::utils::STANDARD_PAGE_SIZE,
                                                                                     false));

        OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer->data()));

        ::operator delete(blob, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
    } else {
        GTEST_SKIP() << "Skip due to incompatible blob format on this platform.";
    }
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphNotAlignedMemoryMallocBlob) {
    std::string platform = ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice);
    size_t pos = platform.find("3720");
    if (pos != std::string::npos) {
        std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
        ASSERT_TRUE(blobStream.is_open());
        size_t size = get_file_size(blobStream);

        uint8_t* blob = static_cast<uint8_t*>(::operator new(size, std::align_val_t(::utils::STANDARD_PAGE_SIZE)));
        blobStream.read(reinterpret_cast<char*>(blob), size);
        blobStream.close();

        OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

        uint32_t initCommandQueueOrdinal = 0;
        OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                               zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                       ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
        OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));

        std::shared_ptr<ZeroMem> buffer;
        OV_ASSERT_NO_THROW(buffer = ZeroMemPool::get_instance().allocate_zero_memory(zeroInitStruct,
                                                                                     size,
                                                                                     ::utils::STANDARD_PAGE_SIZE,
                                                                                     false));

        OV_ASSERT_NO_THROW(
            zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, static_cast<char*>(buffer->data()) + 1));

        ::operator delete(blob, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
    } else {
        GTEST_SKIP() << "Skip due to incompatible blob format on this platform.";
    }
}

TEST_P(ZeroGraphTest, SetUnalignedAddressBlob) {
    // create blob -> compile model first
    void* blob = nullptr;
    void* blobAddressUnaligned = nullptr;
    size_t alignedSize;
    {
        // use local zeGraphExt;
        auto localZeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);
        GraphDescriptor localGraphDescriptor;
        serializeIR();
        OV_ASSERT_NO_THROW(localGraphDescriptor =
                               localZeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));
        const uint8_t* blobPtr = nullptr;
        std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers
        size_t blobSize = 0;
        OV_ASSERT_NO_THROW(localZeGraphExt->getGraphBinary(localGraphDescriptor, blobVec, blobPtr, blobSize));

        alignedSize = calculate_size_with_alignment_padding(blobSize, ::utils::STANDARD_PAGE_SIZE);
        size_t unalignedSize = alignedSize + 16;
        blob = ::operator new(unalignedSize, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
        blobAddressUnaligned = static_cast<uint8_t*>(blob) + 16;
        std::memcpy(blobAddressUnaligned, blobPtr, blobSize);
        localZeGraphExt->destroyGraph(localGraphDescriptor);
    }

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blobAddressUnaligned, alignedSize));
    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));
    ASSERT_FALSE(zeGraphExt->isBlobDataImported(graphDescriptor));

    ::operator delete(blob, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
}

TEST_P(ZeroGraphTest, SetUnalignedSizeBlob) {
    // create blob -> compile model first
    void* blob = nullptr;
    size_t alignedSize;
    size_t unalignedSize;
    {
        // use local zeGraphExt;
        auto localZeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);
        GraphDescriptor localGraphDescriptor;
        serializeIR();
        OV_ASSERT_NO_THROW(localGraphDescriptor =
                               localZeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));
        const uint8_t* blobPtr = nullptr;
        std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers
        size_t blobSize = 0;
        OV_ASSERT_NO_THROW(localZeGraphExt->getGraphBinary(localGraphDescriptor, blobVec, blobPtr, blobSize));

        alignedSize = calculate_size_with_alignment_padding(blobSize, ::utils::STANDARD_PAGE_SIZE);
        unalignedSize = alignedSize + 16;
        blob = ::operator new(unalignedSize, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
        std::memcpy(blob, blobPtr, blobSize);
        localZeGraphExt->destroyGraph(localGraphDescriptor);
    }

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, unalignedSize));

    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));

    ASSERT_FALSE(zeGraphExt->isBlobDataImported(graphDescriptor));

    ::operator delete(blob, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
}

TEST_P(ZeroGraphTest, SetAlignedBlob) {
    // create blob -> compile model first
    void* blob = nullptr;
    size_t alignedSize;
    {
        // use local zeGraphExt;
        auto localZeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);
        GraphDescriptor localGraphDescriptor;
        serializeIR();
        OV_ASSERT_NO_THROW(localGraphDescriptor =
                               localZeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));
        const uint8_t* blobPtr = nullptr;
        std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers
        size_t blobSize = 0;
        OV_ASSERT_NO_THROW(localZeGraphExt->getGraphBinary(localGraphDescriptor, blobVec, blobPtr, blobSize));

        alignedSize = calculate_size_with_alignment_padding(blobSize, ::utils::STANDARD_PAGE_SIZE);
        blob = ::operator new(alignedSize, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
        std::memcpy(blob, blobPtr, blobSize);
        localZeGraphExt->destroyGraph(localGraphDescriptor);
    }

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, alignedSize));
    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));
    if (zeroInitStruct->getGraphDdiTable().version() >= ZE_MAKE_VERSION(1, 13) &&
        zeroInitStruct->isExternalMemoryStandardAllocationSupported()) {
        ASSERT_TRUE(zeGraphExt->isBlobDataImported(graphDescriptor));
    } else {
        ASSERT_FALSE(zeGraphExt->isBlobDataImported(graphDescriptor));
    }

    ::operator delete(blob, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
}

#ifdef _WIN32
TEST_P(ZeroGraphTest, CheckNoThrowOnUnsupportedFeature) {
    if (zeroInitStruct->getGraphDdiTable().version() >= ZE_MAKE_VERSION(1, 11)) {
        // Driver shall return NO_THROW_ON_UNSUPPORTED_FEATURE as supported to go further here
        if (zeroInitStruct->getGraphDdiTable().pfnCompilerIsOptionSupported(zeroInitStruct->getDevice(),
                                                                            ZE_NPU_DRIVER_OPTIONS,
                                                                            "NO_THROW_ON_UNSUPPORTED_FEATURE",
                                                                            nullptr) != ZE_RESULT_SUCCESS) {
            ADD_FAILURE() << "NO_THROW_ON_UNSUPPORTED_FEATURE shall be supported.";
        }
    }
}
#endif

}  // namespace ov::test::behavior
