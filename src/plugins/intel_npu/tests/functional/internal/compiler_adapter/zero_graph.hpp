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
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "vcl_serializer.hpp"
#include "ze_graph_ext_wrappers.hpp"
#include "zero_init_mock.hpp"

using namespace intel_npu;

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
class ZeroGraphTest : public ::testing::TestWithParam<std::tuple<int, int>> {
public:
    std::shared_ptr<ZeroInitStructsHolder> zeroInitStruct;

    std::shared_ptr<ZeGraphExtWrappers> zeGraphExt;

    SerializedIR serializedIR;

    GraphDescriptor graphDescriptor;

    std::shared_ptr<ov::Model> model;

    std::shared_ptr<driver_compiler_utils::VCLSerializerWithWeightsCopy> vclSerializer;

    std::string blobPath;

    int extVersion;

    int graphDescFlag;

    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<int, int>>& obj) {
        int flag, version;
        std::tie(flag, version) = obj.param;
        std::string targetDevice = ov::test::utils::DEVICE_NPU;

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "graphDescriptorFlag=" + std::to_string(flag) << "_";
        result << "extVersion=" + std::to_string(ZE_MAJOR_VERSION(version)) + "." +
                      std::to_string(ZE_MINOR_VERSION(version));
        return result.str();
    }

    void serializeIR() {
        serializedIR = vclSerializer->serialize();
    }

protected:
    void SetUp() override {
        using namespace ::driver_compiler_utils;

        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(graphDescFlag, extVersion) = GetParam();

        const std::string BLOB_NAME = "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
        blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + BLOB_NAME;

        model = ov::test::utils::make_multi_single_conv();

        std::shared_ptr<ZeroInitStructsMock> zeroInitMock = std::make_shared<ZeroInitStructsMock>(extVersion);

        zeroInitStruct = std::reinterpret_pointer_cast<ZeroInitStructsHolder>(zeroInitMock);

        zeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);

        auto compilerProperties = zeroInitStruct->getCompilerProperties();
        const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
        vclSerializer =
            std::make_shared<VCLSerializerWithWeightsCopy>(model, compilerProperties.compilerVersion, maxOpsetVersion);
    }

    void TearDown() override {
        zeGraphExt->destroyGraph(graphDescriptor);
    }
};

using ZeroGraphCompilationTests = ZeroGraphTest;

TEST_P(ZeroGraphCompilationTests, GetGraphInitIR) {
    serializeIR();
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));
}

TEST_P(ZeroGraphCompilationTests, GetInitSetArgsDestroyGraphAlignedMemoryIR) {
    serializeIR();
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));

    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    std::unique_ptr<ZeroMem> buffer;
    OV_ASSERT_NO_THROW(buffer =
                           std::make_unique<ZeroMem>(zeroInitStruct, totalSize, ::utils::STANDARD_PAGE_SIZE, false));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer->data()));
}

TEST_P(ZeroGraphTest, GetGraphInitBlob) {
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
}

TEST_P(ZeroGraphTest, GetNetworkMeta) {
    serializeIR();
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    OV_ASSERT_NO_THROW(NetworkMetadata meta = zeGraphExt->getNetworkMeta(graphDescriptor));
}

TEST_P(ZeroGraphTest, QueryGraph) {
    serializeIR();
    OV_ASSERT_NO_THROW(zeGraphExt->queryGraph(std::move(serializedIR), ""));
}

TEST_P(ZeroGraphTest, GetGraphBinary) {
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
}

TEST_P(ZeroGraphTest, SetGraphArgOnNullBuffer) {
    serializeIR();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));

    ASSERT_ANY_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, nullptr));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphAlignedMemoryMallocBlob) {
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

    std::unique_ptr<ZeroMem> buffer;
    OV_ASSERT_NO_THROW(buffer = std::make_unique<ZeroMem>(zeroInitStruct, size, ::utils::STANDARD_PAGE_SIZE, false));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer->data()));

    ::operator delete(blob, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphNotAlignedMemoryMallocBlob) {
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

    std::unique_ptr<ZeroMem> buffer;
    OV_ASSERT_NO_THROW(buffer = std::make_unique<ZeroMem>(zeroInitStruct, size, ::utils::STANDARD_PAGE_SIZE, false));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, static_cast<char*>(buffer->data()) + 1));

    ::operator delete(blob, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
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
        OV_ASSERT_NO_THROW(localGraphDescriptor = localZeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));
        const uint8_t* blobPtr = nullptr;
        std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers
        size_t blobSize;
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
        OV_ASSERT_NO_THROW(localGraphDescriptor = localZeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));
        const uint8_t* blobPtr = nullptr;
        std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers
        size_t blobSize;
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
        OV_ASSERT_NO_THROW(localGraphDescriptor = localZeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));
        const uint8_t* blobPtr = nullptr;
        std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers
        size_t blobSize;
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

}  // namespace ov::test::behavior
