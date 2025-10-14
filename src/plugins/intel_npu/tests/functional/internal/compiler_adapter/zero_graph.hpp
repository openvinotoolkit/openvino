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
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "ir_serializer.hpp"
#include "ze_graph_ext_wrappers.hpp"
#include "zero_init_mock.hpp"

using namespace intel_npu;

namespace {
size_t calculate_alignment_with_padding(size_t size, size_t alignment) {
    return size + alignment - (size % alignment);
}

size_t get_file_size(std::ifstream& file) {
    auto size = file.tellg();
    file.seekg(0, std::ios::end);
    size = file.tellg() - size;
    file.seekg(0, std::ios::beg);

    return size;
}

void* allocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                           const size_t bytes,
                           const size_t alignment) noexcept {
    size_t size = bytes + alignment - (bytes % alignment);

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                     nullptr,
                                     ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED};
    void* data = nullptr;
    auto result = intel_npu::zeMemAllocHost(init_structs->getContext(), &desc, size, alignment, &data);

    if (result == ZE_RESULT_SUCCESS) {
        return data;
    } else {
        OPENVINO_THROW("L0 zeMemAllocHost result: %s, code %#X - %s",
                       ze_result_to_string(result).c_str(),
                       uint64_t(result),
                       ze_result_to_description(result).c_str());
        return nullptr;
    }
}

void deallocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, void* handle) noexcept {
    auto result = intel_npu::zeMemFree(init_structs->getContext(), handle);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 zeMemFree result: %s, code %#X - %s",
                       ze_result_to_string(result).c_str(),
                       uint64_t(result),
                       ze_result_to_description(result).c_str());
    }
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

    std::shared_ptr<driver_compiler_utils::IRSerializer> irSerializer;

    std::string blobPath;

    int extVersion;

    int graphDescFlag;

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<int, int>> obj) {
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
        auto compilerProperties = zeroInitStruct->getCompilerProperties();
        const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
        const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
        serializedIR = irSerializer->serializeIR(model, compilerVersion, maxOpsetVersion);
    }

protected:
    void SetUp() override {
        using namespace ::driver_compiler_utils;

        std::tie(graphDescFlag, extVersion) = GetParam();

        const std::string BLOB_NAME = "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
        blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + BLOB_NAME;

        model = ov::test::utils::make_multi_single_conv();

        std::shared_ptr<ZeroInitStructsMock> zeroInitMock = std::make_shared<ZeroInitStructsMock>(extVersion);

        zeroInitStruct = std::reinterpret_pointer_cast<ZeroInitStructsHolder>(zeroInitMock);

        zeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);

        auto compilerProperties = zeroInitStruct->getCompilerProperties();
        const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
        irSerializer = std::make_shared<IRSerializer>(IRSerializer(model, maxOpsetVersion));
    }

    void TearDown() override {
        zeGraphExt->destroyGraph(graphDescriptor);
    }
};

TEST_P(ZeroGraphTest, GetGraphInitIR) {
    serializeIR();
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    uint32_t initCommandQueueOrdinal = 0;
    OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                           zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                   ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal));
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

    const uint8_t* blobPtr = nullptr;
    OV_ASSERT_NO_THROW(zeGraphExt->getGraphBinary(graphDescriptor, blob, blobPtr, size));
}

TEST_P(ZeroGraphTest, SetGraphArgOnNullBuffer) {
    serializeIR();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocate_zero_memory(zeroInitStruct, totalSize, ::utils::STANDARD_PAGE_SIZE));

    ASSERT_ANY_THROW(
        zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, nullptr));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, ptr));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphAlignedMemoryIR) {
    serializeIR();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocate_zero_memory(zeroInitStruct, totalSize, ::utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, ptr));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, ptr));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphAlignedMemoryMallocBlob) {
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = get_file_size(blobStream);
    size = calculate_alignment_with_padding(size, ::utils::STANDARD_PAGE_SIZE);

    uint8_t* blob = static_cast<uint8_t*>(::operator new(size, std::align_val_t(::utils::STANDARD_PAGE_SIZE)));
    blobStream.read(reinterpret_cast<char*>(blob), size);
    blobStream.close();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, ::utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphAlignedMemoryZeMemAllocBlob) {
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = get_file_size(blobStream);
    size = calculate_alignment_with_padding(size, ::utils::STANDARD_PAGE_SIZE);

    void* blob = nullptr;
    OV_ASSERT_NO_THROW(blob = allocate_zero_memory(zeroInitStruct, size, ::utils::STANDARD_PAGE_SIZE))
    blobStream.read(reinterpret_cast<char*>(blob), size);
    blobStream.close();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, ::utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphNotAlignedMemoryMallocBlob) {
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = get_file_size(blobStream);

    uint8_t* blob = static_cast<uint8_t*>(::operator new(size, std::align_val_t(::utils::STANDARD_PAGE_SIZE)));
    blobStream.read(reinterpret_cast<char*>(blob), size);
    blobStream.close();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, ::utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, static_cast<char*>(buffer) + 1));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphNotAlignedMemoryZeMemAllocBlob) {
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = get_file_size(blobStream);

    void* blob = nullptr;
    OV_ASSERT_NO_THROW(blob = allocate_zero_memory(zeroInitStruct, size, ::utils::STANDARD_PAGE_SIZE))
    blobStream.read(reinterpret_cast<char*>(blob), size);
    blobStream.close();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, ::utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, static_cast<char*>(buffer) + 1));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}

TEST_P(ZeroGraphTest, SetGraphArgsOnDestroyedBlob) {
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = get_file_size(blobStream);
    size = calculate_alignment_with_padding(size, ::utils::STANDARD_PAGE_SIZE);

    uint8_t* blob = static_cast<uint8_t*>(::operator new(size, std::align_val_t(::utils::STANDARD_PAGE_SIZE)));
    blobStream.read(reinterpret_cast<char*>(blob), size);
    blobStream.close();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, ::utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}
}  // namespace ov::test::behavior
