// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_graph.hpp"

#include <stdlib.h>

#include <common_test_utils/test_assertions.hpp>

#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "ir_serializer.hpp"

namespace {
size_t calculate_alignment_with_padding(size_t size, size_t alignment) {
    return size + alignment - (size % alignment);
}
}  // namespace

using namespace intel_npu::driver_compiler_utils;

void ZeroGraphTest::SetUp() {
    std::tie(graphDescFlag, extVersion) = GetParam();

    model = ov::test::utils::make_multi_single_conv();

    std::shared_ptr<ZeroInitStructsMock> zeroInitMock = std::make_shared<ZeroInitStructsMock>(extVersion);

    zeroInitStruct = std::reinterpret_pointer_cast<ZeroInitStructsHolder>(zeroInitMock);

    zeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);

    auto compilerProperties = zeroInitStruct->getCompilerProperties();
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    irSerializer = std::make_shared<IRSerializer>(IRSerializer(model, maxOpsetVersion));
}

void ZeroGraphTest::TearDown() {
    zeGraphExt->destroyGraph(graphDescriptor);
}

void ZeroGraphTest::serializeIR() {
    auto compilerProperties = zeroInitStruct->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    serializedIR = irSerializer->serializeIR(model, compilerVersion, maxOpsetVersion);
}

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
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH +
                          "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    auto size = blobStream.tellg();
    blobStream.seekg(0, std::ios::end);
    size = blobStream.tellg() - size;
    blobStream.seekg(0, std::ios::beg);

    std::vector<uint8_t> blob(size);
    blobStream.read(reinterpret_cast<char*>(blob.data()), size);

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

    NetworkMetadata meta;
    // init matters?
    OV_ASSERT_NO_THROW(meta = zeGraphExt->getNetworkMeta(graphDescriptor));
}

TEST_P(ZeroGraphTest, QueryGraph) {
    serializeIR();
    OV_ASSERT_NO_THROW(zeGraphExt->queryGraph(std::move(serializedIR), ""));
}

TEST_P(ZeroGraphTest, GetGraphBinary) {
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH +
                          "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = blobStream.tellg();
    blobStream.seekg(0, std::ios::end);
    size = static_cast<size_t>(blobStream.tellg()) - size;
    blobStream.seekg(0, std::ios::beg);

    std::vector<uint8_t> blob(size);
    blobStream.read(reinterpret_cast<char*>(blob.data()), size);

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob.data(), blob.size()));

    const uint8_t* blobPtr = nullptr;
    OV_ASSERT_NO_THROW(zeGraphExt->getGraphBinary(graphDescriptor, blob, blobPtr, size));
}

TEST_P(ZeroGraphTest, SetGraphArgOnNullBuffer) {
    serializeIR();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocate_zero_memory(zeroInitStruct, totalSize, utils::STANDARD_PAGE_SIZE));

    ASSERT_ANY_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, nullptr));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, ptr));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphAlignedMemoryIR) {
    serializeIR();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocate_zero_memory(zeroInitStruct, totalSize, utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, ptr));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, ptr));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphAlignedMemoryMallocBlob) {
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH +
                          "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = blobStream.tellg();
    blobStream.seekg(0, std::ios::end);
    size = static_cast<size_t>(blobStream.tellg()) - size;
    blobStream.seekg(0, std::ios::beg);
    size = calculate_alignment_with_padding(size, utils::STANDARD_PAGE_SIZE);

    uint8_t* blob = static_cast<uint8_t*>(::operator new(size, std::align_val_t(utils::STANDARD_PAGE_SIZE)));
    blobStream.read(reinterpret_cast<char*>(blob), size);

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphAlignedMemoryZeMemAllocBlob) {
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH +
                          "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = blobStream.tellg();
    blobStream.seekg(0, std::ios::end);
    size = static_cast<size_t>(blobStream.tellg()) - size;
    blobStream.seekg(0, std::ios::beg);
    size = calculate_alignment_with_padding(size, utils::STANDARD_PAGE_SIZE);

    void* blob = nullptr;
    OV_ASSERT_NO_THROW(blob = allocate_zero_memory(zeroInitStruct, size, utils::STANDARD_PAGE_SIZE))
    blobStream.read(reinterpret_cast<char*>(blob), size);

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphNotAlignedMemoryIR) {
    serializeIR();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocate_zero_memory(zeroInitStruct, totalSize, utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, static_cast<char*>(ptr) + 1));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, ptr));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphNotAlignedMemoryMallocBlob) {
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH +
                          "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = blobStream.tellg();
    blobStream.seekg(0, std::ios::end);
    size = static_cast<size_t>(blobStream.tellg()) - size;
    blobStream.seekg(0, std::ios::beg);

    uint8_t* blob = static_cast<uint8_t*>(::operator new(size, std::align_val_t(utils::STANDARD_PAGE_SIZE)));
    blobStream.read(reinterpret_cast<char*>(blob), size);

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, static_cast<char*>(buffer) + 1));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphNotAlignedMemoryZeMemAllocBlob) {
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH +
                          "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = blobStream.tellg();
    blobStream.seekg(0, std::ios::end);
    size = static_cast<size_t>(blobStream.tellg()) - size;
    blobStream.seekg(0, std::ios::beg);

    void* blob = nullptr;
    OV_ASSERT_NO_THROW(blob = allocate_zero_memory(zeroInitStruct, size, utils::STANDARD_PAGE_SIZE))
    blobStream.read(reinterpret_cast<char*>(blob), size);

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, static_cast<char*>(buffer) + 1));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}

TEST_P(ZeroGraphTest, SetGraphArgsOnDestroyedBlob) {
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH +
                          "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = blobStream.tellg();
    blobStream.seekg(0, std::ios::end);
    size = static_cast<size_t>(blobStream.tellg()) - size;
    blobStream.seekg(0, std::ios::beg);
    size = calculate_alignment_with_padding(size, utils::STANDARD_PAGE_SIZE);

    uint8_t* blob = static_cast<uint8_t*>(::operator new(size, std::align_val_t(utils::STANDARD_PAGE_SIZE)));
    blobStream.read(reinterpret_cast<char*>(blob), size);

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, size));

    void* buffer = nullptr;
    OV_ASSERT_NO_THROW(buffer = allocate_zero_memory(zeroInitStruct, size, utils::STANDARD_PAGE_SIZE));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, buffer));
}

std::vector<int> graphDescflags = {ZE_GRAPH_FLAG_NONE,
                                   ZE_GRAPH_FLAG_DISABLE_CACHING,
                                   ZE_GRAPH_FLAG_ENABLE_PROFILING,
                                   ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT};

auto extVersions = ::testing::Range(ZE_MAKE_VERSION(1, 5), ZE_GRAPH_EXT_VERSION_CURRENT + 1);

INSTANTIATE_TEST_SUITE_P(something,
                         ZeroGraphTest,
                         ::testing::Combine(::testing::ValuesIn(graphDescflags), extVersions),
                         ZeroGraphTest::getTestCaseName);

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
