// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_graph.hpp"

#include <stdlib.h>

#include <common_test_utils/test_assertions.hpp>

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "ir_serializer.hpp"

void ZeroGraphTest::SetUp() {
    std::tie(graphDescFlag, extVersion) = GetParam();

    model = ov::test::utils::make_multi_single_conv();

    zeroInitMock = std::make_shared<ZeroInitStructsMock>(extVersion);

    zeroInitStruct = std::reinterpret_pointer_cast<ZeroInitStructsHolder>(zeroInitMock);

    zeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);

    auto compilerProperties = zeroInitStruct->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);
}

void ZeroGraphTest::TearDown() {}

TEST_P(ZeroGraphTest, GetGraphDescriptorIR) {
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));
    auto initCommandQueueOrdinal = zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
    zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);
    zeGraphExt->destroyGraph(graphDescriptor);
}

TEST_P(ZeroGraphTest, GetGraphDescriptorBlob) {
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    auto size = blobStream.tellg();
    blobStream.seekg(0, std::ios::end);
    size = blobStream.tellg() - size;
    blobStream.seekg(0, std::ios::beg);

    std::vector<uint8_t> blob(size);
    blobStream.read(reinterpret_cast<char*>(blob.data()), size);

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob.data(), blob.size()));
    zeGraphExt->destroyGraph(graphDescriptor);
}

TEST_P(ZeroGraphTest, GetNetworkMeta) {
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));
    auto initCommandQueueOrdinal = zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
    zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);
    NetworkMetadata meta;
    // asserts on meta contents?
    OV_ASSERT_NO_THROW(meta = zeGraphExt->getNetworkMeta(graphDescriptor));
    zeGraphExt->destroyGraph(graphDescriptor);
}

TEST_P(ZeroGraphTest, GetIODescriptor) {

}

TEST_P(ZeroGraphTest, InitializeGraph) {
    graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag);
    auto initCommandQueueOrdinal = zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
    zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);
    zeGraphExt->destroyGraph(graphDescriptor);
}

// the "fourth" branch is not being tested
// todo: revise this
TEST_P(ZeroGraphTest, QueryGraph) {
    // add checks for set emptyness?
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));
    auto initCommandQueueOrdinal = zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
    zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);
    const auto supportedLayers = zeGraphExt->queryGraph(std::move(serializedIR), "");
    zeGraphExt->destroyGraph(graphDescriptor);
}

TEST_P(ZeroGraphTest, GetGraphBinary) {
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + "blob_compatibility_dummy_model_MTL_ov_2025_1_0_driver_1003967.blob";
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
    ASSERT_TRUE(blobStream.is_open());
    size_t size = blobStream.tellg();
    blobStream.seekg(0, std::ios::end);
    size = static_cast<size_t>(blobStream.tellg()) - size;
    blobStream.seekg(0, std::ios::beg);

    std::vector<uint8_t> blob(size);
    blobStream.read(reinterpret_cast<char*>(blob.data()), size);

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob.data(), blob.size()));
    auto initCommandQueueOrdinal = zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
    zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);

    // maybe without initializeGraph it wont work?
    const uint8_t* blobPtr = nullptr;
    zeGraphExt->getGraphBinary(graphDescriptor, blob, blobPtr, size);
    zeGraphExt->destroyGraph(graphDescriptor);
}

TEST_P(ZeroGraphTest, DestroyGraph) {
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));
    auto initCommandQueueOrdinal = zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
    zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);
    zeGraphExt->destroyGraph(graphDescriptor);
    ASSERT_EQ(graphDescriptor._handle, nullptr);
}

// two test cases: aligned and non-aligned memory given, despite using ...WRITE_COMBINED
TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraph) { // TODO: add asserts for no throws
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", graphDescFlag));

    auto initCommandQueueOrdinal = zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);
    zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);

    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocate_zero_memory(zeroInitStruct, totalSize, utils::STANDARD_PAGE_SIZE));
    // to check if its persistent or not
    // if persistent: and if i destroy the blob after init, then setGraph should fail?
    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, ptr));

    OV_ASSERT_NO_THROW(deallocate_zero_memory(zeroInitStruct, ptr));
    zeGraphExt->destroyGraph(graphDescriptor);
}


std::vector<int> graphDescflags = {ZE_GRAPH_FLAG_NONE,
                                   ZE_GRAPH_FLAG_DISABLE_CACHING,
                                   ZE_GRAPH_FLAG_ENABLE_PROFILING,
                                   ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT};

// combine with range so as it will be compile time
// maybe not all the early versions are supported
std::vector<std::string> extVersion =
    {"1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12", "1.13"};

INSTANTIATE_TEST_SUITE_P(something,
                         ZeroGraphTest,
                         ::testing::Combine(::testing::ValuesIn(graphDescflags), ::testing::ValuesIn(extVersion)),
                         ZeroGraphTest::getTestCaseName);

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
void checkedMemcpy(void* destination, size_t destinationSize, const void* source, size_t numberOfBytes) {
    if (numberOfBytes == 0) {
        return;
    }

    OPENVINO_ASSERT(destination != nullptr, "Memcpy: received a null destination address");
    OPENVINO_ASSERT(source != nullptr, "Memcpy: received a null source address");
    OPENVINO_ASSERT(numberOfBytes <= destinationSize,
                    "Memcpy: the source buffer does not fit inside the destination one");
    OPENVINO_ASSERT(numberOfBytes <= (destination > source ? ((uintptr_t)destination - (uintptr_t)source)
                                                           : ((uintptr_t)source - (uintptr_t)destination)),
                    "Memcpy: the offset between the two buffers does not allow a safe execution of the operation");

    memcpy(destination, source, numberOfBytes);
}

// todo: maybe we can avoid this
// what about a synthetic model?
// does ZeroWrappers display different behaviors on IRv10 vs IRv11?
SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion) {
    driver_compiler_utils::IRSerializer irSerializer(model, supportedOpsetVersion);

    // Contract between adapter and compiler in driver
    const uint32_t maxNumberOfElements = 10;
    const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
    const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

    const uint32_t numberOfInputData = 2;
    const uint64_t xmlSize = static_cast<uint64_t>(irSerializer.getXmlSize());
    const uint64_t weightsSize = static_cast<uint64_t>(irSerializer.getWeightsSize());

    OPENVINO_ASSERT(numberOfInputData < maxNumberOfElements);
    if (xmlSize >= maxSizeOfXML) {
        OPENVINO_THROW("Xml file is too big to process. xmlSize: ", xmlSize, " >= maxSizeOfXML: ", maxSizeOfXML);
    }
    if (weightsSize >= maxSizeOfWeights) {
        OPENVINO_THROW("Bin file is too big to process. xmlSize: ",
                       weightsSize,
                       " >= maxSizeOfWeights: ",
                       maxSizeOfWeights);
    }

    const uint64_t sizeOfSerializedIR = sizeof(compilerVersion) + sizeof(numberOfInputData) + sizeof(xmlSize) +
                                        xmlSize + sizeof(weightsSize) + weightsSize;

    // use array to avoid vector's memory zeroing overhead
    std::shared_ptr<uint8_t> buffer(new uint8_t[sizeOfSerializedIR], std::default_delete<uint8_t[]>());
    uint8_t* serializedIR = buffer.get();

    uint64_t offset = 0;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &compilerVersion, sizeof(compilerVersion));
    offset += sizeof(compilerVersion);

    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    // xml data is filled in serializeModel()
    uint64_t xmlOffset = offset;
    offset += xmlSize;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    // weights data is filled in serializeModel()
    uint64_t weightsOffset = offset;
    offset += weightsSize;

    irSerializer.serializeModelToBuffer(serializedIR + xmlOffset, serializedIR + weightsOffset);

    OPENVINO_ASSERT(offset == sizeOfSerializedIR);

    return std::make_pair(sizeOfSerializedIR, buffer);
}

void* allocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const size_t bytes, const size_t alignment) noexcept {
    size_t size = bytes + alignment - (bytes % alignment);

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED};
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
