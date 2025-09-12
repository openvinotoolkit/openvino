// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "wrappers.hpp"

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "driver_compiler_adapter.hpp"
#include "ir_serializer.hpp"

// there is something wrong in here
void ZeroWrappersTest::SetUp() {
    int flag;
    std::string extVersion;
    std::tie(flag, extVersion) = GetParam();
    model = ov::test::utils::make_multi_single_conv();
    zeroInitStruct = std::make_shared<ZeroInitStructsMock>(extVersion);

    zeGraphExt = std::make_shared<ZeGraphExtWrappers>(std::reinterpret_pointer_cast<ZeroInitStructsHolder>(zeroInitStruct));

    auto compilerProperties = zeroInitStruct->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);
    std::cout << " aaa \n\n";
    buildFlags = "";

    // should this be here?
    graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, flag);
    std::cout << " aaa \n\n";
}

void ZeroWrappersTest::TearDown() {}

// coverage on all ifelse branches
TEST_P(ZeroWrappersTest, QueryGraph) {
    const auto supportedLayers = zeGraphExt->queryGraph(std::move(serializedIR), buildFlags);
}

TEST_P(ZeroWrappersTest, GetGraphBinary) {
    // zeGraphExt->getGraphBinary(graphDescriptor, __, __, __);
}

TEST_P(ZeroWrappersTest, InitializeGraph) {
    // int max?
    auto commandQueueGroupOrdinal =
        zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

    zeGraphExt->initializeGraph(graphDescriptor, commandQueueGroupOrdinal);
}

TEST_P(ZeroWrappersTest, DestroyGraph) {
    zeGraphExt->destroyGraph(graphDescriptor);
}

std::vector<int> _graphDescflags = {ZE_GRAPH_FLAG_NONE,
                                   ZE_GRAPH_FLAG_DISABLE_CACHING,
                                   ZE_GRAPH_FLAG_ENABLE_PROFILING,
                                   ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT};

std::vector<std::string> _extVersion =
    {"1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12", "1.13"};

INSTANTIATE_TEST_SUITE_P(something,
                         ZeroWrappersTest,
                         ::testing::Combine(::testing::ValuesIn(_graphDescflags), ::testing::ValuesIn(_extVersion)),
                         ZeroWrappersTest::getTestCaseName);

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
