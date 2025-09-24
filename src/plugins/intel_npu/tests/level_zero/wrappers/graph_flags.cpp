// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_flags.hpp"

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "driver_compiler_adapter.hpp"
#include "zero_graph.hpp"

void ZeroGraphFlagsTest::SetUp() {
    std::string extVersion = GetParam();

    model = ov::test::utils::make_multi_single_conv();

    zeroInitMock = std::make_shared<ZeroInitStructsMock>(extVersion);

    zeroInitStruct = std::reinterpret_pointer_cast<ZeroInitStructsHolder>(zeroInitMock);

    zeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);

    auto compilerProperties = zeroInitStruct->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);
}

void ZeroGraphFlagsTest::TearDown() {}

std::vector<int> _flags = {ZE_GRAPH_FLAG_NONE,
                           ZE_GRAPH_FLAG_DISABLE_CACHING,
                           ZE_GRAPH_FLAG_ENABLE_PROFILING,
                           ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT};

TEST_P(ZeroGraphFlagsTest, QueryGraph) {
    uint32_t flagsCombined = 0;
    while (std::next_permutation(_flags.begin(), _flags.end())) {
        flagsCombined = 0;
        for (int flag : _flags) {
            flagsCombined |= flag;
        }

        OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", flagsCombined));
        uint32_t initCommandQueueOrdinal = 0;
        OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                               zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                       ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
        zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);
        ASSERT_NE(graphDescriptor._handle, nullptr);

        const auto supportedLayers = zeGraphExt->queryGraph(serializedIR, "");
        ASSERT_NE(supportedLayers.size(), 0);
        zeGraphExt->destroyGraph(graphDescriptor);
    }
}

// info about current tested flags inside test is not printed with getTestCaseName
// maybe some logs would suffice?
TEST_P(ZeroGraphFlagsTest, InitializeGraph) {
    uint32_t flagsCombined = 0;
    while (std::next_permutation(_flags.begin(), _flags.end())) {
        flagsCombined = 0;
        for (int flag : _flags) {
            flagsCombined |= flag;
        }

        OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", flagsCombined));
        uint32_t initCommandQueueOrdinal = 0;
        OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                               zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                       ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
        zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);
        ASSERT_NE(graphDescriptor._handle, nullptr);

        zeGraphExt->destroyGraph(graphDescriptor);
    }
}

TEST_P(ZeroGraphFlagsTest, DestroyGraph) {
    uint32_t flagsCombined = 0;
    while (std::next_permutation(_flags.begin(), _flags.end())) {
        flagsCombined = 0;
        for (int flag : _flags) {
            flagsCombined |= flag;
        }

        OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", flagsCombined));
        uint32_t initCommandQueueOrdinal = 0;
        OV_ASSERT_NO_THROW(initCommandQueueOrdinal =
                               zeroUtils::findCommandQueueGroupOrdinal(zeroInitStruct->getDevice(),
                                                                       ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
        zeGraphExt->initializeGraph(graphDescriptor, initCommandQueueOrdinal);
        ASSERT_NE(graphDescriptor._handle, nullptr);

        zeGraphExt->destroyGraph(graphDescriptor);
        ASSERT_EQ(graphDescriptor._handle, nullptr);
    }
}

std::vector<std::string> __extVersion = {"1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12", "1.13"};

INSTANTIATE_TEST_SUITE_P(something,
                         ZeroGraphFlagsTest,
                         ::testing::ValuesIn(__extVersion),
                         ZeroGraphFlagsTest::getTestCaseName);
