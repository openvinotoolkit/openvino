// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <stdlib.h>

#include <common_test_utils/test_assertions.hpp>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common/zero_init_mock.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "driver_compiler_adapter.hpp"
#include "graph.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_mem.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "model_serializer.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/util/codec_xor.hpp"
#include "plugin_compiler_adapter.hpp"
#include "ze_graph_ext_wrappers.hpp"

using namespace intel_npu;

using CompilationParamsAndExtensionVersion = std::tuple<std::string,  // Device name
                                                        ov::AnyMap,   // Config
                                                        int>;         // Extension Version
namespace {
size_t calculate_size_with_alignment_padding(size_t size, size_t alignment) {
    return size + alignment - (size % alignment);
}
}  // namespace

namespace ov::test::behavior {
class ZeroGraphTest : public ::testing::TestWithParam<CompilationParamsAndExtensionVersion> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParamsAndExtensionVersion>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        uint32_t zeGraphNpuExtVersion;
        std::tie(targetDevice, configuration, zeGraphNpuExtVersion) = obj.param;
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
        result << "zeGraphNpuExtVersion=" + std::to_string(ZE_MAJOR_VERSION(zeGraphNpuExtVersion)) + "." +
                      std::to_string(ZE_MINOR_VERSION(zeGraphNpuExtVersion));

        return result.str();
    }

    size_t make_blob() {
        size_t alignedSize = 0;
        // use local zeGraphExt;
        auto localZeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);
        GraphDescriptor localGraphDescriptor = {};
        serializeIR();
        localGraphDescriptor = localZeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache());
        const uint8_t* blobPtr = nullptr;
        std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers
        size_t blobSize = 0;
        localZeGraphExt->getGraphBinary(localGraphDescriptor, blobVec, blobPtr, blobSize);

        alignedSize = calculate_size_with_alignment_padding(blobSize, ::utils::STANDARD_PAGE_SIZE);
        blob = ::operator new(alignedSize, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
        std::memcpy(blob, blobPtr, blobSize);
        localZeGraphExt->destroyGraph(localGraphDescriptor);

        return alignedSize;
    }

protected:
    void SetUp() override {
        _logGuard.emplace(ov::log::Level::ERR);
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(targetDevice, configuration, zeGraphNpuExtVersion) = this->GetParam();

        model = ov::test::utils::make_multi_single_conv();
        blob = nullptr;

        std::shared_ptr<ZeroInitStructsMock> zeroInitMock =
            std::make_shared<ZeroInitStructsMock>(::intel_npu::test_constants::TARGET_ZE_DRIVER_NPU_EXT_VERSION,
                                                  zeGraphNpuExtVersion);
        zeroInitStruct = std::reinterpret_pointer_cast<ZeroInitStructsHolder>(zeroInitMock);
        zeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);
    }

    void TearDown() override {
        if (zeGraphExt != nullptr) {
            zeGraphExt->destroyGraph(graphDescriptor);
        }
        if (blob) {
            ::operator delete(blob, std::align_val_t(::utils::STANDARD_PAGE_SIZE));
        }
        _logGuard.reset();
    }

    void serializeIR() {
        auto compilerProperties = zeroInitStruct->getCompilerProperties();
        const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
        // The test is not concerned with validating the serialization algorithm. Choose the "all-weights-copy" as the
        // safest version.
        serializedIR =
            ::intel_npu::compiler_utils::serializeIR(model,
                                                     compilerProperties.compilerVersion,
                                                     maxOpsetVersion,
                                                     ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY,
                                                     [](const std::string&, const std::optional<std::string>&) {
                                                         return true;
                                                     });
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

    ::intel_npu::SerializedIR serializedIR;
    GraphDescriptor graphDescriptor;

    std::shared_ptr<ov::Model> model;
    void* blob;

    std::string targetDevice;
    std::string blobPath;
    uint32_t zeGraphNpuExtVersion;
    std::optional<utils::LoggerLevelGuard> _logGuard;
};

using ZeroGraphCompilationTests = ZeroGraphTest;

TEST_P(ZeroGraphCompilationTests, GetGraphInitIR) {
    serializeIR();
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));
}

TEST_P(ZeroGraphCompilationTests, GetInitSetArgsDestroyGraphAlignedMemoryIR) {
    serializeIR();
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));

    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    std::shared_ptr<ZeroMem> buffer;
    OV_ASSERT_NO_THROW(
        buffer = ::intel_npu::zero_mem::allocate_memory(zeroInitStruct, totalSize, ::utils::STANDARD_PAGE_SIZE, false));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer->data()));
}

TEST_P(ZeroGraphTest, GetGraphInitBlob) {
    size_t alignedSize = 0;
    OV_ASSERT_NO_THROW(alignedSize = make_blob());

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, alignedSize));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));
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
    size_t alignedSize = 0;
    OV_ASSERT_NO_THROW(alignedSize = make_blob());

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, alignedSize));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));

    const uint8_t* blobPtr = nullptr;
    std::vector<uint8_t> blobVec;
    OV_ASSERT_NO_THROW(zeGraphExt->getGraphBinary(graphDescriptor, blobVec, blobPtr, alignedSize));
}

TEST_P(ZeroGraphTest, SetGraphArgOnNullBuffer) {
    serializeIR();

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache()));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));

    ASSERT_ANY_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, nullptr));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphAlignedMemoryMallocBlob) {
    size_t alignedSize = 0;
    OV_ASSERT_NO_THROW(alignedSize = make_blob());

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, alignedSize));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));

    std::shared_ptr<ZeroMem> buffer;
    OV_ASSERT_NO_THROW(
        buffer =
            ::intel_npu::zero_mem::allocate_memory(zeroInitStruct, alignedSize, ::utils::STANDARD_PAGE_SIZE, false));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, buffer->data()));
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraphNotAlignedMemoryMallocBlob) {
    size_t alignedSize = 0;
    OV_ASSERT_NO_THROW(alignedSize = make_blob());

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, alignedSize));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));

    std::shared_ptr<ZeroMem> buffer;
    OV_ASSERT_NO_THROW(
        buffer =
            ::intel_npu::zero_mem::allocate_memory(zeroInitStruct, alignedSize, ::utils::STANDARD_PAGE_SIZE, false));

    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, static_cast<char*>(buffer->data()) + 1));
}

TEST_P(ZeroGraphTest, SetUnalignedAddressBlob) {
    // create blob -> compile model first
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
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));
    ASSERT_FALSE(zeGraphExt->isBlobDataImported(graphDescriptor));
}

TEST_P(ZeroGraphTest, SetUnalignedSizeBlob) {
    // create blob -> compile model first
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
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));

    ASSERT_FALSE(zeGraphExt->isBlobDataImported(graphDescriptor));
}

TEST_P(ZeroGraphTest, SetAlignedBlob) {
    // create blob -> compile model first
    size_t alignedSize = 0;
    OV_ASSERT_NO_THROW(alignedSize = make_blob());

    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(blob, alignedSize));
    OV_ASSERT_NO_THROW(zeGraphExt->initializeGraph(graphDescriptor));
    if (zeroInitStruct->getGraphDdiTable().version() >= ZE_MAKE_VERSION(1, 13) &&
        zeroInitStruct->isExternalMemoryStandardAllocationSupported()) {
        ASSERT_TRUE(zeGraphExt->isBlobDataImported(graphDescriptor));
    } else {
        ASSERT_FALSE(zeGraphExt->isBlobDataImported(graphDescriptor));
    }
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

TEST_P(ZeroGraphTest, DontDestroyZeroGraphWhenZeroContextIsDestroyed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::string logs;
    std::mutex logsMutex;

    std::function<void(std::string_view)> logCb = [&](std::string_view msg) {
        std::lock_guard<std::mutex> lock(logsMutex);
        logs.append(msg);
        logs.push_back('\n');
    };

    auto localZeroInitMock = std::make_shared<::intel_npu::ZeroInitStructsMock>();

    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> localZeroInitStruct =
        std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(localZeroInitMock);

    {
        utils::LogCallbackGuard log_callback_guard(logCb);
        utils::LoggerLevelGuard logger_level_guard(ov::log::Level::WARNING);

        auto localZeroGraphExt = std::make_shared<ZeGraphExtWrappers>(localZeroInitStruct);
        serializeIR();
        GraphDescriptor localGraphDescriptor;
        OV_ASSERT_NO_THROW(localGraphDescriptor = localZeroGraphExt->getGraphDescriptor(serializedIR, ""));
        OV_ASSERT_NO_THROW(localZeroGraphExt->initializeGraph(localGraphDescriptor));

        ::intel_npu::ZeroInitStructsMock::destroyContextForInstance(localZeroInitMock);

        try {
            localZeroGraphExt->destroyGraph(localGraphDescriptor);
        } catch (const std::exception& ex) {
            ASSERT_FALSE(true) << ex.what();
        }
    }
    ASSERT_NE(logs.find("Context or graph is null while trying to destroy graph."), std::string::npos);
    ASSERT_EQ(logs.find("failed to destroy graph handle"), std::string::npos);
}

using IsOptionSupported = ZeroGraphTest;

TEST_P(IsOptionSupported, GetCompilerSupportedOptions) {
    std::optional<std::string> compilerSupportedOptions;
    OV_ASSERT_NO_THROW(compilerSupportedOptions = zeGraphExt->getCompilerSupportedOptions());
    if (zeroInitStruct->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 11)) {
        ASSERT_FALSE(compilerSupportedOptions.has_value());
    } else {
        ASSERT_TRUE(compilerSupportedOptions.has_value());
    }
}

TEST_P(IsOptionSupported, PropertyNotSupportedByDriver) {
    std::optional<bool> isOptionSupportedResult;
    OV_ASSERT_NO_THROW(isOptionSupportedResult = zeGraphExt->isOptionSupported("FAKE_OPTION"));
    if (zeroInitStruct->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 11)) {
        ASSERT_FALSE(isOptionSupportedResult.has_value());
    } else {
        ASSERT_TRUE(isOptionSupportedResult.has_value());
        ASSERT_FALSE(isOptionSupportedResult.value());
    }
}

TEST_P(IsOptionSupported, PropertySupportedByDriver) {
    std::optional<bool> isOptionSupportedResult;
    OV_ASSERT_NO_THROW(isOptionSupportedResult = zeGraphExt->isOptionSupported("PERFORMANCE_HINT"));
    if (zeroInitStruct->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 11)) {
        ASSERT_FALSE(isOptionSupportedResult.has_value());
    } else {
        ASSERT_TRUE(isOptionSupportedResult.has_value());
        ASSERT_TRUE(isOptionSupportedResult.value());
    }
}

using EncryptionCallbacks = ZeroGraphTest;

TEST_P(EncryptionCallbacks, EncryptionCallbacksSetSecureCompileFlag) {
    auto localZeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);
    serializeIR();
    if (zeroInitStruct->getGraphDdiTable().version() < ZE_MAKE_VERSION(1, 17)) {
        OV_EXPECT_THROW(
            localZeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache(), /* secureCompile = */ true),
            ov::Exception,
            testing::HasSubstr(
                "Secure compilation was requested, but the current driver version does not support it."));
    } else {
        OV_ASSERT_NO_THROW(
            localZeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache(), /* secureCompile = */ true));
    }
}

// Exercises Graph::set_qos and Graph::get_qos_info directly, both before the graph is initialized (no command queue
// yet, the queue is created from the given parameters) and after (the queue used for execution is queried).
class ZeroGraphQosTest : public ZeroGraphTest {
protected:
    void SetUp() override {
        ZeroGraphTest::SetUp();
        if (IsSkipped()) {
            return;
        }
        if (!zeroInitStruct->isCommandQueueQosSupported()) {
            GTEST_SKIP() << "The QoS feature is not supported by the current driver.\n";
        }
        serializeIR();
    }

    // The Graph takes ownership of its descriptor and destroys it, so each graph gets its own one.
    std::shared_ptr<Graph> createGraph() {
        auto localGraphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, "", bypassUmdCache());
        auto metadata = zeGraphExt->getNetworkMeta(localGraphDescriptor);
        return std::make_shared<Graph>(zeGraphExt,
                                       zeroInitStruct,
                                       localGraphDescriptor,
                                       std::move(metadata),
                                       std::nullopt);
    }

    static Config createConfig(bool sharedCommonQueue, std::optional<ov::QoS> qos = std::nullopt) {
        auto options = std::make_shared<OptionsDesc>();
        options->add<MODEL_PRIORITY>();
        options->add<WORKLOAD_TYPE>();
        options->add<QOS>();
        options->add<TURBO>();
        options->add<RUN_INFERENCES_SEQUENTIALLY>();
        options->add<SHARED_COMMON_QUEUE>();
        options->add<PERF_COUNT>();

        Config config(options);
        config.update({{ov::intel_npu::shared_common_queue.name(), sharedCommonQueue ? "YES" : "NO"}});
        if (qos.has_value()) {
            config.update({{ov::qos.name(), ov::Any(qos.value()).as<std::string>()}});
        }
        return config;
    }

    static CommandQueueParams createParams(bool sharedCommonQueue, std::optional<ov::QoS> qos = std::nullopt) {
        CommandQueueParams params;
        params.sharedCommonQueue = sharedCommonQueue;
        params.qos = qos;
        return params;
    }

    static void expectRequestedQos(const ov::QoSInfo& qosInfo, ov::QoS expectedLevel) {
        EXPECT_EQ(qosInfo.level, expectedLevel);
        EXPECT_TRUE(qosInfo.is_overridden);
    }

    static void expectDriverDefaultQos(const ov::QoSInfo& qosInfo) {
        EXPECT_NE(qosInfo.level, ov::QoS::DEFAULT) << "The effective level is never DEFAULT";
        EXPECT_FALSE(qosInfo.is_overridden);
    }
};

TEST_P(ZeroGraphQosTest, GetQosInfoBeforeInitializeUsesTheGivenParameters) {
    for (bool sharedCommonQueue : {true, false}) {
        auto graph = createGraph();
        ASSERT_FALSE(graph->init_completed());

        ov::QoSInfo qosInfo{};
        OV_ASSERT_NO_THROW(qosInfo = graph->get_qos_info(createParams(sharedCommonQueue)));
        expectDriverDefaultQos(qosInfo);

        OV_ASSERT_NO_THROW(qosInfo = graph->get_qos_info(createParams(sharedCommonQueue, ov::QoS::DEFAULT)));
        expectDriverDefaultQos(qosInfo);

        for (auto level : {ov::QoS::HIGH, ov::QoS::MEDIUM, ov::QoS::LOW, ov::QoS::UTILITY, ov::QoS::ECO}) {
            OV_ASSERT_NO_THROW(qosInfo = graph->get_qos_info(createParams(sharedCommonQueue, level)));
            expectRequestedQos(qosInfo, level);
        }

        // Querying must not initialize the graph as a side effect.
        ASSERT_FALSE(graph->init_completed());
    }
}

TEST_P(ZeroGraphQosTest, SetQosBeforeInitializeUpdatesTheCommandQueueDescriptor) {
    auto graph = createGraph();

    OV_ASSERT_NO_THROW(graph->set_qos(ov::QoS::LOW));
    auto desc = graph->get_command_queue_desc();
    ASSERT_TRUE(desc.qos().has_value());
    EXPECT_EQ(desc.qos().value(), ZE_COMMAND_QUEUE_QOS_EXT_LOW);

    OV_ASSERT_NO_THROW(graph->set_qos(ov::QoS::DEFAULT));
    desc = graph->get_command_queue_desc();
    ASSERT_TRUE(desc.qos().has_value());
    EXPECT_EQ(desc.qos().value(), ZE_COMMAND_QUEUE_QOS_EXT_DEFAULT);

    ASSERT_FALSE(graph->init_completed());
}

TEST_P(ZeroGraphQosTest, InitializeAppliesTheConfiguredQos) {
    for (bool sharedCommonQueue : {true, false}) {
        auto graph = createGraph();
        OV_ASSERT_NO_THROW(graph->initialize(createConfig(sharedCommonQueue, ov::QoS::HIGH)));
        ASSERT_TRUE(graph->init_completed());

        auto desc = graph->get_command_queue_desc();
        ASSERT_TRUE(desc.qos().has_value());
        EXPECT_EQ(desc.qos().value(), ZE_COMMAND_QUEUE_QOS_EXT_HIGH);
        EXPECT_EQ(desc.sharedCommonQueue(), sharedCommonQueue);

        // Once initialized, the queue used for execution is queried and the given parameters are ignored.
        ov::QoSInfo qosInfo{};
        OV_ASSERT_NO_THROW(qosInfo = graph->get_qos_info(createParams(sharedCommonQueue, ov::QoS::ECO)));
        expectRequestedQos(qosInfo, ov::QoS::HIGH);
    }
}

TEST_P(ZeroGraphQosTest, InitializeWithoutQosUsesTheDriverDefault) {
    for (bool sharedCommonQueue : {true, false}) {
        auto graph = createGraph();
        OV_ASSERT_NO_THROW(graph->initialize(createConfig(sharedCommonQueue)));

        EXPECT_FALSE(graph->get_command_queue_desc().qos().has_value());

        ov::QoSInfo qosInfo{};
        OV_ASSERT_NO_THROW(qosInfo = graph->get_qos_info(createParams(sharedCommonQueue)));
        expectDriverDefaultQos(qosInfo);
    }
}

TEST_P(ZeroGraphQosTest, SetQosAfterInitializeIsReportedByGetQosInfo) {
    for (bool sharedCommonQueue : {true, false}) {
        auto graph = createGraph();
        OV_ASSERT_NO_THROW(graph->initialize(createConfig(sharedCommonQueue)));

        for (auto level : {ov::QoS::HIGH, ov::QoS::MEDIUM, ov::QoS::LOW, ov::QoS::UTILITY, ov::QoS::ECO}) {
            OV_ASSERT_NO_THROW(graph->set_qos(level));

            ov::QoSInfo qosInfo{};
            OV_ASSERT_NO_THROW(qosInfo = graph->get_qos_info(createParams(sharedCommonQueue)));
            expectRequestedQos(qosInfo, level);
        }

        OV_ASSERT_NO_THROW(graph->set_qos(ov::QoS::DEFAULT));
        ov::QoSInfo qosInfo{};
        OV_ASSERT_NO_THROW(qosInfo = graph->get_qos_info(createParams(sharedCommonQueue)));
        expectDriverDefaultQos(qosInfo);
    }
}

TEST_P(ZeroGraphQosTest, QosSetBeforeInitializeIsReplacedByTheInitializationConfig) {
    for (bool sharedCommonQueue : {true, false}) {
        auto graph = createGraph();
        OV_ASSERT_NO_THROW(graph->set_qos(ov::QoS::LOW));

        // The plugin keeps its configuration in sync with set_qos, so initialize always receives the latest level.
        OV_ASSERT_NO_THROW(graph->initialize(createConfig(sharedCommonQueue, ov::QoS::UTILITY)));

        ov::QoSInfo qosInfo{};
        OV_ASSERT_NO_THROW(qosInfo = graph->get_qos_info(createParams(sharedCommonQueue)));
        expectRequestedQos(qosInfo, ov::QoS::UTILITY);
    }
}

TEST_P(ZeroGraphQosTest, QosOfOneGraphDoesNotAffectAnother) {
    for (bool sharedCommonQueue : {true, false}) {
        auto graph = createGraph();
        auto otherGraph = createGraph();
        OV_ASSERT_NO_THROW(graph->initialize(createConfig(sharedCommonQueue)));
        OV_ASSERT_NO_THROW(otherGraph->initialize(createConfig(sharedCommonQueue)));

        OV_ASSERT_NO_THROW(graph->set_qos(ov::QoS::ECO));

        expectRequestedQos(graph->get_qos_info(createParams(sharedCommonQueue)), ov::QoS::ECO);
        expectDriverDefaultQos(otherGraph->get_qos_info(createParams(sharedCommonQueue)));
    }
}

}  // namespace ov::test::behavior

namespace {
const std::vector<ov::AnyMap> configsGraphCompilationTests = {{},
                                                              {ov::cache_dir("test")},
                                                              {ov::intel_npu::bypass_umd_caching(true)}};

// tested versions interval is [1.5, CURRENT + 1)
auto graphExtVersions = ::testing::Range(ZE_MAKE_VERSION(1, 5), ZE_GRAPH_EXT_VERSION_CURRENT + 1);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ZeroGraphCompilationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsGraphCompilationTests),
                                            graphExtVersions),
                         ZeroGraphTest::getTestCaseName);

const std::vector<ov::AnyMap> emptyConfigsTests = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ZeroGraphTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(emptyConfigsTests),
                                            graphExtVersions),
                         ZeroGraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         IsOptionSupported,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(emptyConfigsTests),
                                            graphExtVersions),
                         ZeroGraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         EncryptionCallbacks,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(emptyConfigsTests),
                                            graphExtVersions),
                         ZeroGraphTest::getTestCaseName);

// QoS handling does not depend on the graph extension version, so only the current one is used.
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ZeroGraphQosTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(emptyConfigsTests),
                                            ::testing::Values(ZE_GRAPH_EXT_VERSION_CURRENT)),
                         ZeroGraphTest::getTestCaseName);

}  // namespace
