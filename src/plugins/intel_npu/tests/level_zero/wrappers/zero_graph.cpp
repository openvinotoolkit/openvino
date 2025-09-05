// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>
#include "zero_graph.hpp"
#include "zero_memory.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/options.hpp"
#include "wrappers.hpp"
#include <common_test_utils/test_assertions.hpp>
#include <regex>
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"

void ZeroGraphTest::SetUp() {
    std::tie(graphDescFlag, extVersion) = GetParam();
#ifdef _WIN32
    _putenv_s("NPU_ZE_GRAPH_EXT_VERSION", extVersion.c_str());
#else
    setenv("NPU_ZE_GRAPH_EXT_VERSION", extVersion, 1);
#endif

    model = ov::test::utils::make_multi_single_conv();
    auto zeroInitStruct = ZeroInitStructsHolder::getInstance();
    zeGraphExt = std::make_shared<ZeGraphExtWrappers>(zeroInitStruct);

    compilerAdapter = std::make_unique<DriverCompilerAdapter>(zeroInitStruct);

    auto compilerProperties = zeroInitStruct->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

    // auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    // auto cfg = ::intel_npu::Config(opt_desc);
    // buildFlags += cfg.toString();
}

void ZeroGraphTest::TearDown() {
    
}

TEST_P(ZeroGraphTest, GetGraphDescriptor) {
    OV_ASSERT_NO_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, graphDescFlag));
}

TEST_P(ZeroGraphTest, GetGraphDescriptorEmptyIR) {
    SerializedIR emptyIR = {0, {}};
    ASSERT_ANY_THROW(graphDescriptor = zeGraphExt->getGraphDescriptor(emptyIR, buildFlags, graphDescFlag));
}

// maybe for invalid buildFlags too?
TEST_P(ZeroGraphTest, GetGraphDescriptorIOInfoBuildFlags) {
    for (const auto& op : model->get_ops()) {
        if (auto _op = ov::as_type_ptr<ov::op::v0::Parameter>(op)) {
            model->remove_parameter(_op);
            ov::replace_node(_op, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 24, 24}, {-1}));
        }
    }

    ASSERT_EQ(model->get_parameters().empty(), true);
    
    auto compilerProperties = ZeroInitStructsHolder::getInstance()->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));
    buildFlags += serializeIOInfo(model, useIndices);

    graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, graphDescFlag);

    auto meta = zeGraphExt->getNetworkMeta(graphDescriptor);
    ASSERT_EQ(meta.inputs.size(), 0);
    ASSERT_NE(meta.outputs.size(), 0);
    // if (graphDescFlag == ZE_GRAPH_FLAG_ENABLE_PROFILING) {
    //     ASSERT_NE(meta.profilingOutputs.size(), 0);
    // }
}

TEST_P(ZeroGraphTest, GetGraphDescriptorConfigBuildFlags) {
    for (const auto& op : model->get_ops()) {
        if (auto _op = ov::as_type_ptr<ov::op::v0::Parameter>(op)) {
            model->remove_parameter(_op);
            ov::replace_node(_op, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 24, 24}, {-1}));
        }
    }

    ASSERT_EQ(model->get_parameters().empty(), true);
    
    auto compilerProperties = ZeroInitStructsHolder::getInstance()->getCompilerProperties();
    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    auto cfg = ::intel_npu::Config(opt_desc);
    buildFlags += serializeConfig(cfg, compilerProperties.compilerVersion, zeGraphExt);

    graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, graphDescFlag);

    auto meta = zeGraphExt->getNetworkMeta(graphDescriptor);
    ASSERT_EQ(meta.inputs.size(), 0);
    ASSERT_NE(meta.outputs.size(), 0);
    // if (graphDescFlag == ZE_GRAPH_FLAG_ENABLE_PROFILING) {
    //     ASSERT_NE(meta.profilingOutputs.size(), 0);
    // }
}

TEST_P(ZeroGraphTest, GetGraphDescriptorIOInfoConfigBuildFlags) {
    for (const auto& op : model->get_ops()) {
        if (auto _op = ov::as_type_ptr<ov::op::v0::Parameter>(op)) {
            model->remove_parameter(_op);
            ov::replace_node(_op, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 24, 24}, {-1}));
        }
    }

    ASSERT_EQ(model->get_parameters().empty(), true);
    
    auto compilerProperties = ZeroInitStructsHolder::getInstance()->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));
    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    auto cfg = ::intel_npu::Config(opt_desc);
    buildFlags += serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += serializeConfig(cfg, compilerProperties.compilerVersion, zeGraphExt);

    graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, graphDescFlag);

    auto meta = zeGraphExt->getNetworkMeta(graphDescriptor);
    ASSERT_EQ(meta.inputs.size(), 0);
    ASSERT_NE(meta.outputs.size(), 0);
    // if (graphDescFlag == ZE_GRAPH_FLAG_ENABLE_PROFILING) {
    //     ASSERT_NE(meta.profilingOutputs.size(), 0);
    // }
}

TEST_P(ZeroGraphTest, InitializeGraph) {
    auto compilerProperties = ZeroInitStructsHolder::getInstance()->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

    for (int i = 0; i < 2; i++) {
        buildFlags += serializeIOInfo(model, i);
    
        graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, graphDescFlag);
    
        zeGraphExt->initializeGraph(graphDescriptor, 0);
        buildFlags = "";
        buildFlags.clear();
    }
}

TEST_P(ZeroGraphTest, GetInitSetArgsDestroyGraph) {
    // get graph
    for (const auto& op : model->get_ops()) {
        if (auto _op = ov::as_type_ptr<ov::op::v0::Parameter>(op)) {
            model->remove_parameter(_op);
            ov::replace_node(_op, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 24, 24}, {-1}));
        }
    }

    ASSERT_EQ(model->get_parameters().empty(), true);
    
    auto compilerProperties = ZeroInitStructsHolder::getInstance()->getCompilerProperties();
    const ze_graph_compiler_version_info_t& compilerVersion = compilerProperties.compilerVersion;
    const auto maxOpsetVersion = compilerProperties.maxOVOpsetVersionSupported;
    serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));
    buildFlags += serializeIOInfo(model, useIndices);

    graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, graphDescFlag);

    // init graph
    zeGraphExt->initializeGraph(graphDescriptor, 0);

    // set graph args
    auto allocator = std::make_shared<zeroMemory::HostMemAllocator>(ZeroInitStructsHolder::getInstance(), ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
    size_t totalSize = 1 * 3 * 24 * 24 * sizeof(float);
    void* ptr = allocator->allocate(totalSize);
    OV_ASSERT_NO_THROW(zeGraphExt->setGraphArgumentValue(graphDescriptor, 0, ptr));

    // destroy graph
    zeGraphExt->destroyGraph(graphDescriptor);
    allocator->deallocate(ptr, totalSize);
}

TEST_P(ZeroGraphTest, GetGraphDescriptorBadGraphFlag) {
    graphDescriptor = zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, 0xBADBAD);
    ASSERT_EQ(graphDescriptor._handle, nullptr);
}

std::vector<int> graphDescflags = {ZE_GRAPH_FLAG_NONE, ZE_GRAPH_FLAG_DISABLE_CACHING, ZE_GRAPH_FLAG_ENABLE_PROFILING, ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT};

std::vector<std::string> extVersion = {
    "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12", "1.13"
};

INSTANTIATE_TEST_SUITE_P(something, ZeroGraphTest, 
    ::testing::Combine(
        ::testing::ValuesIn(graphDescflags),
        ::testing::ValuesIn(extVersion)
    ),
    ZeroGraphTest::getTestCaseName
);

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
std::string ovPrecisionToLegacyPrecisionString(const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::Type_t::f16:
        return "FP16";
    case ov::element::Type_t::f32:
        return "FP32";
    case ov::element::Type_t::f64:
        return "FP64";
    case ov::element::Type_t::bf16:
        return "BF16";
    case ov::element::Type_t::f8e4m3:
        return "FP8_E4M3";
    case ov::element::Type_t::f8e5m2:
        return "FP8_E5M2";
    case ov::element::Type_t::f8e8m0:
        return "FP8_E8M0";
    case ov::element::Type_t::nf4:
        return "NF4";
    case ov::element::Type_t::i4:
        return "I4";
    case ov::element::Type_t::i8:
        return "I8";
    case ov::element::Type_t::i16:
        return "I16";
    case ov::element::Type_t::i32:
        return "I32";
    case ov::element::Type_t::i64:
        return "I64";
    case ov::element::Type_t::u4:
        return "U4";
    case ov::element::Type_t::u8:
        return "U8";
    case ov::element::Type_t::u16:
        return "U16";
    case ov::element::Type_t::u32:
        return "U32";
    case ov::element::Type_t::u64:
        return "U64";
    case ov::element::Type_t::u1:
        return "BIN";
    case ov::element::Type_t::u2:
        return "U2";
    case ov::element::Type_t::boolean:
        return "BOOL";
    case ov::element::Type_t::dynamic:
        return "DYNAMIC";
    default:
        OPENVINO_THROW("Incorrect precision: ", precision);
    }
}

std::string rankToLegacyLayoutString(const size_t rank) {
    switch (rank) {
    case 0:
        return "**SCALAR**";
    case 1:
        return "C";
    case 2:
        return "NC";
    case 3:
        return "CHW";
    case 4:
        return "NCHW";
    case 5:
        return "NCDHW";
    default:
        return "BLOCKED";
    }
}

std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model, const bool useIndices) {
    const ov::ParameterVector& parameters = model->get_parameters();
    const ov::ResultVector& results = model->get_results();

    std::stringstream inputsPrecisionSS;
    std::stringstream inputsLayoutSS;
    std::stringstream outputsPrecisionSS;
    std::stringstream outputsLayoutSS;

    inputsPrecisionSS << INPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    inputsLayoutSS << INPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    const auto getRankOrThrow = [](const ov::PartialShape& shape) -> size_t {
        if (shape.rank().is_dynamic()) {
            OPENVINO_THROW("Dynamic rank is not supported for NPU plugin");
        }
        return shape.rank().get_length();
    };

    if (!parameters.empty()) {
        size_t parameterIndex = 0;

        for (const std::shared_ptr<ov::op::v0::Parameter>& parameter : parameters) {
            const auto precision = parameter->get_element_type();
            const auto rank = getRankOrThrow(parameter->get_partial_shape());

            if (parameterIndex != 0) {
                inputsPrecisionSS << VALUES_SEPARATOR;
                inputsLayoutSS << VALUES_SEPARATOR;
            }

            if (useIndices) {
                inputsPrecisionSS << parameterIndex;
                inputsLayoutSS << parameterIndex;
            } else {
                const std::string& name = parameter->get_friendly_name();

                inputsPrecisionSS << name;
                // Ticket: E-88902
                inputsLayoutSS << name;
            }

            inputsPrecisionSS << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
            inputsLayoutSS << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);

            ++parameterIndex;
        }
    }

    inputsPrecisionSS << VALUE_DELIMITER;
    inputsLayoutSS << VALUE_DELIMITER;

    outputsPrecisionSS << OUTPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    outputsLayoutSS << OUTPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;

    size_t resultIndex = 0;
    for (const std::shared_ptr<ov::op::v0::Result>& result : results) {
        const auto precision = result->get_element_type();
        const auto rank = getRankOrThrow(result->get_output_partial_shape(0));

        if (resultIndex != 0) {
            outputsPrecisionSS << VALUES_SEPARATOR;
            outputsLayoutSS << VALUES_SEPARATOR;
        }

        if (useIndices) {
            outputsPrecisionSS << resultIndex;
            outputsLayoutSS << resultIndex;
        } else {
            const std::string& name = result->get_input_node_ptr(0)->get_friendly_name();

            outputsPrecisionSS << name;
            outputsLayoutSS << name;
        }

        outputsPrecisionSS << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
        outputsLayoutSS << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);

        ++resultIndex;
    }

    outputsPrecisionSS << VALUE_DELIMITER;
    outputsLayoutSS << VALUE_DELIMITER;

    // One line without spaces to avoid parsing as config option inside CID
    return inputsPrecisionSS.str() + VALUES_SEPARATOR.data() + inputsLayoutSS.str() + VALUES_SEPARATOR.data() +
           outputsPrecisionSS.str() + VALUES_SEPARATOR.data() + outputsLayoutSS.str();
}

std::string serializeConfig(const Config& config, ze_graph_compiler_version_info_t compilerVersion, const std::shared_ptr<intel_npu::ZeGraphExtWrappers> zeGraphExt) {
    Logger logger("serializeConfig", Logger::global().level());

    std::string content = {};

    const FilteredConfig* plgConfig = dynamic_cast<const FilteredConfig*>(&config);
    if (plgConfig != nullptr) {
        content += plgConfig->toStringForCompiler();
        content += plgConfig->toStringForCompilerInternal();
    } else {
        logger.warning("Failed to cast Config to FilteredConfig. Exporting all configs");
        content += config.toString();
    }

    logger.debug("Original content of config: %s", content.c_str());

    // Remove optimization-level and performance-hint-override for old driver which not support them
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 7)) {
        std::string valueOfParams = config.get<COMPILATION_MODE_PARAMS>();
        std::string keyOfOptL("optimization-level");
        std::string keyOfPerfHO("performance-hint-override");
        if (valueOfParams != "" && (valueOfParams.find(keyOfOptL) != std::string::npos ||
                                    valueOfParams.find(keyOfPerfHO) != std::string::npos)) {
            // Remove unsupported options from value
            std::ostringstream optLevelStr;
            optLevelStr << keyOfOptL << KEY_VALUE_SEPARATOR << "\\d+";
            std::ostringstream perfHintStr;
            perfHintStr << keyOfPerfHO << KEY_VALUE_SEPARATOR << "\\S+";
            logger.warning("%s property is not supported by this compiler version. Removing from parameters",
                           keyOfOptL.c_str());
            valueOfParams = std::regex_replace(valueOfParams, std::regex(optLevelStr.str()), "");
            logger.warning("%s property is not supported by this compiler version. Removing from parameters",
                           keyOfPerfHO.c_str());
            valueOfParams = std::regex_replace(valueOfParams, std::regex(perfHintStr.str()), "");

            // Trim space
            valueOfParams = std::regex_replace(valueOfParams, std::regex(R"(^\s+|\s+$)"), "");

            // Replace the value in content with new value
            std::ostringstream compilationParamsStr;
            compilationParamsStr << ov::intel_npu::compilation_mode_params.name() << KEY_VALUE_SEPARATOR
                                 << VALUE_DELIMITER << ".*" << VALUE_DELIMITER;
            if (valueOfParams == "") {
                logger.warning("Clear empty NPU_COMPILATION_MODE_PARAMS. Removing from parameters");
                content = std::regex_replace(content, std::regex(compilationParamsStr.str()), "");
            } else {
                std::ostringstream newValue;
                newValue << ov::intel_npu::compilation_mode_params.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER
                         << valueOfParams << VALUE_DELIMITER;
                logger.warning("Replace value of NPU_COMPILATION_MODE_PARAMS with new value %s",
                               newValue.str().c_str());
                content = std::regex_replace(content, std::regex(compilationParamsStr.str()), newValue.str().c_str());
            }
        }
    }

    // As a consequence of complying to the conventions established in the 2.0 OV API, the set of values corresponding
    // to the "model priority" key has been modified cpu_pinning property is not supported in compilers < v5.2 - need to
    // remove it
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 2)) {
        const auto& getTargetRegex = [](const ov::hint::Priority& priorityValue) -> std::regex {
            std::ostringstream result;
            result << ov::hint::model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << priorityValue
                   << VALUE_DELIMITER;
            return std::regex(result.str());
        };
        const auto& getStringReplacement = [](const ov::intel_npu::LegacyPriority& priorityValue) -> std::string {
            std::ostringstream result;
            result << ov::intel_npu::legacy_model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER
                   << priorityValue << VALUE_DELIMITER;
            return result.str();
        };

        // E.g. (valid as of writing this): MODEL_PRIORITY="MEDIUM" -> MODEL_PRIORITY="MODEL_PRIORITY_MED"
        content = std::regex_replace(content,
                                     getTargetRegex(ov::hint::Priority::LOW),
                                     getStringReplacement(ov::intel_npu::LegacyPriority::LOW));
        content = std::regex_replace(content,
                                     getTargetRegex(ov::hint::Priority::MEDIUM),
                                     getStringReplacement(ov::intel_npu::LegacyPriority::MEDIUM));
        content = std::regex_replace(content,
                                     getTargetRegex(ov::hint::Priority::HIGH),
                                     getStringReplacement(ov::intel_npu::LegacyPriority::HIGH));
    }

    // Special case for compiler Turbo
    // NPU_TURBO is a special option in the sense that by default it is a driver-setting, but certain compilers support
    // and make use of it too If we have turbo in the config string, we check if compiler supports it. If it doesn't
    // support it, we remove it
    if (std::regex_search(content, std::regex("NPU_TURBO"))) {
        bool is_supported = zeGraphExt->isTurboOptionSupported(compilerVersion);

        if (!is_supported) {
            std::ostringstream turbostr;
            turbostr << ov::intel_npu::turbo.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                     << VALUE_DELIMITER;
            logger.info("NPU_TURBO property is not supported by this compiler. Removing from "
                        "parameters");
            content = std::regex_replace(content, std::regex(turbostr.str()), "");
        }
    }

    // FINAL step to convert prefixes of remaining params, to ensure backwards compatibility
    // From 5.0.0, driver compiler start to use NPU_ prefix, the old version uses VPU_ prefix
    if (compilerVersion.major < 5) {
        std::regex reg("NPU_");
        content = std::regex_replace(content, reg, "VPU_");
        // From 4.0.0, driver compiler start to use VPU_ prefix, the old version uses VPUX_ prefix
        if (compilerVersion.major < 4) {
            // Replace VPU_ with VPUX_ for old driver compiler
            std::regex reg("VPU_");
            content = std::regex_replace(content, reg, "VPUX_");
        }
    }

    return "--config " + content;
}
