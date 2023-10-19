// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "filter_cpu_info.hpp"
#include <string>
#include "ie_system_conf.h"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include <exec_graph_info.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include "ie_system_conf.h"

namespace CPUTestUtils {
    enum class nodeType {
        convolution,
        convolutionBackpropData,
        groupConvolution,
        groupConvolutionBackpropData
    };

    inline std::string nodeType2PluginType(nodeType nt) {
        if (nt == nodeType::convolution) return "Convolution";
        if (nt == nodeType::convolutionBackpropData) return "Deconvolution";
        if (nt == nodeType::groupConvolution) return "Convolution";
        if (nt == nodeType::groupConvolutionBackpropData) return "Deconvolution";
        throw std::runtime_error("Undefined node type to convert to plug-in type node!");
    }

    inline std::string nodeType2str(nodeType nt) {
        if (nt == nodeType::convolution) return "Convolution";
        if (nt == nodeType::convolutionBackpropData) return "ConvolutionBackpropData";
        if (nt == nodeType::groupConvolution) return "GroupConvolution";
        if (nt == nodeType::groupConvolutionBackpropData) return "GroupConvolutionBackpropData";
        throw std::runtime_error("Undefined node type to convert to string!");
    }

class CPUTestsBase {
public:
    typedef std::map<std::string, ov::Any> CPUInfo;

public:
    static std::string getTestCaseName(CPUSpecificParams params);
    static const char *cpu_fmt2str(cpu_memory_format_t v);
    static cpu_memory_format_t cpu_str2fmt(const char *str);
    static std::string fmts2str(const std::vector<cpu_memory_format_t> &fmts, const std::string &prefix);
    static ov::PrimitivesPriority impls2primProiority(const std::vector<std::string> &priority);
    static CPUInfo makeCPUInfo(const std::vector<cpu_memory_format_t>& inFmts,
                               const std::vector<cpu_memory_format_t>& outFmts,
                               const std::vector<std::string>& priority);
   //TODO: change to setter method
    static std::string makeSelectedTypeStr(std::string implString, ngraph::element::Type_t elType);
    void updateSelectedType(const std::string& primitiveType, const ov::element::Type netType, const ov::AnyMap& config);

    CPUInfo getCPUInfo() const;
    std::shared_ptr<ngraph::Function> makeNgraphFunction(const ngraph::element::Type &ngPrc,
                                                         ngraph::ParameterVector &params,
                                                         const std::shared_ptr<ngraph::Node> &lastNode,
                                                         std::string name);

    void CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, const std::set<std::string>& nodeType) const;
    void CheckPluginRelatedResults(const ov::CompiledModel &execNet, const std::set<std::string>& nodeType) const;
    void CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, const std::string& nodeType) const;
    void CheckPluginRelatedResults(const ov::CompiledModel &execNet, const std::string& nodeType) const;

    static const char* any_type;

protected:
    virtual void CheckPluginRelatedResultsImpl(const std::shared_ptr<const ov::Model>& function, const std::set<std::string>& nodeType) const;
    /**
     * @brief This function modifies the initial single layer test graph to add any necessary modifications that are specific to the cpu test scope.
     * @param ngPrc Graph precision.
     * @param params Graph parameters vector.
     * @param lastNode The last node of the initial graph.
     * @return The last node of the modified graph.
     */
    virtual std::shared_ptr<ngraph::Node> modifyGraph(const ngraph::element::Type &ngPrc,
                                                      ngraph::ParameterVector &params,
                                                      const std::shared_ptr<ngraph::Node> &lastNode);

    virtual bool primTypeCheck(std::string primType) const;

protected:
    std::string getPrimitiveType() const;
    std::string getISA(bool skip_amx) const;
    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
};

// common parameters
const auto emptyCPUSpec = CPUSpecificParams{{}, {}, {}, {}};
const std::map<std::string, std::string> cpuEmptyPluginConfig;
const ov::AnyMap empty_plugin_config{};
const std::map<std::string, std::string> cpuFP32PluginConfig =
        { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO } };
const std::map<std::string, std::string> cpuBF16PluginConfig =
        { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES } };


// utility functions
std::vector<CPUSpecificParams> filterCPUSpecificParams(const std::vector<CPUSpecificParams>& paramsVector);
void CheckNumberOfNodesWithType(const ov::CompiledModel &compiledModel, const std::string& nodeType, size_t expectedCount);
void CheckNumberOfNodesWithType(InferenceEngine::ExecutableNetwork &execNet, const std::string& nodeType, size_t expectedCount);
void CheckNumberOfNodesWithTypes(const ov::CompiledModel &compiledModel, const std::unordered_set<std::string>& nodeTypes, size_t expectedCount);
void CheckNumberOfNodesWithTypes(InferenceEngine::ExecutableNetwork &execNet, const std::unordered_set<std::string>& nodeTypes, size_t expectedCount);
} // namespace CPUTestUtils
