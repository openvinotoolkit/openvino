// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/any.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

namespace CPUTestUtils {
typedef enum {
    undef,
    a,
    ab,
    acb,
    aBc8b,
    aBc16b,
    abcd,
    acdb,
    aBcd8b,
    aBcd16b,
    abcde,
    acdeb,
    aBcde8b,
    aBcde16b,
    // RNN layouts
    abc,
    bac,
    abdc,
    abdec,

    x = a,
    nc = ab,
    ncw = abc,
    nchw = abcd,
    ncdhw = abcde,
    nwc = acb,
    nhwc = acdb,
    ndhwc = acdeb,
    nCw8c = aBc8b,
    nCw16c = aBc16b,
    nChw8c = aBcd8b,
    nChw16c = aBcd16b,
    nCdhw8c = aBcde8b,
    nCdhw16c = aBcde16b,
    // RNN layouts
    tnc = abc,
    /// 3D RNN data tensor in the format (batch, seq_length, input channels).
    ntc = bac,
    /// 4D RNN states tensor in the format (num_layers, num_directions,
    /// batch, state channels).
    ldnc = abcd,
    /// 5D RNN weights tensor in the format (num_layers, num_directions,
    ///  input_channels, num_gates, output_channels).
    ///
    ///  - For LSTM cells, the gates order is input, forget, candidate
    ///    and output gate.
    ///  - For GRU cells, the gates order is update, reset and output gate.
    ldigo = abcde,
    /// 5D RNN weights tensor in the format (num_layers, num_directions,
    /// num_gates, output_channels, input_channels).
    ///
    ///  - For LSTM cells, the gates order is input, forget, candidate
    ///    and output gate.
    ///  - For GRU cells, the gates order is update, reset and output gate.
    ldgoi = abdec,
    /// 4D LSTM projection tensor in the format (num_layers, num_directions,
    /// num_channels_in_hidden_state, num_channels_in_recurrent_projection).
    ldio = abcd,
    /// 4D LSTM projection tensor in the format (num_layers, num_directions,
    /// num_channels_in_recurrent_projection, num_channels_in_hidden_state).
    ldoi = abdc,
    /// 4D RNN bias tensor in the format (num_layers, num_directions,
    /// num_gates, output_channels).
    ///
    ///  - For LSTM cells, the gates order is input, forget, candidate
    ///    and output gate.
    ///  - For GRU cells, the gates order is update, reset and output gate.
    ldgo = abcd,
} cpu_memory_format_t;

using CPUSpecificParams = std::tuple<std::vector<cpu_memory_format_t>,  // input memomry format
                                     std::vector<cpu_memory_format_t>,  // output memory format
                                     std::vector<std::string>,          // priority
                                     std::string                        // selected primitive type
                                     >;

enum class nodeType { convolution, convolutionBackpropData, groupConvolution, groupConvolutionBackpropData };

inline std::string nodeType2PluginType(nodeType nt) {
    if (nt == nodeType::convolution)
        return "Convolution";
    if (nt == nodeType::convolutionBackpropData)
        return "Deconvolution";
    if (nt == nodeType::groupConvolution)
        return "Convolution";
    if (nt == nodeType::groupConvolutionBackpropData)
        return "Deconvolution";
    throw std::runtime_error("Undefined node type to convert to plug-in type node!");
}

inline std::string nodeType2str(nodeType nt) {
    if (nt == nodeType::convolution)
        return "Convolution";
    if (nt == nodeType::convolutionBackpropData)
        return "ConvolutionBackpropData";
    if (nt == nodeType::groupConvolution)
        return "GroupConvolution";
    if (nt == nodeType::groupConvolutionBackpropData)
        return "GroupConvolutionBackpropData";
    throw std::runtime_error("Undefined node type to convert to string!");
}
bool with_cpu_x86_avx2_vnni_2();
class CPUTestsBase {
public:
    typedef std::map<std::string, ov::Any> CPUInfo;

public:
    static std::string getTestCaseName(CPUSpecificParams params);
    static const char* cpu_fmt2str(cpu_memory_format_t v);
    static cpu_memory_format_t cpu_str2fmt(const char* str);
    static std::string fmts2str(const std::vector<cpu_memory_format_t>& fmts, const std::string& prefix);
    static ov::PrimitivesPriority impls2primProiority(const std::vector<std::string>& priority);
    static CPUInfo makeCPUInfo(const std::vector<cpu_memory_format_t>& inFmts,
                               const std::vector<cpu_memory_format_t>& outFmts,
                               const std::vector<std::string>& priority);
    // TODO: change to setter method
    static std::string makeSelectedTypeStr(std::string implString, ov::element::Type_t elType);
    static ov::element::Type deduce_expected_precision(const ov::element::Type& opPrecision,
                                                       const ov::AnyMap& configuration);
    void updateSelectedType(const std::string& primitiveType,
                            const ov::element::Type netType,
                            const ov::AnyMap& config);

    CPUInfo getCPUInfo() const;
    std::shared_ptr<ov::Model> makeNgraphFunction(const ov::element::Type& ngPrc,
                                                  ov::ParameterVector& params,
                                                  const std::shared_ptr<ov::Node>& lastNode,
                                                  std::string name);

    void CheckPluginRelatedResults(const ov::CompiledModel& execNet, const std::set<std::string>& nodeType) const;
    void CheckPluginRelatedResults(const ov::CompiledModel& execNet, const std::string& nodeType) const;

    static const char* any_type;

protected:
    virtual void CheckPluginRelatedResultsImpl(const std::shared_ptr<const ov::Model>& function,
                                               const std::set<std::string>& nodeType) const;
    /**
     * @brief This function modifies the initial single layer test graph to add any necessary modifications that are
     * specific to the cpu test scope.
     * @param ngPrc Graph precision.
     * @param params Graph parameters vector.
     * @param lastNode The last node of the initial graph.
     * @return The last node of the modified graph.
     */
    virtual std::shared_ptr<ov::Node> modifyGraph(const ov::element::Type& ngPrc,
                                                  ov::ParameterVector& params,
                                                  const std::shared_ptr<ov::Node>& lastNode);

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
const ov::AnyMap empty_plugin_config{};
const ov::AnyMap cpu_f16_plugin_config = {{ov::hint::inference_precision(ov::element::f16)}};
const ov::AnyMap cpu_bf16_plugin_config = {{ov::hint::inference_precision(ov::element::bf16)}};
const ov::AnyMap cpu_f32_plugin_config = {{ov::hint::inference_precision(ov::element::f32)}};

// utility functions
void CheckNumberOfNodesWithType(const ov::CompiledModel& compiledModel,
                                const std::string& nodeType,
                                size_t expectedCount);
void CheckNumberOfNodesWithTypes(const ov::CompiledModel& compiledModel,
                                 const std::unordered_set<std::string>& nodeTypes,
                                 size_t expectedCount);
bool containsNonSupportedFormat(const std::vector<cpu_memory_format_t>& formats,
                                const std::vector<cpu_memory_format_t>& non_supported_f);
bool containsSupportedFormatsOnly(const std::vector<cpu_memory_format_t>& formats,
                                  const std::vector<cpu_memory_format_t>& supported_f);
}  // namespace CPUTestUtils
