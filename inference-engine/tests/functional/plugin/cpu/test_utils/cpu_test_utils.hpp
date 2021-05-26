// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <ngraph/variant.hpp>
#include "ie_system_conf.h"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include <exec_graph_info.hpp>
#include "ie_system_conf.h"

namespace CPUTestUtils {
    typedef enum {
        undef,
        a,
        ab,
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
        nchw = abcd,
        nChw8c = aBcd8b,
        nChw16c = aBcd16b,
        nhwc = acdb,
        ncdhw = abcde,
        nCdhw8c = aBcde8b,
        nCdhw16c = aBcde16b,
        ndhwc = acdeb,
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

    using CPUSpecificParams =  std::tuple<
        std::vector<cpu_memory_format_t>, // input memomry format
        std::vector<cpu_memory_format_t>, // output memory format
        std::vector<std::string>,         // priority
        std::string                       // selected primitive type
    >;

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
    typedef std::map<std::string, std::shared_ptr<ngraph::Variant>> CPUInfo;

public:
    static std::string getTestCaseName(CPUSpecificParams params);
    static const char *cpu_fmt2str(cpu_memory_format_t v);
    static cpu_memory_format_t cpu_str2fmt(const char *str);
    static std::string fmts2str(const std::vector<cpu_memory_format_t> &fmts, const std::string &prefix);
    static std::string impls2str(const std::vector<std::string> &priority);
    static CPUInfo makeCPUInfo(std::vector<cpu_memory_format_t> inFmts,
                               std::vector<cpu_memory_format_t> outFmts,
                               std::vector<std::string> priority);

    CPUInfo getCPUInfo() const;
    std::shared_ptr<ngraph::Function> makeNgraphFunction(const ngraph::element::Type &ngPrc,
                                                         ngraph::ParameterVector &params,
                                                         const std::shared_ptr<ngraph::Node> &lastNode,
                                                         std::string name) const;

protected:
    virtual void CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType) const;
    /**
     * @brief This function modifies the initial single layer test graph to add any necessary modifications that are specific to the cpu test scope.
     * @param ngPrc Graph precision.
     * @param params Graph parameters vector.
     * @param lastNode The last node of the initial graph.
     * @return The last node of the modified graph.
     */
    virtual std::shared_ptr<ngraph::Node> modifyGraph(const ngraph::element::Type &ngPrc,
                                                      ngraph::ParameterVector &params,
                                                      const std::shared_ptr<ngraph::Node> &lastNode) const;

protected:
    std::string getPrimitiveType() const;
    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
};

const auto emptyCPUSpec = CPUSpecificParams{{}, {}, {}, {}};

// utility functions
std::vector<CPUSpecificParams> filterCPUSpecificParams(std::vector<CPUSpecificParams>& paramsVector);
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::vector<CPUSpecificParams> CPUParams);
void CheckNodeOfTypeCount(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType, size_t expectedCount);
} // namespace CPUTestUtils
