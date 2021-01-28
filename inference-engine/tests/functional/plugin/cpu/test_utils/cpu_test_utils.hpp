// Copyright (C) 2020 Intel Corporation
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

        x = a,
        nc = ab,
        nchw = abcd,
        nChw8c = aBcd8b,
        nChw16c = aBcd16b,
        nhwc = acdb,
        ncdhw = abcde,
        nCdhw8c = aBcde8b,
        nCdhw16c = aBcde16b,
        ndhwc = acdeb
    } cpu_memory_format_t;

    using CPUSpecificParams =  std::tuple<
        std::vector<cpu_memory_format_t>, // input memomry format
        std::vector<cpu_memory_format_t>, // output memory format
        std::vector<std::string>,         // priority
        std::string                       // selected primitive type
    >;

class CPUTestsBase {
public:
    typedef std::map<std::string, std::shared_ptr<ngraph::Variant>> CPUInfo;

public:
    static std::string getTestCaseName(CPUSpecificParams params);
    static const char *cpu_fmt2str(cpu_memory_format_t v);
    static cpu_memory_format_t cpu_str2fmt(const char *str);
    static std::string fmts2str(const std::vector<cpu_memory_format_t> &fmts);
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

const auto conv_ref_2D = CPUSpecificParams{{nchw}, {nchw}, {"ref_any"}, "ref_any"};
const auto conv_ref_3D = CPUSpecificParams{{ncdhw}, {ncdhw}, {"ref_any"}, "ref_any"};

const auto conv_gemm_2D = CPUSpecificParams{{nchw}, {nchw}, {"gemm_any"}, "jit_gemm"};
const auto conv_gemm_3D = CPUSpecificParams{{ncdhw}, {ncdhw}, {"gemm_any"}, "jit_gemm"};

const auto conv_sse42_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42"}, "jit_sse42"};
const auto conv_sse42_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_sse42"}, "jit_sse42"};
const auto conv_sse42_dw_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42_dw"}, "jit_sse42_dw"};
const auto conv_sse42_dw_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_sse42_dw"}, "jit_sse42_dw"};

const auto conv_avx2_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2"}, "jit_avx2"};
const auto conv_avx2_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2"};
const auto conv_avx2_dw_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2_dw"}, "jit_avx2_dw"};
const auto conv_avx2_dw_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_avx2_dw"}, "jit_avx2_dw"};

const auto conv_avx512_2D = CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512"}, "jit_avx512"};
const auto conv_avx512_3D = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512"};
const auto conv_avx512_dw_2D = CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512_dw"}, "jit_avx512_dw"};
const auto conv_avx512_dw_3D = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {"jit_avx512_dw"}, "jit_avx512_dw"};

const auto conv_sse42_2D_1x1 = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42_1x1"}, "jit_sse42_1x1"};
const auto conv_avx2_2D_1x1 = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2_1x1"}, "jit_avx2_1x1"};
const auto conv_avx512_2D_1x1 = CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512_1x1"}, "jit_avx512_1x1"};

// utility functions
std::vector<CPUSpecificParams> filterCPUSpecificParams(std::vector<CPUSpecificParams>& paramsVector);
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::vector<CPUSpecificParams> CPUParams);
} // namespace CPUTestUtils
