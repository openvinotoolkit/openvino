// Copyright (C) 2020 Intel Corporation7
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <ngraph/variant.hpp>
#include "ie_system_conf.h"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>
#include <exec_graph_info.hpp>

namespace CPUTestUtils {
    typedef enum {
        nchw,
        nChw8c,
        nChw16c,
        nhwc,
        ncdhw,
        nCdhw8c,
        nCdhw16c,
        ndhwc,
        nc,
        x,
        undef
    } cpu_memory_format_t;

    using CPUSpecificParams =  std::tuple<
        std::vector<cpu_memory_format_t>, //input memomry format
        std::vector<cpu_memory_format_t>, //output memory format
        std::vector<std::string>, //priority
        std::string // selected primitive type
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
    void CheckCPUImpl(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType) const;

protected:
    std::string getPrimitiveType() const;
    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
};

const auto emptyCPUSpec = CPUSpecificParams{{}, {}, {}, {}};

const auto conv_ref_2D = CPUSpecificParams{{nchw}, {nchw}, {"ref_any"}, "ref_any_FP32"};
const auto conv_ref_3D = CPUSpecificParams{{ncdhw}, {ncdhw}, {"ref_any"}, "ref_any_FP32"};

const auto conv_gemm_2D = CPUSpecificParams{{nchw}, {nchw}, {"gemm_any"}, "jit_gemm_FP32"};
const auto conv_gemm_3D = CPUSpecificParams{{ncdhw}, {ncdhw}, {"gemm_any"}, "jit_gemm_FP32"};

const auto conv_sse42_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42"}, "jit_sse42_FP32"};
const auto conv_sse42_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_sse42"}, "jit_sse42_FP32"};
const auto conv_sse42_dw_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42_dw"}, "jit_sse42_dw_FP32"};
const auto conv_sse42_dw_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_sse42_dw"}, "jit_sse42_dw_FP32"};

const auto conv_avx2_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2"}, "jit_avx2_FP32"};
const auto conv_avx2_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2_FP32"};
const auto conv_avx2_dw_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2_dw"}, "jit_avx2_dw_FP32"};
const auto conv_avx2_dw_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_avx2_dw"}, "jit_avx2_dw_FP32"};

const auto conv_avx512_2D = CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512"}, "jit_avx512_FP32"};
const auto conv_avx512_3D = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512_FP32"};
const auto conv_avx512_dw_2D = CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512_dw"}, "jit_avx512_dw_FP32"};
const auto conv_avx512_dw_3D = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {"jit_avx512_dw"}, "jit_avx512_dw_FP32"};

const auto conv_sse42_2D_1x1 = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42_1x1"}, "jit_sse42_1x1_FP32"};
const auto conv_avx2_2D_1x1 = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2_1x1"}, "jit_avx2_1x1_FP32"};
const auto conv_avx512_2D_1x1 = CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512_1x1"}, "jit_avx512_1x1_FP32"};

// utility functions
std::vector<CPUSpecificParams> filterCPUSpecificParams(std::vector<CPUSpecificParams>& paramsVector);

} // namespace CPUTestUtils
