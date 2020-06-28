// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/layer_test_utils.hpp"
#include "ie_system_conf.h"

#include <ngraph/variant.hpp>
#include <exec_graph_info.hpp>

using namespace InferenceEngine;

namespace CPUTestUtils {

typedef enum {
    nchw,
    nChw8c,
    nChw16c,
    ncdhw,
    nCdhw8c,
    nCdhw16c,
    goihw,
    undef
} cpu_memory_format_t;

typedef std::tuple<
        std::vector<cpu_memory_format_t>,
        std::vector<cpu_memory_format_t>,
        std::vector<std::string>,
        std::string> CPUSpecificParams;

namespace GroupConv {

/* CPU PARAMS */
const auto cpuParams_ref_2D = CPUSpecificParams{{nchw}, {nchw}, {"ref_any"}, "ref_any_FP32"};
const auto cpuParams_ref_3D = CPUSpecificParams{{ncdhw}, {ncdhw}, {"ref_any"}, "ref_any_FP32"};

const auto cpuParams_gemm_2D = CPUSpecificParams{{nchw}, {nchw}, {"gemm_any"}, "jit_gemm_FP32"};
const auto cpuParams_gemm_3D = CPUSpecificParams{{ncdhw}, {ncdhw}, {"gemm_any"}, "jit_gemm_FP32"};

const auto cpuParams_sse42_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42"}, "jit_sse42_FP32"};
const auto cpuParams_sse42_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_sse42"}, "jit_sse42_FP32"};
const auto cpuParams_sse42_dw_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_sse42_dw"}, "jit_sse42_dw_FP32"};
const auto cpuParams_sse42_dw_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_sse42_dw"}, "jit_sse42_dw_FP32"};

const auto cpuParams_avx2_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2"}, "jit_avx2_FP32"};
const auto cpuParams_avx2_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2_FP32"};
const auto cpuParams_avx2_dw_2D = CPUSpecificParams{{nChw8c}, {nChw8c}, {"jit_avx2_dw"}, "jit_avx2_dw_FP32"};
const auto cpuParams_avx2_dw_3D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {"jit_avx2_dw"}, "jit_avx2_dw_FP32"};

const auto cpuParams_avx512_2D = CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512"}, "jit_avx512_FP32"};
const auto cpuParams_avx512_3D = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512_FP32"};
const auto cpuParams_avx512_dw_2D = CPUSpecificParams{{nChw16c}, {nChw16c}, {"jit_avx512_dw"}, "jit_avx512_dw_FP32"};
const auto cpuParams_avx512_dw_3D = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {"jit_avx512_dw"}, "jit_avx512_dw_FP32"};


const std::vector<CPUSpecificParams> CPUParams_Planar_2D = {
        cpuParams_gemm_2D
};
const std::vector<CPUSpecificParams> CPUParams_Planar_3D = {
        cpuParams_gemm_3D
};
const std::vector<CPUSpecificParams> CPUParams_Blocked_2D = {
        cpuParams_sse42_2D,
        cpuParams_avx2_2D,
        cpuParams_avx512_2D
};
const std::vector<CPUSpecificParams> CPUParams_Blocked_3D = {
//        cpuParams_sse42_3D, // not supported jit_sse42 for 3d
        cpuParams_avx2_3D,
        cpuParams_avx512_3D
};
const std::vector<CPUSpecificParams> CPUParams_DW_2D = {
        cpuParams_sse42_dw_2D,
        cpuParams_avx2_dw_2D,
        cpuParams_avx512_dw_2D
};
const std::vector<CPUSpecificParams> CPUParams_DW_3D = {
        cpuParams_sse42_dw_3D,
        cpuParams_avx2_dw_3D,
        cpuParams_avx512_dw_3D
};
/* ========== */

} // namespace GroupConv

const char *cpu_fmt2str(cpu_memory_format_t v);
cpu_memory_format_t cpu_str2fmt(const char *str);
std::string fmts2str(const std::vector<cpu_memory_format_t> &fmts);
std::string impls2str(const std::vector<std::string> &priority);
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::vector<CPUSpecificParams> CPUParams);
std::map<std::string, std::shared_ptr<ngraph::Variant>> setCPUInfo(std::vector<cpu_memory_format_t> inFmts, std::vector<cpu_memory_format_t> outFmts,
                                                                   std::vector<std::string> priority);

void inline CheckCPUImpl(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType, std::vector<cpu_memory_format_t> inputMemoryFormats,
                         std::vector<cpu_memory_format_t> outputMemoryFormats, std::string selectedType) {
    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);

    for (const auto &node : function->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);

            return value->get();
        };

        auto getExecValueOutputsLayout = [] (std::shared_ptr<ngraph::Node> node) -> std::string {
            auto rtInfo = node->get_rt_info();
            auto it = rtInfo.find(ExecGraphInfoSerialization::OUTPUT_LAYOUTS);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);

            return value->get();
        };

        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == nodeType) {
            ASSERT_LE(inputMemoryFormats.size(), node->get_input_size());
            ASSERT_LE(outputMemoryFormats.size(), node->get_output_size());
            for (int i = 0; i < inputMemoryFormats.size(); i++) {
                const auto parentPort = node->input_values()[i];
                for (const auto & port : node->inputs()) {
                    if (port.get_tensor_ptr() == parentPort.get_tensor_ptr()) {
                        auto parentNode = parentPort.get_node_shared_ptr();
                        auto actualInputMemoryFormat = getExecValueOutputsLayout(parentNode);
                        ASSERT_EQ(inputMemoryFormats[i], cpu_str2fmt(actualInputMemoryFormat.c_str()));
                        break;
                    }
                }
            }
            for (int i = 0; i < outputMemoryFormats.size(); i++) {
                auto actualOutputMemoryFormat = getExecValue(ExecGraphInfoSerialization::OUTPUT_LAYOUTS);
                ASSERT_EQ(outputMemoryFormats[i], cpu_str2fmt(actualOutputMemoryFormat.c_str()));
            }

            auto primType = getExecValue(ExecGraphInfoSerialization::IMPL_TYPE);
            ASSERT_EQ(selectedType, primType);
        }
    }
}

}  // namespace CPUTestUtils