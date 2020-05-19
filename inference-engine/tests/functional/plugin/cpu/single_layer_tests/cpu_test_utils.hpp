// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <ngraph/variant.hpp>
#include "network_serializer.h"
#include "ie_system_conf.h"

namespace CPUTestUtils {

typedef enum {
    nchw,
    nChw8c,
    nChw16c,
    ncdhw,
    nCdhw8c,
    nCdhw16c,
    undef
} cpu_memory_format_t;

const char *cpu_fmt2str(cpu_memory_format_t v) {
    if (v == nchw) return "nchw";
    if (v == nChw8c) return "nChw8c";
    if (v == nChw16c) return "nChw16c";
    if (v == ncdhw) return "ncdhw";
    if (v == nCdhw8c) return "nCdhw8c";
    if (v == nCdhw16c) return "nCdhw16c";
    assert(!"unknown fmt");
    return "undef";
}

cpu_memory_format_t cpu_str2fmt(const char *str) {
#define CASE(_fmt) do { \
    if (!strcmp(#_fmt, str) \
            || !strcmp("mkldnn_" #_fmt, str)) \
        return _fmt; \
} while (0)
    CASE(nchw);
    CASE(nChw8c);
    CASE(nChw16c);
    CASE(ncdhw);
    CASE(nCdhw8c);
    CASE(nCdhw16c);
#undef CASE
    assert(!"unknown memory format");
    return undef;
}

std::string fmts2str(const std::vector<cpu_memory_format_t> &fmts) {
    std::string str;
    for (auto &fmt : fmts) {
        ((str += "cpu:") += cpu_fmt2str(fmt)) += ",";
    }
    str.erase(str.end() - 1);
    return str;
}

std::string impls2str(const std::vector<std::string> &priority) {
    std::string str;
    for (auto &impl : priority) {
        ((str += "cpu:") += impl) += ",";
    }
    str.erase(str.end() - 1);
    return str;
}

IE_SUPPRESS_DEPRECATED_START
void inline CheckCPUImpl(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType, std::vector<cpu_memory_format_t> inputMemoryFormats,
                         std::vector<cpu_memory_format_t> outputMemoryFormats, std::string selectedType) {
    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    auto nodes = InferenceEngine::Serialization::TopologicalSort(execGraphInfo);
    for (auto &node : nodes) {
        if (node->type == nodeType) {
            ASSERT_LE(inputMemoryFormats.size(), node->insData.size());
            ASSERT_LE(outputMemoryFormats.size(), node->outData.size());
            for (int i = 0; i < inputMemoryFormats.size(); i++) {
                for (auto &parentNode : nodes) {
                    for (int j = 0; j < parentNode->outData.size(); j++) {
                        if (parentNode->outData[j]->getName() == node->insData[i].lock()->getName()) {
                            auto actualInputMemoryFormat = parentNode->params.find("outputLayouts");
                            ASSERT_NE(actualInputMemoryFormat, parentNode->params.end());
                            ASSERT_EQ(inputMemoryFormats[i], cpu_str2fmt(actualInputMemoryFormat->second.c_str()));
                        }
                    }
                }
            }
            for (int i = 0; i < outputMemoryFormats.size(); i++) {
                auto actualOutputMemoryFormat = node->params.find("outputLayouts");
                ASSERT_NE(actualOutputMemoryFormat, node->params.end());
                ASSERT_EQ(outputMemoryFormats[i], cpu_str2fmt(actualOutputMemoryFormat->second.c_str()));
            }

            auto primType = node->params.find("primitiveType");
            ASSERT_NE(primType, node->params.end());
            ASSERT_EQ(selectedType, primType->second);
        }
    }
}
IE_SUPPRESS_DEPRECATED_END

std::map<std::string, std::shared_ptr<ngraph::Variant>> setCPUInfo(std::vector<cpu_memory_format_t> inFmts, std::vector<cpu_memory_format_t> outFmts,
                                      std::vector<std::string> priority) {
    std::map<std::string, std::shared_ptr<ngraph::Variant>> cpuInfo;

    if (!inFmts.empty()) {
        cpuInfo.insert({"InputMemoryFormats", std::make_shared<ngraph::VariantWrapper<std::string>>(fmts2str(inFmts))});
    }
    if (!outFmts.empty()) {
        cpuInfo.insert({"OutputMemoryFormats", std::make_shared<ngraph::VariantWrapper<std::string>>(fmts2str(outFmts))});
    }
    if (!priority.empty()) {
        cpuInfo.insert({"PrimitivesPriority", std::make_shared<ngraph::VariantWrapper<std::string>>(impls2str(priority))});
    }

    return cpuInfo;
}

}  // namespace CPUTestUtils