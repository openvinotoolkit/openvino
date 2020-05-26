// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <ngraph/variant.hpp>
#include "network_serializer.h"
#include "ie_system_conf.h"

namespace CPUTestUtils {

enum cpu_memory_format_t {
    // 2D layouts
    nc,
    // 3D layouts
    ncw,
    nwc,
    nCw4c,
    nCw8c,
    nCw16c,
    // 4D layouts
    nchw,
    nhwc,
    nChw4c,
    nChw8c,
    nChw16c,
    // 5D layouts
    ncdhw,
    ndhwc,
    nCdhw4c,
    nCdhw8c,
    nCdhw16c,
    undef
};


inline
cpu_memory_format_t tailC_format(size_t ndims) {
    switch (ndims) {
        case 2: return nc;
        case 3: return nwc;
        case 4: return nhwc;
        case 5: return ndhwc;
        default:
            THROW_IE_EXCEPTION << "No known tailC layout for " << ndims << "D tensor";
    }
}

inline
cpu_memory_format_t blockedC4_format(size_t ndims) {
    switch (ndims) {
        case 2: return nc;
        case 3: return nCw4c;
        case 4: return nChw4c;
        case 5: return nCdhw4c;
        default:
            THROW_IE_EXCEPTION << "No known blocked C4 layout for " << ndims << "D tensor";
    }
}

inline
cpu_memory_format_t blockedC8_format(size_t ndims) {
    switch (ndims) {
        case 2: return nc;
        case 3: return nCw8c;
        case 4: return nChw8c;
        case 5: return nCdhw8c;
        default:
            THROW_IE_EXCEPTION << "No known blocked C4 layout for " << ndims << "D tensor";
    }
}

inline
cpu_memory_format_t blockedC16_format(size_t ndims) {
    switch (ndims) {
        case 2: return nc;
        case 3: return nCw16c;
        case 4: return nChw16c;
        case 5: return nCdhw16c;
        default:
            THROW_IE_EXCEPTION << "No known blocked C4 layout for " << ndims << "D tensor";
    }
}

inline
const char *cpu_fmt2str(cpu_memory_format_t v) {
    if (v == nc) return "nc";

    if (v == ncw) return "ncw";
    if (v == nwc) return "nwc";
    if (v == nCw8c) return "nCw8c";
    if (v == nCw16c) return "nCw16c";

    if (v == nchw) return "nchw";
    if (v == nhwc) return "nhwc";
    if (v == nChw8c) return "nChw8c";
    if (v == nChw16c) return "nChw16c";

    if (v == ncdhw) return "ncdhw";
    if (v == ndhwc) return "ndhwc";
    if (v == nCdhw8c) return "nCdhw8c";
    if (v == nCdhw16c) return "nCdhw16c";
    assert(!"unknown fmt");
    return "undef";
}

inline
std::ostream& operator<<(std::ostream &os, const cpu_memory_format_t &v) {
    os << cpu_fmt2str(v);
    return os;
}

inline
cpu_memory_format_t cpu_str2fmt(const char *str) {
#define CASE(_fmt) do { \
    if (!strcmp(#_fmt, str) \
            || !strcmp("mkldnn_" #_fmt, str)) \
        return _fmt; \
} while (0)
    CASE(nc);
    CASE(ncw);
    CASE(nwc);
    CASE(nCw8c);
    CASE(nCw16c);
    CASE(nchw);
    CASE(nhwc);
    CASE(nChw8c);
    CASE(nChw16c);
    CASE(ncdhw);
    CASE(ndhwc);
    CASE(nCdhw8c);
    CASE(nCdhw16c);
#undef CASE
    std::cout << str << std::endl;
    assert(!"unknown memory format");
    return undef;
}

inline
std::string fmts2str(const std::vector<cpu_memory_format_t> &fmts) {
    std::string str;
    for (auto &fmt : fmts) {
        ((str += "cpu:") += cpu_fmt2str(fmt)) += ",";
    }
    str.erase(str.end() - 1);
    return str;
}

inline
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
            ASSERT_FALSE(primType->second.compare(0, selectedType.length(), selectedType));
        }
    }
}
IE_SUPPRESS_DEPRECATED_START
size_t inline numOfExecutedNodes(InferenceEngine::ExecutableNetwork &execNet) {
    auto execGraphInfo = execNet.GetExecGraphInfo();
    auto nodes = InferenceEngine::Serialization::TopologicalSort(execGraphInfo);

    size_t num_executed_nodes = 0;
    for (auto &node : nodes) {
        auto primType = node->params["execTimeMcs"];
        if (primType != "not_executed")
            num_executed_nodes++;
    }
    return num_executed_nodes;
}

IE_SUPPRESS_DEPRECATED_END
inline
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