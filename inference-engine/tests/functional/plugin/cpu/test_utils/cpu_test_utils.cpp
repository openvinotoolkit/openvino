// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_test_utils.hpp"

namespace CPUTestUtils {

const char *cpu_fmt2str(cpu_memory_format_t v) {
    if (v == nchw) return "nchw";
    if (v == nChw8c) return "nChw8c";
    if (v == nChw16c) return "nChw16c";
    if (v == ncdhw) return "ncdhw";
    if (v == nCdhw8c) return "nCdhw8c";
    if (v == nCdhw16c) return "nCdhw16c";
    if (v == goihw) return "goihw";
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
    CASE(goihw);
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

std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::vector<CPUSpecificParams> CPUParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);

        if (selectedTypeStr.find("jit") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !with_cpu_x86_avx512f())
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}

}  // namespace CPUTestUtils