// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_test_utils.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"

namespace CPUTestUtils {

using namespace InferenceEngine;

std::vector<CPUSpecificParams> filterCPUInfoForDevice(const std::vector<CPUSpecificParams>& CPUParams) {
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

const char *CPUTestsBase::cpu_fmt2str(cpu_memory_format_t v) {
    if (v == any) return "any";
    if (v == nchw) return "nchw";
    if (v == nChw8c) return "nChw8c";
    if (v == nChw16c) return "nChw16c";
    if (v == nhwc) return "nhwc";
    if (v == ncdhw) return "ncdhw";
    if (v == nCdhw8c) return "nCdhw8c";
    if (v == nCdhw16c) return "nCdhw16c";
    if (v == ndhwc) return "ndhwc";
    if (v == tnc) return "tnc";
    if (v == nc) return "nc";
    if (v == x) return "x";
    assert(!"unknown fmt");
    return "undef";
}

cpu_memory_format_t CPUTestsBase::cpu_str2fmt(const char *str) {
#define CASE(_fmt) do { \
    if (!strcmp(#_fmt, str) \
            || !strcmp("mkldnn_" #_fmt, str)) \
        return _fmt; \
} while (0)
    CASE(nchw);
    CASE(nChw8c);
    CASE(nChw16c);
    CASE(nhwc);
    CASE(ncdhw);
    CASE(nCdhw8c);
    CASE(nCdhw16c);
    CASE(ndhwc);
    CASE(tnc);
    CASE(nc);
    CASE(x);
#undef CASE
    assert(!"unknown memory format");
    return undef;
}

std::string CPUTestsBase::fmts2str(const std::vector<cpu_memory_format_t> &fmts) {
    std::string str;
    for (auto &fmt : fmts) {
        ((str += "cpu:") += cpu_fmt2str(fmt)) += ",";
    }
    if (!str.empty()) {
        str.pop_back();
    }
    return str;
}

std::string CPUTestsBase::impls2str(const std::vector<std::string> &priority) {
    std::string str;
    for (auto &impl : priority) {
        ((str += "cpu:") += impl) += ",";
    }
    if (!str.empty()) {
        str.pop_back();
    }
    return str;
}

void CPUTestsBase::CheckCPUImpl(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType, bool checkForReorders) const {
    IE_SUPPRESS_DEPRECATED_START
    ASSERT_TRUE(!selectedType.empty()) << "Node type is not defined.";
    bool isNodeFound = false;
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

        if (checkForReorders) {
            ASSERT_NE("Reorder", getExecValue(ExecGraphInfoSerialization::LAYER_TYPE));
        }

        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == nodeType) {
            isNodeFound = true;
            ASSERT_LE(inFmts.size(), node->get_input_size());
            ASSERT_LE(outFmts.size(), node->get_output_size());
            for (int i = 0; i < inFmts.size(); i++) {
                const auto parentPort = node->input_values()[i];
                const auto port = node->inputs()[i];
                if ((parentPort.get_tensor_ptr() == port.get_tensor_ptr())) {
                    auto parentNode = parentPort.get_node_shared_ptr();
                    auto actualInputMemoryFormat = getExecValueOutputsLayout(parentNode);
                    ASSERT_EQ(inFmts[i], cpu_str2fmt(actualInputMemoryFormat.c_str()));
                }
            }
            for (int i = 0; i < outFmts.size(); i++) {
                auto actualOutputMemoryFormat = getExecValue(ExecGraphInfoSerialization::OUTPUT_LAYOUTS);
                ASSERT_EQ(outFmts[i], cpu_str2fmt(actualOutputMemoryFormat.c_str()));
            }
            auto primType = getExecValue(ExecGraphInfoSerialization::IMPL_TYPE);
            ASSERT_EQ(selectedType, primType);
        }
    }
    ASSERT_TRUE(isNodeFound) << "Node type name: \"" << nodeType << "\" has not been found.";
    IE_SUPPRESS_DEPRECATED_END
}

std::string CPUTestsBase::getTestCaseName(CPUSpecificParams params) {
    std::ostringstream result;
    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
    std::tie(inFmts, outFmts, priority, selectedType) = params;
    if (!inFmts.empty()) {
        result << "_inFmts=" << fmts2str(inFmts);
    }
    if (!outFmts.empty()) {
        result << "_outFmts=" << fmts2str(outFmts);
    }
    if (!selectedType.empty()) {
        result << "_primitive=" << selectedType;
    }
    return result.str();
}

CPUTestsBase::CPUInfo CPUTestsBase::getCPUInfo() const {
    return makeCPUInfo(inFmts, outFmts, priority);
}

std::string CPUTestsBase::getPrimitiveType() const {
    std::string isaType;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        isaType = "jit_avx512";
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        isaType = "jit_avx2";
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        isaType = "jit_sse42";
    } else {
        isaType = "ref";
    }
    return isaType;
}

CPUTestsBase::CPUInfo
CPUTestsBase::makeCPUInfo(std::vector<cpu_memory_format_t> inFmts, std::vector<cpu_memory_format_t> outFmts, std::vector<std::string> priority) {
    CPUInfo cpuInfo;

    if (!inFmts.empty()) {
        cpuInfo.insert({std::string(ngraph::MLKDNNInputMemoryFormatsAttr),
                        std::make_shared<ngraph::VariantWrapper<ngraph::MLKDNNInputMemoryFormats>>(ngraph::MLKDNNInputMemoryFormats(fmts2str(inFmts)))});
    }
    if (!outFmts.empty()) {
        cpuInfo.insert({std::string(ngraph::MLKDNNOutputMemoryFormatsAttr),
                        std::make_shared<ngraph::VariantWrapper<ngraph::MLKDNNOutputMemoryFormats>>(ngraph::MLKDNNOutputMemoryFormats(fmts2str(outFmts)))});
    }
    if (!priority.empty()) {
        cpuInfo.insert({"PrimitivesPriority", std::make_shared<ngraph::VariantWrapper<std::string>>(impls2str(priority))});
    }

    return cpuInfo;
}

std::vector<CPUSpecificParams> filterCPUSpecificParams(std::vector<CPUSpecificParams> &paramsVector) {
    auto adjustBlockedFormatByIsa = [](std::vector<cpu_memory_format_t>& formats) {
        for (int i = 0; i < formats.size(); i++) {
            if (formats[i] == nChw16c)
                formats[i] = nChw8c;
            if (formats[i] == nCdhw16c)
                formats[i] = nCdhw8c;
        }
    };

    if (!InferenceEngine::with_cpu_x86_avx512f()) {
        for (auto& param : paramsVector) {
            adjustBlockedFormatByIsa(std::get<0>(param));
            adjustBlockedFormatByIsa(std::get<1>(param));
        }
    }

    return paramsVector;
}

TensorDesc CPUTestsBase::createTensorDesc(const SizeVector dims, const Precision prec, cpu_memory_format_t fmt, const size_t offsetPadding) {
    SizeVector order, blockedDims = dims;
    switch (fmt) {
        case nchw: {
            order = {0, 1, 2, 3};
            break;
        }
        case nhwc: {
            order = {0, 2, 3, 1};
            blockedDims = {dims[0], dims[2], dims[3], dims[1]};
            break;
        }
        case nChw8c: {
            order = {0, 1, 2, 3, 1};
            blockedDims[1] = blockedDims[1] / 8 + (blockedDims[1] % 8 ? 1 : 0);
            blockedDims.push_back(8);
            break;
        }
        case nChw16c: {
            order = {0, 1, 2, 3, 1};
            blockedDims[1] = blockedDims[1] / 16 + (blockedDims[1] % 16 ? 1 : 0);
            blockedDims.push_back(16);
            break;
        }
        case ncdhw: {
            order = {0, 1, 2, 3, 4};
            break;
        }
        case ndhwc: {
            order = {0, 2, 3, 4, 1};
            blockedDims = {dims[0], dims[2], dims[3], dims[4], dims[1]};
            break;
        }
        case nCdhw8c: {
            order = {0, 1, 2, 3, 4, 1};
            blockedDims[1] = blockedDims[1] / 8 + (blockedDims[1] % 8 ? 1 : 0);
            blockedDims.push_back(8);
            break;
        }
        case nCdhw16c: {
            order = {0, 1, 2, 3, 4, 1};
            blockedDims[1] = blockedDims[1] / 16 + (blockedDims[1] % 16 ? 1 : 0);
            blockedDims.push_back(16);
            break;
        }
        case tnc: {
            order = {0, 1, 2};
            break;
        }
        case nc: {
            order = {0, 1};
            break;
        }
        case x: {
            order = {0};
            break;
        }
        case any: {
            order.resize(blockedDims.size());
            std::iota(order.begin(), order.end(), 0);
            break;
        }
        default: {
            std::string error = "Unsupported data format " + std::string(cpu_fmt2str(fmt));
            throw std::runtime_error(error);
        }
    }
    return TensorDesc(prec, dims, BlockingDesc(blockedDims, order, offsetPadding));
}

void CPUTestsBase::checkOffsetPadding(const ExecutableNetwork &execNet, int inOffPadding, int outOffPadding) {
    if (inOffPadding != -1) {
        for (const auto &in : execNet.GetInputsInfo()) {
            int actualOffset = in.second->getInputData()->getTensorDesc().getBlockingDesc().getOffsetPadding();
            ASSERT_EQ(inOffPadding, actualOffset);
        }
    }
    if (outOffPadding != -1) {
        for (const auto &out : execNet.GetOutputsInfo()) {
            int actualOffset = out.second->getTensorDesc().getBlockingDesc().getOffsetPadding();
            ASSERT_EQ(outOffPadding, actualOffset);
        }
    }
}

} // namespace CPUTestUtils
