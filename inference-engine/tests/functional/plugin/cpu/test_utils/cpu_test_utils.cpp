// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_test_utils.hpp"

namespace CPUTestUtils {

const char *CPUTestsBase::cpu_fmt2str(cpu_memory_format_t v) {
    if (v == nchw) return "nchw";
    if (v == nChw8c) return "nChw8c";
    if (v == nChw16c) return "nChw16c";
    if (v == nhwc) return "nhwc";
    if (v == ncdhw) return "ncdhw";
    if (v == nCdhw8c) return "nCdhw8c";
    if (v == nCdhw16c) return "nCdhw16c";
    if (v == ndhwc) return "ndhwc";
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
#undef CASE
    assert(!"unknown memory format");
    return undef;
}

std::string CPUTestsBase::fmts2str(const std::vector<cpu_memory_format_t> &fmts) {
    std::string str;
    for (auto &fmt : fmts) {
        ((str += "cpu:") += cpu_fmt2str(fmt)) += ",";
    }
    str.erase(str.end() - 1);
    return str;
}

std::string CPUTestsBase::impls2str(const std::vector<std::string> &priority) {
    std::string str;
    for (auto &impl : priority) {
        ((str += "cpu:") += impl) += ",";
    }
    str.erase(str.end() - 1);
    return str;
}

void CPUTestsBase::CheckCPUImpl(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType,
                                std::vector<cpu_memory_format_t> inputMemoryFormats,
                                std::vector<cpu_memory_format_t> outputMemoryFormats, std::string selectedType) {
    IE_SUPPRESS_DEPRECATED_START
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
                for (const auto & parentPort : node->input_values()) {
                    for (const auto & port : node->inputs()) {
                        if (port.get_tensor_ptr() == parentPort.get_tensor_ptr()) {
                            auto parentNode = parentPort.get_node_shared_ptr();
                            auto actualInputMemoryFormat = getExecValueOutputsLayout(parentNode);
                            ASSERT_EQ(inputMemoryFormats[i], cpu_str2fmt(actualInputMemoryFormat.c_str()));
                        }
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
    IE_SUPPRESS_DEPRECATED_END
}

std::string CPUTestsBase::getTestCaseName(CPUSpecificParams params) {
    std::ostringstream result;
    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
    std::tie(inFmts, outFmts, priority, selectedType) = params;
    result << "_inFmts=" << fmts2str(inFmts);
    result << "_outFmts=" << fmts2str(outFmts);
    result << "_primitive=" << selectedType;
    return result.str();
}

std::map<std::string, std::shared_ptr<ngraph::Variant>> CPUTestsBase::setCPUInfo(std::vector<cpu_memory_format_t> inFmts,
                                                                                 std::vector<cpu_memory_format_t> outFmts,
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

} // namespace CPUTestUtils
