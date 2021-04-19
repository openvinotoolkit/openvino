// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_test_utils.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"

namespace CPUTestUtils {

const char *CPUTestsBase::cpu_fmt2str(cpu_memory_format_t v) {
#define CASE(_fmt) do { \
    if (v == _fmt) return #_fmt; \
} while (0)
    CASE(undef);
    CASE(nchw);
    CASE(nChw8c);
    CASE(nChw16c);
    CASE(nhwc);
    CASE(ncdhw);
    CASE(nCdhw8c);
    CASE(nCdhw16c);
    CASE(ndhwc);
    CASE(nc);
    CASE(x);
    CASE(tnc);
    CASE(ntc);
    CASE(ldnc);
    CASE(ldigo);
    CASE(ldgoi);
    CASE(ldio);
    CASE(ldoi);
    CASE(ldgo);
#undef CASE
    assert(!"unknown fmt");
    return "undef";
}

cpu_memory_format_t CPUTestsBase::cpu_str2fmt(const char *str) {
#define CASE(_fmt) do { \
    if (!strcmp(#_fmt, str) \
            || !strcmp("mkldnn_" #_fmt, str)) \
        return _fmt; \
} while (0)
    CASE(undef);
    CASE(a);
    CASE(ab);
    CASE(abcd);
    CASE(acdb);
    CASE(aBcd8b);
    CASE(aBcd16b);
    CASE(abcde);
    CASE(acdeb);
    CASE(aBcde8b);
    CASE(aBcde16b);
    CASE(abc);
    CASE(bac);
    CASE(abdc);
    CASE(abdec);
    CASE(nchw);
    CASE(nChw8c);
    CASE(nChw16c);
    CASE(nhwc);
    CASE(ncdhw);
    CASE(nCdhw8c);
    CASE(nCdhw16c);
    CASE(ndhwc);
    CASE(nc);
    CASE(x);
    CASE(tnc);
    CASE(ntc);
    CASE(ldnc);
    CASE(ldigo);
    CASE(ldgoi);
    CASE(ldio);
    CASE(ldoi);
    CASE(ldgo);
#undef CASE
    assert(!"unknown memory format");
    return undef;
}

std::string CPUTestsBase::fmts2str(const std::vector<cpu_memory_format_t> &fmts, const std::string &prefix) {
    std::string str;
    for (auto &fmt : fmts) {
        ((str += prefix) += cpu_fmt2str(fmt)) += ",";
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

void CPUTestsBase::CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType) const {
    if (nodeType.empty()) return;

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
        // skip policy
        auto should_be_skipped = [] (const ngraph::Shape &shape, cpu_memory_format_t fmt) {
            bool skip_unsquized_1D =  std::count(shape.begin(), shape.end(), 1) == shape.size() - 1;
            bool permule_of_1 = (fmt == cpu_memory_format_t::nhwc || fmt == cpu_memory_format_t::ndhwc) && shape[1] == 1;
            return skip_unsquized_1D || permule_of_1;
        };

        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == nodeType) {
            isNodeFound = true;
            ASSERT_LE(inFmts.size(), node->get_input_size());
            ASSERT_LE(outFmts.size(), node->get_output_size());
            for (int i = 0; i < inFmts.size(); i++) {
                const auto parentPort = node->input_values()[i];
                const auto port = node->inputs()[i];
                if ((parentPort.get_tensor_ptr() == port.get_tensor_ptr())) {
                    auto parentNode = parentPort.get_node_shared_ptr();
                    auto shape = parentNode->get_output_tensor(0).get_shape();
                    auto actualInputMemoryFormat = getExecValueOutputsLayout(parentNode);

                    if (!should_be_skipped(shape, inFmts[i])) {
                        ASSERT_EQ(inFmts[i], cpu_str2fmt(actualInputMemoryFormat.c_str()));
                    }
                }
            }

            /* actual output formats are represented as a single string, for example 'fmt1' or 'fmt1, fmt2, fmt3'
             * convert it to the list of formats */
            auto getActualOutputMemoryFormats = [] (const std::string& fmtStr) -> std::vector<std::string> {
                std::vector<std::string> result;
                std::stringstream ss(fmtStr);
                std::string str;
                while (std::getline(ss, str, ',')) {
                    result.push_back(str);
                }
                return result;
            };

            auto actualOutputMemoryFormats = getActualOutputMemoryFormats(getExecValueOutputsLayout(node));

            for (size_t i = 0; i < outFmts.size(); i++) {
                const auto actualOutputMemoryFormat = getExecValue(ExecGraphInfoSerialization::OUTPUT_LAYOUTS);
                const auto shape = node->get_output_shape(i);

                if (should_be_skipped(shape, outFmts[i]))
                    continue;

                ASSERT_EQ(outFmts[i], cpu_str2fmt(actualOutputMemoryFormats[i].c_str()));
            }

            auto primType = getExecValue(ExecGraphInfoSerialization::IMPL_TYPE);

            ASSERT_EQ(selectedType, primType);
        }
    }
    ASSERT_TRUE(isNodeFound) << "Node type name: \"" << nodeType << "\" has not been found.";
}

std::string CPUTestsBase::getTestCaseName(CPUSpecificParams params) {
    std::ostringstream result;
    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
    std::tie(inFmts, outFmts, priority, selectedType) = params;
    if (!inFmts.empty()) {
        result << "_inFmts=" << fmts2str(inFmts, "");
    }
    if (!outFmts.empty()) {
        result << "_outFmts=" << fmts2str(outFmts, "");
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
                std::make_shared<ngraph::VariantWrapper<ngraph::MLKDNNInputMemoryFormats>>(ngraph::MLKDNNInputMemoryFormats(fmts2str(inFmts, "cpu:")))});
    }
    if (!outFmts.empty()) {
        cpuInfo.insert({std::string(ngraph::MLKDNNOutputMemoryFormatsAttr),
                std::make_shared<ngraph::VariantWrapper<ngraph::MLKDNNOutputMemoryFormats>>(ngraph::MLKDNNOutputMemoryFormats(fmts2str(outFmts, "cpu:")))});
    }
    if (!priority.empty()) {
        cpuInfo.insert({"PrimitivesPriority", std::make_shared<ngraph::VariantWrapper<std::string>>(impls2str(priority))});
    }

    return cpuInfo;
}

std::shared_ptr<ngraph::Function>
CPUTestsBase::makeNgraphFunction(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params,
                                 const std::shared_ptr<ngraph::Node> &lastNode, std::string name) const {
   auto newLastNode = modifyGraph(ngPrc, params, lastNode);
   ngraph::ResultVector results;

   for (int i = 0; i < newLastNode->get_output_size(); i++)
        results.push_back(std::make_shared<ngraph::opset1::Result>(newLastNode->output(i)));

   return std::make_shared<ngraph::Function>(results, params, name);
}

std::shared_ptr<ngraph::Node>
CPUTestsBase::modifyGraph(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params, const std::shared_ptr<ngraph::Node> &lastNode) const {
    lastNode->get_rt_info() = getCPUInfo();
    return lastNode;
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

void CheckNodeOfTypeCount(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType, size_t expectedCount) {
    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    size_t actualNodeCount = 0;
    for (const auto &node : function->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);
            return value->get();
        };
        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == nodeType) {
            actualNodeCount++;
        }
    }

    ASSERT_EQ(expectedCount, actualNodeCount) << "Unexpected count of the node type '" << nodeType << "' ";
}
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::vector<CPUSpecificParams> CPUParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);

        if (selectedTypeStr.find("jit") != std::string::npos && !InferenceEngine::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !InferenceEngine::with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx") != std::string::npos && !InferenceEngine::with_cpu_x86_avx())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !InferenceEngine::with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !InferenceEngine::with_cpu_x86_avx512f())
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}
} // namespace CPUTestUtils
