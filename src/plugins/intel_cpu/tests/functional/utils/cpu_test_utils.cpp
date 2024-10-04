// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <regex>

#include "cpu_test_utils.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "utils/general_utils.h"
#include "utils/rt_info/memory_formats_attribute.hpp"

namespace CPUTestUtils {
const char* CPUTestsBase::any_type = "any_type";

const char* CPUTestsBase::cpu_fmt2str(cpu_memory_format_t v) {
#define CASE(_fmt)                    \
    case (cpu_memory_format_t::_fmt): \
        return #_fmt;
    switch (v) {
        CASE(undef);
        CASE(ncw);
        CASE(nCw8c);
        CASE(nCw16c);
        CASE(nwc);
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
        CASE(ntc);
        CASE(ldgoi);
        CASE(ldoi);
    }
#undef CASE
    assert(!"unknown fmt");
    return "undef";
}

cpu_memory_format_t CPUTestsBase::cpu_str2fmt(const char* str) {
#define CASE(_fmt)                                              \
    do {                                                        \
        if (!strcmp(#_fmt, str) || !strcmp("dnnl_" #_fmt, str)) \
            return _fmt;                                        \
    } while (0)
    CASE(undef);
    CASE(a);
    CASE(ab);
    CASE(abc);
    CASE(acb);
    CASE(aBc8b);
    CASE(aBc16b);
    CASE(abcd);
    CASE(acdb);
    CASE(aBcd8b);
    CASE(aBcd16b);
    CASE(abcde);
    CASE(acdeb);
    CASE(aBcde8b);
    CASE(aBcde16b);
    CASE(bac);
    CASE(abdc);
    CASE(abdec);
    CASE(ncw);
    CASE(nCw8c);
    CASE(nCw16c);
    CASE(nwc);
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

std::string CPUTestsBase::fmts2str(const std::vector<cpu_memory_format_t>& fmts, const std::string& prefix) {
    std::string str;
    for (auto& fmt : fmts) {
        ((str += prefix) += cpu_fmt2str(fmt)) += ",";
    }
    if (!str.empty()) {
        str.pop_back();
    }
    return str;
}

ov::PrimitivesPriority CPUTestsBase::impls2primProiority(const std::vector<std::string>& priority) {
    std::string str;
    for (auto& impl : priority) {
        if (!impl.empty())
            ((str += "cpu:") += impl) += ",";
    }
    if (!str.empty()) {
        str.pop_back();
    }
    return ov::PrimitivesPriority(str);
}

void CPUTestsBase::CheckPluginRelatedResults(const ov::CompiledModel& execNet,
                                             const std::set<std::string>& nodeType) const {
    if (!execNet || nodeType.empty())
        return;

    ASSERT_TRUE(!selectedType.empty()) << "Node type is not defined.";
    auto function = execNet.get_runtime_model();
    CheckPluginRelatedResultsImpl(function, nodeType);
}

void CPUTestsBase::CheckPluginRelatedResults(const ov::CompiledModel& execNet, const std::string& nodeType) const {
    CheckPluginRelatedResults(execNet, std::set<std::string>{nodeType});
}

void CPUTestsBase::CheckPluginRelatedResultsImpl(const std::shared_ptr<const ov::Model>& function,
                                                 const std::set<std::string>& nodeType) const {
    ASSERT_NE(nullptr, function);
    for (const auto& node : function->get_ops()) {
        const auto& rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };
        auto getExecValueOutputsLayout = [](const std::shared_ptr<ov::Node>& node) -> std::string {
            auto rtInfo = node->get_rt_info();
            auto it = rtInfo.find(ov::exec_model_info::OUTPUT_LAYOUTS);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };
        // skip policy
        auto should_be_skipped = [](const ov::PartialShape& partialShape, cpu_memory_format_t fmt) {
            if (partialShape.is_dynamic()) {
                return false;
            }

            auto shape = partialShape.get_shape();
            bool skip_unsquized_1D = static_cast<size_t>(std::count(shape.begin(), shape.end(), 1)) == shape.size() - 1;
            bool permule_of_1 = (fmt == cpu_memory_format_t::nhwc || fmt == cpu_memory_format_t::ndhwc ||
                                 fmt == cpu_memory_format_t::nwc) &&
                                shape[1] == 1;
            return skip_unsquized_1D || permule_of_1;
        };

        if (nodeType.count(getExecValue(ov::exec_model_info::LAYER_TYPE))) {
            ASSERT_LE(inFmts.size(), node->get_input_size());
            ASSERT_LE(outFmts.size(), node->get_output_size());
            for (size_t i = 0; i < inFmts.size(); i++) {
                const auto parentPort = node->input_values()[i];
                const auto port = node->inputs()[i];
                if ((parentPort.get_tensor_ptr() == port.get_tensor_ptr())) {
                    auto parentNode = parentPort.get_node_shared_ptr();
                    auto shape = parentNode->get_output_tensor(0).get_partial_shape();
                    auto actualInputMemoryFormat = getExecValueOutputsLayout(parentNode);

                    if (!should_be_skipped(shape, inFmts[i])) {
                        ASSERT_EQ(inFmts[i], cpu_str2fmt(actualInputMemoryFormat.c_str()));
                    }
                }
            }

            /* actual output formats are represented as a single string, for example 'fmt1' or 'fmt1, fmt2, fmt3'
             * convert it to the list of formats */
            auto getActualOutputMemoryFormats = [](const std::string& fmtStr) -> std::vector<std::string> {
                std::vector<std::string> result;
                std::stringstream ss(fmtStr);
                std::string str;
                while (std::getline(ss, str, ',')) {
                    result.push_back(str);
                }
                return result;
            };

            auto actualOutputMemoryFormats = getActualOutputMemoryFormats(getExecValueOutputsLayout(node));

            bool isAllEqual = true;
            for (size_t i = 1; i < outFmts.size(); i++) {
                if (outFmts[i - 1] != outFmts[i]) {
                    isAllEqual = false;
                    break;
                }
            }
            size_t fmtsNum = outFmts.size();
            if (isAllEqual) {
                fmtsNum = fmtsNum == 0 ? 0 : 1;
            } else {
                ASSERT_EQ(fmtsNum, actualOutputMemoryFormats.size());
            }
            for (size_t i = 0; i < fmtsNum; i++) {
                const auto actualOutputMemoryFormat = getExecValue(ov::exec_model_info::OUTPUT_LAYOUTS);
                const auto shape = node->get_output_partial_shape(i);

                if (should_be_skipped(shape, outFmts[i]))
                    continue;
                ASSERT_EQ(outFmts[i], cpu_str2fmt(actualOutputMemoryFormats[i].c_str()));
            }

            auto primType = getExecValue(ov::exec_model_info::IMPL_TYPE);

            ASSERT_TRUE(primTypeCheck(primType))
                << "primType is unexpected : " << primType << " Expected : " << selectedType;
        }
    }
}

bool CPUTestsBase::primTypeCheck(std::string primType) const {
#ifndef NDEBUG
    std::cout << "selectedType: " << selectedType << std::endl;
#endif
    if (selectedType.find("FP") != std::string::npos)
        return selectedType.find(CPUTestsBase::any_type) != std::string::npos ||
               std::regex_match(primType,
                                std::regex(std::regex_replace(selectedType, std::regex("FP"), "f"), std::regex::icase));
    else
        return selectedType.find(CPUTestsBase::any_type) != std::string::npos ||
               std::regex_match(primType, std::regex(selectedType, std::regex::icase));
}

std::string CPUTestsBase::getTestCaseName(CPUSpecificParams params) {
    std::ostringstream result;
    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
    std::tie(inFmts, outFmts, priority, selectedType) = params;
    if (!inFmts.empty()) {
        auto str = fmts2str(inFmts, "");
        std::replace(str.begin(), str.end(), ',', '.');
        result << "_inFmts=" << str;
    }
    if (!outFmts.empty()) {
        auto str = fmts2str(outFmts, "");
        std::replace(str.begin(), str.end(), ',', '.');
        result << "_outFmts=" << str;
    }
    if (!selectedType.empty()) {
        result << "_primitive=" << selectedType;
    }
    return result.str();
}

CPUTestsBase::CPUInfo CPUTestsBase::getCPUInfo() const {
    return makeCPUInfo(inFmts, outFmts, priority);
}

#if defined(OV_CPU_WITH_ACL)
std::string CPUTestsBase::getPrimitiveType() const {
    return "acl";
}
#else
std::string CPUTestsBase::getPrimitiveType() const {
    std::string isaType;
    if (ov::with_cpu_x86_avx512f()) {
        isaType = "jit_avx512";
    } else if (ov::with_cpu_x86_avx2()) {
        isaType = "jit_avx2";
    } else if (ov::with_cpu_x86_sse42()) {
        isaType = "jit_sse42";
    } else {
        isaType = "ref";
    }
    return isaType;
}
#endif

std::string CPUTestsBase::getISA(bool skip_amx) const {
    std::string isaType;
    if (!skip_amx && ov::with_cpu_x86_avx512_core_amx()) {
        isaType = "avx512_amx";
    } else if (ov::with_cpu_x86_avx512f()) {
        isaType = "avx512";
    } else if (ov::with_cpu_x86_avx2()) {
        isaType = "avx2";
    } else if (ov::with_cpu_x86_sse42()) {
        isaType = "sse42";
    } else {
        isaType = "";
    }
    return isaType;
}

static std::string setToString(const std::unordered_set<std::string> s) {
    if (s.empty())
        return {};

    std::string result;
    result.append("{");
    for (const auto& str : s) {
        result.append(str);
        result.append(",");
    }
    result.append("}");

    return result;
}

CPUTestsBase::CPUInfo CPUTestsBase::makeCPUInfo(const std::vector<cpu_memory_format_t>& inFmts,
                                                const std::vector<cpu_memory_format_t>& outFmts,
                                                const std::vector<std::string>& priority) {
    CPUInfo cpuInfo;

    if (!inFmts.empty()) {
        cpuInfo.insert({ov::intel_cpu::InputMemoryFormats::get_type_info_static(),
                        ov::intel_cpu::InputMemoryFormats(fmts2str(inFmts, "cpu:"))});
    }
    if (!outFmts.empty()) {
        cpuInfo.insert({ov::intel_cpu::OutputMemoryFormats::get_type_info_static(),
                        ov::intel_cpu::OutputMemoryFormats(fmts2str(outFmts, "cpu:"))});
    }
    if (!priority.empty()) {
        cpuInfo.emplace(ov::PrimitivesPriority::get_type_info_static(), impls2primProiority(priority));
    }
    cpuInfo.insert({"enforceBF16evenForGraphTail", true});

    return cpuInfo;
}

std::shared_ptr<ov::Model> CPUTestsBase::makeNgraphFunction(const ov::element::Type& ngPrc,
                                                            ov::ParameterVector& params,
                                                            const std::shared_ptr<ov::Node>& lastNode,
                                                            std::string name) {
    auto newLastNode = modifyGraph(ngPrc, params, lastNode);
    ov::ResultVector results;

    for (size_t i = 0; i < newLastNode->get_output_size(); i++)
        results.push_back(std::make_shared<ov::op::v0::Result>(newLastNode->output(i)));

    return std::make_shared<ov::Model>(results, params, name);
}

std::shared_ptr<ov::Node> CPUTestsBase::modifyGraph(const ov::element::Type& ngPrc,
                                                    ov::ParameterVector& params,
                                                    const std::shared_ptr<ov::Node>& lastNode) {
    lastNode->get_rt_info() = getCPUInfo();
    return lastNode;
}

std::string CPUTestsBase::makeSelectedTypeStr(std::string implString, ov::element::Type_t elType) {
    implString.push_back('_');
    implString += ov::element::Type(elType).get_type_name();
    return implString;
}

void CPUTestsBase::updateSelectedType(const std::string& primitiveType,
                                      const ov::element::Type netType,
                                      const ov::AnyMap& config) {
    if (selectedType.empty()) {
        selectedType = primitiveType;
    }

    if (selectedType.find("*") != std::string::npos) {
        selectedType = primitiveType + "_" + selectedType;
        return;
    }

    if (selectedType.find("$/") != std::string::npos) {
        selectedType = selectedType.substr(0, selectedType.find("$/"));
        return;
    }

    auto getExecType = [&]() {
        // inference_precision affects only floating point type networks
        if (!netType.is_real()) {
            if (netType == ov::element::u8) {
                // Node::getPrimitiveDescriptorType() returns i8 for u8
                return ov::element::i8;
            }
            if (netType == ov::element::u32) {
                // Node::getPrimitiveDescriptorType() returns i32 for u32
                return ov::element::i32;
            }
            return netType;
        }

        const auto it = config.find(ov::hint::inference_precision.name());
        if (it == config.end())
            return netType;

        const auto inference_precision_type = it->second.as<ov::element::Type>();
        // currently plugin only allows to change precision from higher to lower (i.e. f32 -> f16 or f32 -> bf16)
        if (netType.bitwidth() < inference_precision_type.bitwidth()) {
            return netType;
        }

        return inference_precision_type;
    };

    const auto execType = getExecType();
    selectedType.push_back('_');
    selectedType += execType.get_type_name();
}

inline void CheckNumberOfNodesWithTypeImpl(std::shared_ptr<const ov::Model> function,
                                           const std::unordered_set<std::string>& nodeTypes,
                                           size_t expectedCount) {
    ASSERT_NE(nullptr, function);
    size_t actualNodeCount = 0;
    for (const auto& node : function->get_ops()) {
        const auto& rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        if (nodeTypes.count(getExecValue(ov::exec_model_info::LAYER_TYPE))) {
            actualNodeCount++;
        }
    }

    ASSERT_EQ(expectedCount, actualNodeCount)
        << "Unexpected count of the node types '" << setToString(nodeTypes) << "' ";
}

void CheckNumberOfNodesWithTypes(const ov::CompiledModel& compiledModel,
                                 const std::unordered_set<std::string>& nodeTypes,
                                 size_t expectedCount) {
    if (!compiledModel)
        return;

    std::shared_ptr<const ov::Model> function = compiledModel.get_runtime_model();

    CheckNumberOfNodesWithTypeImpl(function, nodeTypes, expectedCount);
}

void CheckNumberOfNodesWithType(const ov::CompiledModel& compiledModel,
                                const std::string& nodeType,
                                size_t expectedCount) {
    CheckNumberOfNodesWithTypes(compiledModel, {nodeType}, expectedCount);
}


// deduce the actual precision of the operation given the ngraph level operation precision and the plugin config
ov::element::Type
CPUTestsBase::deduce_expected_precision(const ov::element::Type& opPrecision,
                                        const ov::AnyMap& configuration) {
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    return opPrecision;
#endif
#if defined(OPENVINO_ARCH_RISCV64)
    return opPrecision;
#endif
#if defined(OPENVINO_ARCH_X86_64)
    // if is not float
    if (!opPrecision.is_real()) {
        return opPrecision;
    }
    ov::element::Type inferencePrecision = ov::element::f32;
    bool inferencePrecisionSetExplicitly = false;
    const std::string precisionKey = ov::hint::inference_precision.name();
    const auto& it = configuration.find(precisionKey);
    if (it != configuration.end()) {
        auto inferencePrecisionConfig = it->second.as<ov::element::Type>();
        inferencePrecisionSetExplicitly = true;
        // TODO also need to check (dnnl::impl::cpu::x64::avx2_vnni_2)
        if ((inferencePrecisionConfig == ov::element::bf16 && ov::with_cpu_x86_avx512_core())
                || (inferencePrecisionConfig == ov::element::f16 && ov::with_cpu_x86_avx512_core_fp16())
                || (inferencePrecisionConfig == ov::element::f32)
                || (inferencePrecisionConfig == ov::element::undefined)) {
            inferencePrecision = inferencePrecisionConfig;
        }
    }
    if (!inferencePrecisionSetExplicitly) {
        const std::string executionModeKey = ov::hint::execution_mode.name();
        const auto& configIt = configuration.find(executionModeKey);
        if (configIt != configuration.end() && configIt->second.as<ov::hint::ExecutionMode>() == ov::hint::ExecutionMode::PERFORMANCE) {
            inferencePrecision = ov::element::f32;
            if (ov::with_cpu_x86_bfloat16()) {
                inferencePrecision = ov::element::bf16;
            }
        } else {
            inferencePrecision = ov::element::undefined;
        }
    }

    ov::element::Type deducedType = opPrecision;
    // enforceInferPrecision stage
    if (inferencePrecision == ov::element::bf16) {
        deducedType = ov::with_cpu_x86_avx512_core() ? ov::element::bf16 : ov::element::f32;
    }

    // ngraph transform pipeline stage
    if (inferencePrecision == ov::element::f16) {
        if (deducedType == ov::element::f32) {
            deducedType = ov::element::f16;
        }
    }
    if (deducedType == ov::element::bf16) {
        deducedType = ov::with_cpu_x86_avx512_core() ? ov::element::bf16 : ov::element::f32;
    } else if (deducedType == ov::element::f16) {
        if (inferencePrecision != ov::element::f16 && inferencePrecision != ov::element::undefined) {
            deducedType = ov::element::f32;
        }
    } else {
        deducedType = ov::element::f32;
    }

    return deducedType;
#endif
}

bool containsNonSupportedFormat(const std::vector<cpu_memory_format_t>& formats, const std::vector<cpu_memory_format_t>& non_supported_f) {
    for (const auto& format : formats) {
        if (std::find(non_supported_f.begin(), non_supported_f.end(), format) != non_supported_f.end()) {
            return true;
        }
    }
    return false;
}

bool containsSupportedFormatsOnly(const std::vector<cpu_memory_format_t>& formats, const std::vector<cpu_memory_format_t>& supported_f) {
    for (const auto& format : formats) {
        if (std::find(supported_f.begin(), supported_f.end(), format) == supported_f.end()) {
            return false;
        }
    }
    return true;
}

}  // namespace CPUTestUtils
