// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime_requirements.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config.h"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "snippets/op/subgraph.hpp"
#include "utils/precision_support.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "openvino/op/matmul.hpp"
#    include "openvino/runtime/system_conf.hpp"
#    include "ov_ops/fully_connected_compressed.hpp"
#    include "ov_ops/gather_matmul_compressed.hpp"
#    include "ov_ops/rms.hpp"
#    include "snippets/op/brgemm.hpp"
#    include "transformations/cpu_opset/x64/op/llm_mlp.hpp"
#    include "transformations/cpu_opset/x64/op/qkv_proj.hpp"
#    include "transformations/snippets/x64/op/brgemm_utils.hpp"
#elif defined(OPENVINO_ARCH_ARM64)
#    include "cpu/aarch64/cpu_isa_traits.hpp"
#elif defined(OPENVINO_ARCH_RISCV64)
#    include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#endif

namespace ov::intel_cpu {

namespace {

constexpr std::string_view meta_version = "2.0";

constexpr std::string_view cpu_architecture() {
#if defined(OPENVINO_ARCH_X86_64)
    return "x86_64";
#elif defined(OPENVINO_ARCH_X86)
    return "x86";
#elif defined(OPENVINO_ARCH_ARM)
    return "arm";
#elif defined(OPENVINO_ARCH_ARM64)
    return "arm64";
#elif defined(OPENVINO_ARCH_RISCV64)
    return "riscv64";
#else
#    error "Undefined system processor"
#endif
}

std::string runtime_version() {
    std::string version = ov::get_openvino_version().buildNumber;
    std::replace(version.begin(), version.end(), ';', '_');
    return version;
}

// Host capabilities a transformed graph may depend on.
using Predicate = bool (*)();
const std::map<std::string, Predicate, std::less<>>& capabilities() {
    static const std::map<std::string, Predicate, std::less<>> caps = {
        {"bf16",
         [] {
             return hasHardwareSupport(ov::element::bf16);
         }},
        {"f16",
         [] {
             return hasHardwareSupport(ov::element::f16);
         }},
#if defined(OPENVINO_ARCH_X86_64)
        {"ov.avx2", ov::with_cpu_x86_avx2},
        {"ov.amx_bf16", ov::with_cpu_x86_avx512_core_amx},
        {"ov.amx_fp16", ov::with_cpu_x86_avx512_core_amx_fp16},
        {"dnnl.avx2",
         [] {
             return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2);
         }},
        {"brgemm_bf16", brgemm_utils::is_bf16_supported},
        {"brgemm_f16", brgemm_utils::is_fp16_supported},
        {"brgemm_i8", brgemm_utils::is_i8_supported},
        {"fp8_bf16",
         [] {
             return hasFp8WeightsDecompressionSupport(ov::element::bf16);
         }},
        {"fp8_f16",
         [] {
             return hasFp8WeightsDecompressionSupport(ov::element::f16);
         }},
#elif defined(OPENVINO_ARCH_ARM64)
        {"dnnl.asimd",
         [] {
             return dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd);
         }},
#elif defined(OPENVINO_ARCH_RISCV64)
        {"ov.rvv",
         [] {
             return ov::intel_cpu::riscv64::mayiuse(ov::intel_cpu::riscv64::gv);
         }},
#endif
    };
    return caps;
}

std::string subgraph_capability() {
#if defined(OPENVINO_ARCH_X86_64)
    return "dnnl.avx2";
#elif defined(OPENVINO_ARCH_ARM64)
    return "dnnl.asimd";
#elif defined(OPENVINO_ARCH_RISCV64)
    return "ov.rvv";
#else
    return {};
#endif
}

std::optional<bool> has_capability(std::string_view name) {
    const auto& caps = capabilities();
    const auto it = caps.find(name);
    if (it == caps.end()) {
        return std::nullopt;
    }
    return it->second();
}

class RequirementsCollector {
public:
    explicit RequirementsCollector(const Config& config) : m_config(config) {}

    void collect(const std::shared_ptr<const ov::Model>& model) {
        collect_model(model, false);
        if (m_has_bf16_tensors && has_capability("bf16").value_or(false)) {
            m_caps.insert("bf16");
        }
        if (m_config.inferencePrecision == ov::element::f16) {
            m_caps.insert("f16");
        }
    }

    [[nodiscard]] std::string caps() const {
        std::ostringstream ss;
        for (auto cap = m_caps.begin(); cap != m_caps.end(); ++cap) {
            ss << (cap == m_caps.begin() ? "" : ",") << *cap;
        }
        return ss.str();
    }

private:
    void collect_model(const std::shared_ptr<const ov::Model>& model, bool in_snippets) {
        for (const auto& node : model->get_ordered_ops()) {
            collect_caps(*node, in_snippets);
            if (const auto multi_subgraph = ov::as_type_ptr<const ov::op::util::MultiSubGraphOp>(node)) {
                const bool is_snippets = in_snippets || ov::is_type<const ov::snippets::op::Subgraph>(node);
                for (const auto& body : multi_subgraph->get_functions()) {
                    collect_model(body, is_snippets);
                }
            }
        }
    }

    void collect_caps(const ov::Node& node, [[maybe_unused]] bool in_snippets) {
        if (!ov::is_type<ov::op::v0::Parameter>(&node) && !ov::is_type<ov::op::v0::Result>(&node)) {
            for (const auto& output : node.outputs()) {
                m_has_bf16_tensors |= output.get_element_type() == ov::element::bf16;
            }
        }
        if (ov::is_type<ov::snippets::op::Subgraph>(&node)) {
            if (auto cap = subgraph_capability(); !cap.empty()) {
                m_caps.insert(std::move(cap));
            }
        }
#if defined(OPENVINO_ARCH_X86_64)
        if (ov::is_type<ov::op::internal::RMS>(&node)) {
            m_caps.insert("ov.avx2");
        }
        if (in_snippets && (ov::is_type<ov::op::v0::MatMul>(&node) || ov::is_type<ov::snippets::op::Brgemm>(&node))) {
            for (const auto& input : node.inputs()) {
                const auto precision = input.get_element_type();
                if (precision == ov::element::bf16) {
                    m_caps.insert("brgemm_bf16");
                } else if (precision == ov::element::f16) {
                    m_caps.insert("brgemm_f16");
                } else if (precision == ov::element::i8 || precision == ov::element::u8) {
                    m_caps.insert("brgemm_i8");
                }
            }
        }
        if (ov::is_type<ov::intel_cpu::LLMMLPNode>(&node) || ov::is_type<ov::intel_cpu::QKVProjectionNode>(&node)) {
            m_caps.insert(node.get_input_element_type(0) == ov::element::f16 ? "ov.amx_fp16" : "ov.amx_bf16");
        }
        if (ov::is_type<ov::op::internal::FullyConnectedCompressed>(&node) ||
            ov::is_type<ov::op::internal::GatherMatmulCompressed>(&node)) {
            collect_compressed_matmul_caps(node);
        }
#endif
    }

#if defined(OPENVINO_ARCH_X86_64)
    // The compressed matmuls are executed with oneDNN, see FullyConnected::isSupportedCompressedOperation() and
    // GatherMatmul::isSupportedCompressedOperation()
    void collect_compressed_matmul_caps(const ov::Node& node) {
        m_caps.insert("dnnl.avx2");
        const auto weights_precision = node.get_input_element_type(1);
        if (weights_precision == ov::element::f8e4m3 || weights_precision == ov::element::f8e5m2) {
            m_caps.insert(node.get_input_element_type(0) == ov::element::f16 ? "fp8_f16" : "fp8_bf16");
        }
    }
#endif

    const Config& m_config;
    std::set<std::string> m_caps;
    bool m_has_bf16_tensors = false;
};

std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> result;
    if (str.empty()) {
        return result;
    }
    std::string item;
    std::istringstream ss(str);
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
    if (str.back() == delim) {
        result.emplace_back();
    }
    return result;
}

std::string check_caps(const std::string& caps) {
    for (const auto& cap : split(caps, ',')) {
        const auto present = has_capability(cap);
        if (!present.has_value()) {
            return "unknown capability '" + cap + "'";
        }
        if (!present.value()) {
            return "host lacks required capability '" + cap + "'";
        }
    }
    return {};
}

}  // namespace

std::string build_runtime_requirements(const std::shared_ptr<const ov::Model>& model, const Config& config) {
    RequirementsCollector collector(config);
    collector.collect(model);

    std::ostringstream ss;
    ss << "meta=" << meta_version << ";ov=" << runtime_version() << ";arch=" << cpu_architecture()
       << ";caps=" << collector.caps();
    return ss.str();
}

std::string check_runtime_requirements(const std::string& requirements) {
    std::map<std::string, std::string> fields;
    for (const auto& field : split(requirements, ';')) {
        const auto pos = field.find('=');
        if (pos == std::string::npos || !fields.emplace(field.substr(0, pos), field.substr(pos + 1)).second) {
            return "malformed requirements field '" + field + "'";
        }
    }

    const auto meta = fields.find("meta");
    if (meta == fields.end() || meta->second != meta_version) {
        return "unsupported requirements format (expected meta=" + std::string(meta_version) + ")";
    }
    for (const auto& [key, value] : fields) {
        if (key != "meta" && key != "ov" && key != "arch" && key != "caps") {
            return "unknown requirements field '" + key + "'";
        }
    }
    for (const auto* key : {"ov", "arch", "caps"}) {
        if (fields.count(key) == 0) {
            return std::string("missing requirements field '") + key + "'";
        }
    }

    if (fields["ov"] != runtime_version()) {
        return "blob was produced by OpenVINO " + fields["ov"] + ", the current version is " + runtime_version();
    }

    if (fields["arch"] != cpu_architecture()) {
        return "blob was produced for " + fields["arch"] + " architecture";
    }

    return check_caps(fields["caps"]);
}

bool is_runtime_requirements_compatible(const std::string& requirements) {
    return check_runtime_requirements(requirements).empty();
}

}  // namespace ov::intel_cpu
