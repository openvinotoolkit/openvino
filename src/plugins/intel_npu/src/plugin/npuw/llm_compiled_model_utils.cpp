// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_compiled_model_utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <unordered_set>

#include "intel_npu/config/config.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

bool ov::npuw::util::has_input(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    auto inputs = model->inputs();
    auto it = std::find_if(inputs.begin(), inputs.end(), [&](const auto& port) {
        return port.get_names().count(name) != 0;
    });
    return it != inputs.end();
}

std::shared_ptr<ov::Node> ov::npuw::util::find_kv_cache_concat(const std::shared_ptr<ov::Node>& sdpa) {
    // The key is SDPA input 1, reached as Concat -> Broadcast -> Reshape -> SDPA when the model
    // carries a GQA KV cache.
    auto* reshape_node = sdpa->input(1).get_source_output().get_node();
    if (reshape_node == nullptr || strstr(reshape_node->get_type_name(), "Reshape") == nullptr) {
        return nullptr;
    }
    auto* broadcast_node = reshape_node->input(0).get_source_output().get_node();
    if (broadcast_node == nullptr || strstr(broadcast_node->get_type_name(), "Broadcast") == nullptr) {
        return nullptr;
    }
    auto* concat_node = broadcast_node->input(1).get_source_output().get_node();
    if (concat_node == nullptr || strstr(concat_node->get_type_name(), "Concat") == nullptr) {
        return nullptr;
    }
    return concat_node->shared_from_this();
}

bool ov::npuw::util::is_encoder_embedding_model(const std::shared_ptr<ov::Model>& model) {
    // The autoregressive embedding path reconstructs a prefill/KV model out of the KV-cache
    // concat feeding each SDPA key. A bidirectional encoder (BERT) has SDPA but no such concat,
    // so there is nothing to reconstruct and it has to run as a single forward instead. Routing
    // on the same predicate the reconstruction uses keeps the two from ever disagreeing.
    bool has_sdpa = false;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) {
            has_sdpa = true;
            if (find_kv_cache_concat(op)) {
                // Autoregressive (Qwen3-Embedding-style), handled by PrepareTextEmbeddingModel.
                return false;
            }
        }
    }
    return has_sdpa;
}

void ov::npuw::util::validate_encoder_embedding_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& param : model->get_parameters()) {
        for (const auto& name : param->get_output_tensor(0).get_names()) {
            OPENVINO_ASSERT(name.find("past_key_values") == std::string::npos,
                            "Encoder embedding model has an unexpected autoregressive KV-cache input '",
                            name,
                            "'.");
        }
    }
}

std::optional<uint32_t> ov::npuw::util::get_max_position_embeddings(const std::shared_ptr<ov::Model>& model) {
    namespace opp = ov::pass::pattern;

    // Every embedding table in a BERT-like encoder is read by the same shape of subgraph: a rank-2
    // constant [rows, hidden] behind a Gather, with whatever Convert (and, for compressed weights,
    // Subtract and Multiply) decompression left in place. Spell those intermediates out rather
    // than walking a fixed number of hops through whatever happens to sit above the Gather.
    // int4 and nf4 exports also convert back up to the activation type after scaling, so allow a
    // Convert on either side of the dequantization.
    auto table = opp::wrap_type<ov::op::v0::Constant>();
    auto converted = opp::optional<ov::op::v0::Convert>({table->output(0)});
    auto unzeroed = opp::optional<ov::op::v1::Subtract>({converted->output(0), opp::any_input()});
    auto scaled = opp::optional<ov::op::v1::Multiply>({unzeroed->output(0), opp::any_input()});
    auto upcast = opp::optional<ov::op::v0::Convert>({scaled->output(0)});
    auto lookup = opp::wrap_type<ov::op::v8::Gather>({upcast->output(0), opp::any_input(), opp::any_input()});

    // The word, token-type and position tables are all the same lookup, so the topology on its own
    // cannot tell them apart and the exported name has to do it. It now chooses among candidates
    // the pattern has already established are embedding tables, instead of being the only thing
    // between us and an unrelated Gather. Picking the wrong table here would be expensive: the
    // token-type one has two rows and would clamp the sequence length to nothing.
    opp::Matcher matcher(lookup, "npuw::PositionEmbeddingTable");
    for (const auto& op : model->get_ops()) {
        if (!ov::is_type<ov::op::v8::Gather>(op) ||
            op->get_friendly_name().find("position_embeddings") == std::string::npos) {
            continue;
        }
        if (!matcher.match(op->output(0))) {
            continue;
        }
        const auto& table_shape = matcher.get_pattern_value_map().at(table).get_partial_shape();
        if (table_shape.rank().is_static() && table_shape.size() == 2 && table_shape[0].is_static()) {
            return static_cast<uint32_t>(table_shape[0].get_length());
        }
    }
    return std::nullopt;
}

std::optional<ov::npuw::util::NPUDesc> ov::npuw::util::extract_npu_descriptor(
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& config) {
    if (!plugin->get_core()) {
        return std::nullopt;
    }
    std::vector<std::string> all_devices;
    try {
        all_devices = plugin->get_core()->get_property("NPU", ov::available_devices);
    } catch (const ov::Exception& ex) {
        LOG_WARN("Failed to query NPU capabilities, defaulting LLM config to backend-agnostic path: " << ex.what());
        return std::nullopt;
    }
    if (all_devices.empty()) {
        return std::nullopt;
    }
    NPUDesc desc;
    desc.arch = plugin->get_property(ov::device::architecture.name(), ov::AnyMap{}).as<std::string>();
    desc.max_tiles = plugin->get_property(ov::intel_npu::max_tiles.name(), ov::AnyMap{}).as<int64_t>();

    // Don't use reference here!
    const auto supported_properties =
        plugin->get_property(ov::supported_properties.name(), ov::AnyMap{}).as<std::vector<ov::PropertyName>>();
    if (std::find(supported_properties.begin(), supported_properties.end(), "NPU_COMPILER_DYNAMIC_QUANTIZATION") !=
        supported_properties.end()) {
        desc.compiler_dq = true;
    }

    // Get compiler version based on NPU_COMPILER_TYPE configuration
    // If NPU_COMPILER_TYPE is not specified in config, use default compiler version
    auto compiler_type_it = config.find(ov::intel_npu::compiler_type.name());
    if (compiler_type_it == config.end()) {
        // NPU_COMPILER_TYPE is not specified in config, use default compiler version
        desc.compiler_ver = plugin->get_property(ov::intel_npu::compiler_version.name(), ov::AnyMap{}).as<int64_t>();
    } else {
        // NPU_COMPILER_TYPE is specified in config, get compiler version for the specified compiler type
        auto target_compiler_type = compiler_type_it->second.as<std::string>();
        desc.compiler_ver = plugin
                                ->get_property(ov::intel_npu::compiler_version.name(),
                                               ov::AnyMap{{ov::intel_npu::compiler_type.name(), target_compiler_type}})
                                .as<int64_t>();
    }
    LOG_INFO("Compiler version: " << ONEAPI_VERSION_MAJOR(desc.compiler_ver) << "."
                                  << ONEAPI_VERSION_MINOR(desc.compiler_ver));

    constexpr std::string_view compiler_gate_support_msg =
        "Compiler: accurate gated matmul (MatMul -> Divide -> Tanh -> Multiply -> Result) : ";

    if (desc.compiler_ver >= ONEAPI_MAKE_VERSION(7, 28)) {
        // accuracy for gated matmul fixed at 7.28
        desc.compiler_matmul_gate = true;
        LOG_INFO(compiler_gate_support_msg << "supported");
    } else {
        LOG_WARN(compiler_gate_support_msg << "unsupported");
    }

    static const std::unordered_set<std::string_view> flash_attention_tile_supported_platforms = {
        ov::intel_npu::Platform::NPU5010,
        ov::intel_npu::Platform::NPU6010};

    // Flash attention tile with GQA is supported starting from compiler version 8.1 on supported platforms
    desc.support_flash_attention_tile =
        flash_attention_tile_supported_platforms.count(desc.arch) && desc.compiler_ver >= ONEAPI_MAKE_VERSION(8, 1);

    return std::make_optional(std::move(desc));
}

namespace {
ov::AnyMap get_baseline_common_config(const std::optional<ov::npuw::util::NPUDesc>& npudesc) {
    ov::AnyMap config = {
        {"NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm"},
        {"NPUW_DEVICES", "NPU"},
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_FOLD", "YES"},
        {"NPUW_DCOFF_TYPE", "f16"},
        {"NPUW_DCOFF_SCALE", "YES"},
        {"NPUW_SLICE_OUT", "YES"},
        {"NPUW_FUNCALL_ASYNC", "YES"}};
    // FIXME: this config logic is getting more and more complex
    if (npudesc.has_value() && npudesc->compiler_dq) {
        config.emplace("NPUW_DQ", "YES");
        config.emplace("NPUW_DQ_FULL", "NO");
        config.emplace("NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES");
        config.erase("NPUW_DCOFF_TYPE");
        config.erase("NPUW_DCOFF_SCALE");
    }

    // default value is ON
    // for compiler versions >= 7.28 value is ON
    // for other compiler versions value is OFF
    if (npudesc.has_value()) {
        config.emplace("NPUW_MM_GATED", (npudesc->compiler_matmul_gate ? "YES" : "NO"));
    }
    return config;
}
}  // namespace

ov::AnyMap ov::npuw::util::get_default_common_config(const std::optional<NPUDesc>& npudesc) {
    // FIXME: add `if_model_contain_slice()` condition for `SLICE_OUT` option.
    auto config = get_baseline_common_config(npudesc);
    const char* npu_l0 = std::getenv("DISABLE_OPENVINO_GENAI_NPU_L0");
    if (npu_l0 && std::atoi(npu_l0) == 1) {
        config.emplace("NPUW_WEIGHTS_BANK_ALLOC", "CPU");
    } else {
        config.emplace("NPUW_FUNCALL_FOR_ALL", "YES");
    }
    auto set_max_tiles_based_on_arch = [&config, &npudesc]() {
        if (!npudesc.has_value()) {
            return;
        }
        LOG_DEBUG("NPU architecture detected: " << npudesc->arch << ", max tiles: " << npudesc->max_tiles
                                                << ", compiler DQ support: " << (npudesc->compiler_dq ? "YES" : "NO"));

        // Platform parameter has a higher priority than deviceID
        std::string npu_platform;
        if (npudesc->arch != ov::intel_npu::Platform::AUTO_DETECT) {
            npu_platform = ov::intel_npu::Platform::standardize(npudesc->arch);
        } else {
            npu_platform = npudesc->arch;
        }

        bool set_npu_tiles = false;
        std::string arch_added_compilation_param;
        if (npu_platform == ov::intel_npu::Platform::NPU3720) {
            // Keep baseline settings
        } else if (npu_platform == ov::intel_npu::Platform::NPU4000) {
            set_npu_tiles = true;
            arch_added_compilation_param = "optimization-level=3";
        } else if (npu_platform == ov::intel_npu::Platform::NPU5010 ||
                   npu_platform == ov::intel_npu::Platform::NPU5020) {
            set_npu_tiles = true;
        } else if (npu_platform == ov::intel_npu::Platform::NPU6010) {
            arch_added_compilation_param = "performance-hint-override=latency";
        } else if (npu_platform == ov::intel_npu::Platform::AUTO_DETECT) {
            // Keep baseline settings
        } else {
            LOG_WARN("Unknown NPU platform: " << npu_platform << ". Default config will be used.");
        }

        if (set_npu_tiles) {
            config["NPU_TILES"] = npudesc->max_tiles;
        }

        if (!arch_added_compilation_param.empty()) {
            config["NPU_COMPILATION_MODE_PARAMS"] =
                config["NPU_COMPILATION_MODE_PARAMS"].as<std::string>() + " " + arch_added_compilation_param;
        }
    };
    set_max_tiles_based_on_arch();
    return config;
}
