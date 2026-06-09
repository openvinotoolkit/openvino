// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/frontend.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>

#include "builders/builder.hpp"
#include "gguf_reader.hpp"
#include "input_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/frontend/common/path_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "rt_info_keys.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

constexpr std::array<uint8_t, 4> kGGUFMagic = {'G', 'G', 'U', 'F'};

bool has_gguf_magic(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open())
        return false;
    std::array<char, 4> header{};
    if (!stream.read(header.data(), header.size()))
        return false;
    return std::equal(kGGUFMagic.begin(), kGGUFMagic.end(), reinterpret_cast<const uint8_t*>(header.data()));
}

void populate_rt_info(const std::shared_ptr<ov::Model>& model, const GGUFReader& reader) {
    const std::string arch = reader.architecture();
    auto set = [&](const ov::Any& value, std::vector<std::string> segments) {
        segments.insert(segments.begin(), rt_info::top);
        model->set_rt_info(value, segments);
    };
    auto a = [&](const char* suffix) {
        return arch_key(arch, suffix);
    };

    // --- general ---
    set(reader.version(), {"version"});
    set(arch, {"architecture"});
    if (reader.has(file_keys::general_name))
        set(reader.get_str(file_keys::general_name), {"model_name"});
    set(static_cast<uint32_t>(reader.get_u64(file_keys::general_file_type)), {"file_type"});

    // --- architecture hyper-parameters ---
    set(reader.get_u64(a(arch_suffix::context_length)), {"context_length"});
    set(reader.get_u64(a(arch_suffix::embedding_length)), {"embedding_length"});
    set(reader.get_u64(a(arch_suffix::block_count)), {"block_count"});
    set(reader.get_u64(a(arch_suffix::attention_head_count)), {"attention", "head_count"});
    set(reader.get_u64(a(arch_suffix::attention_head_count_kv)), {"attention", "head_count_kv"});
    set(static_cast<float>(reader.get_f64(a(arch_suffix::attention_layer_norm_rms_eps))),
        {"attention", "layer_norm_rms_epsilon"});
    // rope.dimension_count: many qwen3 GGUFs omit it and instead carry attention.key_length
    // (the per-head dimension RoPE is applied over). Derive it from key_length, then from
    // embedding_length / head_count, so real-world qwen3 files load. The rt-info key is still
    // always populated (SPEC.md §3.3).
    uint64_t rope_dim = 0;
    if (reader.has(a(arch_suffix::rope_dimension_count))) {
        rope_dim = reader.get_u64(a(arch_suffix::rope_dimension_count));
    } else if (reader.has(a(arch_suffix::attention_key_length))) {
        rope_dim = reader.get_u64(a(arch_suffix::attention_key_length));
    } else {
        const uint64_t head_count = reader.get_u64(a(arch_suffix::attention_head_count));
        OPENVINO_ASSERT(head_count > 0, "[GGUF Frontend] '", a(arch_suffix::attention_head_count), "' must be > 0.");
        rope_dim = reader.get_u64(a(arch_suffix::embedding_length)) / head_count;
    }
    set(rope_dim, {"rope", "dimension_count"});
    if (reader.has(a(arch_suffix::rope_freq_base)))
        set(static_cast<float>(reader.get_f64(a(arch_suffix::rope_freq_base))), {"rope", "freq_base"});
    if (reader.has(a(arch_suffix::rope_scaling_type)))
        set(reader.get_str(a(arch_suffix::rope_scaling_type)), {"rope", "scaling", "type"});
    if (reader.has(a(arch_suffix::rope_scaling_factor)))
        set(static_cast<float>(reader.get_f64(a(arch_suffix::rope_scaling_factor))), {"rope", "scaling", "factor"});

    // --- tokenizer ---
    set(reader.get_str(file_keys::tokenizer_model), {"tokenizer", "ggml", "model"});
    // Pre-tokenizer regex family (e.g. "qwen2"): needed by GenAI to pick the BPE split regex. Not in
    // the original SPEC §3.3 table; added so the native FE can fully feed the GenAI tokenizer.
    if (reader.has(file_keys::tokenizer_pre))
        set(reader.get_str(file_keys::tokenizer_pre), {"tokenizer", "ggml", "pre"});
    set(reader.raw(file_keys::tokenizer_tokens), {"tokenizer", "ggml", "tokens"});
    if (reader.has(file_keys::tokenizer_scores))
        set(reader.raw(file_keys::tokenizer_scores), {"tokenizer", "ggml", "scores"});
    if (reader.has(file_keys::tokenizer_token_type))
        set(reader.raw(file_keys::tokenizer_token_type), {"tokenizer", "ggml", "token_type"});
    if (reader.has(file_keys::tokenizer_merges))
        set(reader.raw(file_keys::tokenizer_merges), {"tokenizer", "ggml", "merges"});
    set(static_cast<uint32_t>(reader.get_u64(file_keys::tokenizer_bos_id)), {"tokenizer", "ggml", "bos_token_id"});
    set(static_cast<uint32_t>(reader.get_u64(file_keys::tokenizer_eos_id)), {"tokenizer", "ggml", "eos_token_id"});
    if (reader.has(file_keys::tokenizer_chat_template))
        set(reader.get_str(file_keys::tokenizer_chat_template), {"tokenizer", "chat_template"});

    set(reader.source_file_hash(), {"source_file_hash"});
}

}  // namespace

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    if (variants.empty())
        return false;
    const auto path = ov::frontend::get_path_from_any(variants[0]);
    if (!path)
        return false;
    const auto& ext = path->extension();
    if (ext == ".gguf")
        return true;
    return has_gguf_magic(*path);
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    FRONT_END_GENERAL_CHECK(!variants.empty(), "[GGUF Frontend] No input model is provided.");
    const auto path = ov::frontend::get_path_from_any(variants[0]);
    FRONT_END_GENERAL_CHECK(path.has_value(), "[GGUF Frontend] Expected a path to a .gguf file.");

    // mmap is enabled by default; core appends the resolved enable_mmap flag as the last variant.
    const bool mmap_enable = variants.back().is<bool>() ? variants.back().as<bool>() : true;

    auto reader = std::make_shared<GGUFReader>(path->string(), mmap_enable);
    return std::make_shared<InputModel>(reader);
}

std::shared_ptr<ov::Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    const auto gguf_model = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(gguf_model != nullptr, "[GGUF Frontend] Unexpected input model type.");
    const auto& reader = gguf_model->reader();
    FRONT_END_GENERAL_CHECK(reader != nullptr, "[GGUF Frontend] Input model has no parsed GGUF reader.");

    const std::string arch = reader->architecture();
    FRONT_END_OP_CONVERSION_CHECK(arch == "qwen3",
                                  "[GGUF Frontend] Architecture '",
                                  arch,
                                  "' is not supported in this release (qwen3 only).");

    auto ov_model = build_qwen3_model(*reader);
    populate_rt_info(ov_model, *reader);
    return ov_model;
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (const auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension))
        m_telemetry = telemetry;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
