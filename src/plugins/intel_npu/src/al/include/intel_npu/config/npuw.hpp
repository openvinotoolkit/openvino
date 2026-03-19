// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/npuw_private_properties.hpp"

namespace ov {
namespace npuw {
namespace s11n {
// FIXME: likely shouldn't be here as it was initially a part of npuw::s11n
// but we need to somehow serialize AnyMap right here for several properties.
enum class AnyType : int {
    STRING = 0,
    CHARS,
    INT,
    UINT32,
    INT64,
    UINT64,
    SIZET,
    FLOAT,
    BOOL,
    CACHE_MODE,
    ELEMENT_TYPE,
    ANYMAP,
    PERFMODE
};

std::string anyToString(const ov::Any& var);
ov::Any stringToAny(const std::string& var);
std::string anyMapToString(const ov::AnyMap& var);
ov::AnyMap stringToAnyMap(const std::string& var);
}  // namespace s11n
}  // namespace npuw
}  // namespace ov

namespace intel_npu {

//
// register
//

void registerNPUWOptions(OptionsDesc& desc);
void registerNPUWLLMOptions(OptionsDesc& desc);
void registerNPUWKokoroOptions(OptionsDesc& desc);

#define DEFINE_OPT(Name, Type, DefaultValue, PropertyKey, Mode) \
    struct Name final : OptionBase<Name, Type> {                \
        static std::string_view key() {                         \
            return ov::intel_npu::PropertyKey.name();           \
        }                                                       \
                                                                \
        static Type defaultValue() {                            \
            return DefaultValue;                                \
        }                                                       \
                                                                \
        static OptionMode mode() {                              \
            return OptionMode::Mode;                            \
        }                                                       \
    };

#define DEFINE_ANYMAP_OPT(Name, PropertyKey)                         \
    struct Name final : OptionBase<Name, ov::AnyMap> {               \
        static std::string_view key() {                              \
            return ov::intel_npu::PropertyKey.name();                \
        }                                                            \
                                                                     \
        static constexpr std::string_view getTypeName() {            \
            return "::intel_npu::" #PropertyKey;                     \
        }                                                            \
                                                                     \
        static ov::AnyMap defaultValue() {                           \
            return {};                                               \
        }                                                            \
                                                                     \
        static ov::AnyMap parse(std::string_view val) {              \
            return ov::npuw::s11n::stringToAnyMap(std::string(val)); \
        }                                                            \
                                                                     \
        static std::string toString(const ov::AnyMap& val) {         \
            return ov::npuw::s11n::anyMapToString(val);              \
        }                                                            \
                                                                     \
        static OptionMode mode() {                                   \
            return OptionMode::RunTime;                              \
        }                                                            \
                                                                     \
        static bool isPublic() {                                     \
            return false;                                            \
        }                                                            \
    };

DEFINE_OPT(NPU_USE_NPUW, bool, false, use_npuw, RunTime);
DEFINE_OPT(NPUW_DEVICES, std::string, "NPU,CPU", npuw::devices, RunTime);
DEFINE_OPT(NPUW_SUBMODEL_DEVICE, std::string, "", npuw::submodel_device, RunTime);
DEFINE_OPT(NPUW_ONLINE_PIPELINE, std::string, "REG", npuw::partitioning::online::pipeline, RunTime);
DEFINE_OPT(NPUW_ONLINE_AVOID, std::string, "", npuw::partitioning::online::avoid, RunTime);
DEFINE_OPT(NPUW_ONLINE_ISOLATE, std::string, "", npuw::partitioning::online::isolate, RunTime);
DEFINE_OPT(NPUW_ONLINE_NO_FOLD, std::string, "", npuw::partitioning::online::nofold, RunTime);
DEFINE_OPT(NPUW_ONLINE_MIN_SIZE, std::size_t, 10, npuw::partitioning::online::min_size, RunTime);
DEFINE_OPT(NPUW_ONLINE_KEEP_BLOCKS, std::size_t, 5, npuw::partitioning::online::keep_blocks, RunTime);
DEFINE_OPT(NPUW_ONLINE_KEEP_BLOCK_SIZE, std::size_t, 10, npuw::partitioning::online::keep_block_size, RunTime);
DEFINE_OPT(NPUW_ONLINE_DUMP_PLAN, std::string, "", npuw::partitioning::online::dump_plan, RunTime);
DEFINE_OPT(NPUW_PLAN, std::string, "", npuw::partitioning::plan, RunTime);
DEFINE_OPT(NPUW_FOLD, bool, false, npuw::partitioning::fold, RunTime);
DEFINE_OPT(NPUW_CWAI, bool, false, npuw::partitioning::cwai, RunTime);
DEFINE_OPT(NPUW_DQ, bool, false, npuw::partitioning::dyn_quant, RunTime);
DEFINE_OPT(NPUW_DQ_FULL, bool, true, npuw::partitioning::dyn_quant_full, RunTime);
DEFINE_OPT(NPUW_PMM, std::string, "2", npuw::partitioning::par_matmul_merge_dims, RunTime);
DEFINE_OPT(NPUW_MM_GATED, bool, true, npuw::partitioning::matmul_gate_preserve_constants, RunTime);
DEFINE_OPT(NPUW_SLICE_OUT, bool, false, npuw::partitioning::slice_out, RunTime);
DEFINE_OPT(NPUW_HOST_GATHER, bool, true, npuw::partitioning::host_gather, RunTime);
DEFINE_OPT(NPUW_SPATIAL, bool, false, npuw::partitioning::spatial, RunTime);
DEFINE_OPT(NPUW_F16IC, bool, true, npuw::partitioning::f16_interconnect, RunTime);
DEFINE_OPT(NPUW_SPATIAL_NWAY, std::size_t, 128, npuw::partitioning::spatial_nway, RunTime);
DEFINE_OPT(NPUW_SPATIAL_DYN, bool, true, npuw::partitioning::spatial_dyn, RunTime);
DEFINE_OPT(NPUW_MOE_TOKEN_CHUNK_SIZE, uint64_t, 0, npuw::partitioning::moe_token_chunk_size, RunTime);
DEFINE_OPT(NPUW_MOE_POOL_SIZE, std::size_t, 8, npuw::partitioning::moe_pool_size, RunTime);
DEFINE_OPT(NPUW_ATTN, std::string, "STATIC", npuw::partitioning::attn, RunTime);
DEFINE_OPT(NPUW_ATTN_DYN, bool, true, npuw::partitioning::attn_dyn, RunTime);
DEFINE_OPT(NPUW_ATTN_NO_COPY, bool, false, npuw::partitioning::attn_no_copy, RunTime);
DEFINE_OPT(NPUW_ATTN_HFA_FUSED, bool, false, npuw::partitioning::attn_hfa_fused, RunTime);
DEFINE_OPT(NPUW_DCOFF_TYPE, std::string, "", npuw::partitioning::dcoff_type, RunTime);
DEFINE_OPT(NPUW_DCOFF_SCALE, bool, false, npuw::partitioning::dcoff_with_scale, RunTime);
DEFINE_OPT(NPUW_FUNCALL_FOR_ALL, bool, false, npuw::partitioning::funcall_for_all, RunTime);
DEFINE_OPT(NPUW_PARALLEL_COMPILE, bool, false, npuw::parallel_compilation, RunTime);
DEFINE_OPT(NPUW_WEIGHTS_BANK, std::string, "", npuw::weights_bank, RunTime);
DEFINE_OPT(NPUW_WEIGHTS_BANK_ALLOC, std::string, "", npuw::weights_bank_alloc, RunTime);
DEFINE_OPT(NPUW_CACHE_DIR, std::string, "", npuw::cache_dir, RunTime);
DEFINE_OPT(NPUW_FUNCALL_ASYNC, bool, false, npuw::funcall_async, RunTime);
DEFINE_OPT(NPUW_UNFOLD_IREQS, bool, false, npuw::unfold_ireqs, RunTime);
DEFINE_OPT(NPUW_FALLBACK_EXEC, bool, true, npuw::fallback_exec, RunTime);
DEFINE_OPT(NPUW_ACC_CHECK, bool, false, npuw::accuracy::check, RunTime);
DEFINE_OPT(NPUW_ACC_THRESH, double, 0.01, npuw::accuracy::threshold, RunTime);
DEFINE_OPT(NPUW_ACC_DEVICE, std::string, "", npuw::accuracy::reference_device, RunTime);
DEFINE_OPT(NPUW_DUMP_FULL, bool, false, npuw::dump::full, RunTime);
DEFINE_OPT(NPUW_DUMP_SUBS, std::string, "", npuw::dump::subgraphs, RunTime);
DEFINE_OPT(NPUW_DUMP_SUBS_DIR, std::string, "", npuw::dump::subgraphs_dir, RunTime);
DEFINE_OPT(NPUW_DUMP_SUBS_ON_FAIL, std::string, "", npuw::dump::subgraphs_on_fail, RunTime);
DEFINE_OPT(NPUW_DUMP_IO, std::string, "", npuw::dump::inputs_outputs, RunTime);
DEFINE_OPT(NPUW_DUMP_IO_ITERS, bool, false, npuw::dump::io_iters, RunTime);
DEFINE_OPT(NPUW_LLM, bool, false, npuw::llm::enabled, RunTime);
DEFINE_OPT(NPUW_LLM_BATCH_DIM, uint32_t, 0, npuw::llm::batch_dim, RunTime);
DEFINE_OPT(NPUW_LLM_SEQ_LEN_DIM, uint32_t, 2, npuw::llm::seq_len_dim, RunTime);
DEFINE_OPT(NPUW_LLM_MAX_PROMPT_LEN, uint32_t, 1024, npuw::llm::max_prompt_len, RunTime);
DEFINE_OPT(NPUW_LLM_MAX_GENERATION_TOKEN_LEN, uint32_t, 1, npuw::llm::max_generation_token_len, RunTime);
DEFINE_OPT(NPUW_LLM_MIN_RESPONSE_LEN, uint32_t, 128, npuw::llm::min_response_len, RunTime);
DEFINE_OPT(NPUW_LLM_OPTIMIZE_V_TENSORS, bool, true, npuw::llm::optimize_v_tensors, RunTime);
DEFINE_OPT(NPUW_LLM_CACHE_ROPE, bool, true, npuw::llm::cache_rope, RunTime);
DEFINE_OPT(NPUW_LLM_GENERATE_PYRAMID, bool, false, npuw::llm::generate_pyramid, RunTime);
DEFINE_OPT(NPUW_LLM_PREFILL_CHUNK_SIZE, uint64_t, 1024, npuw::llm::prefill_chunk_size, RunTime);
DEFINE_OPT(NPUW_LLM_SHARED_HEAD, bool, true, npuw::llm::shared_lm_head, RunTime);
DEFINE_OPT(NPUW_KOKORO, bool, false, npuw::kokoro::enabled, RunTime);
DEFINE_OPT(NPUW_KOKORO_BLOCK_SIZE, uint64_t, 200, npuw::kokoro::block_size, RunTime);
DEFINE_OPT(NPUW_KOKORO_OVERLAP_SIZE, uint64_t, 20, npuw::kokoro::overlap_size, RunTime);
DEFINE_OPT(NPUW_LLM_MAX_LORA_RANK, uint32_t, 32, npuw::llm::max_lora_rank, RunTime);
DEFINE_OPT(NPUW_LLM_OPTIMIZE_FP8, bool, false, npuw::llm::optimize_fp8, RunTime);
DEFINE_OPT(NPUW_LLM_ENABLE_PREFIX_CACHING, bool, false, npuw::llm::enable_prefix_caching, RunTime);
DEFINE_OPT(NPUW_LLM_PREFIX_CACHING_BLOCK_SIZE, uint64_t, 256, npuw::llm::prefix_caching_block_size, RunTime);
DEFINE_OPT(NPUW_LLM_PREFIX_CACHING_MAX_NUM_BLOCKS, uint64_t, 128, npuw::llm::prefix_caching_max_num_blocks, RunTime);
DEFINE_OPT(NPUW_WHISPER, bool, false, npuw::whisper::enabled, RunTime);
DEFINE_OPT(NPUW_WHISPER_EOS_TOKEN, uint64_t, 50257, npuw::whisper::whisper_eos_token, RunTime);
DEFINE_OPT(NPUW_EAGLE, bool, false, npuw::eagle::enabled, RunTime);
DEFINE_OPT(NPUW_TEXT_EMBED, bool, false, npuw::text_embed::enabled, RunTime);
DEFINE_ANYMAP_OPT(NPUW_LLM_PREFILL_CONFIG, npuw::llm::prefill_config);
DEFINE_ANYMAP_OPT(NPUW_LLM_ADDITIONAL_PREFILL_CONFIG, npuw::llm::additional_prefill_config);
DEFINE_ANYMAP_OPT(NPUW_LLM_GENERATE_CONFIG, npuw::llm::generate_config);
DEFINE_ANYMAP_OPT(NPUW_LLM_ADDITIONAL_GENERATE_CONFIG, npuw::llm::additional_generate_config);
DEFINE_ANYMAP_OPT(NPUW_LLM_SHARED_LM_HEAD_CONFIG, npuw::llm::shared_lm_head_config);
DEFINE_ANYMAP_OPT(NPUW_LLM_ADDITIONAL_SHARED_LM_HEAD_CONFIG, npuw::llm::additional_shared_lm_head_config);

namespace npuw {
namespace llm {
enum class PrefillHint { DYNAMIC, STATIC };
enum class GenerateHint { FAST_COMPILE, BEST_PERF };
enum class AttentionHint { DYNAMIC, STATIC, PYRAMID, HFA };
enum class MoEHint { DENSE, HOST_ROUTED, DEVICE_ROUTED };
}  // namespace llm
}  // namespace npuw

struct NPUW_LLM_PREFILL_HINT final : OptionBase<NPUW_LLM_PREFILL_HINT, ::intel_npu::npuw::llm::PrefillHint> {
    static std::string_view key() {
        return ov::intel_npu::npuw::llm::prefill_hint.name();
    }

    static constexpr std::string_view getTypeName() {
        return "::intel_npu::npuw::llm::PrefillHint";
    }

    static ::intel_npu::npuw::llm::PrefillHint defaultValue() {
        return ::intel_npu::npuw::llm::PrefillHint::DYNAMIC;
    }

    static ::intel_npu::npuw::llm::PrefillHint parse(std::string_view val) {
        if (val == "DYNAMIC") {
            return ::intel_npu::npuw::llm::PrefillHint::DYNAMIC;
        } else if (val == "STATIC") {
            return ::intel_npu::npuw::llm::PrefillHint::STATIC;
        }
        OPENVINO_THROW("Unsupported \"PREFILL_HINT\" provided: ", val);
        return {};
    }

    static std::string toString(const ::intel_npu::npuw::llm::PrefillHint& val) {
        switch (val) {
        case ::intel_npu::npuw::llm::PrefillHint::DYNAMIC:
            return "DYNAMIC";
        case ::intel_npu::npuw::llm::PrefillHint::STATIC:
            return "STATIC";
        default:
            OPENVINO_THROW("Can't convert provided \"PREFILL_HINT\" : ", int(val), " to string.");
        }
        return {};
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct ATTN_HINT_BASE : OptionBase<ATTN_HINT_BASE, ::intel_npu::npuw::llm::AttentionHint> {
    static constexpr std::string_view getTypeName() {
        return "::intel_npu::npuw::llm::AttentionHint";
    }

    static ::intel_npu::npuw::llm::AttentionHint defaultValue() {
        return ::intel_npu::npuw::llm::AttentionHint::STATIC;
    }

    static ::intel_npu::npuw::llm::AttentionHint parse(std::string_view val) {
        if (val == "DYNAMIC") {
            return ::intel_npu::npuw::llm::AttentionHint::DYNAMIC;
        } else if (val == "STATIC") {
            return ::intel_npu::npuw::llm::AttentionHint::STATIC;
        } else if (val == "PYRAMID") {
            return ::intel_npu::npuw::llm::AttentionHint::PYRAMID;
        } else if (val == "HFA") {
            return ::intel_npu::npuw::llm::AttentionHint::HFA;
        }
        OPENVINO_THROW("Unsupported attention hint provided: ", val);
        return {};
    }

    static std::string toString(const ::intel_npu::npuw::llm::AttentionHint& val) {
        switch (val) {
        case ::intel_npu::npuw::llm::AttentionHint::DYNAMIC:
            return "DYNAMIC";
        case ::intel_npu::npuw::llm::AttentionHint::STATIC:
            return "STATIC";
        case ::intel_npu::npuw::llm::AttentionHint::PYRAMID:
            return "PYRAMID";
        case ::intel_npu::npuw::llm::AttentionHint::HFA:
            return "HFA";
        default:
            OPENVINO_THROW("Can't convert provided attention hint : ", int(val), " to string.");
        }
        return {};
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_LLM_GENERATE_ATTENTION_HINT final : ATTN_HINT_BASE {
    static std::string_view key() {
        return ov::intel_npu::npuw::llm::generate_attn_hint.name();
    }
};

struct NPUW_LLM_PREFILL_ATTENTION_HINT final : ATTN_HINT_BASE {
    static std::string_view key() {
        return ov::intel_npu::npuw::llm::prefill_attn_hint.name();
    }
};

struct MOE_HINT_BASE : OptionBase<MOE_HINT_BASE, ::intel_npu::npuw::llm::MoEHint> {
    static constexpr std::string_view getTypeName() {
        return "::intel_npu::npuw::llm::MoEHint";
    }

    static ::intel_npu::npuw::llm::MoEHint defaultValue() {
        return ::intel_npu::npuw::llm::MoEHint::HOST_ROUTED;
    }

    static ::intel_npu::npuw::llm::MoEHint parse(std::string_view val) {
        if (val == "DENSE") {
            return ::intel_npu::npuw::llm::MoEHint::DENSE;
        } else if (val == "HOST_ROUTED") {
            return ::intel_npu::npuw::llm::MoEHint::HOST_ROUTED;
        } else if (val == "DEVICE_ROUTED") {
            return ::intel_npu::npuw::llm::MoEHint::DEVICE_ROUTED;
        }
        OPENVINO_THROW("Unsupported MoE hint provided: ", val);
        return {};
    }

    static std::string toString(const ::intel_npu::npuw::llm::MoEHint& val) {
        switch (val) {
        case ::intel_npu::npuw::llm::MoEHint::DENSE:
            return "DENSE";
        case ::intel_npu::npuw::llm::MoEHint::HOST_ROUTED:
            return "HOST_ROUTED";
        case ::intel_npu::npuw::llm::MoEHint::DEVICE_ROUTED:
            return "DEVICE_ROUTED";
        default:
            OPENVINO_THROW("Can't convert provided MoE hint : ", int(val), " to string.");
        }
        return {};
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return false;
    }
};

struct NPUW_LLM_PREFILL_MOE_HINT final : MOE_HINT_BASE {
    static std::string_view key() {
        return ov::intel_npu::npuw::llm::prefill_moe_hint.name();
    }
};

struct NPUW_LLM_GENERATE_MOE_HINT final : MOE_HINT_BASE {
    static std::string_view key() {
        return ov::intel_npu::npuw::llm::generate_moe_hint.name();
    }
};

struct NPUW_LLM_GENERATE_HINT final : OptionBase<NPUW_LLM_GENERATE_HINT, ::intel_npu::npuw::llm::GenerateHint> {
    static std::string_view key() {
        return ov::intel_npu::npuw::llm::generate_hint.name();
    }

    static constexpr std::string_view getTypeName() {
        return "::intel_npu::npuw::llm::GenerateHint";
    }

    static ::intel_npu::npuw::llm::GenerateHint defaultValue() {
        return ::intel_npu::npuw::llm::GenerateHint::FAST_COMPILE;
    }

    static ::intel_npu::npuw::llm::GenerateHint parse(std::string_view val) {
        ::intel_npu::npuw::llm::GenerateHint res;

        if (val == "FAST_COMPILE") {
            res = ::intel_npu::npuw::llm::GenerateHint::FAST_COMPILE;
        } else if (val == "BEST_PERF") {
            res = ::intel_npu::npuw::llm::GenerateHint::BEST_PERF;
        } else {
            OPENVINO_THROW("Unsupported \"GENERATE_HINT\" provided: ",
                           val,
                           ". Please select either \"FAST_COMPILE\" or \"BEST_PERF\".");
        }
        return res;
    }

    static std::string toString(const ::intel_npu::npuw::llm::GenerateHint& val) {
        std::string res;
        switch (val) {
        case ::intel_npu::npuw::llm::GenerateHint::FAST_COMPILE:
            res = "FAST_COMPILE";
            break;
        case ::intel_npu::npuw::llm::GenerateHint::BEST_PERF:
            res = "BEST_PERF";
            break;
        default:
            OPENVINO_THROW("Can't convert provided \"GENERATE_HINT\" : ", int(val), " to string.");
        }
        return res;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return false;
    }
};
}  // namespace intel_npu

// Single-source NPUW option inventory for registration and property publication.
// Format:
//   APPLY(EMIT, option_type, group, surface, caching, build)
// where:
//   group   = ROOT | LLM | KOKORO
//   surface = EXPOSED | HIDDEN
//   caching = CACHED | UNCACHED
//   build   = ALL | DEV
#define INTEL_NPU_FOR_EACH_NPUW_OPTION(APPLY, EMIT)                                                       \
    APPLY(EMIT, NPU_USE_NPUW, ROOT, EXPOSED, CACHED, ALL)                                                 \
    APPLY(EMIT, NPUW_DEVICES, ROOT, EXPOSED, CACHED, ALL)                                                 \
    APPLY(EMIT, NPUW_SUBMODEL_DEVICE, ROOT, EXPOSED, CACHED, ALL)                                         \
    APPLY(EMIT, NPUW_ONLINE_PIPELINE, ROOT, EXPOSED, CACHED, ALL)                                         \
    APPLY(EMIT, NPUW_ONLINE_AVOID, ROOT, EXPOSED, CACHED, ALL)                                            \
    APPLY(EMIT, NPUW_ONLINE_ISOLATE, ROOT, EXPOSED, CACHED, ALL)                                          \
    APPLY(EMIT, NPUW_ONLINE_NO_FOLD, ROOT, EXPOSED, CACHED, ALL)                                          \
    APPLY(EMIT, NPUW_ONLINE_MIN_SIZE, ROOT, EXPOSED, CACHED, ALL)                                         \
    APPLY(EMIT, NPUW_ONLINE_KEEP_BLOCKS, ROOT, EXPOSED, CACHED, ALL)                                      \
    APPLY(EMIT, NPUW_ONLINE_KEEP_BLOCK_SIZE, ROOT, EXPOSED, CACHED, ALL)                                  \
    APPLY(EMIT, NPUW_ONLINE_DUMP_PLAN, ROOT, HIDDEN, UNCACHED, ALL)                                       \
    APPLY(EMIT, NPUW_PLAN, ROOT, HIDDEN, UNCACHED, ALL)                                                   \
    APPLY(EMIT, NPUW_FOLD, ROOT, EXPOSED, CACHED, ALL)                                                    \
    APPLY(EMIT, NPUW_CWAI, ROOT, EXPOSED, CACHED, ALL)                                                    \
    APPLY(EMIT, NPUW_DQ, ROOT, EXPOSED, CACHED, ALL)                                                      \
    APPLY(EMIT, NPUW_DQ_FULL, ROOT, EXPOSED, CACHED, ALL)                                                 \
    APPLY(EMIT, NPUW_PMM, ROOT, EXPOSED, CACHED, ALL)                                                     \
    APPLY(EMIT, NPUW_MM_GATED, ROOT, EXPOSED, CACHED, ALL)                                                \
    APPLY(EMIT, NPUW_SLICE_OUT, ROOT, EXPOSED, CACHED, ALL)                                               \
    APPLY(EMIT, NPUW_SPATIAL, ROOT, EXPOSED, CACHED, ALL)                                                 \
    APPLY(EMIT, NPUW_SPATIAL_NWAY, ROOT, EXPOSED, CACHED, ALL)                                            \
    APPLY(EMIT, NPUW_SPATIAL_DYN, ROOT, EXPOSED, CACHED, ALL)                                             \
    APPLY(EMIT, NPUW_MOE_TOKEN_CHUNK_SIZE, ROOT, EXPOSED, CACHED, ALL)                                    \
    APPLY(EMIT, NPUW_MOE_POOL_SIZE, ROOT, EXPOSED, CACHED, ALL)                                           \
    APPLY(EMIT, NPUW_ATTN, ROOT, EXPOSED, CACHED, ALL)                                                    \
    APPLY(EMIT, NPUW_ATTN_DYN, ROOT, HIDDEN, UNCACHED, ALL)                                               \
    APPLY(EMIT, NPUW_ATTN_NO_COPY, ROOT, HIDDEN, UNCACHED, ALL)                                           \
    APPLY(EMIT, NPUW_ATTN_HFA_FUSED, ROOT, EXPOSED, CACHED, ALL)                                          \
    APPLY(EMIT, NPUW_HOST_GATHER, ROOT, EXPOSED, CACHED, ALL)                                             \
    APPLY(EMIT, NPUW_F16IC, ROOT, EXPOSED, CACHED, ALL)                                                   \
    APPLY(EMIT, NPUW_DCOFF_TYPE, ROOT, EXPOSED, CACHED, ALL)                                              \
    APPLY(EMIT, NPUW_DCOFF_SCALE, ROOT, EXPOSED, CACHED, ALL)                                             \
    APPLY(EMIT, NPUW_FUNCALL_FOR_ALL, ROOT, EXPOSED, CACHED, ALL)                                         \
    APPLY(EMIT, NPUW_PARALLEL_COMPILE, ROOT, HIDDEN, UNCACHED, ALL)                                       \
    APPLY(EMIT, NPUW_WEIGHTS_BANK, ROOT, EXPOSED, CACHED, ALL)                                            \
    APPLY(EMIT, NPUW_WEIGHTS_BANK_ALLOC, ROOT, EXPOSED, CACHED, ALL)                                      \
    APPLY(EMIT, NPUW_CACHE_DIR, ROOT, HIDDEN, UNCACHED, ALL)                                              \
    APPLY(EMIT, NPUW_FUNCALL_ASYNC, ROOT, EXPOSED, CACHED, ALL)                                           \
    APPLY(EMIT, NPUW_UNFOLD_IREQS, ROOT, EXPOSED, CACHED, ALL)                                            \
    APPLY(EMIT, NPUW_FALLBACK_EXEC, ROOT, EXPOSED, CACHED, ALL)                                           \
    APPLY(EMIT, NPUW_ACC_CHECK, ROOT, HIDDEN, UNCACHED, ALL)                                              \
    APPLY(EMIT, NPUW_ACC_THRESH, ROOT, HIDDEN, UNCACHED, ALL)                                             \
    APPLY(EMIT, NPUW_ACC_DEVICE, ROOT, HIDDEN, UNCACHED, ALL)                                             \
    APPLY(EMIT, NPUW_DUMP_FULL, ROOT, HIDDEN, UNCACHED, DEV)                                              \
    APPLY(EMIT, NPUW_DUMP_SUBS, ROOT, HIDDEN, UNCACHED, DEV)                                              \
    APPLY(EMIT, NPUW_DUMP_SUBS_DIR, ROOT, HIDDEN, UNCACHED, DEV)                                          \
    APPLY(EMIT, NPUW_DUMP_SUBS_ON_FAIL, ROOT, HIDDEN, UNCACHED, DEV)                                      \
    APPLY(EMIT, NPUW_DUMP_IO, ROOT, HIDDEN, UNCACHED, DEV)                                                \
    APPLY(EMIT, NPUW_DUMP_IO_ITERS, ROOT, HIDDEN, UNCACHED, DEV)                                          \
    APPLY(EMIT, NPUW_LLM, LLM, EXPOSED, CACHED, ALL)                                                      \
    APPLY(EMIT, NPUW_LLM_BATCH_DIM, LLM, EXPOSED, CACHED, ALL)                                            \
    APPLY(EMIT, NPUW_LLM_SEQ_LEN_DIM, LLM, EXPOSED, CACHED, ALL)                                          \
    APPLY(EMIT, NPUW_LLM_MAX_PROMPT_LEN, LLM, EXPOSED, CACHED, ALL)                                       \
    APPLY(EMIT, NPUW_LLM_MIN_RESPONSE_LEN, LLM, EXPOSED, CACHED, ALL)                                     \
    APPLY(EMIT, NPUW_LLM_MAX_LORA_RANK, LLM, EXPOSED, CACHED, ALL)                                        \
    APPLY(EMIT, NPUW_LLM_OPTIMIZE_V_TENSORS, LLM, EXPOSED, CACHED, ALL)                                   \
    APPLY(EMIT, NPUW_LLM_OPTIMIZE_FP8, LLM, EXPOSED, CACHED, ALL)                                         \
    APPLY(EMIT, NPUW_LLM_CACHE_ROPE, LLM, EXPOSED, CACHED, ALL)                                           \
    APPLY(EMIT, NPUW_LLM_PREFILL_MOE_HINT, LLM, EXPOSED, CACHED, ALL)                                     \
    APPLY(EMIT, NPUW_LLM_GENERATE_MOE_HINT, LLM, EXPOSED, CACHED, ALL)                                    \
    APPLY(EMIT, NPUW_LLM_GENERATE_PYRAMID, LLM, EXPOSED, CACHED, ALL)                                     \
    APPLY(EMIT, NPUW_LLM_PREFILL_CHUNK_SIZE, LLM, EXPOSED, CACHED, ALL)                                   \
    APPLY(EMIT, NPUW_LLM_ENABLE_PREFIX_CACHING, LLM, EXPOSED, CACHED, ALL)                                \
    APPLY(EMIT, NPUW_LLM_PREFIX_CACHING_BLOCK_SIZE, LLM, EXPOSED, CACHED, ALL)                            \
    APPLY(EMIT, NPUW_LLM_PREFIX_CACHING_MAX_NUM_BLOCKS, LLM, EXPOSED, CACHED, ALL)                        \
    APPLY(EMIT, NPUW_LLM_MAX_GENERATION_TOKEN_LEN, LLM, EXPOSED, CACHED, ALL)                             \
    APPLY(EMIT, NPUW_LLM_PREFILL_HINT, LLM, EXPOSED, CACHED, ALL)                                         \
    APPLY(EMIT, NPUW_LLM_GENERATE_HINT, LLM, EXPOSED, CACHED, ALL)                                        \
    APPLY(EMIT, NPUW_LLM_PREFILL_ATTENTION_HINT, LLM, EXPOSED, CACHED, ALL)                               \
    APPLY(EMIT, NPUW_LLM_GENERATE_ATTENTION_HINT, LLM, EXPOSED, CACHED, ALL)                              \
    APPLY(EMIT, NPUW_LLM_SHARED_HEAD, LLM, EXPOSED, CACHED, ALL)                                          \
    APPLY(EMIT, NPUW_WHISPER, LLM, EXPOSED, CACHED, ALL)                                                  \
    APPLY(EMIT, NPUW_WHISPER_EOS_TOKEN, LLM, EXPOSED, UNCACHED, ALL)                                      \
    APPLY(EMIT, NPUW_EAGLE, LLM, EXPOSED, CACHED, ALL)                                                    \
    APPLY(EMIT, NPUW_TEXT_EMBED, LLM, EXPOSED, CACHED, ALL)                                               \
    APPLY(EMIT, NPUW_LLM_PREFILL_CONFIG, LLM, EXPOSED, CACHED, ALL)                                       \
    APPLY(EMIT, NPUW_LLM_ADDITIONAL_PREFILL_CONFIG, LLM, EXPOSED, CACHED, ALL)                            \
    APPLY(EMIT, NPUW_LLM_GENERATE_CONFIG, LLM, EXPOSED, CACHED, ALL)                                      \
    APPLY(EMIT, NPUW_LLM_ADDITIONAL_GENERATE_CONFIG, LLM, EXPOSED, CACHED, ALL)                           \
    APPLY(EMIT, NPUW_LLM_SHARED_LM_HEAD_CONFIG, LLM, EXPOSED, CACHED, ALL)                                \
    APPLY(EMIT, NPUW_LLM_ADDITIONAL_SHARED_LM_HEAD_CONFIG, LLM, EXPOSED, CACHED, ALL)                     \
    APPLY(EMIT, NPUW_KOKORO, KOKORO, EXPOSED, UNCACHED, ALL)                                              \
    APPLY(EMIT, NPUW_KOKORO_BLOCK_SIZE, KOKORO, EXPOSED, UNCACHED, ALL)                                   \
    APPLY(EMIT, NPUW_KOKORO_OVERLAP_SIZE, KOKORO, EXPOSED, UNCACHED, ALL)

#define INTEL_NPU_NPUW_IF_BUILD_ALL(SELECT, EMIT, OPT) SELECT(EMIT, OPT)
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
#define INTEL_NPU_NPUW_IF_BUILD_DEV(SELECT, EMIT, OPT) SELECT(EMIT, OPT)
#else
#define INTEL_NPU_NPUW_IF_BUILD_DEV(SELECT, EMIT, OPT)
#endif

#define INTEL_NPU_NPUW_IF_GROUP_ROOT_ROOT(EMIT, OPT) EMIT(OPT)
#define INTEL_NPU_NPUW_IF_GROUP_ROOT_LLM(EMIT, OPT)
#define INTEL_NPU_NPUW_IF_GROUP_ROOT_KOKORO(EMIT, OPT)
#define INTEL_NPU_NPUW_IF_GROUP_LLM_ROOT(EMIT, OPT)
#define INTEL_NPU_NPUW_IF_GROUP_LLM_LLM(EMIT, OPT) EMIT(OPT)
#define INTEL_NPU_NPUW_IF_GROUP_LLM_KOKORO(EMIT, OPT)
#define INTEL_NPU_NPUW_IF_GROUP_KOKORO_ROOT(EMIT, OPT)
#define INTEL_NPU_NPUW_IF_GROUP_KOKORO_LLM(EMIT, OPT)
#define INTEL_NPU_NPUW_IF_GROUP_KOKORO_KOKORO(EMIT, OPT) EMIT(OPT)

#define INTEL_NPU_NPUW_IF_SURFACE_EXPOSED(EMIT, OPT) EMIT(OPT)
#define INTEL_NPU_NPUW_IF_SURFACE_HIDDEN(EMIT, OPT)
#define INTEL_NPU_NPUW_IF_CACHING_CACHED(EMIT, OPT) EMIT(OPT)
#define INTEL_NPU_NPUW_IF_CACHING_UNCACHED(EMIT, OPT)

#define INTEL_NPU_NPUW_SELECT_ROOT(EMIT, OPT, GROUP, SURFACE, CACHING, BUILD) \
    INTEL_NPU_NPUW_IF_BUILD_##BUILD(INTEL_NPU_NPUW_IF_GROUP_ROOT_##GROUP, EMIT, OPT)
#define INTEL_NPU_NPUW_SELECT_LLM(EMIT, OPT, GROUP, SURFACE, CACHING, BUILD) \
    INTEL_NPU_NPUW_IF_BUILD_##BUILD(INTEL_NPU_NPUW_IF_GROUP_LLM_##GROUP, EMIT, OPT)
#define INTEL_NPU_NPUW_SELECT_KOKORO(EMIT, OPT, GROUP, SURFACE, CACHING, BUILD) \
    INTEL_NPU_NPUW_IF_BUILD_##BUILD(INTEL_NPU_NPUW_IF_GROUP_KOKORO_##GROUP, EMIT, OPT)
#define INTEL_NPU_NPUW_SELECT_EXPOSED(EMIT, OPT, GROUP, SURFACE, CACHING, BUILD) \
    INTEL_NPU_NPUW_IF_BUILD_##BUILD(INTEL_NPU_NPUW_IF_SURFACE_##SURFACE, EMIT, OPT)
#define INTEL_NPU_NPUW_SELECT_CACHED(EMIT, OPT, GROUP, SURFACE, CACHING, BUILD) \
    INTEL_NPU_NPUW_IF_BUILD_##BUILD(INTEL_NPU_NPUW_IF_CACHING_##CACHING, EMIT, OPT)

#define INTEL_NPU_FOR_EACH_ROOT_NPUW_OPTION(EMIT) INTEL_NPU_FOR_EACH_NPUW_OPTION(INTEL_NPU_NPUW_SELECT_ROOT, EMIT)
#define INTEL_NPU_FOR_EACH_LLM_NPUW_OPTION(EMIT) INTEL_NPU_FOR_EACH_NPUW_OPTION(INTEL_NPU_NPUW_SELECT_LLM, EMIT)
#define INTEL_NPU_FOR_EACH_KOKORO_NPUW_OPTION(EMIT) \
    INTEL_NPU_FOR_EACH_NPUW_OPTION(INTEL_NPU_NPUW_SELECT_KOKORO, EMIT)
#define INTEL_NPU_FOR_EACH_EXPOSED_NPUW_OPTION(EMIT) \
    INTEL_NPU_FOR_EACH_NPUW_OPTION(INTEL_NPU_NPUW_SELECT_EXPOSED, EMIT)
#define INTEL_NPU_FOR_EACH_CACHED_NPUW_OPTION(EMIT) \
    INTEL_NPU_FOR_EACH_NPUW_OPTION(INTEL_NPU_NPUW_SELECT_CACHED, EMIT)
