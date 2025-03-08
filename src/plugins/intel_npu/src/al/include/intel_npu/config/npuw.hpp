// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

#include "common.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/npuw_private_properties.hpp"

namespace intel_npu {

//
// register
//

void registerNPUWOptions(OptionsDesc& desc);
void registerNPUWLLMOptions(OptionsDesc& desc);

#define DEFINE_OPT(Name, Type, DefaultValue, PropertyKey, Mode)                     \
    struct Name final : OptionBase<Name, Type> {                                    \
        static std::string_view key() { return ov::intel_npu::PropertyKey.name(); } \
                                                                                    \
        static Type defaultValue() { return DefaultValue; }                         \
                                                                                    \
        static OptionMode mode() { return OptionMode::Mode; }                       \
    };

DEFINE_OPT(NPU_USE_NPUW, bool, false, use_npuw, CompileTime);
DEFINE_OPT(NPUW_DEVICES, std::string, "NPU,CPU", npuw::devices, CompileTime);
DEFINE_OPT(NPUW_SUBMODEL_DEVICE, std::string, "", npuw::submodel_device, CompileTime);
DEFINE_OPT(NPUW_ONLINE_PIPELINE, std::string, "REG", npuw::partitioning::online::pipeline, CompileTime);
DEFINE_OPT(NPUW_ONLINE_AVOID, std::string, "", npuw::partitioning::online::avoid, CompileTime);
DEFINE_OPT(NPUW_ONLINE_ISOLATE, std::string, "", npuw::partitioning::online::isolate, CompileTime);
DEFINE_OPT(NPUW_ONLINE_NO_FOLD, std::string, "", npuw::partitioning::online::nofold, CompileTime);
DEFINE_OPT(NPUW_ONLINE_MIN_SIZE, std::size_t, 10, npuw::partitioning::online::min_size, CompileTime);
DEFINE_OPT(NPUW_ONLINE_KEEP_BLOCKS, std::size_t, 5, npuw::partitioning::online::keep_blocks, CompileTime);
DEFINE_OPT(NPUW_ONLINE_KEEP_BLOCK_SIZE, std::size_t, 10, npuw::partitioning::online::keep_block_size, CompileTime);
DEFINE_OPT(NPUW_ONLINE_DUMP_PLAN, std::string, "", npuw::partitioning::online::dump_plan, CompileTime);
DEFINE_OPT(NPUW_PLAN, std::string, "", npuw::partitioning::plan, CompileTime);
DEFINE_OPT(NPUW_FOLD, bool, false, npuw::partitioning::fold, CompileTime);
DEFINE_OPT(NPUW_CWAI, bool, false, npuw::partitioning::cwai, CompileTime);
DEFINE_OPT(NPUW_DQ, bool, false, npuw::partitioning::dyn_quant, CompileTime);
DEFINE_OPT(NPUW_DQ_FULL, bool, true, npuw::partitioning::dyn_quant_full, CompileTime);
DEFINE_OPT(NPUW_PMM, std::string, "2", npuw::partitioning::par_matmul_merge_dims, CompileTime);
DEFINE_OPT(NPUW_SLICE_OUT, bool, false, npuw::partitioning::slice_out, CompileTime);
DEFINE_OPT(NPUW_HOST_GATHER, bool, true, npuw::partitioning::host_gather, CompileTime);
DEFINE_OPT(NPUW_SPATIAL, bool, false, npuw::partitioning::spatial, CompileTime);
DEFINE_OPT(NPUW_F16IC, bool, false, npuw::partitioning::f16_interconnect, CompileTime);
DEFINE_OPT(NPUW_SPATIAL_NWAY, std::size_t, 128, npuw::partitioning::spatial_nway, CompileTime);
DEFINE_OPT(NPUW_SPATIAL_DYN, bool, true, npuw::partitioning::spatial_dyn, CompileTime);
DEFINE_OPT(NPUW_DCOFF_TYPE, std::string, "", npuw::partitioning::dcoff_type, CompileTime);
DEFINE_OPT(NPUW_DCOFF_SCALE, bool, false, npuw::partitioning::dcoff_with_scale, CompileTime);
DEFINE_OPT(NPUW_FUNCALL_FOR_ALL, bool, false, npuw::partitioning::funcall_for_all, CompileTime);
DEFINE_OPT(NPUW_PARALLEL_COMPILE, bool, false, npuw::parallel_compilation, CompileTime);
DEFINE_OPT(NPUW_WEIGHTS_BANK, std::string, "", npuw::weights_bank, CompileTime);
DEFINE_OPT(NPUW_WEIGHTS_BANK_ALLOC, std::string, "", npuw::weights_bank_alloc, CompileTime);
DEFINE_OPT(NPUW_CACHE_DIR, std::string, "", npuw::cache_dir, CompileTime);
DEFINE_OPT(NPUW_FUNCALL_ASYNC, bool, false, npuw::funcall_async, RunTime);
DEFINE_OPT(NPUW_UNFOLD_IREQS, bool, false, npuw::unfold_ireqs, RunTime);
DEFINE_OPT(NPUW_ACC_CHECK, bool, false, npuw::accuracy::check, RunTime);
DEFINE_OPT(NPUW_ACC_THRESH, double, 0.01, npuw::accuracy::threshold, RunTime);
DEFINE_OPT(NPUW_ACC_DEVICE, std::string, "", npuw::accuracy::reference_device, RunTime);
DEFINE_OPT(NPUW_DUMP_FULL, bool, false, npuw::dump::full, CompileTime);
DEFINE_OPT(NPUW_DUMP_SUBS, std::string, "", npuw::dump::subgraphs, CompileTime);
DEFINE_OPT(NPUW_DUMP_SUBS_ON_FAIL, std::string, "", npuw::dump::subgraphs_on_fail, CompileTime);
DEFINE_OPT(NPUW_DUMP_IO, std::string, "", npuw::dump::inputs_outputs, RunTime);
DEFINE_OPT(NPUW_DUMP_IO_ITERS, bool, false, npuw::dump::io_iters, RunTime);
DEFINE_OPT(NPUW_LLM, bool, false, npuw::llm::enabled, CompileTime);
DEFINE_OPT(NPUW_LLM_BATCH_DIM, uint32_t, 0, npuw::llm::batch_dim, CompileTime);
DEFINE_OPT(NPUW_LLM_SEQ_LEN_DIM, uint32_t, 2, npuw::llm::seq_len_dim, CompileTime);
DEFINE_OPT(NPUW_LLM_MAX_PROMPT_LEN, uint32_t, 1024, npuw::llm::max_prompt_len, CompileTime);
DEFINE_OPT(NPUW_LLM_MIN_RESPONSE_LEN, uint32_t, 128, npuw::llm::min_response_len, CompileTime);
DEFINE_OPT(NPUW_LLM_OPTIMIZE_V_TENSORS, bool, true, npuw::llm::optimize_v_tensors, CompileTime);

namespace npuw {
namespace llm {
enum class PrefillHint { DYNAMIC, STATIC };
enum class GenerateHint { FAST_COMPILE, BEST_PERF };
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
        return ::intel_npu::npuw::llm::PrefillHint::STATIC;
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
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
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
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }
};
}  // namespace intel_npu
