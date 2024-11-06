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
DEFINE_OPT(NPUW_PMM, std::string, "2", npuw::partitioning::par_matmul_merge_dims, CompileTime);
DEFINE_OPT(NPUW_SLICE_OUT, bool, false, npuw::partitioning::slice_out, CompileTime);
DEFINE_OPT(NPUW_HOST_GATHER, bool, true, npuw::partitioning::host_gather, CompileTime);
DEFINE_OPT(NPUW_SPATIAL, bool, false, npuw::partitioning::spatial, CompileTime);
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
DEFINE_OPT(NPUW_ACC_CHECK, bool, false, npuw::accuracy::check, RunTime);
DEFINE_OPT(NPUW_ACC_THRESH, double, 0.01, npuw::accuracy::threshold, RunTime);
DEFINE_OPT(NPUW_ACC_DEVICE, std::string, "", npuw::accuracy::reference_device, RunTime);
DEFINE_OPT(NPUW_DUMP_FULL, bool, false, npuw::dump::full, CompileTime);
DEFINE_OPT(NPUW_DUMP_SUBS, std::string, "", npuw::dump::subgraphs, CompileTime);
DEFINE_OPT(NPUW_DUMP_SUBS_ON_FAIL, std::string, "", npuw::dump::subgraphs_on_fail, CompileTime);
DEFINE_OPT(NPUW_DUMP_IO, std::string, "", npuw::dump::inputs_outputs, RunTime);
DEFINE_OPT(NPUW_DUMP_IO_ITERS, bool, false, npuw::dump::io_iters, RunTime);
}  // namespace intel_npu
