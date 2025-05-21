// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/npuw.hpp"

using namespace intel_npu;
using namespace ov::intel_npu;

//
// register
//

void intel_npu::registerNPUWOptions(OptionsDesc& desc) {
    desc.add<NPU_USE_NPUW>();
    desc.add<NPUW_DEVICES>();
    desc.add<NPUW_SUBMODEL_DEVICE>();
    desc.add<NPUW_ONLINE_PIPELINE>();
    desc.add<NPUW_ONLINE_AVOID>();
    desc.add<NPUW_ONLINE_ISOLATE>();
    desc.add<NPUW_ONLINE_NO_FOLD>();
    desc.add<NPUW_ONLINE_MIN_SIZE>();
    desc.add<NPUW_ONLINE_KEEP_BLOCKS>();
    desc.add<NPUW_ONLINE_KEEP_BLOCK_SIZE>();
    desc.add<NPUW_ONLINE_DUMP_PLAN>();
    desc.add<NPUW_PLAN>();
    desc.add<NPUW_FOLD>();
    desc.add<NPUW_CWAI>();
    desc.add<NPUW_DQ>();
    desc.add<NPUW_DQ_FULL>();
    desc.add<NPUW_PMM>();
    desc.add<NPUW_SLICE_OUT>();
    desc.add<NPUW_SPATIAL>();
    desc.add<NPUW_SPATIAL_NWAY>();
    desc.add<NPUW_SPATIAL_DYN>();
    desc.add<NPUW_HOST_GATHER>();
    desc.add<NPUW_F16IC>();
    desc.add<NPUW_DCOFF_TYPE>();
    desc.add<NPUW_DCOFF_SCALE>();
    desc.add<NPUW_FUNCALL_FOR_ALL>();
    desc.add<NPUW_PARALLEL_COMPILE>();
    desc.add<NPUW_FUNCALL_ASYNC>();
    desc.add<NPUW_UNFOLD_IREQS>();
    desc.add<NPUW_WEIGHTS_BANK>();
    desc.add<NPUW_WEIGHTS_BANK_ALLOC>();
    desc.add<NPUW_CACHE_DIR>();
    desc.add<NPUW_ACC_CHECK>();
    desc.add<NPUW_ACC_THRESH>();
    desc.add<NPUW_ACC_DEVICE>();
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    desc.add<NPUW_DUMP_FULL>();
    desc.add<NPUW_DUMP_SUBS>();
    desc.add<NPUW_DUMP_SUBS_ON_FAIL>();
    desc.add<NPUW_DUMP_IO>();
    desc.add<NPUW_DUMP_IO_ITERS>();
#endif
}

void intel_npu::registerNPUWLLMOptions(OptionsDesc& desc) {
    desc.add<NPUW_LLM>();
    desc.add<NPUW_LLM_BATCH_DIM>();
    desc.add<NPUW_LLM_SEQ_LEN_DIM>();
    desc.add<NPUW_LLM_MAX_PROMPT_LEN>();
    desc.add<NPUW_LLM_MIN_RESPONSE_LEN>();
    desc.add<NPUW_LLM_OPTIMIZE_V_TENSORS>();
    desc.add<NPUW_LLM_PREFILL_HINT>();
    desc.add<NPUW_LLM_GENERATE_HINT>();
}
