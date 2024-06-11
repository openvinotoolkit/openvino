// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/al/config/npuw.hpp"

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
    desc.add<NPUW_ONLINE_MIN_SIZE>();
    desc.add<NPUW_ONLINE_DUMP_PLAN>();
    desc.add<NPUW_PLAN>();
    desc.add<NPUW_FOLD>();
    desc.add<NPUW_CWAI>();
    desc.add<NPUW_DCOFF_TYPE>();
    desc.add<NPUW_DCOFF_SCALE>();
    desc.add<NPUW_FUNCALL_FOR_ALL>();
    desc.add<NPUW_PARALLEL_COMPILE>();
    desc.add<NPUW_FUNCALL_ASYNC>();
    desc.add<NPUW_ACC_CHECK>();
    desc.add<NPUW_ACC_THRESH>();
    desc.add<NPUW_ACC_DEVICE>();
    desc.add<NPUW_DUMP_FULL>();
    desc.add<NPUW_DUMP_SUBS>();
    desc.add<NPUW_DUMP_SUBS_ON_FAIL>();
    desc.add<NPUW_DUMP_IO>();
    desc.add<NPUW_DUMP_IO_ITERS>();
}
