// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define SSM_JIT_KERNEL paged_selective_ssm_jit_integrated
#define SSM_JIT_USE_SLM 0
#include "paged_selective_ssm_jit_common.cl"
#undef SSM_JIT_USE_SLM
#undef SSM_JIT_KERNEL
