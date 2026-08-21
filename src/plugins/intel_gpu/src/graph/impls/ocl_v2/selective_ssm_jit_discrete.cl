// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define SSM_JIT_KERNEL selective_ssm_jit_discrete
#define SSM_JIT_USE_SLM 1
#include "selective_ssm_jit_common.cl"
#undef SSM_JIT_USE_SLM
#undef SSM_JIT_KERNEL
