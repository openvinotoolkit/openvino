// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluation_base.glsl"

#if ELTWISE_FUSED_POST_OP
#include "post_operations.glsl"
#elif ELTWISE_FUSED
#include "fused_evaluation.glsl"
#elif ELTWISE_FUSED_CHAIN
#include "fused_chain_evaluation.glsl"
#endif
