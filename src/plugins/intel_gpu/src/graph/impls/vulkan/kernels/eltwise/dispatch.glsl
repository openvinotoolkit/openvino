// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if ELTWISE_PACKED_VECTOR_WIDTH == 2 || ELTWISE_PACKED_VECTOR_WIDTH == 4
#include "dispatch_packed.glsl"
#elif ELTWISE_F32_VECTOR_WIDTH == 2 || ELTWISE_F32_VECTOR_WIDTH == 4
#include "dispatch_f32_vector.glsl"
#else
#include "dispatch_scalar.glsl"
#endif
