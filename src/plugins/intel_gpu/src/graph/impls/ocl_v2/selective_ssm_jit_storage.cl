// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Shared data conversion and recurrence-state storage policy. The device-specific
// entrypoint or its JIT constants select private memory or SLM before including it.
#if INPUT0_IS_FP
#    define SSM_TO_FLOAT(v) convert_float(v)
#else
#    define SSM_TO_FLOAT(v) _convert_as_bfloat16_float(v)
#endif

#if SSM_JIT_USE_SLM
#    define SSM_STATE_AT(p_offset, step, state_element) slm_state[(p_offset) * SSM_STATE_SIZE + (state_element)]
#else
#    define SSM_STATE_AT(p_offset, step, state_element) private_state[p_offset][step]
#endif
