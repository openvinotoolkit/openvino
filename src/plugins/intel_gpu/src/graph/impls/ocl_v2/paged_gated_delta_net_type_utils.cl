// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/bf16_utils.cl"

// bf16 arrives as ushort, so the storage type is selected by (TYPE_SIZE, IS_FP).
#define PGDN_KT_4_1          F32
#define PGDN_KT_2_1          F16
#define PGDN_KT_2_0          BF16
#define PGDN_KT(size, is_fp) CAT(PGDN_KT_, CAT(size, CAT(_, is_fp)))
#define Q_KT PGDN_KT(INPUT0_TYPE_SIZE, INPUT0_IS_FP)
#define K_KT PGDN_KT(INPUT1_TYPE_SIZE, INPUT1_IS_FP)
#define V_KT PGDN_KT(INPUT2_TYPE_SIZE, INPUT2_IS_FP)
#define S_KT PGDN_KT(INPUT3_TYPE_SIZE, INPUT3_IS_FP)
#define G_KT PGDN_KT(INPUT4_TYPE_SIZE, INPUT4_IS_FP)
#define B_KT PGDN_KT(INPUT5_TYPE_SIZE, INPUT5_IS_FP)
#define O_KT PGDN_KT(OUTPUT_TYPE_SIZE, OUTPUT_IS_FP)

#define PGDN_DECV_F16(v, n)  CAT(convert_float, n)(v)
#define PGDN_DECV_F32(v, n)  (v)
#define PGDN_DECV_BF16(v, n) CONVERT_AS_BFLOAT16_FLOAT(v, n)
#define PGDN_DECV(KT, v, n)  CAT(PGDN_DECV_, KT)(v, n)

#define PGDN_ENCV_F16(v, n)  CAT(convert_half, n)(v)
#define PGDN_ENCV_F32(v, n)  (v)
#define PGDN_ENCV_BF16(v, n) CONVERT_BFLOAT16_AS_USHORT(v, n)
#define PGDN_ENCV(KT, v, n)  CAT(PGDN_ENCV_, KT)(v, n)

#define PGDN_SDEC_F16(x)  convert_float(x)
#define PGDN_SDEC_F32(x)  (x)
#define PGDN_SDEC_BF16(x) _convert_as_bfloat16_float(x)
#define PGDN_SENC_F16(f)  convert_half(f)
#define PGDN_SENC_F32(f)  (f)
#define PGDN_SENC_BF16(f) _convert_bfloat16_as_ushort(f)
#define PGDN_SDEC(KT, x)  CAT(PGDN_SDEC_, KT)(x)
#define PGDN_SENC(KT, f)  CAT(PGDN_SENC_, KT)(f)
